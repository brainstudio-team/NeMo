/*! \file gather.cu Gather kernel */

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "types.h"

#include "log.cu_h"

#include "bitvector.cu"
#include "double_buffer.cu"
#include "fixedpoint.cu"
#include "globalQueue.cu"
#include "parameters.cu"
#include "current.cu"


/*! Gather incoming current from all spikes due for delivery \e now
 *
 * The whole spike delivery process is described in more detail in \ref
 * cuda_delivery and cuda_gather.
 *
 * \param[in] cycle
 * 		Current cycle
 * \param[in] g_fcm
 *		Forward connectivity matrix in global memory
 * \param[in] g_gqFill
 *		Fill rate for global queue
 * \param[in] g_gqData
 *		Pointer to full global memory double-buffered global queue
 * \param[out] s_current
 *		per-neuron vector with accumulated current in fixed point format.
 */
__device__
void
gather( unsigned cycle,
		const param_t& s_params,
		synapse_t* g_fcm,
		gq_entry_t* g_gqData,
		unsigned* g_gqFill,
		float sf_currentE[],
		float sf_currentI[])
{
	/* Per-neuron bit-vectors. See bitvector.cu for accessors */
	__shared__ uint32_t s_overflowE[S_BV_PITCH];
	__shared__ uint32_t s_overflowI[S_BV_PITCH];

	bv_clear(s_overflowE);
	bv_clear(s_overflowI);

	/* Accumulation is done using fixed-point, but the conversion back to
	 * floating point is done in-place. */
	fix_t* s_currentE = (fix_t*) sf_currentE;
	fix_t* s_currentI = (fix_t*) sf_currentI;

	for(int bNeuron=0; bNeuron < MAX_PARTITION_SIZE; bNeuron += THREADS_PER_BLOCK) {
		unsigned neuron = bNeuron + threadIdx.x;
		s_currentE[neuron] = 0U;
		s_currentI[neuron] = 0U;
	}
	// __syncthreads();

	__shared__ unsigned s_incomingCount;
	if(threadIdx.x == 0) {
		size_t addr = gq_fillOffset(CURRENT_PARTITION, readBuffer(cycle));
		s_incomingCount = g_gqFill[addr];
		g_gqFill[addr] = 0;
	}
	__syncthreads();

	__shared__ synapse_t* s_warpAddress[THREADS_PER_BLOCK];

	/* Process the incoming warps in fixed size groups */
	for(unsigned bGroup = 0; bGroup < s_incomingCount; bGroup += THREADS_PER_BLOCK) {

		unsigned group = bGroup + threadIdx.x;

		/* In each loop iteration we process /up to/ THREADS_PER_BLOCK warps.
		 * For the last iteration of the outer loop we process fewer */
		__shared__ unsigned s_groupSize;
		if(threadIdx.x == 0) {
			s_groupSize =
				(bGroup + THREADS_PER_BLOCK) > s_incomingCount
				? s_incomingCount % THREADS_PER_BLOCK
				: THREADS_PER_BLOCK;
			DEBUG_MSG_SYNAPSE("c%u: group size=%u, incoming=%u\n", cycle, s_groupSize, s_incomingCount);
		}
		__syncthreads();

		if(threadIdx.x < s_groupSize) {
			gq_entry_t sgin = gq_read(readBuffer(cycle), group, g_gqData);
			s_warpAddress[threadIdx.x] = g_fcm + gq_warpOffset(sgin) * WARP_SIZE;
			DEBUG_MSG_SYNAPSE("c%u w%u -> p%u\n", cycle, gq_warpOffset(sgin), CURRENT_PARTITION);
		}

		__syncthreads();

		for(unsigned gwarp_base = 0; gwarp_base < s_groupSize; gwarp_base += WARPS_PER_BLOCK) {

			unsigned bwarp = threadIdx.x / WARP_SIZE; // warp index within a block
			unsigned gwarp = gwarp_base + bwarp;      // warp index within the global schedule

			unsigned postsynaptic;
			fix_t weight = 0;

			synapse_t* base = s_warpAddress[gwarp] + threadIdx.x % WARP_SIZE;

			/* only warps at the very end of the group are invalid here */
			if(gwarp < s_groupSize) {
				postsynaptic = *base;
				ASSERT(postsynaptic < MAX_PARTITION_SIZE);
				weight = *((unsigned*)base + s_params.fcmPlaneSize);
			}

			bool excitatory = weight >= 0;
			fix_t* s_current = excitatory ? s_currentE : s_currentI;
			uint32_t* s_overflow = excitatory ? s_overflowE : s_overflowI;

			if(weight != 0) {
				bool overflow = fx_atomicAdd(s_current + postsynaptic, weight);
				bv_atomicSetPredicated(overflow, postsynaptic, s_overflow);
#ifndef NEMO_WEIGHT_FIXED_POINT_SATURATION
				ASSERT(!overflow);
#endif
				DEBUG_MSG_SYNAPSE("c%u p?n? -> p%un%u %+f [warp %u]\n",
						s_cycle, CURRENT_PARTITION, postsynaptic,
						fx_tofloat(weight), (s_warpAddress[gwarp] - g_fcm) / WARP_SIZE);
			}
		}
		__syncthreads(); // to avoid overwriting s_groupSize
	}

	fx_arrSaturatedToFloat(s_overflowE, false, s_currentE, sf_currentE, s_params.fixedPointScale);
	fx_arrSaturatedToFloat(s_overflowI,  true, s_currentI, sf_currentI, s_params.fixedPointScale);
}



__global__
void
gather( uint32_t cycle,
		unsigned* g_partitionSize,
		param_t* g_params,
		synapse_t* g_fcm,
		gq_entry_t* g_gqData,      // pitch = c_gqPitch
		unsigned* g_gqFill,
		float* g_current)
{
	__shared__ float s_currentE[MAX_PARTITION_SIZE];
	__shared__ float s_currentI[MAX_PARTITION_SIZE];

	__shared__ param_t s_params;

	/* Per-partition parameters */
	__shared__ unsigned s_partitionSize;

	if(threadIdx.x == 0) {
#ifdef NEMO_CUDA_DEBUG_TRACE
		s_cycle = cycle;
#endif
		s_partitionSize = g_partitionSize[CURRENT_PARTITION];
	} // sync done in loadParameters
	loadParameters(g_params, &s_params);

	gather(cycle, s_params, g_fcm, g_gqData, g_gqFill, s_currentE, s_currentI);

	/* Write back to global memory The global memory roundtrip is so that the
	 * gather and fire steps can be done in separate kernel invocations. */

	float* g_currentE = incomingExcitatory(g_current, PARTITION_COUNT, CURRENT_PARTITION, s_params.pitch32);
	float* g_currentI = incomingInhibitory(g_current, PARTITION_COUNT, CURRENT_PARTITION, s_params.pitch32);
	for(unsigned bNeuron=0; bNeuron < s_partitionSize; bNeuron += THREADS_PER_BLOCK) {
		unsigned neuron = bNeuron + threadIdx.x;
		g_currentE[neuron] = s_currentE[neuron];
		g_currentI[neuron] = s_currentI[neuron];
	}
}




__host__
cudaError_t
gather( cudaStream_t stream,
		unsigned cycle,
		unsigned partitionCount,
		unsigned* d_partitionSize,
		param_t* d_params,
		float* d_current,
		synapse_t* d_fcm,
		gq_entry_t* d_gqData,
		unsigned* d_gqFill)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(partitionCount);
	gather<<<dimGrid, dimBlock, 0, stream>>>(cycle, d_partitionSize, d_params, d_fcm, d_gqData, d_gqFill, d_current);
	return cudaGetLastError();
}
