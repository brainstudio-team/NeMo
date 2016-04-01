#ifndef NEMO_CUDA_STDP_CU
#define NEMO_CUDA_STDP_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file stdp.cu STDP update kernel */

#include "log.cu_h"
#include "device_assert.cu"
#include "parameters.cu"
#include "rcm.cu"


/* STDP parameters
 *
 * The STDP parameters apply to all neurons in a network. One might, however,
 * want to do this differently for different neuron populations. This is not
 * yet supported.
 */


/* The STDP window is separated into two distinct region: potentiation and
 * depression. In the common assymetrical case the pre-fire part of the STDP
 * window will be potentiation, while the post-fire part of the window will be
 * depression. Other schemes are supported, however. */

/*! The STDP parameters are stored in constant memory, and is loaded into shared
 * memory during execution. \a configureStdp should be called before simulation
 * starts, and \a loadStdpParameters should be called at the beginning of the
 * kernel */


/* Mask of cycles within the STDP for which we consider potentiating synapses */
__constant__ uint64_t c_stdpPotentiation;

/* Mask of cycles within the STDP for which we consider depressing synapses */
__constant__ uint64_t c_stdpDepression;

/* The STDP function sampled at integer cycle points within the STDP window */
__constant__ weight_dt c_stdpFn[STDP_WINDOW_SIZE];

/* Length of the window (in cycles) which is post firing */
__constant__ unsigned c_stdpPostFireWindow;

/* Length of the window (in cycles) which is pre firing */
__constant__ unsigned c_stdpPreFireWindow;

__constant__ unsigned c_stdpWindow;


__shared__ uint64_t s_stdpPotentiation;
__shared__ uint64_t s_stdpDepression;
__shared__ weight_dt s_stdpFn[STDP_WINDOW_SIZE];
__shared__ unsigned s_stdpPostFireWindow;
__shared__ unsigned s_stdpPreFireWindow;


#define SET_STDP_PARAMETER(symbol, val) {                                      \
        cudaError status;                                                      \
        status = cudaMemcpyToSymbol(symbol, &val, sizeof(val),                 \
                                    0, cudaMemcpyHostToDevice);                \
        if(cudaSuccess != status) {                                            \
            return status;                                                     \
        }                                                                      \
    }


__host__
cudaError
configureStdp(
		unsigned preFireWindow,
		unsigned postFireWindow,
		uint64_t potentiationBits, // remainder are depression
		uint64_t depressionBits,   // remainder are depression
		weight_dt* stdpFn)
{
	SET_STDP_PARAMETER(c_stdpPreFireWindow, preFireWindow);
	SET_STDP_PARAMETER(c_stdpPostFireWindow, postFireWindow);
	SET_STDP_PARAMETER(c_stdpWindow, preFireWindow + postFireWindow);
	SET_STDP_PARAMETER(c_stdpPotentiation, potentiationBits);
	SET_STDP_PARAMETER(c_stdpDepression, depressionBits);
	unsigned window = preFireWindow + postFireWindow;
	assert(window <= STDP_WINDOW_SIZE);
	return cudaMemcpyToSymbol(c_stdpFn, stdpFn, sizeof(weight_dt)*window, 0, cudaMemcpyHostToDevice);
}



/* In the kernel we load the parameters into shared memory. These variables can
 * then be accessed using broadcast */



#define LOAD_STDP_PARAMETER(symbol) s_ ## symbol = c_ ## symbol


__device__
void
loadStdpParameters_()
{
    if(threadIdx.x == 0) {
        LOAD_STDP_PARAMETER(stdpPotentiation);
        LOAD_STDP_PARAMETER(stdpDepression);
        LOAD_STDP_PARAMETER(stdpPreFireWindow);
        LOAD_STDP_PARAMETER(stdpPostFireWindow);
    }

	ASSERT(MAX_STDP_DELAY <= THREADS_PER_BLOCK);
	int dt = threadIdx.x;
	if(dt < STDP_WINDOW_SIZE) {
		s_stdpFn[dt] = c_stdpFn[dt];
	}
    __syncthreads();
}





/* Update a single synapse according to STDP rule
 *
 * Generally, the synapse is potentiated if the postsynaptic neuron fired
 * shortly after the spike arrived. Conversely, the synapse is depressed if the
 * postsynaptic neuron fired shortly before the spike arrived.
 *
 * Both potentiation and depression is applied with some delay after the neuron
 * actually fired. This is so that all the relevant history has taken place.
 *
 * We determine which synapses are potentiated and which are depressed by
 * inspecting the recent firing history of both the pre and post-synaptic
 * neuron. Consider the firing history at the presynatic neuron:
 *
 *    |---P---||---D---||--delay--|
 * XXXPPPPPPPPPDDDDDDDDDFFFFFFFFFFF
 * 31      23      15      7      0
 *
 * where
 *	X: cycles not of interest as spikes would have reached postsynaptic outside
 *	   the STDP window 
 *  D: (D)epressing spikes which would have reached after postsynaptic firing
 *  P: (P)otentiating spikes which would have reached before postsynaptic firing
 *	F: In-(F)light spikes which have not yet reached.
 *
 * Within the P spikes (if any) we only consider the *last* spike. Within the D
 * spikes (if any) we only consider the *first* spike.
 */


#define STDP_NO_APPLICATION (~0)


/*! \return
 *		shortest delay between spike arrival and firing of this neuron or
 *		largest representable delay if STDP not appliccable.
 *
 * STDP is not applicable if the postsynaptic neuron also fired closer to the
 * incoming spike than the firing currently under consideration. */
__device__
unsigned
closestPreFire(uint64_t spikes)
{
	int dt =  __ffsll(spikes >> s_stdpPostFireWindow);
	return dt ? (unsigned) dt-1 : STDP_NO_APPLICATION;
}



__device__
unsigned
closestPostFire(uint64_t spikes)
{
	int dt = __clzll(spikes << (64 - s_stdpPostFireWindow));
	return spikes ? (unsigned) dt : STDP_NO_APPLICATION;
}



#if NEMO_CUDA_DEBUG_TRACE >= 0x8

__device__
void
logStdp(int dt, weight_dt w_diff, unsigned targetNeuron, rsynapse_t r_synapse)
{
	const char* type[] = { "ltd", "ltp" };

	if(w_diff != 0) {
		// cuPrintf is limited to ten arguments, so split up the printing here
		DEBUG_MSG_STDP("c%u %s: %u-%u -> %u-%u %+f ",
				s_cycle, type[size_t(w_diff > 0)],
				sourcePartition(r_synapse), sourceNeuron(r_synapse),
				CURRENT_PARTITION, targetNeuron,
				fx_tofloat(w_diff));
		DEBUG_MSG_STDP("(dt=%d, delay=%u, prefire@%u, postfire@%u)\n",
				dt, r_delay1(r_synapse),
				s_cycle - s_stdpPostFireWindow + dt,
				s_cycle - s_stdpPostFireWindow);
	}
}

#endif


__device__
weight_dt
updateRegion(
		uint64_t spikes,
		unsigned targetNeuron,
		rsynapse_t r_synapse) // used for logging only
{
	/* The potentiation can happen on either side of the firing. We want to
	 * find the one closest to the firing. We therefore need to compute the
	 * prefire and postfire dt's separately. */
	unsigned dt_pre = closestPreFire(spikes);
	unsigned dt_post = closestPostFire(spikes);

	/* For logging. Positive values: post-fire, negative values: pre-fire */
#if NEMO_CUDA_DEBUG_TRACE >= 0x8
	int dt_log;
#endif

	weight_dt w_diff = 0;
	if(spikes) {
		if(dt_pre < dt_post) {
			w_diff = s_stdpFn[s_stdpPreFireWindow - 1 - dt_pre];
#if NEMO_CUDA_DEBUG_TRACE >= 0x8
			dt_log = -int(dt_pre);
#endif
		} else if(dt_post < dt_pre) {
			w_diff = s_stdpFn[s_stdpPreFireWindow+dt_post];
#if NEMO_CUDA_DEBUG_TRACE >= 0x8
			dt_log = int(dt_post);
#endif
		}
		// if neither is applicable dt_post == dt_pre
	}
#if NEMO_CUDA_DEBUG_TRACE >= 0x8
	logStdp(dt_log, w_diff, targetNeuron, r_synapse);
#endif
	return w_diff;
}



/*! Update a synapse according to the user-specified STDP function. Both
 * potentiation and depression takes place.
 *
 * \return weight modifcation (additive term)
 */
__device__
weight_dt
updateSynapse(
		rsynapse_t r_synapse,
		unsigned targetNeuron,
		uint64_t* g_sourceFiring)
{
	int inFlight = r_delay0(r_synapse);
	/* -1 since we do spike arrival before neuron-update and STDP in a single
	 * simulation cycle */

	uint64_t sourceFiring = g_sourceFiring[sourceNeuron(r_synapse)] >> inFlight;
	uint64_t p_spikes = sourceFiring & s_stdpPotentiation;
	uint64_t d_spikes = sourceFiring & s_stdpDepression;

	return updateRegion(p_spikes, targetNeuron, r_synapse)
           + updateRegion(d_spikes, targetNeuron, r_synapse);
}



/*! Update STDP statistics for all neurons */
__device__
void
updateSTDP_(
	uint32_t cycle,
	uint32_t* s_dfired,
	uint64_t* g_recentFiring,
	const param_t& s_params,
	unsigned partitionSize,
	rcm_dt& g_rcm,
	nidx_dt* s_firingIdx) // s_NIdx, so can handle /all/ neurons firing
{
	/* Determine what postsynaptic neurons needs processing in small batches */
	for(unsigned nbase = 0; nbase < partitionSize; nbase += THREADS_PER_BLOCK) {
		unsigned target = nbase + threadIdx.x;

		__shared__ rcm_index_address_t s_rcmIndexAddress[THREADS_PER_BLOCK];

		uint64_t targetRecentFiring =
			g_recentFiring[(readBuffer(cycle) * PARTITION_COUNT + CURRENT_PARTITION) * s_params.pitch64 + target];

		const int processingDelay = s_stdpPostFireWindow - 1;

		bool fired = targetRecentFiring & (0x1 << processingDelay);

		/* Write updated history to double buffer */
		g_recentFiring[(writeBuffer(cycle) * PARTITION_COUNT + CURRENT_PARTITION) * s_params.pitch64 + target] =
				(targetRecentFiring << 1) | (bv_isSet(target, s_dfired) ? 0x1 : 0x0);

		__shared__ unsigned s_firingCount;
		if(threadIdx.x == 0) {
			s_firingCount = 0;
		}
		__syncthreads();

		if(fired && target < partitionSize) {
			unsigned i = atomicAdd(&s_firingCount, 1);
			s_firingIdx[i] = target;
			s_rcmIndexAddress[i] = rcm_indexAddress(target, g_rcm);
		}
		__syncthreads();

		for(unsigned i=0; i<s_firingCount; ++i) {

			unsigned target = s_firingIdx[i];

			rcm_index_address_t row = s_rcmIndexAddress[i];

			for(unsigned bIndex=0 ; bIndex < rcm_indexRowLength(row);
					bIndex += THREADS_PER_BLOCK/WARP_SIZE) {
				__shared__ rcm_address_t warp[THREADS_PER_BLOCK/WARP_SIZE];

				if(threadIdx.x < THREADS_PER_BLOCK/WARP_SIZE) {
					warp[threadIdx.x] =
						rcm_address(rcm_indexRowStart(row), bIndex + threadIdx.x, g_rcm);
				}
				__syncthreads();

				size_t word = rcm_offset(warp[threadIdx.x/WARP_SIZE]);
				rsynapse_t r_sdata = g_rcm.data[word];

				if(r_sdata != INVALID_REVERSE_SYNAPSE) {

					weight_dt w_diff =
						updateSynapse(
							r_sdata,
							target,
							g_recentFiring + (readBuffer(cycle) * PARTITION_COUNT + sourcePartition(r_sdata)) * s_params.pitch64);

					//! \todo perhaps stage diff in output buffers
					//! \todo add saturating arithmetic here
					if(w_diff != 0) {
						g_rcm.accumulator[word] += w_diff;
					}
				}
			}
			__syncthreads();
		}
		__syncthreads();
	}
}



/*! Load per-neuron bit-vector for fired neurons from global memory
 *
 * \param[in] g_dfired
 *		Per-neuron bit-vector in global memory for fired neurons.
 * \param[out] s_dfired
 *		Per-neuron bit-vector in shared memory for fired neurons.
 *
 * \see storeDenseFiring
 */
__device__
void
loadDenseFiring(size_t pitch1, uint32_t* g_dfired, uint32_t* s_dfired)
{
	bv_copy(g_dfired + CURRENT_PARTITION * pitch1, s_dfired);
}



__global__
void
updateStdp(
		uint32_t cycle,
		unsigned* g_partitionSize,
		param_t* g_params,
		rcm_dt g_rcm,
		uint64_t* g_recentFiring,
		uint32_t* g_dfired,        // dense firing. pitch = c_bvPitch.
		unsigned* g_nFired,        // device-only buffer.
		nidx_dt* g_fired)          // device-only buffer, sparse output. pitch = c_pitch32.
{
	__shared__ unsigned s_nFired;
	__shared__ nidx_dt s_fired[MAX_PARTITION_SIZE];
	__shared__ uint32_t s_dfired[S_BV_PITCH];
	__shared__ param_t s_params;

	loadParameters(g_params, &s_params);

	/* If the STDP update kernel is merged with the scatter
	 * kernel, we'd only need to load this once per simulation
	 * step, rather than twice. */
	loadSparseFiring(g_nFired, s_params.pitch32, g_fired, &s_nFired, s_fired);
	loadDenseFiring(s_params.pitch1, g_dfired, s_dfired);
	loadStdpParameters_();
	updateSTDP_(
			cycle,
			s_dfired,
			g_recentFiring,
			s_params,
			g_partitionSize[CURRENT_PARTITION],
			g_rcm,
			s_fired);
}



__host__
cudaError_t
updateStdp(
		cudaStream_t stream,
		unsigned cycle,
		unsigned partitionCount,
		unsigned* d_partitionSize,
		param_t* d_parameters,
		rcm_dt* d_rcm,
		uint64_t* d_recentFiring,
		uint32_t* d_dfired,
		unsigned* d_nFired,
		nidx_dt* d_fired)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(partitionCount);
	updateStdp<<<dimGrid, dimBlock, 0, stream>>>(cycle, d_partitionSize, d_parameters, 
			*d_rcm, d_recentFiring, d_dfired, d_nFired, d_fired);
	return cudaGetLastError();
}


#endif
