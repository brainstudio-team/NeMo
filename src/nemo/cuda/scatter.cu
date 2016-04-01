#ifndef NEMO_CUDA_SCATTER_CU
#define NEMO_CUDA_SCATTER_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file scatter.cu Spike scatter kernel */

#include "localQueue.cu"
#include "parameters.cu"


/*! Load sparse firing from global memory buffer
 *
 * The global memory roundtrip is required to support having 'fire' and
 * 'scatter' in separate kernels.
 *
 * \param[in] g_nFired global memory per-partition vector of firing counts
 * \param[in] g_fired global memory per-neuron vector of fired neuron indices.
 * \param[out] s_nFired number of neurons in this partition which fired this cycle
 * \param[out] s_fired shared memory vector of the relevant neuron indices.
 * 		Only the first \a nFired entries contain valid data.
 * 
 * \see storeSparseFiring
 */
__device__
void
loadSparseFiring(unsigned* g_nFired, size_t pitch32, nidx_dt* g_fired, unsigned* s_nFired, nidx_dt* s_fired)
{
	if(threadIdx.x == 0) {
		*s_nFired = g_nFired[CURRENT_PARTITION];
	}
	__syncthreads();

	for(unsigned b=0; b < *s_nFired; b += THREADS_PER_BLOCK) {
		unsigned i = b + threadIdx.x;
		if(i < *s_nFired) {
			s_fired[i] = g_fired[CURRENT_PARTITION * pitch32 + i];
		}
	}
	__syncthreads();
}



/*! Enque all recently fired neurons in the local queue
 *
 * See the section on \ref cuda_local_delivery "local spike delivery" for more
 * details.
 *
 * \param cycle
 * \param nFired number of valid entries in s_fired
 * \param s_fired shared memory buffer containing the recently fired neurons
 * \param g_delays delay bits for each neuron
 * \param g_lqFill queue fill for local queue
 * \param g_queue global memory for the local queue
 */
__device__
void
scatterLocal(
		unsigned cycle,
		const param_t& s_params,
		unsigned nFired,
		const nidx_dt* s_fired,
		unsigned g_ndFill[],
		delay_dt g_delays[],
		unsigned g_lqFill[],
		lq_entry_t g_queue[])
{
	/* This shared memory vector is quite small, so no need to reuse */
	__shared__ unsigned s_lqFill[MAX_DELAY];

	lq_loadQueueFill(s_params.maxDelay, g_lqFill, s_lqFill);
	__syncthreads();

	/*! \todo do more than one neuron at a time. We can deal with
	 * THREADS_PER_BLOCK/MAX_DELAY per iteration. */
	for(unsigned bFired = 0; bFired < nFired; bFired+=THREADS_PER_BLOCK) {

		__shared__ unsigned s_ndFill[THREADS_PER_BLOCK];

		if(bFired + threadIdx.x < nFired) {
			/* Non-coalesced load: */
			s_ndFill[threadIdx.x] = nd_loadFill(s_fired[bFired + threadIdx.x], g_ndFill);
		}
		__syncthreads();

		for(unsigned iFired = 0;
				iFired < THREADS_PER_BLOCK && bFired + iFired < nFired;
				++iFired) {

			__shared__ delay_dt s_delays[MAX_DELAY];

			unsigned neuron = s_fired[bFired + iFired];
			unsigned ndFill = s_ndFill[iFired];

			nd_loadDelays(neuron, ndFill, g_delays, s_delays);
			__syncthreads();

			for(unsigned bDelay = 0; bDelay < ndFill; bDelay += THREADS_PER_BLOCK) {
				unsigned iDelay = bDelay + threadIdx.x;
				if(iDelay < ndFill) {
					/* This write operation will almost certainly be non-coalesced.
					 * It would be possible to stage data in smem, e.g. one warp
					 * per queue slot. 64 slots would require 64 x 32 x 4B = 8kB.
					 * Managaging this data can be costly, however, as we need to
					 * flush buffers as we go. */
					lq_enque(neuron, cycle, s_params.maxDelay, s_delays[iDelay], s_lqFill, g_queue);
					DEBUG_MSG_SYNAPSE("c%u[local scatter]: enqueue n%u d%u\n", cycle, neuron, s_delays[iDelay]+1);
				}
			}
		}
	}
	__syncthreads();
	lq_storeQueueFill(s_params.maxDelay, s_lqFill, g_lqFill);
}



/*! Echange spikes between partitions
 *
 * See the section on \ref cuda_global_delivery "global spike delivery" for
 * more details.
 */
__device__
void
scatterGlobal(unsigned cycle,
		const param_t& s_params,
		unsigned* g_lqFill,
		lq_entry_t* g_lq,
		outgoing_addr_t* g_outgoingAddr,
		outgoing_t* g_outgoing,
		unsigned* g_gqFill,
		gq_entry_t* g_gqData)
{
	__shared__ unsigned s_fill[MAX_PARTITION_COUNT]; // 512

	/* Instead of iterating over fired neurons, load all fired data from a
	 * single local queue entry. Iterate over the neuron/delay pairs stored
	 * there. */
	__shared__ unsigned s_nLq;

	if(threadIdx.x == 0) {
		s_nLq = lq_getAndClearCurrentFill(cycle, s_params.maxDelay, g_lqFill);
	}
	__syncthreads();

	for(unsigned bLq = 0; bLq < s_nLq; bLq += THREADS_PER_BLOCK) {

		unsigned iLq = bLq + threadIdx.x;

		//! \todo share this memory with other stages
#ifdef NEMO_CUDA_DEBUG_TRACE
		__shared__ lq_entry_t s_lq[THREADS_PER_BLOCK];   // 1KB
#endif
		__shared__ unsigned s_offset[THREADS_PER_BLOCK]; // 1KB
		__shared__ unsigned s_len[THREADS_PER_BLOCK];    // 1KB

		s_len[threadIdx.x] = 0;

		/* Load local queue entries (neuron/delay pairs) and the associated
		 * outgoing lengths into shared memory */
		if(iLq < s_nLq) {
			ASSERT(iLq < c_lqPitch);
			lq_entry_t entry = g_lq[lq_offset(cycle, s_params.maxDelay, 0) + iLq];
#ifdef NEMO_CUDA_DEBUG_TRACE
			s_lq[threadIdx.x] = entry;
#endif
			short delay0 = entry.y;
			ASSERT(delay0 < s_params.maxDelay);

			short neuron = entry.x;
			ASSERT(neuron < MAX_PARTITION_SIZE);

			/* Outgoing counts is cachable. It is not too large and is runtime
			 * constant. It is too large for constant memory however. The
			 * alternatives are thus texture memory or the L1 cache (on Fermi) */
			outgoing_addr_t addr = outgoingAddr(neuron, delay0, g_outgoingAddr);
			s_offset[threadIdx.x] = addr.x;
			s_len[threadIdx.x] = addr.y;
			ASSERT(s_len[threadIdx.x] <= s_params.outgoingPitch);
			DEBUG_MSG_SYNAPSE("c%u[global scatter]: dequeued n%u d%u from local queue (%u warps from %u)\n",
					cycle, neuron, delay0, s_len[threadIdx.x], s_offset[threadIdx.x]);
		}
		__syncthreads();

		/* Now loop over all the entries we just loaded from the local queue.
		 * Read a number of entries in one go, if possible. Note that a large
		 * spread in the range of outgoing row lengths (e.g. one extremely long
		 * one) will adveresly affect performance here. */
		unsigned jLqMax = min(THREADS_PER_BLOCK, s_nLq-bLq);
		for(unsigned jbLq = 0; jbLq < jLqMax; jbLq += s_params.outgoingStep) {

			/* jLq should be in [0, 256) so that we can point to s_len
			 * e.g.     0,8,16,24,...,248 + 0,1,...,8 */
			unsigned jLq = jbLq + threadIdx.x / s_params.outgoingPitch;
			ASSERT(jLq < THREADS_PER_BLOCK);

			/* There may be more than THREADS_PER_BLOCK entries in this
			 * outgoing row, although the common case should be just a single
			 * loop iteration here */
			unsigned nOut = s_len[jLq];
			if(threadIdx.x < PARTITION_COUNT) {
				s_fill[threadIdx.x] = 0;
			}
			__syncthreads();

			/* Load row of outgoing data (specific to neuron/delay pair) */
			unsigned iOut = threadIdx.x % s_params.outgoingPitch;
			unsigned targetPartition = 0;
			unsigned warpOffset = 0;
			unsigned localOffset = 0;
			bool valid = bLq + jLq < s_nLq && iOut < nOut;
			if(valid) {
				outgoing_t sout = g_outgoing[s_offset[jLq] + iOut];
				targetPartition = outgoingTargetPartition(sout);
				ASSERT(targetPartition < PARTITION_COUNT);
				warpOffset = outgoingWarpOffset(sout);
				ASSERT(warpOffset != 0);
				localOffset = atomicAdd(s_fill + targetPartition, 1);
			}
			__syncthreads();

			/* Update s_fill to store actual offset */
			if(threadIdx.x < PARTITION_COUNT) {
				size_t fillAddr = gq_fillOffset(threadIdx.x, writeBuffer(cycle));
				s_fill[threadIdx.x] = atomicAdd(g_gqFill + fillAddr, s_fill[threadIdx.x]);
			}
			__syncthreads();

			if(valid) {
				unsigned offset = s_fill[targetPartition] + localOffset;
				size_t base = gq_bufferStart(targetPartition, writeBuffer(cycle));
				ASSERT(offset < c_gqPitch);
				ASSERT(base < 2 * PARTITION_COUNT * c_gqPitch);
				g_gqData[base + offset] = warpOffset;
				DEBUG_MSG_SYNAPSE("c%u[global scatter]: enqueued warp %u (p%un%u -> p%u with d%u) to global queue (buffer entry %u/%lu)\n",
						cycle, warpOffset,
						CURRENT_PARTITION, s_lq[jLq].x, targetPartition, s_lq[jLq].y,
						offset, c_gqPitch);
				/* The writes to the global queue are non-coalesced. It would
				 * be possible to stage this data in smem for each partition.
				 * However, this would require a fair amount of smem (1), and
				 * handling buffer overflow is complex and introduces
				 * sequentiality. Overall, it's probably not worth it.
				 *
				 * (1) 128 partitions, warp-sized buffers, 4B/entry = 16KB
				 */
			}
			__syncthreads(); // to protect s_fill
		}
		__syncthreads(); // to protect s_len
	}
}



__global__
void
scatter(uint32_t cycle,
		param_t* g_params,
		outgoing_addr_t* g_outgoingAddr,
		outgoing_t* g_outgoing,
		gq_entry_t* g_gqData,      // pitch = c_gqPitch
		unsigned* g_gqFill,
		lq_entry_t* g_lqData,      // pitch = c_lqPitch
		unsigned* g_lqFill,
		delay_dt g_ndData[],
		unsigned g_ndFill[],
		unsigned* g_nFired,        // device-only buffer.
		nidx_dt* g_fired)          // device-only buffer, sparse output. pitch = c_pitch32.
{
	__shared__ unsigned s_nFired;
	__shared__ nidx_dt s_fired[MAX_PARTITION_SIZE];
	__shared__ param_t s_params;

	loadParameters(g_params, &s_params);
	loadSparseFiring(g_nFired, s_params.pitch32, g_fired, &s_nFired, s_fired);

	scatterLocal(cycle, s_params,
			s_nFired, s_fired,
			g_ndFill, g_ndData,
			g_lqFill, g_lqData);

	scatterGlobal(cycle,
			s_params,
			g_lqFill, g_lqData,
			g_outgoingAddr,
			g_outgoing,
			g_gqFill, g_gqData);
}



__host__
cudaError_t
scatter(cudaStream_t stream,
		unsigned cycle,
		unsigned partitionCount,
		param_t* d_globalParameters,
		unsigned* d_nFired,
		nidx_dt* d_fired,
		outgoing_addr_t* d_outgoingAddr,
		outgoing_t* d_outgoing,
		gq_entry_t* d_gqData,
		unsigned* d_gqFill,
		lq_entry_t* d_lqData,
		unsigned* d_lqFill,
		delay_dt d_ndData[],
		unsigned d_ndFill[])
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(partitionCount);

	scatter<<<dimGrid, dimBlock, 0, stream>>>(
			cycle,
			d_globalParameters,
			// spike delivery
			d_outgoingAddr, d_outgoing,
			d_gqData, d_gqFill,
			d_lqData, d_lqFill, 
			d_ndData, d_ndFill,
			// firing data
			d_nFired, d_fired);

	return cudaGetLastError();
}


#endif
