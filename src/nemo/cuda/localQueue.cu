#ifndef NEMO_CUDA_LOCAL_QUEUE_CU
#define NEMO_CUDA_LOCAL_QUEUE_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "kernel.cu_h"
#include "localQueue.cu_h"

/* Local queue pitch in terms of words for each delay/partition pair */
__constant__ size_t c_lqPitch;



__host__
cudaError
setLocalQueuePitch(size_t pitch)
{
	return cudaMemcpyToSymbol(c_lqPitch,
				&pitch, sizeof(size_t), 0, cudaMemcpyHostToDevice);
}


/*! \return offset into local queue fill data (in gmem) for the current
 * partition and the given slot. */
__device__
unsigned
lq_globalFillOffset(unsigned slot, unsigned maxDelay)
{
	return CURRENT_PARTITION * maxDelay + slot;
}


/*!
 * \param cycle current simulation cycle
 * \param maxDelay maximum delay in the network
 * \param g_fill global memory containing the fill rate for each slot in the local queue.
 *
 * \return queue fill for the current partition slot due for delivery now, and
 * 		reset the relevant slot.
 *
 * \see nemo::cuda::LocalQueue
 */
__device__
unsigned
lq_getAndClearCurrentFill(unsigned cycle, unsigned maxDelay, unsigned* g_fill)
{
	return atomicExch(g_fill + lq_globalFillOffset(cycle % maxDelay, maxDelay), 0);
}



/*! Load current queue fill from gmem to smem. 
 * 
 * \param maxDelay maximum delay in the network
 * \param[in] g_fill
 * \param[out] s_fill
 *
 * \pre s_fill has capacity for at least maxDelay elements
 */ 
__device__
void
lq_loadQueueFill(unsigned maxDelay, unsigned* g_fill, unsigned* s_fill)
{
	unsigned slot = threadIdx.x;
	//! \todo this could be based on the actual max delay rather than the absolute maximum.
	if(slot < maxDelay) {
		s_fill[slot] = g_fill[lq_globalFillOffset(slot, maxDelay)];
	}
}


/*! Store updated queue fill back to gmem
 *
 * \param maxDelay maximum delay in the network
 * \param[in] s_fill
 * \param[out] s_fill
 */
__device__
void
lq_storeQueueFill(unsigned maxDelay, unsigned* s_fill, unsigned* g_fill)
{
	unsigned slot = threadIdx.x;
	if(slot < maxDelay) {
		g_fill[lq_globalFillOffset(slot, maxDelay)] = s_fill[slot];
	}
}



/*! \return the buffer number to use for the given delay, given current cycle */
__device__
unsigned
lq_delaySlot(unsigned cycle, unsigned maxDelay, unsigned delay0)
{
	return (cycle + delay0) % maxDelay;
}


/*! \return the full address to the start of a queue entry (for a
 * partition/delay pair) given a precomputed slot number */
__device__
unsigned
lq_offsetOfSlot(unsigned maxDelay, unsigned slot)
{
	ASSERT(slot < maxDelay);
	return (CURRENT_PARTITION * maxDelay + slot) * c_lqPitch;
}


/*! \return the full address to the start of a queue entry for the current
 * partition and the given delay (relative to the current cycle). */
__device__
unsigned
lq_offset(unsigned cycle, unsigned maxDelay, unsigned delay0)
{
	return lq_offsetOfSlot(maxDelay, lq_delaySlot(cycle, maxDelay, delay0));
}



/*! \return offset to next free queue slot in gmem queue for current partition
 * 		and the given \a delay0 offset from the given \a cycle.
 *
 * \param cycle
 * \param delay0
 * \param s_fill shared memory buffer which should have been previously filled
 *        using lq_loadQueueFill
 */
__device__
unsigned
lq_nextFree(unsigned cycle, unsigned maxDelay, delay_t delay0, unsigned* s_fill)
{
	/* The buffer should be sized such that we never overflow into the next
	 * queue slot (or out of the queue altogether). However, even if this is
	 * not the case the wrap-around in the atomic increment ensures that we
	 * just overwrite our own firing data rather than someone elses */
	unsigned delaySlot = lq_delaySlot(cycle, maxDelay, delay0);
	ASSERT(delaySlot < maxDelay);
	ASSERT(lq_offsetOfSlot(maxDelay, delaySlot) < PARTITION_COUNT * maxDelay * c_lqPitch);
	unsigned next = atomicInc(s_fill + delaySlot, c_lqPitch-1);
	ASSERT(next < c_lqPitch);
	return lq_offsetOfSlot(maxDelay, delaySlot) + next;
}



/*! Enqueue a single neuron/delay pair in the local queue
 *
 * This operation will almost certainly be non-coalesced.
 */
__device__
void
lq_enque(nidx_dt neuron,
		unsigned cycle,
		unsigned maxDelay,
		delay_t delay0,
		unsigned* s_fill,
		lq_entry_t* g_queue)
{
	g_queue[lq_nextFree(cycle, maxDelay, delay0, s_fill)] = make_short2(neuron, delay0);
}

#endif
