#ifndef NEMO_CUDA_STATE_CU
#define NEMO_CUDA_STATE_CU

/*! Rotating queue scheme for state variables
 *
 * In order to support 1) accessing state from other neurons 2) accessing
 * previous state of neurons, the neuron state variables are stored in a
 * rotating buffer.
 *
 * All state variables are replicated in the same manner. All the state from a
 * single time step is stored contigously. For example if we have state
 * variables U and V, two partitions of four neurons, and we keep a history of
 * four cycles, we have the following layout.
 *
 *        p0   p1 
 * 0 t-2 UUUU UUUU
 * 0 t-2 VVVV VVVV
 * 1 t-1 UUUU UUUU
 * 1 t-1 VVVV VVVV
 * 2 t   UUUU UUUU
 * 2 t   VVVV VVVV
 * 3 t+1 UUUU UUUU
 * 3 t+1 VVVV VVVV
 *
 * Note that during the neuron update phase, a partition will write from slot t
 * to slot t+1. Other partitions can thus reliably read the slots t, t-1, and
 * t-2. In general if the history length is 'h', 'h-1' slots can be reliably
 * read by neurons outside a partition.
 */

#include "kernel.cu_h"
#include <nemo/util.h>

/*! \return pointer to beginning of one variables current cycle's data
 *
 * \tparam H history length, which should be a power of two
 * \tparam N number of state variables for the current neuron model
 * \tparam V desired state variable
 * 
 * \todo pass in pitch instead
 * \param s_params global parmeter set
 * \param cycle current simulation cycle
 * \param g_state global memory for all state for all time steps
 *
 * \pre history length is a power of two
 */
//! \todo should share this with host
template<unsigned H, unsigned N, unsigned V>
__device__
float*
state(unsigned cycle, size_t pitch, float* g_state)
{
	ASSERT(IS_POWER_OF_TWO(H));
	return g_state + (((cycle % H) * N + V) * PARTITION_COUNT + CURRENT_PARTITION) * pitch; 
}



/*! Get a neuron state at a previous point in time

 * This function is intended for random access to the neuron state, i.e.
 * accesses which cannot be coalesced.
 *
 * \tparam H history length, which should be a power of two
 * \tparam N number of state variables for the current neuron model
 * \tparam V desired state variable
 *
 * \param cycle
 *		Desired time-slot. This is not checked for validity. Generally H-1 or
 *		H-2 cycles worth of previous data are valid.
 */
template<unsigned H, unsigned N, unsigned V>
__device__
float
state(unsigned cycle, size_t pitch, pidx_t partition, nidx_t neuron, const float* g_state)
{
	return g_state[(((cycle % H) * N + V) * PARTITION_COUNT + partition) * pitch + neuron];
}



#endif
