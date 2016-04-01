#ifndef NEMO_CUDA_DELAYS_CU
#define NEMO_CUDA_DELAYS_CU

#include "types.h"
#include "device_assert.cu"


/*! Load length of row in delays data structure 
 *
 * \param[in] g_ndFill vector of row fill. Not offset based on partition.
 */
__device__
unsigned
nd_loadFill(nidx_dt neuron, unsigned g_ndFill[])
{
	return g_ndFill[CURRENT_PARTITION * MAX_PARTITION_SIZE + neuron];
}



/*! Load a number of delay entries in parallel
 *
 * \param[in] nDelays row fill
 * \param[in] g_delays matrix of delays for all neurons
 * \param[out] s_delays vector of delays for the given \a neuron
 *
 * \pre len(s_nd) >= MAX_DELAY
 *
 * It's possible to load the data here on a 32b basis. However, this was found
 * to make no difference in practice, so code is left in the current, slightly
 * simpler form.
 */
__device__
void
nd_loadDelays(nidx_dt neuron, unsigned nDelays, delay_dt g_delays[], delay_dt s_delays[])
{
	ASSERT(nDelays < MAX_DELAY);
	for(unsigned b=0; b < nDelays; b += THREADS_PER_BLOCK) {
		unsigned i = b + threadIdx.x;
		if(i < nDelays) {
			unsigned n = CURRENT_PARTITION * MAX_PARTITION_SIZE + neuron;
			s_delays[i] = g_delays[n*MAX_DELAY + i];
		}
	}
}


#endif
