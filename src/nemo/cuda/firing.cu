#ifndef NEMO_CUDA_FIRING_CU
#define NEMO_CUDA_FIRING_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file firing.cu Firing-related functions*
 * 
 * The functions in this file are utility functions to be used by neuron update
 * kernels to manipulate the two forms of the firing data: the per-neuron
 * bit-vector of firing and the sparse firing vector.
 */

#include "bitvector.cu"


/*! Set per-neuron bit-vector for fired neurons in both shared and global memory
 *
 * \param[in] nfired
 *		number of neurons in current partition which fired this cycle.
 * \param[in] partition
 *		global index of current partition
 * \param[in] s_fired
 *		vector of indices of the fired neuron. The first \a nfired entries
 *		should be set.
 * \param[out] g_dfired
 *		per-neuron bit-vector in global memory for fired neurons.
 *
 * \see loadDenseFiring
 */
__device__
void
storeDenseFiring(unsigned nfired,
		unsigned partition,
		size_t pitch1,
		nidx_dt* s_fired,
		uint32_t* g_dfired)
{
	__shared__ uint32_t s_dfired[S_BV_PITCH];

	bv_clear_(s_dfired);

	for(unsigned nbase=0; nbase < nfired; nbase += THREADS_PER_BLOCK) {
		unsigned i = nbase + threadIdx.x;
		unsigned neuron = s_fired[i];
		bv_atomicSetPredicated(i < nfired, neuron, s_dfired);
	}
	__syncthreads();

	bv_copy(s_dfired, g_dfired + partition * pitch1);
}


/*! Store sparse firing in global memory buffer
 *
 * The global memory roundtrip is required to support having 'fire' and
 * 'scatter' in separate kernels.
 *
 * \param[in]  nFired     number of neurons in this partition which fired this cycle
 * \param[in]  partition  global index of current partition
 * \param[in]  s_fired    shared memory vector of the relevant neuron indices.
 * \param[out] g_nFired   global memory per-partition vector of firing counts
 * \param[out] g_fired    global memory per-neuron vector of fired neuron indices.
 * 		For each partition, only the first \a nFired entries contain valid data.
 *
 * \see loadSparseFiring
 */
__device__
void
storeSparseFiring(unsigned nFired,
		unsigned partition,
		size_t pitch32,
		nidx_dt* s_fired,
		unsigned* g_nFired,
		nidx_dt* g_fired)
{
	for(unsigned b=0; b < nFired; b += THREADS_PER_BLOCK) {
		unsigned i = b + threadIdx.x;
		if(i < nFired) {
			g_fired[partition * pitch32 + i] = s_fired[i];
		}
	}

	if(threadIdx.x == 0) {
		g_nFired[partition] = nFired;
	}
}



/*! The external firing stimulus is (possibly) provided in a per-neuron
 * bit-vector
 *
 * \param partition global partition index
 */
__device__
void
loadFiringInput(unsigned partition, size_t pitch1, uint32_t* g_firing, uint32_t* s_firing)
{
	if(g_firing != NULL) {
		bv_copy(g_firing + partition * pitch1, s_firing);
	} else {
		bv_clear(s_firing);
	}
	__syncthreads();
}


#endif
