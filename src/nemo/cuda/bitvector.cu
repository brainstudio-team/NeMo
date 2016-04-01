#ifndef BIT_VECTOR_CU
#define BIT_VECTOR_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "bitvector.cu_h"
#include "types.h"

#include "device_assert.cu"


// pitch for shared memory bit-vectors (no padding)
#define S_BV_PITCH (MAX_PARTITION_SIZE/32)



/*! Clear whole bitvector */
__device__
void
bv_clear(uint32_t* s_vec)
{
	ASSERT(THREADS_PER_BLOCK >= S_BV_PITCH);
	if(threadIdx.x < S_BV_PITCH) {
		s_vec[threadIdx.x] = 0;
	}
}


/*! Clear whole bitvector */
__device__
void
bv_clear_(uint32_t* s_vec)
{
	bv_clear(s_vec);
	__syncthreads();
}


/*! Check if a particular bit is set */
__device__
bool
bv_isSet(nidx_t neuron, uint32_t* s_vec)
{
	return (s_vec[neuron/32] >> (neuron % 32)) & 0x1;
}



/*! Set bit vector for \a neuron */
__device__
void
bv_atomicSet(nidx_t neuron, uint32_t* s_vec)
{
	unsigned word = neuron / 32;
	uint32_t mask = 0x1 << (neuron % 32);
	atomicOr(s_vec + word, mask);
}



/*! Set bit vector for \a neuron given that \a condition is true */
__device__
void
bv_atomicSetPredicated(bool condition, nidx_t neuron, uint32_t* s_vec)
{
	unsigned word = neuron / 32;
	uint32_t mask = 0x1 << (neuron % 32);
	if(condition) {
		atomicOr(s_vec + word, mask);
	}
}


/*! Copy bit vector */
__device__
void
bv_copy(uint32_t* src, uint32_t* dst)
{
	ASSERT(THREADS_PER_BLOCK >= S_BV_PITCH);
	if(threadIdx.x < S_BV_PITCH) {
		dst[threadIdx.x] =  src[threadIdx.x];
	}
}


#endif
