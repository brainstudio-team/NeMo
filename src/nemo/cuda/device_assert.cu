#ifndef DEVICE_ASSERT_CU
#define DEVICE_ASSERT_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "kernel.cu_h"


#ifdef NEMO_CUDA_DEVICE_ASSERTIONS

#define DEVICE_ASSERTION_MEMSZ (MAX_PARTITION_COUNT * THREADS_PER_BLOCK)

__device__ uint32_t g_assertions[DEVICE_ASSERTION_MEMSZ];


__device__ __host__
size_t
assertion_offset(size_t partition, size_t thread)
{
    return partition * THREADS_PER_BLOCK + thread;
}



#ifdef __DEVICE_EMULATION__
#	define ASSERT(cond) assert(cond)
#else
#	define ASSERT(cond) \
        if(!(cond)) {\
			g_assertions[assertion_offset(CURRENT_PARTITION, threadIdx.x)] = __LINE__;\
        }
#endif
#else // NEMO_CUDA_DEVICE_ASSERTIONS
#   define ASSERT(cond)
#endif


__host__
cudaError_t
getDeviceAssertions(unsigned partitions, uint32_t* h_assertions)
{
#ifdef NEMO_CUDA_DEVICE_ASSERTIONS
	size_t bytes = partitions * THREADS_PER_BLOCK * sizeof(uint32_t);
	return cudaMemcpyFromSymbol(h_assertions, g_assertions, bytes, cudaMemcpyDeviceToHost);
#else
	return cudaSuccess;
#endif
}


__host__
cudaError_t
clearDeviceAssertions()
{
#ifdef NEMO_CUDA_DEVICE_ASSERTIONS
	void* addr;
	cudaError_t err = cudaGetSymbolAddress(&addr, g_assertions);
	if(err != cudaSuccess) {
		return err;
	}
	return cudaMemset(addr, 0, DEVICE_ASSERTION_MEMSZ*sizeof(uint32_t));
#else
	return cudaSuccess;
#endif
}

#endif
