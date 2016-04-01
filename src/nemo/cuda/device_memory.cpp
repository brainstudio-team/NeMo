/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file device_memory.cpp Device memory methods */

#include <boost/format.hpp>

#include "device_memory.hpp"

namespace nemo {
	namespace cuda {


void
throw_allocation_error(const char* structname, size_t bytes, cudaError err)
{
	using boost::format;
	throw nemo::exception(NEMO_CUDA_MEMORY_ERROR,
			str(format("Failed to allocate %uB for %s.\nCuda error: %s\n")
				% bytes % structname % cudaGetErrorString(err)));
}



void
safeCall(cudaError_t err, unsigned error = NEMO_CUDA_MEMORY_ERROR)
{
	if(cudaSuccess != err) {
		throw nemo::exception(error, cudaGetErrorString(err));
	}
}


void
d_malloc(void** d_ptr, size_t sz, const char* name)
{
	cudaError_t err = cudaMalloc(d_ptr, sz);
	if(cudaSuccess != err) {
		throw_allocation_error(name, sz, err);
	}
}



void
d_free(void* arr)
{
	safeCall(cudaFree(arr));
}



void
d_mallocPitch(void** d_ptr, size_t* bpitch, size_t width, size_t height, const char* name)
{
	if(width == 0) {
		*d_ptr = NULL;
		*bpitch = 0;
		return;
	}
	cudaError_t err = cudaMallocPitch(d_ptr, bpitch, width, height);
	if(cudaSuccess != err) {
		throw_allocation_error(name, height * width, err);
	}
}



void
memcpyBytesToDevice(void* dst, const void* src, size_t bytes)
{
	safeCall(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}


void
memcpyBytesToDeviceAsync(void* dst, const void* src, size_t bytes, cudaStream_t stream)
{
	safeCall(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
}


void
memcpyBytesFromDevice(void* dst, const void* src, size_t bytes)
{
	safeCall(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}


void
memcpyBytesFromDeviceAsync(void* dst, const void* src, size_t bytes, cudaStream_t stream)
{
	safeCall(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream));
}


void
d_memset(void* d_ptr, int value, size_t count)
{
	safeCall(cudaMemset(d_ptr, value, count));
}


void
d_memset2D(void* d_ptr, size_t pitch, int value, size_t height)
{
	safeCall(cudaMemset2D(d_ptr, pitch, value, pitch, height));
}


void
mallocPinned(void** h_ptr, size_t sz)
{
	safeCall(cudaMallocHost(h_ptr, sz));
}


void
freePinned(void* arr)
{
	safeCall(cudaFreeHost(arr));
}


} 	} // end namespaces
