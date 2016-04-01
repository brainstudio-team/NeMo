#ifndef NEMO_CUDA_COMPACT_CU
#define NEMO_CUDA_COMPACT_CU

#include <nemo/config.h>

/*! Compact a boolean vector of firing to the per-partition compact
 * representation used internally.
 *
 * \param d_partitionSize
 *		size of each partition in the network
 * \param[in] g_fired
 *		global memory with one entry per neuron with non-zero words indicating
 *		fired neurons.
 * \param[out] g_nFired
 *		number of valid entries for each partition in \a g_firedCompact
 * \param[out] g_firedCompact
 *		global memory with one entry per neuron, but compacted on a
 *		per-partition basis. Non-zero entries are neuron indices of fired
 *		neurons. The partition index is implicit in the data structure.
 */
__global__
void
compact(
	unsigned* g_partitionSize,
	param_t* g_params,
	unsigned g_fired[],
	unsigned g_nFired[],
	nidx_dt g_firedCompact[])
{
	__shared__ nidx_dt s_firedCompact[MAX_PARTITION_SIZE];
	__shared__ unsigned s_nFired;
	__shared__ unsigned s_partitionSize;
	__shared__ param_t s_params;

	loadParameters(g_params, &s_params);

	if(threadIdx.x == 0) {
		s_partitionSize = g_partitionSize[CURRENT_PARTITION];
		s_nFired = 0;
	}
	__syncthreads();

	for(unsigned bNeuron = 0; bNeuron < s_partitionSize; bNeuron += blockDim.x) {
		unsigned neuron = bNeuron + threadIdx.x;
		if(g_fired[CURRENT_PARTITION * MAX_PARTITION_SIZE + neuron]
				&& neuron < s_partitionSize) {
			unsigned i = atomicAdd(&s_nFired, 1);
			s_firedCompact[i] = neuron;
		}
	}
	__syncthreads();

	for(unsigned b=0; b < s_nFired; b += blockDim.x) {
		unsigned i = b + threadIdx.x;
		if(i < s_nFired) {
			g_firedCompact[CURRENT_PARTITION * s_params.pitch32 + i] = s_firedCompact[i];
		}
	}

	if(threadIdx.x == 0) {
		g_nFired[CURRENT_PARTITION] = s_nFired;
	}
}


__host__
cudaError_t
compact(cudaStream_t stream,
		unsigned* d_partitionSize,
		param_t* d_params,
		unsigned partitionCount,
		unsigned d_fired[],
		unsigned d_nFired[],
		nidx_dt d_firedCompact[])
{
	dim3 dimBlock(1024);
	dim3 dimGrid(partitionCount);
	compact<<<dimGrid, dimBlock, 0, stream>>>(d_partitionSize,
		d_params, d_fired, d_nFired, d_firedCompact);
	return cudaGetLastError();
}

#endif
