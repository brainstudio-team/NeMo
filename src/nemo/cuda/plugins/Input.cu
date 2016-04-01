#ifndef NEMO_CUDA_PLUGINS_INPUT_CU
#define NEMO_CUDA_PLUGINS_INPUT_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file Input.cu Input neuron update kernel */

#include <nemo/config.h>
#ifdef NEMO_CUDA_PLUGIN_DEBUG_TRACE
#	include <log.cu_h>
#endif
#include <bitvector.cu>
#include <firing.cu>
#include <parameters.cu>
#include <rng.cu_h>
#include "neuron_model.h"


/*! \brief Update input neurons
 *
 * Input neurons have no internal dynamics, and serve merely as an originator
 * for spikes generated externelly. The whole implementation here is a bit
 * redundant, in that it takes a sparse firing vector (provided by the user),
 * turns it into a dense vector (on the host side), and then here turns it back
 * into a sparse firing vector (for use by scatter). However, writing the
 * neuron in this way means that it can use the same plugin infrastructure as
 * regular neurons.
 */
__global__
void
updateNeurons(
		unsigned basePartition,
		unsigned* g_partitionSize,
		param_t* g_params,
		uint32_t* g_valid,
		uint32_t* g_fstim,
		uint32_t* g_firingOutput, // dense output, already offset to current cycle
		unsigned* g_nFired,       // device-only buffer
		nidx_dt* g_fired)         // device-only buffer, sparse output
{
	__shared__ nidx_dt s_fired[MAX_PARTITION_SIZE];

	__shared__ unsigned s_nFired;
	__shared__ unsigned s_partitionSize;
	__shared__ unsigned s_globalPartition; 

	if(threadIdx.x == 0) {
		s_nFired = 0;
		s_globalPartition = basePartition + CURRENT_PARTITION;
		s_partitionSize = g_partitionSize[s_globalPartition];
    }
	__syncthreads();

	__shared__ param_t s_params;
	loadParameters(g_params, &s_params);

	__shared__ uint32_t s_fstim[S_BV_PITCH];
	loadFiringInput(s_globalPartition, s_params.pitch1, g_fstim, s_fstim);

	__shared__ uint32_t s_valid[S_BV_PITCH];
	bv_copy(g_valid + CURRENT_PARTITION * s_params.pitch1, s_valid);
	__syncthreads();

	for(unsigned nbase=0; nbase < s_partitionSize; nbase += THREADS_PER_BLOCK) {
		unsigned neuron = nbase + threadIdx.x;
		bool fired = bv_isSet(neuron, s_valid) && bv_isSet(neuron, s_fstim);
		if(fired) {
                s_fired[atomicAdd(&s_nFired, 1)] = neuron;
#ifdef NEMO_CUDA_PLUGIN_DEBUG_TRACE
				DEBUG_MSG_NEURON("c? ?+%u-%u fired\n", CURRENT_PARTITION, neuron);
#endif
		}
		__syncthreads();
	}
	storeDenseFiring(s_nFired, s_globalPartition, s_params.pitch1, s_fired, g_firingOutput);
	storeSparseFiring(s_nFired, s_globalPartition, s_params.pitch32, s_fired, g_nFired, g_fired);
}



/*! Wrapper for the __global__ call that performs a single simulation step */
extern "C"
NEMO_PLUGIN_DLL_PUBLIC
cudaError_t
cuda_update_neurons(
		cudaStream_t stream,
		unsigned /* cycle */,
		unsigned globalPartitionCount,
		unsigned localPartitionCount,
		unsigned basePartition,
		unsigned* d_partitionSize,
		param_t* d_params,
		float* /* df_neuronParameters */,
		float* /* df_neuronState */,
		nrng_t /* d_nrng */,
		uint32_t* d_valid,
		uint32_t* d_fstim,
		float* /* d_istim */,
		float* /* d_current */,
		uint32_t* d_fout,
		unsigned* d_nFired,
		nidx_dt* d_fired,
		struct rcm_dt* /* unused */)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(localPartitionCount);
	updateNeurons<<<dimGrid, dimBlock, 0, stream>>>(
			basePartition, d_partitionSize, d_params, d_valid, d_fstim, 
			d_fout, d_nFired, d_fired);
	return cudaGetLastError();
}

cuda_update_neurons_t* test_update = &cuda_update_neurons;

#include "default_init.c"

#endif
