#ifndef NEMO_CUDA_PLUGINS_POISSON_SOURCE_CU
#define NEMO_CUDA_PLUGINS_POISSON_SOURCE_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <nemo/config.h>
#ifdef NEMO_CUDA_PLUGIN_DEBUG_TRACE
#	include <log.cu_h>
#endif
#include <bitvector.cu>
#include <firing.cu>
#include <neurons.cu>
#include <parameters.cu>
#include <rng.cu>

#include "neuron_model.h"


__global__
void
updateNeurons(
		uint32_t cycle,
		unsigned globalPartitionCount,
		unsigned basePartition,
		unsigned* g_partitionSize,
		param_t* g_params,
		// neuron state
		float* gf_neuronParameters,
		nrng_t g_nrng,
		uint32_t* g_valid,
		// firing stimulus
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
#ifdef NEMO_CUDA_PLUGIN_DEBUG_TRACE
		s_cycle = cycle;
#endif
		s_nFired = 0;
		s_globalPartition = basePartition + CURRENT_PARTITION;
		s_partitionSize = g_partitionSize[CURRENT_PARTITION];
    }
	__syncthreads();

	__shared__ param_t s_params;
	loadParameters(g_params, &s_params);

	__shared__ uint32_t s_fstim[S_BV_PITCH];
	loadFiringInput(s_globalPartition, s_params.pitch1, g_fstim, s_fstim);

	__shared__ uint32_t s_valid[S_BV_PITCH];
	bv_copy(g_valid + CURRENT_PARTITION * s_params.pitch1, s_valid);
	__syncthreads();

	const float* g_rate = gf_neuronParameters + CURRENT_PARTITION * s_params.pitch32;

	for(unsigned nbase=0; nbase < s_partitionSize; nbase += THREADS_PER_BLOCK) {

		unsigned neuron = nbase + threadIdx.x;

		/* if index space is contigous, no warp divergence here */
		if(bv_isSet(neuron, s_valid)) {

			/* Instantenous rate of firing */
			float rate = g_rate[neuron];

			unsigned p0 = unsigned(rate * float(1<<16));
			unsigned p1 = urand(globalPartitionCount, s_globalPartition, neuron, g_nrng) & 0xffff;
			bool fired = p1 < p0;
			bool forceFiring = bv_isSet(neuron, s_fstim); // (smem broadcast)

			if(fired || forceFiring) {
#ifdef NEMO_CUDA_PLUGIN_DEBUG_TRACE
				DEBUG_MSG_NEURON("c%u %u-%u fired (forced: %u)\n",
						s_cycle, s_globalPartition, neuron, forceFiring);
#endif
				s_fired[atomicAdd(&s_nFired, 1)] = neuron;
			}
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
		unsigned cycle,
		unsigned globalPartitionCount,
		unsigned localPartitionCount,
		unsigned basePartition,
		unsigned* d_partitionSize,
		param_t* d_params,
		float* df_neuronParameters,
		float* /* df_neuronState */,
		nrng_t d_nrng,
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
			cycle, globalPartitionCount, basePartition, d_partitionSize, d_params,
			df_neuronParameters, d_nrng, d_valid,
			d_fstim,   // firing stimulus
			d_fout, d_nFired, d_fired);

	return cudaGetLastError();
}


cuda_update_neurons_t* test_update = &cuda_update_neurons;

#include "default_init.c"

#endif
