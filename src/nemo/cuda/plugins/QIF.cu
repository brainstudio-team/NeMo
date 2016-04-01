#ifndef NEMO_CUDA_PLUGINS_QIF_CU
#define NEMO_CUDA_PLUGINS_QIF_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file QIF.cu Quadratic intergrate and fire neuron update kernel */

#include <nemo/config.h>
#ifdef NEMO_CUDA_PLUGIN_DEBUG_TRACE
#	include <log.cu_h>
#endif
#include <bitvector.cu>
#include <current.cu>
#include <firing.cu>
#include <fixedpoint.cu>
#include <neurons.cu>
#include <parameters.cu>
#include <rng.cu>


#include "neuron_model.h"

#include <nemo/internal_types.h>


__device__
float
fx_saturate_pos(float value)
{
         fix_t bits = fix_t(1) << 31;
         return value < 0.0f ? fx_tofloat(~bits,20) : value;
}

__device__
float
fx_saturate_neg(float value)
{
         fix_t bits = fix_t(1) << 31;
         return value > 0.0f ? fx_tofloat(bits,20) : value;
}



/*! Update state of all neurons
 *
 * Update the state of all neurons in partition according to the Quadratic intergrate and fire equations 
 *
 * - the neuron parameters (a)
 * - the neuron state (v)
 * - input current (from other neurons, random input current, or externally provided)
 * - per-neuron specific firing stimulus
 *
 * The neuron state is updated using the Euler method.
 *
 * \param[in] s_partitionSize
 *		number of neurons in current partition
 * \param[in] g_neuronParameters
 *		global memory containing neuron parameters (see \ref nemo::cuda::Neurons)
 * \param[in] g_neuronState
 *		global memory containing neuron state (see \ref nemo::cuda::Neurons)
 * \param[in] s_current
 *		shared memory vector containing input current for all neurons in
 *		partition
 * \param[in] s_fstim
 *		shared memory bit vector where set bits indicate neurons which should
 *		be forced to fire
 * \param[out] s_nFired
 *		output variable which will be set to the number of	neurons which fired
 *		this cycle
 * \param[out] s_fired
 *		shared memory vector containing local indices of neurons which fired.
 *		s_fired[0:s_nFired-1] will contain valid data, whereas remaining
 *		entries may contain garbage.
 */
__device__
void
updateNeurons(
	uint32_t cycle,
	const param_t& s_params,
	unsigned globalPartitionCount,
	unsigned s_globalPartition,
	unsigned s_partitionSize,
	float* g_neuronParameters,
	float* g_neuronState,
	uint32_t* s_valid,   // bitvector for valid neurons
	// input
	nrng_t g_nrng,
	float* g_currentE,
	float* g_currentI,
	float* s_currentExt,    // external input current
	// buffers
	uint32_t* s_fstim,
	// output
	unsigned* s_nFired,
	nidx_dt* s_fired)    // s_NIdx, so can handle /all/ neurons firing
{
	#define PARAM_A 0
	#define STATE_V 0

	//! \todo could set these in shared memory
	size_t neuronParametersSize = PARTITION_COUNT * s_params.pitch32;
	const float* g_a = g_neuronParameters + PARAM_A * neuronParametersSize;
	//! \todo avoid repeated computation of the same data here
	const float* g_v0 = state<1, 1, STATE_V>(cycle, s_params.pitch32, g_neuronState);
	float* g_v1 = state<1, 1, STATE_V>(cycle+1, s_params.pitch32, g_neuronState);


	float dt = 0.1f;
	float a1 = 0.5f*dt;
	float a2 = dt;
	float b1 = dt/6.0f;
	int inc_max= (int)(1/dt);
	//float RevE = 0.0f;
	//float RevI = -70.0f;

	for(unsigned nbase=0; nbase < s_partitionSize; nbase += THREADS_PER_BLOCK) {

		unsigned neuron = nbase + threadIdx.x;

		/* if index space is contigous, no warp divergence here */
		if(bv_isSet(neuron, s_valid)) {

			float v = g_v0[neuron];
			float a = g_a[neuron];
			
			float Excit = g_currentE[neuron];
			float Inhib = g_currentI[neuron];
 			float Exter = s_currentExt[neuron];

			
			
			
			// Update v and u using QIF model in increments of tau
			bool fired = false;
			for(int k=1; k<=inc_max; ++k)
			{  
				if(!fired)
				{
					//new version
					//float vShift = (v*20)-65;
					//float I = (Excit*(RevE-vShift)) + (Inhib*((RevI-vShift)/-1)) + Exter;

					//old version
					float I = Excit + Inhib + Exter;
					
					I=I*0.015f;
					
					float k1 = a*v*(v-1)+I;
					float x2 = v+a1*k1;
					float k2 = a*x2*(x2-1)+I;
					float x3 = v+a1*k2;
					float k3 = a*x3*(x3-1)+I;
					float x4 = v+a2*k3;
					float k4 = a*x4*(x4-1)+I;

					v = v + b1*(k1+2*k2+2*k3+k4);

					fired = v>=1;
			   	}
			}      
			



			bool forceFiring = bv_isSet(neuron, s_fstim); // (smem broadcast)

			if(fired || forceFiring) {

				/* Only a subset of the neurons fire and thus require c/d
				 * fetched from global memory. One could therefore deal with
				 * all the fired neurons separately. This was found, however,
				 * to slow down the fire step by 50%, due to extra required
				 * synchronisation.  */
				//! \todo could probably hard-code c
				v = 0;
				
#ifdef NEMO_CUDA_PLUGIN_DEBUG_TRACE
				DEBUG_MSG("c%u ?+%u-%u fired (forced: %u)\n",
						s_cycle, CURRENT_PARTITION, neuron, forceFiring);
#endif

				//! \todo consider *only* updating this here, and setting u and v separately
				unsigned i = atomicAdd(s_nFired, 1);

				/* can overwrite current as long as i < neuron. See notes below
				 * on synchronisation and declaration of s_current/s_fired. */
				s_fired[i] = neuron;
			}

		
			g_v1[neuron] = v;
			
		}

		/* synchronise to ensure accesses to s_fired and s_current (which use
	fix_t
fx_saturate_pos(float value)
{
         fix_t bits = fix_t(1) << 31;
         return value < 0.0f ? bits : value;
}

fix_t
fx_saturate_neg(float value)
{
         fix_t bits = fix_t(1) << 31;
         return value > 0.0f ? ~bits : value;
}
	 * the same underlying buffer) do not overlap. Even in the worst case
		 * (all neurons firing) the write to s_fired will be at least one
		 * before the first unconsumed s_current entry. */
		__syncthreads();
	}
}



/*! Update the state of all neurons in the network */
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
		float* gf_neuronState,
		nrng_t g_nrng,
		uint32_t* g_valid,
		// firing stimulus
		uint32_t* g_fstim,
		float* g_istim,
		float* g_current,
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
		s_partitionSize = g_partitionSize[s_globalPartition];
    }
	__syncthreads();

	__shared__ param_t s_params;
	loadParameters(g_params, &s_params);

	float* g_currentE = incomingExcitatory(g_current, globalPartitionCount, s_globalPartition, s_params.pitch32);
	float* g_currentI = incomingInhibitory(g_current, globalPartitionCount, s_globalPartition, s_params.pitch32);

	__shared__ float s_current[MAX_PARTITION_SIZE];
	loadCurrentStimulus(s_globalPartition, s_partitionSize, s_params.pitch32, g_istim, s_current);

	__shared__ uint32_t s_fstim[S_BV_PITCH];
	loadFiringInput(s_globalPartition, s_params.pitch1, g_fstim, s_fstim);

	__shared__ uint32_t s_valid[S_BV_PITCH];
	bv_copy(g_valid + CURRENT_PARTITION * s_params.pitch1, s_valid);
	__syncthreads();

	updateNeurons(
			cycle,
			s_params,
			globalPartitionCount,
			s_globalPartition,
			s_partitionSize,
			//! \todo use consistent parameter passing scheme here
			gf_neuronParameters + CURRENT_PARTITION * s_params.pitch32,
			gf_neuronState,
			s_valid,
			g_nrng,
			g_currentE, g_currentI,
			s_current, s_fstim,
			&s_nFired,
			s_fired);

	__syncthreads();

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
		float* df_neuronState,
		nrng_t d_nrng,
		uint32_t* d_valid,
		uint32_t* d_fstim,
		float* d_istim,
		float* d_current,
		uint32_t* d_fout,
		unsigned* d_nFired,
		nidx_dt* d_fired,
		struct rcm_dt* /* unused */)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(localPartitionCount);

	updateNeurons<<<dimGrid, dimBlock, 0, stream>>>(
			cycle, globalPartitionCount, basePartition,
			d_partitionSize, d_params,
			df_neuronParameters, df_neuronState, d_nrng, d_valid,
			d_fstim,   // firing stimulus
			d_istim,   // current stimulus
			d_current, // internal input current
			d_fout, d_nFired, d_fired);

	return cudaGetLastError();
}


#include "default_init.c"

#endif
