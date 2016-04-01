#ifndef NEMO_CUDA_PLUGINS_IF_CURR_EXP_CU
#define NEMO_CUDA_PLUGINS_IF_CURR_EXP_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file IF_curr_exp.cu Neuron update CUDA kernel for current-based
 * exponential decay integrate-and-fire neurons. */

#include <nemo/config.h>
#ifdef NEMO_CUDA_PLUGIN_DEBUG_TRACE
#	include <log.cu_h>
#endif
#include <bitvector.cu>
#include <current.cu>
#include <firing.cu>
#include <neurons.cu>
#include <parameters.cu>

#include <nemo/plugins/IF_curr_exp.h>
#include "neuron_model.h"



/*! Update state of all neurons
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
	//! \todo could set these in shared memory
	size_t neuronParametersSize = PARTITION_COUNT * s_params.pitch32;

	const float* g_v_rest     = g_neuronParameters + PARAM_V_REST     * neuronParametersSize;
	const float* g_c_m        = g_neuronParameters + PARAM_C_M        * neuronParametersSize;
	const float* g_tau_m      = g_neuronParameters + PARAM_TAU_M      * neuronParametersSize;
	const float* g_tau_refrac = g_neuronParameters + PARAM_TAU_REFRAC * neuronParametersSize;
	const float* g_tau_syn_E  = g_neuronParameters + PARAM_TAU_SYN_E  * neuronParametersSize;
	const float* g_tau_syn_I  = g_neuronParameters + PARAM_TAU_SYN_I  * neuronParametersSize;
	const float* g_I_offset   = g_neuronParameters + PARAM_I_OFFSET   * neuronParametersSize;
	const float* g_v_reset    = g_neuronParameters + PARAM_V_RESET    * neuronParametersSize;
	const float* g_v_thresh   = g_neuronParameters + PARAM_V_THRESH   * neuronParametersSize;

	//! \todo avoid repeated computation of the same data here
	const float* g_v0         = state<HISTORY_LENGTH, STATE_COUNT, STATE_V        >(cycle, s_params.pitch32, g_neuronState);
	const float* g_Ie0        = state<HISTORY_LENGTH, STATE_COUNT, STATE_IE       >(cycle, s_params.pitch32, g_neuronState);
	const float* g_Ii0        = state<HISTORY_LENGTH, STATE_COUNT, STATE_II       >(cycle, s_params.pitch32, g_neuronState);
	const float* g_lastfired0 = state<HISTORY_LENGTH, STATE_COUNT, STATE_LASTFIRED>(cycle, s_params.pitch32, g_neuronState);

	float* g_v1         = state<HISTORY_LENGTH, STATE_COUNT, STATE_V        >(cycle+1, s_params.pitch32, g_neuronState);
	float* g_Ie1        = state<HISTORY_LENGTH, STATE_COUNT, STATE_IE       >(cycle+1, s_params.pitch32, g_neuronState);
	float* g_Ii1        = state<HISTORY_LENGTH, STATE_COUNT, STATE_II       >(cycle+1, s_params.pitch32, g_neuronState);
	float* g_lastfired1 = state<HISTORY_LENGTH, STATE_COUNT, STATE_LASTFIRED>(cycle+1, s_params.pitch32, g_neuronState);

	for(unsigned bNeuron=0; bNeuron < s_partitionSize; bNeuron += THREADS_PER_BLOCK) {

		unsigned neuron = bNeuron + threadIdx.x;

		/* if index space is contigous, no warp divergence here */
		if(bv_isSet(neuron, s_valid)) {

			//! \todo consider pre-multiplying g_tau_syn_E
			float Ie = ((1.0f - 1.0f/g_tau_syn_E[neuron]) * g_Ie0[neuron]) + g_currentE[neuron];
			float Ii = ((1.0f - 1.0f/g_tau_syn_I[neuron]) * g_Ii0[neuron]) + g_currentI[neuron];

			/* Update the incoming currents */
			float I = Ie + Ii + s_currentExt[neuron] + g_I_offset[neuron];

			g_Ie1[neuron] = Ie;
			g_Ii1[neuron] = Ii;

			float v = g_v0[neuron];
			float lastfired = g_lastfired0[neuron];
			bool refractory = lastfired <= g_tau_refrac[neuron];
			lastfired += 1;

			if(!refractory) {
				float c_m = g_c_m[neuron];
				float v_rest = g_v_rest[neuron];
				float tau_m = g_tau_m[neuron];
				//! \todo make integration step size a model parameter as well
				v += I / c_m + (v_rest - v) / tau_m;
			}

			/* Firing can be forced externally, even during refractory period */

			bool firedInternal = !refractory && v >= g_v_thresh[neuron];
			bool firedExternal = bv_isSet(neuron, s_fstim); // (smem broadcast)

			if(firedInternal || firedExternal) {
				v = g_v_reset[neuron];
				lastfired = 1;
#ifdef NEMO_CUDA_PLUGIN_DEBUG_TRACE
				DEBUG_MSG("c%u ?+%u-%u fired (forced: %u)\n",
						s_cycle, CURRENT_PARTITION, neuron, firedExternal);
#endif
				unsigned i = atomicAdd(s_nFired, 1);
				s_fired[i] = neuron;
			}

			g_v1[neuron] = v;
			g_lastfired1[neuron] = lastfired;
		}

		__syncthreads();
	}
}



/*! Update the state of all Izhikevich neurons in the network */
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

cuda_update_neurons_t* test_update = &cuda_update_neurons;

#include "default_init.c"

#endif
