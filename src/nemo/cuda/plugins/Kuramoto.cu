#ifndef NEMO_CUDA_PLUGINS_KURAMOTO_CU
#define NEMO_CUDA_PLUGINS_KURAMOTO_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file Kuramoto.cu Kuramoto oscillator kernel */ 

#include <math.h>

#include <nemo/config.h>
#include <rng.cu_h>
#include <log.cu_h>

#include <bitvector.cu>
#include <neurons.cu>
#include <parameters.cu>
#include <rcm.cu>

#include <nemo/plugins/Kuramoto.h>

#include "neuron_model.h"

#define INCOMING_BUFFER_SIZE 1024


//! \todo use more optimal reduction here
__device__
void
sum256(const float g_Cmean, float* sdata, float *s_out)
{  
    unsigned tid = threadIdx.x;

    for(unsigned s=THREADS_PER_BLOCK/2; s>0; s>>=1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        } 
        __syncthreads();
    }

    if(tid == 0) {
		*s_out += sdata[0]/g_Cmean;

    }
}


/* Compute the phase shift induced in a single oscillator
 *
 * \return theta_i = sum_j { w_ij * sin(theta_j - theta_i) }
 */
__device__
void
sumN(const float g_Cmean, const float s_weight[],
		const float s_sourcePhase[],
		unsigned indegree,
		float targetPhase,		
		float* out)
{
	unsigned tid = threadIdx.x;
	__shared__ float s_tmp[THREADS_PER_BLOCK]; // partial sums
	s_tmp[tid] = 0.0f;
	for(unsigned i=0; i<indegree; i += THREADS_PER_BLOCK) {
		unsigned sid = i+tid;
		if(sid < indegree) {
			s_tmp[tid] = s_weight[sid] * sinf(s_sourcePhase[sid] - targetPhase);
		}
		
	}
	__syncthreads();
	sum256(g_Cmean, s_tmp, out);

	__syncthreads();
}



/*! Load up to INCOMING_BUFFER_SIZE couping strength/source phase pairs for a single target oscillator 
 *
 * \pre s_sourcePhase has INCOMING_BUFFER_SIZE elements
 * \pre s_weight has INCOMING_BUFFER_SIZE elements
 */
__device__
void
loadIncoming(
		unsigned cycle,
		unsigned inWarps,
		const param_t& s_params,
		float* g_state,
		rcm_dt g_rcm,
		rcm_index_address_t s_row,
		float s_sourcePhase[],
		float s_weight[])
{
    unsigned tid = threadIdx.x;
	/*! \todo add a second loop and pre-load THREADS_PER_BLOCK warp
	 * addresses */
	for(unsigned bIndex=0 ; bIndex < inWarps; bIndex += THREADS_PER_BLOCK/WARP_SIZE) {

		__shared__ rcm_address_t warp[THREADS_PER_BLOCK/WARP_SIZE];

		if(tid < THREADS_PER_BLOCK/WARP_SIZE) {
			warp[tid] = rcm_address(rcm_indexRowStart(s_row), bIndex + tid, g_rcm);
		}
		__syncthreads();

		size_t r_offset = rcm_offset(warp[tid/WARP_SIZE]);
		rsynapse_t synapse = g_rcm.data[r_offset];

		size_t sid = bIndex * WARP_SIZE + tid;
		s_weight[sid] = 0.0f;
		s_sourcePhase[sid] = 0.0f;

		if(synapse != INVALID_REVERSE_SYNAPSE) {
			ASSERT(r_delay1(synapse) < MAX_HISTORY_LENGTH-1);
			/* Reading the source state here is non-coalesced. Much of this
			 * should be cachable, however, so for > 2.0 devices we should use
			 * a large L1. */
			s_sourcePhase[sid] =
				state<MAX_HISTORY_LENGTH, 1, STATE_PHASE>(
						int(cycle)-int(r_delay0(synapse)),
						s_params.pitch32,
						sourcePartition(synapse), sourceNeuron(synapse),
						g_state);
			s_weight[sid] = g_rcm.weights[r_offset];
		}
		__syncthreads(); // to protect 'warp'
	}
}



/*! Compute the influence from other oscillators, for a single target oscillator.
 */
__device__
void
computePhaseShift(const float* g_frequency, const float g_Cmean,
	unsigned cycle,
	unsigned partitionSize,
	const param_t& s_params,
	float* g_state,
	rcm_dt g_rcm,
	rcm_index_address_t s_row,
	float targetPhase,
	float targetFrequency,
	float h,
	float* s_phaseShift)
{
    	unsigned tid = threadIdx.x;
	//! \todo pre-load index addresses for a bunch of targets, and pass this in

	__shared__ float s_sourcePhase[INCOMING_BUFFER_SIZE];
	__shared__ float s_weight[INCOMING_BUFFER_SIZE];
     
	const unsigned inWarps = rcm_indexRowLength(s_row);

	loadIncoming(cycle, inWarps, s_params, g_state, g_rcm, s_row, s_sourcePhase, s_weight);

	__shared__ float s_k[4];
	if(tid < 4) {
		s_k[tid] = targetFrequency;
	}
	__syncthreads();


	unsigned indegree = inWarps * WARP_SIZE; // possible some padding
	sumN(g_Cmean, s_weight, s_sourcePhase, indegree, targetPhase            , s_k+0);
	sumN(g_Cmean, s_weight, s_sourcePhase, indegree, targetPhase+s_k[0]*0.5f*h, s_k+1);
	sumN(g_Cmean, s_weight, s_sourcePhase, indegree, targetPhase+s_k[1]*0.5f*h, s_k+2);
	sumN(g_Cmean, s_weight, s_sourcePhase, indegree, targetPhase+s_k[2]*h     , s_k+3);

	//! \todo use precomputed factor and multiply
	if(tid == 0) {
		*s_phaseShift = h*(s_k[0] + 2*s_k[1] + 2*s_k[2] + s_k[3])/6.0f;
	}
	__syncthreads();
}




/*! Update state for all oscillators in the current partition */
__global__
void
updateOscillators( 
		uint32_t cycle,
		unsigned* g_partitionSize,
		param_t* g_params,
		float* g_nparams,
		float* g_nstate,
		uint32_t* g_valid,
		rcm_dt g_rcm)
{
	__shared__ unsigned s_partitionSize;
	__shared__ param_t s_params;
	__shared__ uint32_t s_valid[S_BV_PITCH];

	unsigned tid = threadIdx.x;

	loadParameters(g_params, &s_params);
	if(tid == 0) {
		s_partitionSize = g_partitionSize[CURRENT_PARTITION];
    }
	__syncthreads();

	bv_copy(g_valid + CURRENT_PARTITION * s_params.pitch1, s_valid);
	__syncthreads();

	/* Natural frequency of oscillations */
	//const float* g_frequency = g_nparams + CURRENT_PARTITION * s_params.pitch32;

	const float* neuronParameters = g_nparams + CURRENT_PARTITION * s_params.pitch32;
	size_t neuronParametersSize = PARTITION_COUNT * s_params.pitch32;	
     	const float* g_frequency = neuronParameters + 0 * neuronParametersSize;
       	const float* g_Cmean = neuronParameters + 1 * neuronParametersSize;
  
	/* Current phase */
	const float* g_phase0 =
		state<MAX_HISTORY_LENGTH, 1, STATE_PHASE>(cycle, s_params.pitch32, g_nstate);

	/* Next phase */
	float* g_phase1 =
		state<MAX_HISTORY_LENGTH, 1, STATE_PHASE>(cycle+1, s_params.pitch32, g_nstate);

	/*! \todo consider adding a way to provide external stimulus here */

	for(unsigned bOscillator=0; bOscillator < s_partitionSize; bOscillator += THREADS_PER_BLOCK) {

		__shared__ float s_Cmean[THREADS_PER_BLOCK];
		__shared__ float s_phase0[THREADS_PER_BLOCK];
		__shared__ float s_phaseShift[THREADS_PER_BLOCK];
		__shared__ float s_frequency[THREADS_PER_BLOCK];
		__shared__ float s_h[THREADS_PER_BLOCK];
		__shared__ rcm_index_address_t s_row[THREADS_PER_BLOCK];

		unsigned oscillator = bOscillator + tid;

		s_Cmean[tid] = g_Cmean[oscillator] > 0 ?  g_Cmean[oscillator] : 1;
		s_phase0[tid] = g_phase0[oscillator];
		s_frequency[tid] = g_frequency[oscillator];
		s_phaseShift[tid] = 0.0f;
		s_row[tid] = rcm_indexAddress(oscillator, g_rcm);
		__syncthreads();

		float h = 0.05;

		/* now compute the incoming phase for each sequentially */
		//! \todo cut loop short when we get to end?
		for(unsigned iTarget=0; iTarget < THREADS_PER_BLOCK; iTarget+= 1) {
			if(bv_isSet(bOscillator+iTarget, s_valid)) {
				computePhaseShift(g_frequency, s_Cmean[iTarget], cycle, s_partitionSize,
						s_params, g_nstate, g_rcm,
						s_row[iTarget], s_phase0[iTarget], s_frequency[iTarget], h,
						s_phaseShift + iTarget);
			}
		}

		/* Set next state for THREADS_PER_BLOCK oscillators */
		//! \todo remove this test. Use the earlier escape.
		if(bv_isSet(oscillator, s_valid)) {
			float phase = s_phase0[tid] + s_phaseShift[tid];
			g_phase1[oscillator] = fmodf(phase, 2.0f*M_PI) + (phase < 0.0f ? 2.0f*M_PI: 0.0f);
		}
	}

	/* Normally the firing is written back to global memory here. The
	 * oscillators do not fire, so just leave it as it is */
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
		nrng_t /* ignored */,
		uint32_t* d_valid,
		uint32_t* /* ignored */,
		float* d_istim,
		float* d_current,
		uint32_t* /* ignored */,
		unsigned* /* ignored */,
		nidx_dt* /* ignored */,
		rcm_dt* d_rcm)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(localPartitionCount);

	/*! \todo add support for 'singleton' types, which do not mix */
	assert(globalPartitionCount == localPartitionCount);

	updateOscillators<<<dimGrid, dimBlock, 0, stream>>>(
			cycle, d_partitionSize, d_params,
			df_neuronParameters, df_neuronState, d_valid,
			*d_rcm);

	return cudaGetLastError();
}


cuda_update_neurons_t* test_update = &cuda_update_neurons;




/* Run model backwards without coupling in order to fill history */
__global__
void
initOscillators(
		unsigned* g_partitionSize,
		param_t* g_params,
		float* g_nparams,
		float* g_nstate,
		uint32_t* g_valid)
{
	__shared__ unsigned s_partitionSize;
	__shared__ param_t s_params;
	__shared__ uint32_t s_valid[S_BV_PITCH];

	unsigned tid = threadIdx.x;

	loadParameters(g_params, &s_params);
	if(tid == 0) {
		s_partitionSize = g_partitionSize[CURRENT_PARTITION];
    }
	__syncthreads();

	bv_copy(g_valid + CURRENT_PARTITION * s_params.pitch1, s_valid);
	__syncthreads();

	/* Natural frequency of oscillations */
	//const float* g_frequency = g_nparams + CURRENT_PARTITION * s_params.pitch32;

	const float* neuronParameters = g_nparams + CURRENT_PARTITION * s_params.pitch32;
	size_t neuronParametersSize = PARTITION_COUNT * s_params.pitch32;	
     	const float* g_frequency = neuronParameters + 0 * neuronParametersSize;
    

	for(unsigned t=0; t < MAX_HISTORY_LENGTH-1; t++) {

		/* These unsigned values wrap around */
		unsigned current = 0U - t;
		unsigned previous = 0U - (t+1U);

		/* Current phase */
		const float* g_phase0 =
			state<MAX_HISTORY_LENGTH, 1, STATE_PHASE>(current, s_params.pitch32, g_nstate);

		/* Next phase */
		float* g_phase1 =
			state<MAX_HISTORY_LENGTH, 1, STATE_PHASE>(previous, s_params.pitch32, g_nstate);

		for(unsigned bOscillator=0; bOscillator < s_partitionSize; bOscillator += THREADS_PER_BLOCK) {
			unsigned oscillator = bOscillator + tid;
			if(bv_isSet(oscillator, s_valid)) {
				/* negate frequency to run backwards */
				float phase = g_phase0[oscillator] - g_frequency[oscillator];
				g_phase1[oscillator] = fmodf(phase, 2.0f*M_PI) + (phase < 0.0f ? 2.0f*M_PI: 0.0f);
			}
		}
	}
}



extern "C"
NEMO_PLUGIN_DLL_PUBLIC
cudaError_t
cuda_init_neurons(
		unsigned globalPartitionCount,
		unsigned localPartitionCount,
		unsigned basePartition,
		unsigned* d_partitionSize,
		param_t* d_params,
		float* df_neuronParameters,
		float* df_neuronState,
		nrng_t /* rng */,
		uint32_t* d_valid)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(localPartitionCount);

	/*! \todo add support for 'singleton' types, which do not mix */
	assert(globalPartitionCount == localPartitionCount);

	initOscillators<<<dimGrid, dimBlock>>>(
			d_partitionSize, d_params, df_neuronParameters, df_neuronState, d_valid);

	return cudaGetLastError();
}

cuda_init_neurons_t* test_init = &cuda_init_neurons;

#endif
