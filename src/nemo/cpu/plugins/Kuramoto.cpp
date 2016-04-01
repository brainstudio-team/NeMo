#include <vector>
#include <cmath>

#include <nemo/util.h>
#include <nemo/fixedpoint.hpp>
#include <nemo/types.hpp>
#include <nemo/runtime/RCM.hpp>

#include <nemo/plugins/Kuramoto.h>

#include "neuron_model.h"



/* Compute the phase shift induced in a single oscillator
 *
 * \return theta_i = sum_j { w_ij * sin(theta_j - theta_i) }
 */
float
sumN(const std::vector<float>& weight,
		const std::vector<float>& sourcePhase,
		unsigned indegree,
		float targetPhase)
{
	float sum = 0.0f;
	for(unsigned sid=0; sid < indegree; ++sid){
		sum += weight[sid] * sinf(sourcePhase[sid] - targetPhase);
		
	}
	return sum;
}



/*! Get phase for a particular oscillator at a particular time */
float*
phase(float* base, size_t stride, unsigned cycle)
{
	size_t b0 = cycle % MAX_HISTORY_LENGTH;
	return base + b0 * stride;
}



/*
 * \return in-degree of the neuron
 */
unsigned
loadIncoming(const nemo::runtime::RCM& rcm,
		unsigned target,
		int cycle,
		float* phaseBase,
		size_t phaseStride,
		std::vector<float>& weight,
		std::vector<float>& sourcePhase)
{
	unsigned indegree = rcm.indegree(target);
	weight.resize(indegree);
	sourcePhase.resize(indegree);

	if(indegree) {
		unsigned si = 0U;
		const std::vector<size_t>& warps = rcm.warps(target);
		for(std::vector<size_t>::const_iterator wi = warps.begin();
				wi != warps.end(); ++wi) {

			const nemo::RSynapse* rsynapse_p = rcm.data(*wi);
			const float* weight_p = rcm.weight(*wi);

			for(unsigned ri=0; ri < rcm.WIDTH && si < indegree; ri++, si++) {
				weight[si] = weight_p[ri];
				sourcePhase[si] = phase(phaseBase, phaseStride, cycle-int(rsynapse_p[ri].delay-1))[rsynapse_p[ri].source];
			}
		}
	}
	return indegree;
}



extern "C"
NEMO_PLUGIN_DLL_PUBLIC
void
cpu_update_neurons(
		unsigned start, unsigned end,
		unsigned cycle,
		float* paramBase, size_t paramStride,
		float* stateBase, size_t stateHistoryStride, size_t stateVarStride,
		unsigned fbits,
		unsigned fstim[],
		RNG rng[],
		float /*currentEPSP*/[],
		float /*currentIPSP*/[],
		float /*currentExternal*/[],
		uint64_t recentFiring[],
		unsigned fired[],
		void* rcm_ptr)
{
	const nemo::runtime::RCM& rcm = *static_cast<nemo::runtime::RCM*>(rcm_ptr);

	const float* frequency = paramBase + 0 * paramStride;
 	const float* g_Cmean = paramBase + 1 * paramStride;

	const float* phase0 = phase(stateBase, stateHistoryStride, cycle);// current
	float* phase1 = phase(stateBase, stateHistoryStride, cycle+1);    // next

	std::vector<float> weight;
	std::vector<float> sourcePhase;

	for(unsigned n=start; n < end; n++) {
		float h = 0.05;	
		const float f = frequency[n];
		float targetPhase = phase0[n];

		const unsigned indegree = loadIncoming(rcm, n, cycle, stateBase, stateHistoryStride, weight, sourcePhase);
		float Cmean = g_Cmean[n] > 0 ?  g_Cmean[n] : 1;
		

		float k0 = f + (sumN(weight, sourcePhase, indegree, targetPhase          )/Cmean);
		float k1 = f + (sumN(weight, sourcePhase, indegree, targetPhase+k0*0.5f*h)/Cmean);
		float k2 = f + (sumN(weight, sourcePhase, indegree, targetPhase+k1*0.5f*h)/Cmean);
		float k3 = f + (sumN(weight, sourcePhase, indegree, targetPhase+k2*h     )/Cmean);

		//! \todo ensure neuron is valid
		//! \todo use precomputed factor and multiply
		targetPhase += h*(k0 + 2*k1 + 2*k2 + k3)/6.0f;
		phase1[n] = fmodf(targetPhase, 2.0f*M_PI) + (targetPhase < 0.0f ? 2.0f*M_PI: 0.0f);
	}
}


cpu_update_neurons_t* test_update = &cpu_update_neurons;


/* Run model backwards without coupling in order to fill history */
extern "C"
NEMO_PLUGIN_DLL_PUBLIC
void
cpu_init_neurons(
		unsigned start, unsigned end,
		float* paramBase, size_t paramStride,
		float* stateBase, size_t stateHistoryStride, size_t stateVarStride,
		RNG rng[])
{
	const float* frequency = paramBase;

	for(unsigned t=0; t < MAX_HISTORY_LENGTH-1; t++) {
		/* The C standard ensures that unsigned values wrap around in the
		 * expected manner */
		unsigned current = 0U - t;
		unsigned previous = 0U - (t+1U);
		const float* phase0 = phase(stateBase, stateHistoryStride, current);
		float* phase1 = phase(stateBase, stateHistoryStride, previous);
		for(unsigned n=start; n < end; n++) {
			//! \todo ensure neuron is valid
			float phase = phase0[n] - frequency[n]; // negate to run backwards
			phase1[n] = fmodf(phase, 2.0f*M_PI) + (phase < 0.0f ? 2.0f*M_PI: 0.0f);
		}
	}
}


cpu_init_neurons_t* test_init = &cpu_init_neurons;
