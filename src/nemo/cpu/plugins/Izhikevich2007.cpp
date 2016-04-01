#include <cassert>
#include <nemo/fixedpoint.hpp>
#include <nemo/plugins/Izhikevich2007.h>

#include "neuron_model.h"

const unsigned SUBSTEPS = 4;
const float SUBSTEP_MULT = 0.25f;

#include <iostream>

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
		float currentEPSP[],
		float currentIPSP[],
		float currentExternal[],
		uint64_t recentFiring[],
		unsigned fired[],
		void* /* rcm */)
{
	const float* a = paramBase + PARAM_2007A * paramStride;
	const float* b = paramBase + PARAM_2007B * paramStride;
	const float* c = paramBase + PARAM_2007C * paramStride;
	const float* d = paramBase + PARAM_2007D * paramStride;
	const float* vpeak = paramBase + PARAM_2007VPEAK * paramStride;
	const float* Vr = paramBase + PARAM_2007VR * paramStride;
	const float* Vt = paramBase + PARAM_2007VT * paramStride;
	const float* k = paramBase + PARAM_2007K * paramStride;
	const float* C = paramBase + PARAM_2007CCAP * paramStride;
	const float* sigma = paramBase + PARAM_2007SIGMA * paramStride;
	const float* tMdtOt_exc = paramBase + PARAM_2007TauMinusDtOverTau_Exc * paramStride;
	const float* G_exc = paramBase + PARAM_2007G_Exc * paramStride;
	const float* E_exc = paramBase + PARAM_2007E_Exc * paramStride;
	const float* tMdtOt_inh = paramBase + PARAM_2007TauMinusDtOverTau_Inh * paramStride;
	const float* G_inh = paramBase + PARAM_2007G_Inh * paramStride;
	const float* E_inh = paramBase + PARAM_2007E_Inh * paramStride;


	const size_t historyLength = 1;

	/* Current state */
	size_t b0 = cycle % historyLength;
	const float* u0 = stateBase + b0 * stateHistoryStride + STATE_2007U * stateVarStride;
	const float* v0 = stateBase + b0 * stateHistoryStride + STATE_2007V * stateVarStride;
	const float* ge0 = stateBase + b0 * stateHistoryStride + STATE_2007Ge * stateVarStride;
	const float* gi0 = stateBase + b0 * stateHistoryStride + STATE_2007Gi * stateVarStride;

	/* Next state */
	size_t b1 = (cycle+1) % historyLength;
	float* u1 = stateBase + b1 * stateHistoryStride + STATE_2007U * stateVarStride;
	float* v1 = stateBase + b1 * stateHistoryStride + STATE_2007V * stateVarStride;
	float* ge1 = stateBase + b1 * stateHistoryStride + STATE_2007Ge * stateVarStride;
	float* gi1 = stateBase + b1 * stateHistoryStride + STATE_2007Gi * stateVarStride;

	/* Each neuron has two indices: a local index (within the group containing
	 * neurons of the same type) and a global index. */

	int nn = end-start;
	assert(nn >= 0);

#pragma omp parallel for default(shared)
	for(int nl=0; nl < nn; nl++) {

		unsigned ng = start + nl;

		float ge = ge0[nl];
		float gi = gi0[nl];
		float u = u0[nl];
		float v = v0[nl];

		// -- UPDATE SYNAPSES ----------------------------------------------------

		// STEP 1: Decrease g accordning to tau.
		ge = ge * tMdtOt_exc[nl];
		gi = gi * tMdtOt_inh[nl];

		// STEP 2: Increment g according to inputs.
        if(currentEPSP[ng] > 0.0) ge = 1.0f;
		if(currentIPSP[ng] > 0.0) gi = 1.0f;

		ge1[nl] = ge;
		gi1[nl] = gi;

		// Update input
		float I = G_exc[nl]*ge*(E_exc[nl] - v) + 
                  G_inh[nl]*gi*(E_inh[nl] - v) + currentExternal[ng];

		/* no need to clear current?PSP. */

		//! \todo clear this outside kernel
		currentExternal[ng] = 0.0f;

		if(sigma[nl] != 0.0f) {
			I += C[nl] * sigma[nl] * nrand(&rng[nl]);
		}

		fired[ng] = 0;

		float oneOverC = 1.0f/C[nl];

		for(unsigned t=0; t<SUBSTEPS; ++t) {
			if(!fired[ng]) {
				// Equations
                		v += SUBSTEP_MULT * oneOverC * (k[nl] * (v - Vr[nl]) * (v - Vt[nl])  - u + I);
                		u += SUBSTEP_MULT * (a[nl] * (b[nl] * (v - Vr[nl]) - u));
				fired[ng] = v >= vpeak[nl];
			}
		}

		fired[ng] |= fstim[ng];
		fstim[ng] = 0;
		recentFiring[ng] = (recentFiring[ng] << 1) | (uint64_t) fired[ng];

		if(fired[ng]) {
			v = c[nl];
			u += d[nl];
			// LOG("c%lu: n%u fired\n", elapsedSimulation(), m_mapper.globalIdx(n));
		}

		u1[nl] = u;
		v1[nl] = v;
	}
}


cpu_update_neurons_t* test = &cpu_update_neurons;


#include "default_init.c"



















