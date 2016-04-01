#include <cassert>
#include <nemo/fixedpoint.hpp>
#include <nemo/plugins/Izhikevich2007DopSyn.h>

#include "neuron_model.h"

const unsigned SUBSTEPS = 4;
const float SUBSTEP_MULT = 0.25f;


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
	const float* d1 = paramBase + PARAM_2007D1 * paramStride;
	const float* d2 = paramBase + PARAM_2007D2 * paramStride;
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

		// STEP 2: Increment g according to inputs. (and later to short-term facilitation)!
		ge += (1.0f + MSN_BETA1 * d1[nl]) * (1.0f - MSN_BETA2 * d2[nl]) * currentEPSP[ng];
		gi += currentIPSP[ng];

		// STEP 3: Boundaries.
		/*if(ge < 0.00001)           ge = 0;  // ZAF: Whould this restriction be plausible??
		else */if(ge > G_exc[nl])    ge = G_exc[nl];

		/*if(gi > 0.00001)           gi = 0;
		else */if(gi < -G_inh[nl])   gi = -G_inh[nl];

		ge1[nl] = ge;
		gi1[nl] = gi;

		// Update input
      /*float I = (1.0f + MSN_BETA1 * d1[nl]) * (1.0f - MSN_BETA2 * d2[nl]) * currentEPSP[ng] 
                + currentIPSP[ng] 
                + currentExternal[ng];*/
      float I = ge*(E_exc[nl] - v) - gi*(E_inh[nl] - v) + currentExternal[ng];
		// ZAF: SOS!!!!! TO CHANGE THIS BACK! - Update: It seems to work like that.
		//																Justify why!!
		// -----------------------------------------------------------------------

		/* no need to clear current?PSP. */

		//! \todo clear this outside kernel
		currentExternal[ng] = 0.0f;

		if(sigma[nl] != 0.0f) {
			I += sigma[nl] * nrand(&rng[nl]);
		}

		fired[ng] = 0;

		float oneOverC = 1.0f/C[nl];
      float msn_Vr, msn_d, msn_k;

		for(unsigned t=0; t<SUBSTEPS; ++t) {
			if(!fired[ng]) {

                // Effects of dopamine in MSN neurons
                msn_Vr = Vr[nl] * (1 + MSN_K * d1[nl]);
                msn_d = d[nl] * (1 - MSN_L * d1[nl]);
                msn_k = k[nl] * (1 - MSN_ALPHA * d2[nl]);

                //v += SUBSTEP_MULT * ((0.04* v + 5.0) * v + 140.0- u + I);
                v += SUBSTEP_MULT * oneOverC * (msn_k * (v - msn_Vr) * (v - Vt[nl])  - u + I);
				//u += SUBSTEP_MULT * (a[nl] * (b[nl] * v - u));
                u += SUBSTEP_MULT * (a[nl] * (b[nl] * (v - msn_Vr) - u));
				fired[ng] = v >= vpeak[nl];
			}
		}

		fired[ng] |= fstim[ng];
		fstim[ng] = 0;
		recentFiring[ng] = (recentFiring[ng] << 1) | (uint64_t) fired[ng];

		if(fired[ng]) {
			v = c[nl];
            u += msn_d;
			// LOG("c%lu: n%u fired\n", elapsedSimulation(), m_mapper.globalIdx(n));
		}

		u1[nl] = u;
		v1[nl] = v;
	}
}


cpu_update_neurons_t* test = &cpu_update_neurons;


#include "default_init.c"



















