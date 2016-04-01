#include <cassert>
#include <nemo/fixedpoint.hpp>
#include <nemo/plugins/Izhikevich.h>

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
	const float* a = paramBase + PARAM_A * paramStride;
	const float* b = paramBase + PARAM_B * paramStride;
	const float* c = paramBase + PARAM_C * paramStride;
	const float* d = paramBase + PARAM_D * paramStride;
	const float* sigma = paramBase + PARAM_SIGMA * paramStride;
	const float* d1 = paramBase + PARAM_D1 * paramStride;
	const float* d2 = paramBase + PARAM_D2 * paramStride;

	const size_t historyLength = 1;

	/* Current state */
	size_t b0 = cycle % historyLength;
	const float* u0 = stateBase + b0 * stateHistoryStride + STATE_U * stateVarStride;
	const float* v0 = stateBase + b0 * stateHistoryStride + STATE_V * stateVarStride;

	/* Next state */
	size_t b1 = (cycle+1) % historyLength;
	float* u1 = stateBase + b1 * stateHistoryStride + STATE_U * stateVarStride;
	float* v1 = stateBase + b1 * stateHistoryStride + STATE_V * stateVarStride;

	/* Each neuron has two indices: a local index (within the group containing
	 * neurons of the same type) and a global index. */

	int nn = end-start;
	assert(nn >= 0);

#pragma omp parallel for default(shared)
	for(int nl=0; nl < nn; nl++) {

		unsigned ng = start + nl;

		// Zaf: I guess this means inhibitory/excitatory post synaptic potential??
		float I = (1.0f + d1[nl]) * (1.0f - d2[nl]) * currentEPSP[ng] + currentIPSP[ng] + currentExternal[ng];

		/* no need to clear current?PSP. */

		//! \todo clear this outside kernel
		currentExternal[ng] = 0.0f;

		if(sigma[nl] != 0.0f) {
			I += sigma[nl] * nrand(&rng[nl]);
		}

		fired[ng] = 0;

		float u = u0[nl];
		float v = v0[nl];

		for(unsigned t=0; t<SUBSTEPS; ++t) {
			if(!fired[ng]) {
				v += SUBSTEP_MULT * ((0.04* v + 5.0) * v + 140.0- u + I);
				u += SUBSTEP_MULT * (a[nl] * (b[nl] * v - u));
				fired[ng] = v >= 30.0;
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

