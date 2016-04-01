#include <nemo/fixedpoint.hpp>
#include <nemo/plugins/QIF.h>

#include <nemo/config.h>
#include "neuron_model.h"
#include <nemo/internal_types.h>

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

	const float* a0 = paramBase + PARAM_A * paramStride;

	const size_t historyLength = 1;

	/* Current state */
	size_t b0 = cycle % historyLength;

	const float* v0 = stateBase + b0 * stateHistoryStride + STATE_V * stateVarStride;

	/* Next state */
	size_t b1 = (cycle+1) % historyLength;
	float* v1 = stateBase + b1 * stateHistoryStride + STATE_V * stateVarStride;

	float dt = 0.1f;
	float a1 = 0.5f*dt;
	float a2 = dt;
	float bb1 = dt/6.0f;
	//float RevE = 0.0f;
	//float RevI = -70.0f;
	unsigned int inc_max= (unsigned int)(1/dt);

	for(unsigned int n=start; n < end; ++n) {

		float Excit = currentEPSP[n];
		float Inhib = currentIPSP[n];
 		float Exter = currentExternal[n];


		/* no need to clear current?PSP. */

		//! \todo clear this outside kernel
		currentExternal[n] = 0.0f;

		fired[n] = 0;

		float v = v0[n];
		float a = a0[n];

		for(unsigned t=0; t<inc_max; ++t) {
			if(!fired[n])
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

				v = v + bb1*(k1+2*k2+2*k3+k4);

				fired[n] = v>=1;
		       	}
		}

		fired[n] |= fstim[n];
		fstim[n] = 0;
		recentFiring[n] = (recentFiring[n] << 1) | (uint64_t) fired[n];

		if(fired[n]) {
			v = 0;
			// LOG("c%lu: n%u fired\n", elapsedSimulation(), m_mapper.globalIdx(n));
		}

		v1[n] = v;
	}
}


cpu_update_neurons_t* test = &cpu_update_neurons;


#include "default_init.c"
