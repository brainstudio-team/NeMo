#include <nemo/fixedpoint.hpp>
#include <nemo/config.h>
#include "neuron_model.h"
#include <nemo/internal_types.h>
#include <math.h>

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
	#define STATE_V 0
	#define STATE_N 1
	#define STATE_M 2
	#define STATE_H 3
	#define STATE_DIR 4

	const size_t historyLength = 1;

	/* Current state */
	size_t b0 = cycle % historyLength;

	const float* v0 = stateBase + b0 * stateHistoryStride + STATE_V * stateVarStride;
	const float* n0 = stateBase + b0 * stateHistoryStride + STATE_N * stateVarStride;
	const float* m0 = stateBase + b0 * stateHistoryStride + STATE_M * stateVarStride;
	const float* h0 = stateBase + b0 * stateHistoryStride + STATE_H * stateVarStride;
	const float* dir0 = stateBase + b0 * stateHistoryStride + STATE_DIR * stateVarStride;
	

	/* Next state */
	size_t b1 = (cycle+1) % historyLength;

	float* v1 = stateBase + b1 * stateHistoryStride + STATE_V * stateVarStride;
	float* n1 = stateBase + b1 * stateHistoryStride + STATE_N * stateVarStride;	
	float* m1 = stateBase + b1 * stateHistoryStride + STATE_M * stateVarStride;	
	float* h1 = stateBase + b1 * stateHistoryStride + STATE_H * stateVarStride;	
	float* dir1 = stateBase + b1 * stateHistoryStride + STATE_DIR * stateVarStride;
	
	float dt = 0.001f; // Simulation time increment
	float gNa = 120.0f;
	float gK = 36.0f;
	float gL = 0.3f;
	float ENa = 115.0f-65.0f;
	float EK = -12.0f-65.0f;
	float EL = 10.6f-65.0f;
	float C = 1.0f;
	float RevE = 0.0f;
	float RevI = -70.0f;
	int inc_max= (int)(1/dt);

	for(unsigned int nn=start; nn < end; ++nn) {

		float v = v0[nn];
		float n = n0[nn];
		float m = m0[nn];
		float h = h0[nn];
		float dir = dir0[nn];

		
		float Excit = currentEPSP[nn];
		float Inhib = currentIPSP[nn];
 		float Exter = currentExternal[nn];
		
		/* no need to clear current?PSP. */

		//! \todo clear this outside kernel
		currentExternal[nn] = 0.0f;
		fired[nn] = 0;

		for(unsigned t=0; t<inc_max; ++t) {

			float I = (Excit*(RevE-v)) + (Inhib*((RevI-v)/-1)) + Exter;

		    
			float alphan = (0.1f-0.01f*(v+65.0f))/(exp(1.0f-0.1f*(v+65.0f))-1.0f);
			float alpham = (2.5f-0.1f*(v+65.0f))/(exp(2.5f-0.1f*(v+65.0f))-1.0f);
			float alphah = 0.07f*exp(-(v+65.0f)/20.0f);

			float betan = 0.125f*exp(-(v+65.0f)/80.0f);
			float betam = 4.0f*exp(-(v+65.0f)/18.0f);
			float betah = 1.0f/(exp(3.0f-0.1f*(v+65.0f))+1.0f);


			m = m + dt*(alpham*(1.0f-m)-betam*m);
			n = n + dt*(alphan*(1.0f-n)-betan*n);
			h = h + dt*(alphah*(1.0f-h)-betah*h);

			float Ik = gNa*(m*m*m)*h*(v-ENa) + gK*(n*n*n*n)*(v-EK) + gL*(v-EL);


			float newv = v + dt*(-Ik+I)/C;

			float new_dir = (newv-v);
			float change = dir<0 | newv<-45 ? 0 : new_dir;
			dir = new_dir;

		
			if(!fired[nn] && cycle >= 10)
				fired[nn] = change < -0.000000001;
				
			v=newv;

			
		}

		fired[nn] |= fstim[nn];
		fstim[nn] = 0;
		recentFiring[nn] = (recentFiring[nn] << 1) | (uint64_t) fired[nn];

		if(fired[nn]) {
			// LOG("c%lu: n%u fired\n", elapsedSimulation(), m_mapper.globalIdx(n));
		}

		v1[nn] = v;
		n1[nn] = n;
		m1[nn] = m;
		h1[nn] = h;
		dir1[nn] = dir;	
		
	}
}


cpu_update_neurons_t* test = &cpu_update_neurons;


#include "default_init.c"
