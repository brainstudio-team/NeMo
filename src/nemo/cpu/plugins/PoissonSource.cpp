#include "neuron_model.h"

extern "C"
NEMO_PLUGIN_DLL_PUBLIC
void
cpu_update_neurons(
		unsigned start, unsigned end,
		unsigned /* cycle */,
		float* paramBase, size_t /* paramStride */,
		float* /* stateBase */, size_t /* stateHistoryStride */, size_t /* stateVarStride */,
		unsigned /* fbits */,
		unsigned fstim[],
		RNG rng[],
		float /*currentEPSP*/[],
		float /*currentIPSP*/[],
		float /*currentExternal*/[],
		uint64_t recentFiring[],
		unsigned fired[],
		void* /* rcm */)
{
	const float* rate = paramBase;

	for(unsigned ng=start, nl=0U; ng < end; ng++, nl++) {
		unsigned p0 = unsigned(rate[nl] * float(1<<16));
		unsigned p1 = urand(&rng[nl]) & 0xffff;
		fired[ng] = p1 < p0 || fstim[ng];
		fstim[ng] = 0;
		recentFiring[ng] = (recentFiring[ng] << 1) | (uint64_t) fired[ng];
	}
}


cpu_update_neurons_t* test = &cpu_update_neurons;

#include "default_init.c"
