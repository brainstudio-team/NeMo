#include "neuron_model.h"

extern "C"
NEMO_PLUGIN_DLL_PUBLIC
void
cpu_update_neurons(
		unsigned start, unsigned end,
		unsigned /* cycle */,
		float* /* paramBase */, size_t /* paramStride */,
		float* /* stateBase */, size_t /* stateHistoryStride */, size_t /* stateVarStride */,
		unsigned /* fbits */,
		unsigned fstim[],
		RNG /*rng*/[],
		float /*currentEPSP*/[],
		float /*currentIPSP*/[],
		float /*currentExternal */[],
		uint64_t recentFiring[],
		unsigned fired[],
		void* /* rcm */)
{
	for(unsigned n=start; n < end; n++) {
		fired[n] = fstim[n];
		fstim[n] = 0;
		recentFiring[n] = (recentFiring[n] << 1) | (uint64_t) fired[n];
	}
}


cpu_update_neurons_t* test = &cpu_update_neurons;

#include "default_init.c"
