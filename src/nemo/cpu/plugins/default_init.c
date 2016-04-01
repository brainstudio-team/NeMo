#include "neuron_model.h"

extern "C"
NEMO_PLUGIN_DLL_PUBLIC
void
cpu_init_neurons(
		unsigned start, unsigned end,
		float* paramBase, size_t paramStride,
		float* stateBase, size_t stateHistoryStride, size_t stateVarStride,
		RNG rng[])
{
	;
}

cpu_init_neurons_t* test_init = &cpu_init_neurons;
