#ifndef NEMO_CPU_PLUGIN_NEURON_MODEL_H
#define NEMO_CPU_PLUGIN_NEURON_MODEL_H

/* Common API for CPU neuron model plugins */

#include <nemo/config.h>
#include <nemo/internal_types.h>
#include <nemo/RNG.hpp>

#ifdef __cplusplus
extern "C" {
#endif

/*! Update a number of neurons in a contigous range
 *
 * \param currentEPSP input current due to EPSPs
 * \param currentIPSP input current due to IPSPs
 * \param currentExternal externally (user-provided input current)
 * \param cycle current simulation cycle
 *
 * \post currentExternal contain all 0
 *
 * There is no need to clear currentEPSP and currentIPSP
 */
typedef void cpu_update_neurons_t(
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
		void* rcm_ptr);


/*! Initialise all neurons in the network 
 *
 * For neuron types which requires some state history this may be required,
 * whereas for types which only stores the current value, this step is
 * redundant.
 */
typedef void cpu_init_neurons_t(
		unsigned start, unsigned end,
		float* paramBase, size_t paramStride,
		float* stateBase, size_t stateHistoryStride, size_t stateVarStride,
		RNG rng[]);


#ifdef __cplusplus
}
#endif

#endif
