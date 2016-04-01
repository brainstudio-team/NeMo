#ifndef NEMO_RNG_HPP
#define NEMO_RNG_HPP

#include <vector>
#include <nemo/internal_types.h>
#include <nemo/config.h>

struct NEMO_BASE_DLL_PUBLIC RNG {
	unsigned state[4];
};


#ifdef __cplusplus
extern "C" {
#endif

/*! \return uniform random 32-bit number */
NEMO_BASE_DLL_PUBLIC unsigned urand(RNG* rng);

/*! \return normal random number drawn from N(0, 1) */
NEMO_BASE_DLL_PUBLIC float nrand(RNG* rng);

#ifdef __cplusplus
}
#endif


namespace nemo {

/* Generates RNG seeds for neurons in the range [0, maxIdx], and writes the
 * seeds for [minIdx, maxIdx] to the output vector (indices [0, maxIdx -
 * minIdx). Generating and discarding the initial seed values is done in order
 * to always have a fixed mapping from global neuron index to RNG seed values.
 */
NEMO_BASE_DLL_PUBLIC
void
initialiseRng(nidx_t minNeuronIdx, nidx_t maxNeuronIdx, std::vector<RNG>& rngs);

} // end namespace

#endif
