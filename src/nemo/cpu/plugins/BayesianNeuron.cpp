#include <cassert>
#include "math.h"
#include <nemo/fixedpoint.hpp>
#include <nemo/plugins/BayesianNeuron.h>

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
  const float* rOn  = paramBase + PARAM_R_ON * paramStride;
  const float* rOff = paramBase + PARAM_R_OFF * paramStride;
  const float* g0   = paramBase + PARAM_G0 * paramStride;

  const size_t historyLength = 1;

  /* Current state */
  size_t b0 = cycle % historyLength;
  const float* L0 = stateBase + b0 * stateHistoryStride + STATE_L * stateVarStride;
  const float* G0 = stateBase + b0 * stateHistoryStride + STATE_G * stateVarStride;

  /* Next state */
  size_t b1 = (cycle+1) % historyLength;
  float* L1 = stateBase + b1 * stateHistoryStride + STATE_L * stateVarStride;
  float* G1 = stateBase + b1 * stateHistoryStride + STATE_G * stateVarStride;

  /* Each neuron has two indices: a local index (within the group containing
   * neurons of the same type) and a global index. */

  int nn = end-start;
  assert(nn >= 0);

#pragma omp parallel for default(shared)
  for(int nl=0; nl < nn; nl++) {

    unsigned ng = start + nl;

    float I = currentEPSP[ng] + currentIPSP[ng] + currentExternal[ng];

    /* no need to clear current?PSP. */

    //! \todo clear this outside kernel
    currentExternal[ng] = 0.0f;

    fired[ng] = 0;

    float L = L0[nl];
    float G = G0[nl];
    float rOn_this = rOn[nl];
    float rOff_this = rOff[nl];
    float g0_this = g0[nl];

    for(unsigned t=0; t<SUBSTEPS; ++t) {
      if(!fired[ng]) {
        L += SUBSTEP_MULT * (rOn_this * (1 + exp(-L)) - rOff_this * (1 + exp(L)) + I);
        G += SUBSTEP_MULT * (rOn_this * (1 + exp(-G)) - rOff_this * (1 + exp(G)));
        fired[ng] = L > (G + g0_this/2.0);
      }
    }

    fired[ng] |= fstim[ng];
    fstim[ng] = 0;
    recentFiring[ng] = (recentFiring[ng] << 1) | (uint64_t) fired[ng];

    if(fired[ng]) {
      G += g0_this;
    }

    L1[nl] = L;
    G1[nl] = G;
  }
}


cpu_update_neurons_t* test = &cpu_update_neurons;


#include "default_init.c"

