#ifndef NEMO_CUDA_TYPES_HPP
#define NEMO_CUDA_TYPES_HPP

#include <nemo/internal_types.h>

typedef unsigned warp_idx;

typedef fix_t weight_dt; // on the device
typedef unsigned int pidx_t; // partition index 

/* On the device both address and weight data are squeezed into 32b */
//! \todo use a union type here?
typedef uint32_t synapse_t;

/* Type for storing (within-partition) neuron indices on the device. We could
 * use uint16_t here to save some shared memory, in exchange for slightly
 * poorer shared memory access patterns */
typedef uint32_t nidx_dt;

typedef uint16_t delay_dt;

#endif
