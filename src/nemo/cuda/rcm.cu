#ifndef NEMO_CUDA_RCM_CU
#define NEMO_CUDA_RCM_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <assert.h>

#include <nemo/util.h>

#include "kernel.cu_h"
#include "rcm.cu_h"

/* Bit shifts for packed synapses */
const rsynapse_t R_NEURON_SHIFT = DELAY_BITS;
const rsynapse_t R_PARTITION_SHIFT = R_NEURON_SHIFT + NEURON_BITS;

/* Bit masks for packed synapses */
const rsynapse_t NEURON_MASK = MASK(NEURON_BITS);
const rsynapse_t PARTITION_MASK = MASK(PARTITION_BITS);
const rsynapse_t DELAY_MASK = MASK(DELAY_BITS);


__host__
rsynapse_t
make_rsynapse(unsigned sourcePartition, unsigned sourceNeuron, unsigned delay)
{
	assert(!(sourcePartition & ~PARTITION_MASK));
	assert(!(sourceNeuron & ~NEURON_MASK));
	assert(!(delay & ~DELAY_MASK));
	rsynapse_t s = 0;
	s |= sourcePartition << R_PARTITION_SHIFT;
	s |= sourceNeuron    << R_NEURON_SHIFT;
	s |= delay;
	return s;
}



__device__ __host__
unsigned
sourceNeuron(rsynapse_t rsynapse)
{
    return (rsynapse >> R_NEURON_SHIFT) & NEURON_MASK;
}



__device__ __host__
unsigned
sourcePartition(rsynapse_t rsynapse)
{
    return (rsynapse >> R_PARTITION_SHIFT) & PARTITION_MASK;
}



__device__ __host__
unsigned
r_delay1(rsynapse_t rsynapse)
{
    return rsynapse & DELAY_MASK;
}



__device__
unsigned
r_delay0(rsynapse_t rsynapse)
{
	return r_delay1(rsynapse) - 1;
}



__host__ __device__
size_t
rcm_metaIndexAddress(pidx_t partition, nidx_t neuron)
{
	return partition * MAX_PARTITION_SIZE + neuron;
}



__device__
unsigned
rcm_indexRowStart(rcm_index_address_t addr)
{
	return addr.x;
}


__device__
unsigned
rcm_indexRowLength(rcm_index_address_t addr)
{
	return addr.y;
}



/*! \return address in RCM index for a neuron in current partition */
__device__
rcm_index_address_t
rcm_indexAddress(nidx_t neuron, const rcm_dt& rcm)
{
	return rcm.meta_index[rcm_metaIndexAddress(CURRENT_PARTITION, neuron)];
}


__device__
rcm_address_t
rcm_address(unsigned rowStart, unsigned rowOffset, const rcm_dt& rcm)
{
	return rcm.index[rowStart + rowOffset];
}



/*! \return word offset into RCM for a particular synapse */
__device__
size_t
rcm_offset(rcm_address_t warpOffset)
{
	return warpOffset * WARP_SIZE + threadIdx.x % WARP_SIZE;
}


#endif
