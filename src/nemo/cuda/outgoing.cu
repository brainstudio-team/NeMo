#ifndef NEMO_CUDA_OUTGOING_CU
#define NEMO_CUDA_OUTGOING_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "outgoing.cu_h"

__device__ unsigned outgoingTargetPartition(outgoing_t out) { return out.x; } 
__device__ unsigned outgoingWarpOffset(outgoing_t out) { return out.y; }



/*! \return
 *		Address to the address info (meta-address?) for a particular neuron/delay pair
 */
__host__ __device__
size_t
outgoingAddrOffset(unsigned partition, short neuron, short delay0)
{
	return (partition * MAX_PARTITION_SIZE + neuron) * MAX_DELAY + delay0;
}



/*! \return
 *		The address info (offset/length) required to fetch the outgoing warp
 *		entries for a particular neuron/delay pair
 */
__device__
outgoing_addr_t
outgoingAddr(short neuron, short delay0, outgoing_addr_t* g_addr)
{
	return g_addr[outgoingAddrOffset(CURRENT_PARTITION, neuron, delay0)];
}



#endif
