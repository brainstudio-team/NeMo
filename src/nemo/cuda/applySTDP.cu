#ifndef APPLY_STDP_CU
#define APPLY_STDP_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <cuda.h>

#include <nemo/util.h>
#include <nemo/config.h>

#include "parameters.cu_h"
#include "fcm.cu_h"
#include "fixedpoint.cu"


/*! Apply STDP 
 * 
 * The STDP statistics are stored in reverse CM order with potentiation and
 * depression already combined. This data needs to be re-ordered into the
 * forward order when updating the weight.
 *
 * The new weight is limited by a maximum weight, and is not allowed to fall
 * below 0.
 *
 * prefix r: reverse matrix
 * prefix f: forward matrix
 */
__global__
void
applyStdp(
	unsigned* g_partitionSize,
	param_t* g_params,
	synapse_t* g_fcm,
	rcm_dt g_rcm,
	weight_dt minExcitatoryWeight,
	weight_dt maxExcitatoryWeight,
	weight_dt minInhibitoryWeight,
	weight_dt maxInhibitoryWeight,
	weight_dt reward)
{
	__shared__ unsigned s_partitionSize;

	__shared__ param_t s_params;
	loadParameters(g_params, &s_params);

	if(threadIdx.x == 0) {
		s_partitionSize = g_partitionSize[CURRENT_PARTITION];
	}
	__syncthreads();

	for(unsigned target=0; target < s_partitionSize; ++target) {

		__shared__ rcm_index_address_t row;
		if(threadIdx.x == 0) {
			row = rcm_indexAddress(target, g_rcm);
		}
		__syncthreads();

		/*! \todo add a second loop and pre-load THREADS_PER_BLOCK warp
		 * addresses */
		for(unsigned bIndex=0 ; bIndex < rcm_indexRowLength(row);
				bIndex += THREADS_PER_BLOCK/WARP_SIZE) {

			__shared__ rcm_address_t warp[THREADS_PER_BLOCK/WARP_SIZE];

			if(threadIdx.x < THREADS_PER_BLOCK/WARP_SIZE) {
				warp[threadIdx.x] =
					rcm_address(rcm_indexRowStart(row), bIndex + threadIdx.x, g_rcm);
			}
			__syncthreads();

			size_t r_offset = rcm_offset(warp[threadIdx.x/WARP_SIZE]);
			size_t f_offset = g_rcm.forward[r_offset];
#ifdef NEMO_CUDA_DEBUG_TRACE
			rsynapse_t rsynapse = g_rcm.data[r_offset];
#endif

			if(f_offset != 0) {

				weight_dt w_diff =
					fx_mul(g_rcm.accumulator[r_offset], reward,
							s_params.fixedPointFractionalBits);

				if(w_diff != 0) {
					g_rcm.accumulator[r_offset] = 0;

					weight_dt* gf_weight = (weight_dt*) g_fcm + s_params.fcmPlaneSize * FCM_WEIGHT;

					weight_dt w_old = gf_weight[f_offset];
					weight_dt w_new = 0;
					if(w_old > 0) {
						w_new = min(maxExcitatoryWeight, max(w_old + w_diff, minExcitatoryWeight));
					} else if(w_old < 0) {
						w_new = min(minInhibitoryWeight, max(w_old + w_diff, maxInhibitoryWeight));
					}
					if(w_old != w_new) {
						gf_weight[f_offset] = w_new;
						DEBUG_MSG_STDP("stdp (%u-%u -> %u-%u) %f %+f = %f\n",
								sourcePartition(rsynapse), sourceNeuron(rsynapse),
								CURRENT_PARTITION, target,
								fx_tofloat(w_old), fx_tofloat(w_diff), fx_tofloat(w_new));
					}
				}
			}
			__syncthreads(); // to protect 'warp'
		}
		//! \todo remove sync?
		__syncthreads();
	}
}


__host__
void
applyStdp(
		cudaStream_t stream,
		unsigned partitionCount,
		unsigned* d_partitionSize,
		unsigned fractionalBits,
		param_t* d_params,
		synapse_t* d_fcm,
		rcm_dt* d_rcm,
		float minExcitatoryWeight,
		float maxExcitatoryWeight,
		float minInhibitoryWeight,
		float maxInhibitoryWeight,
		float reward)
{
	dim3 dimBlock(THREADS_PER_BLOCK);
	dim3 dimGrid(partitionCount);

	applyStdp<<<dimGrid, dimBlock, 0, stream>>>(
			d_partitionSize,
			d_params,
			d_fcm,
			*d_rcm,
			fx_toFix(minExcitatoryWeight, fractionalBits),
			fx_toFix(maxExcitatoryWeight, fractionalBits),
			fx_toFix(minInhibitoryWeight, fractionalBits),
			fx_toFix(maxInhibitoryWeight, fractionalBits),
			fx_toFix(reward, fractionalBits));
}


#endif
