#ifndef NEMO_CUDA_PARAMETERS_CU
#define NEMO_CUDA_PARAMETERS_CU

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "parameters.cu_h"

/*! Load global parameters from global memory to shared memory */
__device__
void
loadParameters(const param_t* g_param, param_t* s_param)
{
	int *src = (int*) g_param;
	int *dst = (int*) s_param;
	for(unsigned i=threadIdx.x; i < sizeof(param_t)/sizeof(int); i+=blockDim.x) {
		dst[i] = src[i];
	}
	__syncthreads();
}


#endif
