#ifndef NEMO_CUDA_NVECTOR_CU
#define NEMO_CUDA_NVECTOR_CU

/*! \file nvector.cu Access functions for per-neuron data.
 *
 * See NVector.hpp/NVector.ipp for host-side functionality
 */

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */


/*! \return 32-bit datum for a single neuron in the current partition */
template<typename T>
__device__
T
nv_load32(unsigned neuron, unsigned plane, size_t pitch32, T* g_data)
{
	return g_data[(plane * PARTITION_COUNT + CURRENT_PARTITION) * pitch32 + neuron];
}


/*! \return 64-bit datum for a single neuron in the current partition */
template<typename T>
__device__
T
nv_load64(unsigned neuron, unsigned plane, size_t pitch64, T* g_data)
{
	return g_data[(plane * PARTITION_COUNT + CURRENT_PARTITION) * pitch64 + neuron];
}

#endif
