#ifndef NEMO_CUDA_CURRENT_CU
#define NEMO_CUDA_CURRENT_CU

/*! \file current.cu Functions related to neuron input current */

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */


/*! \brief Load externally provided current stimulus from gmem
 *
 * The user can provide per-neuron current stimulus
 * (nemo::cuda::Simulation::addCurrentStimulus).
 *
 * \param[in] partition
 *		\i global index of partition
 * \param[in] psize
 *		number of neurons in current partition
 * \param[in] pitch
 *		pitch of g_current, i.e. distance in words between each partitions data
 * \param[in] g_current
 *		global memory vector containing current for all neurons in partition.
 *		If set to NULL, no input current will be delivered.
 * \param[out] s_current
 *		shared memory vector which will be set to contain input stimulus (or
 *		zero, if there's no stimulus).
 *
 * \pre neuron < size of current partition
 * \pre all shared memory buffers have at least as many entries as the size of
 * 		the current partition
 *
 * \see nemo::cuda::Simulation::addCurrentStimulus
 */
__device__
void
loadCurrentStimulus(
		unsigned partition,
		unsigned psize,
		size_t pitch,
		const float* g_current,
		float* s_current)
{
	if(g_current != NULL) {
		for(unsigned nbase=0; nbase < psize; nbase += THREADS_PER_BLOCK) {
			unsigned neuron = nbase + threadIdx.x;
			unsigned pstart = partition * pitch;
			float stimulus = g_current[pstart + neuron];
			s_current[neuron] = stimulus;
#ifdef NEMO_CUDA_PLUGIN_DEBUG_TRACE
			DEBUG_MSG_SYNAPSE("c%u %u-%u: +%f (external)\n",
					s_cycle, partition, neuron, g_current[pstart + neuron]);
#endif
		}
	} else {
		for(unsigned nbase=0; nbase < psize; nbase += THREADS_PER_BLOCK) {
			unsigned neuron = nbase + threadIdx.x;
			s_current[neuron] = 0;
		}
	}
	__syncthreads();
}



/*
 * \param pcount
 *		partition count considering \em all neuron types. Note that for some
 *		kernels PARTITION_COUNT refers to this global count, whereas for other
 *		kernels it refers to a local count.
 * \param partition
 *		\em global index of current partition
 *
 * \return gmem pointer to accumulated incoming excitatory current
 */
__device__
float*
incomingExcitatory(float* g_base, unsigned /* pcount */, unsigned partition, size_t pitch32)
{
	return g_base + partition * pitch32;
}



/*
 * \param pcount
 *		partition count considering \em all neuron types. Note that for some
 *		kernels PARTITION_COUNT refers to this global count, whereas for other
 *		kernels it refers to a local count.
 * \param partition
 *		\em global index of current partition

 * \return gmem pointer to accumulated incoming inhbitory current
 */
__device__
float*
incomingInhibitory(float* g_base, unsigned pcount, unsigned partition, size_t pitch32)
{
	return g_base + (pcount + partition) * pitch32;
}



#endif
