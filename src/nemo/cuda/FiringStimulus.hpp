#ifndef NEMO_CUDA_FIRING_STIMULUS_HPP
#define NEMO_CUDA_FIRING_STIMULUS_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <boost/shared_array.hpp>

#include <cuda_runtime.h>

#include "Mapper.hpp"

namespace nemo {
	namespace cuda {

/*! \brief Host and Device buffer for firing stimulus
 *
 * Firing stimulus is provided as a bit vector with set bits for neurons which
 * should be stimulated. 
 *
 * Usage:
 * \code
 * for(<every simulation cycle>) {
 *     fstim.set(mapper, <stimulus vector>, stream);
 *     ... do whatever
 *     cudaEventSynchronize(event);
 *     ... perform fire step
 *     fstim.reset()
 * \endcode
 */
class FiringStimulus
{
	public :

		FiringStimulus(size_t partitionCount);

		/*! Initiate transfer of firing stimulus from host to device, recording 
		 * event when the transfer is done. The copy is asynchrous, i.e. the
		 * host can continue working while the copy takes place. */
		void set(const Mapper& mapper, const std::vector<unsigned>& nidx, cudaStream_t);

		//! \todo replace by a step function
		void reset();

		/*! \return device pointer to buffer containing most recent cycle's
		 * worth of firing stimulus, or NULL if there is no stimulus */
		uint32_t* d_buffer() const;

		/*! \return number of bytes of allocated device memory */
		size_t d_allocated() const;

		size_t wordPitch() const { return mw_pitch; }

	private :

		boost::shared_array<uint32_t> md_arr;
		boost::shared_array<uint32_t> mh_arr;

		size_t mw_pitch;
		size_t mw_allocated;

		bool m_haveStimulus;
};

	}
}

#endif
