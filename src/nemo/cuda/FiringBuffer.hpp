#ifndef NEMO_CUDA_FIRING_BUFFER
#define NEMO_CUDA_FIRING_BUFFER

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <boost/shared_ptr.hpp>

#include <cuda_runtime.h>

#include <nemo/internal_types.h>
#include <nemo/FiringBuffer.hpp>
#include "Mapper.hpp"


namespace nemo {
	namespace cuda {

/*! \brief Buffer transferring firing data from device to host
 *
 * This buffer uses pinned memory on the host as traffic through it is likely
 * to be heavy. It is a bit-vector, i.e. a single bit of storage is used for
 * each neuron. The size of data transfers is therefore always fixed (but
 * small) regardless of the firing rate. The conversion from this dense format
 * to a more user-friendly sparse format is handled internally, and the sparse
 * data is returned using \ref readFiring.
 *
 * The firing buffer internally contains a queue with one entry for each
 * cycles' worth of firing. \ref sync() enqueues a new entry to the back, while
 * \ref readFiring deques the oldest entry from the front.
 *
 * On the device side, the firing buffer is filled by \ref storeFiringOutput.
 */
class FiringBuffer {

	public:

		/*! Set up data on both host and device for probing firing */
		FiringBuffer(const Mapper& mapper);

		~FiringBuffer();

		/*! Read firing data from device to host buffer. This should be called
		 * every simulation cycle. */
		void sync(cudaStream_t stream);

		/*! Return oldest buffered cycle's worth of firing */
		FiredList readFiring();

		/*! \return device pointer to the firing buffer */
		uint32_t* d_buffer() const { return md_buffer.get(); }

		/*! \return bytes of allocated device memory */
		size_t d_allocated() const { return mb_allocated; }

		size_t wordPitch() const { return m_pitch; }

	private:

		/* Dense firing buffers on device and host */
		boost::shared_ptr<uint32_t> md_buffer;
		boost::shared_ptr<uint32_t> mh_buffer; // pinned, same size as device buffer
		size_t m_pitch; // in words

		size_t mb_allocated;

		void populateSparse(const uint32_t* hostBuffer);

		Mapper m_mapper;

		nemo::FiringBuffer m_outputBuffer;

		cudaEvent_t m_copyDone;
};

	} // end namespace cuda
} // end namespace nemo

#endif
