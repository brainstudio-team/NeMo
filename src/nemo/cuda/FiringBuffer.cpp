/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <string.h>

#include "device_memory.hpp"
#include "exception.hpp"
#include "FiringBuffer.hpp"

#include "bitvector.cu_h"

namespace nemo {
	namespace cuda {


FiringBuffer::FiringBuffer(const Mapper& mapper):
	m_pitch(0),
	mb_allocated(0),
	m_mapper(mapper)
{
	size_t width = BV_BYTE_PITCH;
	size_t height = m_mapper.partitionCount();

	size_t bytePitch;
	void* d_buffer;
	d_mallocPitch(&d_buffer, &bytePitch, width, height, "firing output");
	md_buffer = boost::shared_ptr<uint32_t>(static_cast<uint32_t*>(d_buffer), d_free);
	d_memset2D(md_buffer.get(), bytePitch, 0, height);
	m_pitch = bytePitch / sizeof(uint32_t);

	size_t mb_allocated = bytePitch * height;
	void* h_buffer;

	mallocPinned(&h_buffer, mb_allocated);
	mh_buffer = boost::shared_ptr<uint32_t>(static_cast<uint32_t*>(h_buffer), freePinned);
	memset(mh_buffer.get(), 0, mb_allocated);

	CUDA_SAFE_CALL(cudaEventCreate(&m_copyDone));
}



FiringBuffer::~FiringBuffer()
{
	cudaEventDestroy(m_copyDone);
}



void
FiringBuffer::sync(cudaStream_t stream)
{
	memcpyFromDeviceAsync(mh_buffer.get(), md_buffer.get(),
			m_mapper.partitionCount() * m_pitch, stream);
	CUDA_SAFE_CALL(cudaEventRecord(m_copyDone, stream));
	CUDA_SAFE_CALL(cudaEventSynchronize(m_copyDone));
	populateSparse(mh_buffer.get());
}



FiredList
FiringBuffer::readFiring()
{
	return m_outputBuffer.dequeueCycle();
}


void
FiringBuffer::populateSparse(const uint32_t* hostBuffer)
{
	unsigned pcount = m_mapper.partitionCount();
	m_outputBuffer.enqueueCycle();

	//! \todo consider processing this using multiple threads
	for(size_t partition=0; partition < pcount; ++partition) {

		size_t partitionOffset = partition * m_pitch;

		for(size_t nword=0; nword < m_pitch; ++nword) {

			/* Within a partition we might go into the padding part of the
			 * firing buffer. We rely on the device not leaving any garbage
			 * in the unused entries */
			uint32_t word = hostBuffer[partitionOffset + nword];
			if(word == 0)
				continue;

			/*! \todo use bitops here to speed up processing. No need to
			 * iterate over every bit. */
			for(size_t nbit=0; nbit < 32; ++nbit) {
				bool fired = (word & (1 << nbit)) != 0;
				if(fired) {
					m_outputBuffer.addFiredNeuron(m_mapper.globalIdx(DeviceIdx(partition, nword*32 + nbit)));
				}
			}
		}
	}
}


	} // end namespace cuda
} // end namespace nemo
