/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <algorithm>

#include "FiringStimulus.hpp"
#include "device_memory.hpp"
#include "bitvector.cu_h"


namespace nemo {
	namespace cuda {

FiringStimulus::FiringStimulus(size_t partitionCount) :
	mw_pitch(0),
	mw_allocated(0)
{
	//! \todo use 1D data here instead, for more compact form
	size_t height = partitionCount;
	size_t b_pitch = 0;
	void* d_ptr = NULL;
	d_mallocPitch(&d_ptr, &b_pitch, BV_WORD_PITCH * sizeof(uint32_t), height, "Firing stimulus");
	md_arr = boost::shared_array<uint32_t>(static_cast<uint32_t*>(d_ptr), d_free);

	mw_pitch = b_pitch / sizeof(uint32_t);
	mw_allocated = mw_pitch* height;

	d_memset2D(d_ptr, b_pitch, 0x0, height);

	void* h_ptr = NULL;
	mallocPinned(&h_ptr, height * b_pitch);
	mh_arr = boost::shared_array<uint32_t>(static_cast<uint32_t*>(h_ptr), freePinned);
}



void
FiringStimulus::set(
		const Mapper& mapper,
		const std::vector<unsigned>& nidx,
		cudaStream_t stream)
{
	m_haveStimulus = !nidx.empty();

	if(m_haveStimulus) {

		//! \todo clear only the set bits in reset
		std::fill(mh_arr.get(), mh_arr.get() + mw_allocated, 0);	

		for(std::vector<unsigned>::const_iterator i = nidx.begin();
				i != nidx.end(); ++i) {
			//! \todo should check that this neuron exists
			DeviceIdx dev = mapper.deviceIdx(*i);
			size_t word = dev.partition * mw_pitch + dev.neuron / 32;
			size_t bit = dev.neuron % 32;
			mh_arr[word] |= 1 << bit;
		}

		memcpyToDeviceAsync(md_arr.get(), mh_arr.get(), mw_allocated, stream);
	}
}



uint32_t*
FiringStimulus::d_buffer() const
{
	return m_haveStimulus ? md_arr.get() : NULL;
}


void
FiringStimulus::reset()
{
	m_haveStimulus = false;
}


size_t
FiringStimulus::d_allocated() const
{
	return mw_allocated * sizeof(uint32_t);
}


	}
}
