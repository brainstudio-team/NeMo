/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Bitvector.hpp"
#include <nemo/exception.hpp>

#include "device_memory.hpp"
#include "bitvector.cu_h"

namespace nemo {
	namespace cuda {


Bitvector::Bitvector(size_t partitionCount, bool hostAlloc) :
	mw_pitch(0),
	mw_allocated(0)
{
	size_t height = partitionCount;
	size_t b_pitch = 0;
	void* d_ptr = NULL;
	d_mallocPitch(&d_ptr, &b_pitch, BV_WORD_PITCH * sizeof(uint32_t), height, "Bit vector");
	md_arr = boost::shared_array<uint32_t>(static_cast<uint32_t*>(d_ptr), d_free);
	mw_pitch = b_pitch / sizeof(uint32_t);
	mw_allocated = mw_pitch * height;
	d_memset2D(d_ptr, b_pitch, 0x0, height);

	if(hostAlloc) {
		mh_arr = boost::shared_array<uint32_t>(new uint32_t[mw_allocated]);
		std::fill(mh_arr.get(), mh_arr.get() + mw_allocated, 0x0);
	}
}



void
Bitvector::setNeuron(const DeviceIdx& n)
{
	size_t word = n.partition * mw_pitch + n.neuron / 32;
	size_t bit = n.neuron % 32;
	mh_arr[word] |= 1 << bit;
}



void
Bitvector::copyToDevice()
{
	if(mh_arr.get() == 0) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "attempt to copy empty host bitvector to device"); 
	}
	memcpyToDevice(md_arr.get(), mh_arr.get(), mw_allocated);
}



void
Bitvector::moveToDevice()
{
	copyToDevice();
	mh_arr.reset();
}


	}
}
