#ifndef NEMO_CUDA_BITVECTOR_HPP
#define NEMO_CUDA_BITVECTOR_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/shared_array.hpp>

#include "Mapper.hpp"

namespace nemo {
	namespace cuda {


/*! \brief Host and device data for per-neuron bitvector
 *
 * These bit vectors stores per-neuron data in a compact format with a single
 * bit per neuron. The device code for manipulating this data is found in
 * bitvector.cu.
 */
class Bitvector
{
	public :

		/*! Create bit vector and initialise to zero */
		Bitvector(size_t partitionCount, bool hostAlloc);

		/*! \return pointer to device data */
		uint32_t* d_data() const { return md_arr.get(); }

		/*! Set value (in host buffer) for a single neuron */
		void setNeuron(const DeviceIdx& neuron);

		/*! \return number of 32-bit words used for each partition */
		size_t wordPitch() const { return mw_pitch; }

		/*! \return number of bytes of allocated device memory */
		size_t d_allocated() const { return mw_allocated * sizeof(uint32_t); }

		/*! Copy entire host buffer to device and deallocote host memory */
		void moveToDevice();
		
		/*! Copy entire host buffer to the device */
		void copyToDevice();

	private :

		/* scoped_array seems a better fit, but does not support a custom destructor */
		boost::shared_array<uint32_t> md_arr;
		boost::shared_array<uint32_t> mh_arr;

		size_t mw_pitch;
		size_t mw_allocated;
};

	}
}

#endif
