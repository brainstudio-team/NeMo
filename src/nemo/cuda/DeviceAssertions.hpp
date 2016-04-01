#ifndef DEVICE_ASSERTIONS_HPP
#define DEVICE_ASSERTIONS_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \brief Run-time assertions on the GPU
 *
 * If the kernel is compiled with device assertions (CPP flag
 * DEVICE_ASSERTIONS), the kernel can perform run-time assertions, logging
 * location data to global memory. Only the line-number is recorded, so some
 * guess-work my be required to work out exactly what assertion failed. There
 * is only one assertion failure slot per thread, so it's possible to overwrite
 * an assertion failure.
 *
 * \author Andreas Fidjeland
 */

#include <vector>

#include <nemo/exception.hpp>

#ifdef _MSC_VER
// visual studio warning re non-implementation of throw specifiers
#pragma warning(disable: 4290)
#endif

namespace nemo {
	namespace cuda {

class DeviceAssertions
{
	public :

		/* Allocate device memory for device assertions */
		DeviceAssertions(unsigned partitionCount);

		/* Check whether any device assertions have failed. Only the last
		 * assertion failure for each thread will be reported. Checking device
		 * assertions require reading device memory and can therefore be
		 * costly. */
		void check(unsigned cycle) throw(nemo::exception);

	private :

		unsigned m_partitionCount;

		std::vector<uint32_t> mh_mem;
};

	} // end namespace cuda
} // end namespace nemo

#endif
