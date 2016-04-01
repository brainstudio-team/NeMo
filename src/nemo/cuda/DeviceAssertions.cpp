/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "DeviceAssertions.hpp"

#include <cuda_runtime.h>
#include <boost/format.hpp>

#include <nemo/config.h>
#include "device_assert.cu_h"
#include "exception.hpp"
#include "kernel.cu_h"

namespace nemo {
	namespace cuda {


DeviceAssertions::DeviceAssertions(unsigned partitionCount) :
	m_partitionCount(partitionCount),
	mh_mem(partitionCount * THREADS_PER_BLOCK, 0)
{
	//! \todo could probably inline the clearing here
	CUDA_SAFE_CALL(::clearDeviceAssertions());
}



void
DeviceAssertions::check(unsigned cycle) throw (nemo::exception)
{
#ifdef NEMO_CUDA_DEVICE_ASSERTIONS
	using boost::format;

	uint32_t* h_mem = &mh_mem[0];
	CUDA_SAFE_CALL(::getDeviceAssertions(m_partitionCount, h_mem));

	for(unsigned partition=0; partition < m_partitionCount; ++partition) {
		for(unsigned thread=0; thread < THREADS_PER_BLOCK; ++thread) {
			uint32_t line = h_mem[assertion_offset(partition, thread)];
			if(line != 0) {
				throw nemo::exception(NEMO_CUDA_ASSERTION_FAILURE,
						str(format("Device assertion failure for partition %u thread %u in line %u during cycle %u. Only the first assertion failure is reported and the exact file is not known")
							//% partition % thread % line % cycle));
							% partition % thread % (line >> 16) % cycle));
			}
		}
	}
#endif
	return;
}

	} // end namespace cuda
} // end namespace nemo
