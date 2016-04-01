/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Delays.hpp"

#include <nemo/util.h>
#include <nemo/cuda/kernel.cu_h>
#include <nemo/cuda/device_memory.hpp>
#include <nemo/cuda/construction/FcmIndex.hpp>

namespace nemo {
	namespace cuda {
		namespace runtime {


Delays::Delays(unsigned partitionCount, const construction::FcmIndex& index) :
	mb_allocated(0)
{
	using namespace boost::tuples;

	size_t width = ALIGN(MAX_DELAY, 32);
	size_t height = partitionCount * MAX_PARTITION_SIZE;

	/* Allocate temporary host-side data */
	std::vector<delay_dt> h_data(height * width);
	std::vector<unsigned> h_fill(height, 0U);

	/* Populate */
	//! \todo get this from somewhere
	for(construction::FcmIndex::iterator i = index.begin(); i != index.end(); ++i) {
		const construction::FcmIndex::index_key& k = i->first;
		pidx_t p  = get<0>(k);
		nidx_t nl = get<1>(k);
		nidx_t n = p * MAX_PARTITION_SIZE + nl;
		delay_t delay1 = get<2>(k);
		h_data[n * width + h_fill[n]] = delay1-1;
		h_fill[n] += 1;
	}

	/* Allocate device data for fill */
	md_fill = d_array<unsigned>(height, true, "delays fill");
	mb_allocated += height * sizeof(unsigned);
	memcpyToDevice(md_fill.get(), h_fill);
	
	/* Allocate device data for data proper */
	md_data = d_array<delay_dt>(height*width, true, "delays");
	mb_allocated += height * width * sizeof(delay_dt);
	memcpyToDevice(md_data.get(), h_data);
}


}	}	} // end namespace
