/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <boost/tuple/tuple.hpp>

#include <nemo/util.h>
#include <nemo/construction/RCM.hpp>
#include <nemo/cuda/runtime/RCM.hpp>
#include <nemo/cuda/device_memory.hpp>
#include <nemo/cuda/parameters.cu_h>
#include <nemo/cuda/rcm.cu_h>
#include <nemo/cuda/kernel.cu_h>

namespace nemo {
	namespace cuda {
		namespace runtime {


inline
rcm_index_address_t
make_rcm_index_address(unsigned start, unsigned len)
{
	return make_uint2(start, len);
}



RCM::RCM(size_t partitionCount, construction_t& h_rcm):
	m_allocated(0),
	m_planeSize(h_rcm.size())
{
	using namespace boost::tuples;

	/* Even if there are no connections, we still need an index if the kernel
	 * assumes there /may/ be connections present. */
	bool empty = h_rcm.synapseCount() == 0;

	std::vector<uint32_t>& h_data = h_rcm.m_data;
	std::vector<uint32_t>& h_forward = h_rcm.m_forward;

	if(h_rcm.m_useData && !empty) {
		md_data = d_array<rsynapse_t>(m_planeSize, false, "rcm (data)");
		memcpyToDevice(md_data.get(), h_data, m_planeSize);
		h_data.clear();
		m_allocated += m_planeSize * sizeof(rsynapse_t);
	}

	if(h_rcm.m_stdpEnabled && !empty) {
		md_accumulator = d_array<weight_dt>(m_planeSize, true, "rcm (accumulator)");
		m_allocated += m_planeSize * sizeof(weight_dt);
	}

	if(h_rcm.m_useWeights && !empty) {
		md_weights = d_array<float>(m_planeSize, false, "rcm (weights)");
		memcpyToDevice(md_weights.get(), h_rcm.m_weights, m_planeSize);
		m_allocated += m_planeSize * sizeof(float);
	}

	if(h_rcm.m_useForward && !empty) {
		md_forward = d_array<uint32_t>(m_planeSize, false, "rcm (forward address)");
		memcpyToDevice(md_forward.get(), h_forward, m_planeSize);
		h_forward.clear();
		m_allocated += m_planeSize * sizeof(uint32_t);
	}

	const size_t maxNeuronCount = partitionCount * MAX_PARTITION_SIZE;
	std::vector<rcm_index_address_t> h_address(maxNeuronCount, INVALID_RCM_INDEX_ADDRESS);
	std::vector<rcm_address_t> h_index;

	/* Populate the host-side data structures */

	typedef construction_t::warp_map::const_iterator iterator;

	size_t allocated = 0; // words, so far

	for(iterator i = h_rcm.m_warps.begin(); i != h_rcm.m_warps.end(); ++i) {

		const key_t k = i->first;
		const unsigned targetPartition = get<0>(k);
		const unsigned targetNeuron = get<1>(k);
		const std::vector<size_t>& row = i->second;

		h_address.at(rcm_metaIndexAddress(targetPartition, targetNeuron)) =
			make_rcm_index_address(allocated, row.size());

		/* Each row in the index is padded out to the next warp boundary */
		size_t nWarps = row.size();
		size_t nWords = ALIGN(nWarps, WARP_SIZE);
		size_t nPadding = nWords - nWarps;

		std::copy(row.begin(), row.end(), std::back_inserter(h_index));          // data
		std::fill_n(std::back_inserter(h_index), nPadding, INVALID_RCM_ADDRESS); // padding

		allocated += nWords;
	}

	/* Copy meta index to device */
	md_metaIndex = d_array<rcm_index_address_t>(h_address.size(), false, "RCM index addresses");
	memcpyToDevice(md_metaIndex.get(), h_address);
	m_allocated += h_address.size() * sizeof(rcm_index_address_t);

	/* Copy index data to device */
	if(allocated != 0) {
		md_index = d_array<rcm_address_t>(allocated, false, "RCM index data");
		memcpyToDevice(md_index.get(), h_index);
		m_allocated += allocated * sizeof(rcm_address_t);
	}

	md_rcm.data = md_data.get();
	md_rcm.forward = md_forward.get();
	md_rcm.accumulator = md_accumulator.get();
	md_rcm.weights = md_weights.get();
	md_rcm.index = md_index.get();
	md_rcm.meta_index = md_metaIndex.get();
}



void
RCM::clearAccumulator()
{
	d_memset(md_accumulator.get(), 0, m_planeSize*sizeof(weight_dt));
}


		}
	}
}
