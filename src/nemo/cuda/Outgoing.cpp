/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Outgoing.hpp"

#include <map>
#include <vector>
#include <cuda_runtime.h>
#include <boost/format.hpp>

#include <nemo/util.h>
#include <nemo/bitops.h>
#include <nemo/cuda/construction/FcmIndex.hpp>

#include "device_memory.hpp"
#include "exception.hpp"
#include "kernel.cu_h"
#include "parameters.cu_h"

namespace nemo {
	namespace cuda {

Outgoing::Outgoing() : m_pitch(0), m_allocated(0), m_maxIncomingWarps(0) {}


Outgoing::Outgoing(size_t partitionCount, const construction::FcmIndex& index) :
		m_pitch(0),
		m_allocated(0),
		m_maxIncomingWarps(0)
{
	init(partitionCount, index);
}



bool
compare_warp_counts(
		const std::pair<pidx_t, size_t>& lhs,
		const std::pair<pidx_t, size_t>& rhs)
{
	return lhs.second < rhs.second;
}



/*! Create a new outgoing entry
 *
 * \param partition source partition index
 * \param warpOffset offset into \ref fcm FCM in terms of number of warps
 */
inline
outgoing_t
make_outgoing(pidx_t partition, unsigned warpOffset)
{
	using boost::format;

	if(partition > MAX_PARTITION_COUNT) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Partition index (%u) out of bounds. Kernel supports at most %u partitions")
					% partition % MAX_PARTITION_COUNT));
	}

	return make_uint2(partition, (unsigned) warpOffset);
}



outgoing_addr_t
make_outgoing_addr(unsigned offset, unsigned len)
{
	return make_uint2(offset, len);
}



void
Outgoing::init(size_t partitionCount, const construction::FcmIndex& index)
{
	using namespace boost::tuples;

	size_t height = partitionCount * MAX_PARTITION_SIZE * MAX_DELAY;

	/* allocate device and host memory for row lengths */
	void* d_addr = NULL;
	d_malloc(&d_addr, height * sizeof(outgoing_addr_t), "outgoing spikes (row lengths)");
	md_rowLength = boost::shared_ptr<outgoing_addr_t>(
			static_cast<outgoing_addr_t*>(d_addr), d_free);
	m_allocated = height * sizeof(outgoing_addr_t);
	std::vector<outgoing_addr_t> h_addr(height, make_outgoing_addr(0,0));

	/* allocate temporary host memory for table */
	std::vector<outgoing_t> h_data;

	/* accumulate the number of incoming warps for each partition, such that we
	 * can size the global queue correctly */
	std::map<pidx_t, size_t> incoming;

	size_t allocated = 0; // words, so far
	unsigned wpitch = 0;  // maximum, so far

	/* populate host memory */
	for(construction::FcmIndex::iterator ti = index.begin(); ti != index.end(); ++ti) {

		const construction::FcmIndex::index_key& k = ti->first;

		pidx_t sourcePartition = get<0>(k);
		nidx_t sourceNeuron = get<1>(k);
		delay_t delay1 = get<2>(k);

		/* Allocate memory for this row. Add padding to ensure each row starts
		 * at warp boundaries */
		unsigned nWarps = index.indexRowLength(k);
		unsigned nWords = ALIGN(nWarps, WARP_SIZE);
		wpitch = std::max(wpitch, nWords);
		assert(nWords >= nWarps);
		h_data.resize(allocated + nWords, INVALID_OUTGOING);

		unsigned rowBegin = allocated;
		unsigned col = 0;

		/* iterate over target partitions in a row */
		const construction::FcmIndex::row_t& r = ti->second;
		for(construction::FcmIndex::row_iterator ri = r.begin(); ri != r.end(); ++ri) {

			pidx_t targetPartition = ri->first;
			const std::vector<size_t>& warps = ri->second;
			size_t len = warps.size();
			incoming[targetPartition] += len;
			outgoing_t* p = &h_data[rowBegin + col];
			col += len;
			assert(col <= nWarps);

			/* iterate over warps specific to a target partition */
			for(std::vector<size_t>::const_iterator wi = warps.begin();
					wi != warps.end(); ++wi, ++p) {
				*p = make_outgoing(targetPartition, *wi);
			}
		}

		/* Set address info here, since both start and length are now known.
		 * Col points to next free entry, which is also the length. */
		size_t r_addr = outgoingAddrOffset(sourcePartition, sourceNeuron, delay1-1);
		h_addr[r_addr] = make_outgoing_addr(rowBegin, col);
		allocated += nWords;
	}

	memcpyToDevice(md_rowLength.get(), h_addr);

	/* allocate device memory for table */
	if(allocated != 0) {
		void* d_arr = NULL;
		d_malloc(&d_arr, allocated*sizeof(outgoing_t), "outgoing spikes");
		md_arr = boost::shared_ptr<outgoing_t>(static_cast<outgoing_t*>(d_arr), d_free);
		memcpyToDevice(md_arr.get(), h_data, allocated);
		m_allocated += allocated * sizeof(outgoing_t);
	}

	setParameters(wpitch);

	//! \todo compute this on forward pass (in construction::FcmIndex)
	m_maxIncomingWarps = incoming.size() ? std::max_element(incoming.begin(), incoming.end(), compare_warp_counts)->second : 0;
}



/*! Set parameters for the outgoing data
 *
 * In the inner loop in scatterGlobal the kernel processes potentially multiple
 * rows of outgoing data. We set the relevant loop parameters in constant
 * memory, namely the pitch (max width of row) and step (the number of rows a
 * thread block can process in parallel).
 *
 * It would be possible, and perhaps desirable, to store the pitch/row length
 * on a per-partition basis rather than use a global maximum.
 *
 * \todo store pitch/step on per-partition basis
 * \todo support handling of pitch greater than THREADS_PER_BLOCK
 */
void
Outgoing::setParameters(unsigned maxWarpsPerNeuronDelay)
{
	using boost::format;

	/* We need the step to exactly divide the pitch, in order for the inner
	 * loop in scatterGlobal to work out. */
	m_pitch = std::max(1U, unsigned(ceilPowerOfTwo(maxWarpsPerNeuronDelay)));

	/* Additionally scatterGlobal assumes that m_pitch <= THREADS_PER_BLOCK. It
	 * would possible to modify scatterGLobal to handle the other case as well,
	 * with different looping logic. Separate kernels might be more sensible. */
	assert_or_throw(m_pitch <= THREADS_PER_BLOCK,
			str(format("Outgoing pitch too wide (%u, max %u)") % m_pitch % THREADS_PER_BLOCK));

	m_step = THREADS_PER_BLOCK / m_pitch;
	assert_or_throw(m_step * m_pitch == THREADS_PER_BLOCK, "Invalid outgoing pitch/step");
}



void
Outgoing::setParameters(param_t* params) const
{
	params->outgoingPitch = m_pitch;
	params->outgoingStep = m_step;
}



	} // end namespace cuda
} // end namespace nemo
