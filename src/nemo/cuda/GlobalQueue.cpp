/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/format.hpp>

#include "GlobalQueue.hpp"

#include <nemo/util.h>

#include "kernel.cu_h"
#include "exception.hpp"
#include "device_memory.hpp"

namespace nemo {
	namespace cuda {

GlobalQueue::GlobalQueue() : mb_allocated(0) {}


void
GlobalQueue::allocate(size_t partitionCount, size_t maxIncomingWarps, double sizeMultiplier)
{
	using boost::format;

	// allocate space for the incoming count (double-buffered)
	void* d_fill;
	size_t len = ALIGN(partitionCount * 2, 32) * sizeof(unsigned);
	d_malloc(&d_fill, len, "global queue fill");
	md_fill = boost::shared_ptr<unsigned>(static_cast<unsigned*>(d_fill), d_free);
	d_memset(d_fill, 0, len);
	mb_allocated = len;

	/* The queue has one entry for incoming spikes for each partition */
	if(partitionCount > MAX_PARTITION_COUNT) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Network contains %u partitions, but kernel supports at most %u")
					% partitionCount % MAX_PARTITION_COUNT));
	}
	size_t height = partitionCount * 2; // double buffered

	/* Each buffer entry (for a particular target partition) is of a fixed size.
	 * The sizing of this is very conservative. In fact the buffer is large
	 * enough that every neuron can fire every cycle. */
	/*! \todo relax this constraint. We'll end up using a very large amount of
	 * space when using a large number of partitions */
	assert(sizeMultiplier > 0.0);
	double mult = std::min(1.0, sizeMultiplier);
	size_t width = size_t(mult * maxIncomingWarps * sizeof(gq_entry_t));

	void* d_buffer;
	size_t bpitch;

	d_mallocPitch(&d_buffer, &bpitch, width, height, "global queue");
	mb_allocated += bpitch * height;

	md_buffer = boost::shared_ptr<gq_entry_t>(static_cast<gq_entry_t*>(d_buffer), d_free);

	/* We don't need to clear the queue. It will generally be full of garbage
	 * anyway. The queue heads must be used to determine what's valid data */

	size_t wpitch = bpitch / sizeof(gq_entry_t);
	CUDA_SAFE_CALL(setGlobalQueuePitch(wpitch));
}

	} // end namespace cuda
} // end namespace nemo
