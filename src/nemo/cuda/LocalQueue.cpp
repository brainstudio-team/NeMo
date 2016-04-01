/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <nemo/util.h>
#include "kernel.cu_h"

#include "LocalQueue.hpp"
#include "device_memory.hpp"
#include "exception.hpp"


namespace nemo {
	namespace cuda {




LocalQueue::LocalQueue(
		size_t partitionCount,
		size_t partitionSize,
		unsigned maxDelay) :
	mb_allocated(0)
{
	/* In the worst case all neurons have synapses at all delays and all
	 * neurons constantly fire. In practice this will be approximately
	 * 1024 x 64 x 4B = 256kB. A single partition requires MAX_DELAYS (e.g. 64)
	 * such entries for a total of 16MB. For 128 partitions we require 2GB of
	 * queue space.
	 *
	 * The worst-case assumptions are pretty severe, however, so we should be
	 * able to reduce this by a factor of ten, say.
	 */
	/*! \todo count the actual number of delay bits set for any partition. Use
	 * this number instead of partitionSize * MAX_DELAY */
	size_t width = ALIGN(partitionSize * maxDelay, 32) * sizeof(lq_entry_t);

	/* We need one such queue for each partition and each delay. We can thus
	 * end up with 8k separate queues. This could result in a total of 268MB
	 * used for this queue alone. */
	size_t height = partitionCount * maxDelay;

	/* allocate space for the queue fill. For simplicity we allocate this based
	 * on the maximum possible delay, rather than the maximum actual delay. */
	void* d_fill;
	size_t len = height * sizeof(unsigned);
	d_malloc(&d_fill, len, "local queue fill");
	md_fill = boost::shared_ptr<unsigned>(static_cast<unsigned*>(d_fill), d_free);
	d_memset(d_fill, 0, len);
	mb_allocated += len;

	void* d_data;
	size_t bpitch;
	d_mallocPitch(&d_data, &bpitch, width, height, "local queue");
	mb_allocated += bpitch * height;
	md_data = boost::shared_ptr<lq_entry_t>(static_cast<lq_entry_t*>(d_data), d_free);

	/* We don't need to clear the queue. It will generally be full of garbage
	 * anyway. The queue fill struct must be used to determine what's valid
	 * data */

	size_t wpitch = bpitch / sizeof(lq_entry_t);
	CUDA_SAFE_CALL(setLocalQueuePitch(wpitch));
}

	} // end namespace cuda
} // end namespace nemo
