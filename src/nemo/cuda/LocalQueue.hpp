#ifndef NEMO_CUDA_LOCAL_QUEUE_HPP
#define NEMO_CUDA_LOCAL_QUEUE_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/shared_ptr.hpp>
#include "localQueue.cu_h"

namespace nemo {
	namespace cuda {


/*! Queue for sorting spikes temporally for each partition
 *
 * Each partition has a rotating queue with MAX_DELAY slots, each of which is
 * filled with new neuron/delay pairs during the scatter step. This local queue
 * data is later enqueued in a larger global queue. The reason for this
 * two-step operation is that enqueueing directly into the global queue leads
 * to the global queue being much larger (MAX_DELAY x larger) plus the global
 * queue operations being harder to coalesce.
 */
class LocalQueue
{
	public :

		/*! Allocate device data for queue and set it to empty */
		LocalQueue(size_t partitionCount, size_t partitionSize, unsigned maxDelay);

		/*! \return device pointer to queue data */
		lq_entry_t* d_data() const { return md_data.get(); }

		/*! \return device pointer to queue fill data */
		unsigned* d_fill() const { return md_fill.get(); }

		/*! \return bytes of allocated memory */
		size_t allocated() const { return mb_allocated; }

	private :

		/* On the device there a buffer for incoming spike groups for each
		 * (target) partition */
		boost::shared_ptr<lq_entry_t> md_data;

		/* At run-time, we keep track of how many incoming spike groups are
		 * queued for each target partition */
		boost::shared_ptr<unsigned> md_fill;

		size_t mb_allocated;
};

	} // end namespace cuda
} // end namespace nemo

#endif
