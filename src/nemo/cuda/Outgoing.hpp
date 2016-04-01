#ifndef NEMO_CUDA_OUTGOING_HPP
#define NEMO_CUDA_OUTGOING_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/shared_ptr.hpp>

#include "outgoing.cu_h"

struct param_t;

namespace nemo {
	namespace cuda {

		namespace construction {
			class FcmIndex;
		}

/*! \brief Runtime mapping from neuron/delay pairs to synapse warp addresses
 *
 * This class wraps the runtime data structures that performs this mapping,
 * which is required during the \ref cuda_global_delivery "global spike
 * delivery step". The synapse warp is a number of synapses which shares the
 * same source partition, source neuron, and delay, the organisation of which
 * is found in \ref nemo::cuda::ConnectivityMatrix.
 *
 * The outgoing data structure is organised in rows which share the same source
 * partition, source neuron, and delay. Each row is padded to a warp-boundary,
 * with invalid entries set to \a INVALID_OUTGOING. Non-empty rows are
 * allocated back-to-back for compactness. Since the data structure is
 * compacted in this manner, lookups are indirect; A separate data structure
 * contains the start address and row length for each row. This is in a
 * non-compact format, so addresses can be computed based on the index (source
 * partition, source neuron, and delay).
 *
 * The \e construction time data structure which is used to build up the
 * mapping found herein is the \ref nemo::cuda::construction::FcmIndex
 * "FCM index".
 *
 * \see nemo::cuda::construction::FcmIndex
 */
class Outgoing
{
	public :

		Outgoing();

		/*! Set the device data containing the outgoing spike groups. */
		Outgoing(size_t partitionCount, const construction::FcmIndex&);

		/*! \return device pointer to the outgoing data */
		outgoing_t* d_data() const { return md_arr.get(); }

		/*! \return device pointer to address table */
		outgoing_addr_t* d_addr() const { return md_rowLength.get(); }

		/*! \return bytes of allocated memory */
		size_t allocated() const { return m_allocated; }

		/* \return
		 * 		the maximum number of incoming warps for any one partition.
		 * 		This is a worst-case value, which assumes that every source
		 * 		neuron fires every cycle for some time. */
		size_t maxIncomingWarps() const { return m_maxIncomingWarps; }

		/*! Fill in the relevant fields of the global parameters struct */
		void setParameters(param_t*) const;

	private :

		void init(size_t partitionCount, const class construction::FcmIndex&);

		boost::shared_ptr<outgoing_t> md_arr; // device data
		size_t m_pitch;                       // max pitch
		unsigned m_step;

		/* Store offset/length pairs here in order to address arbitrary entries
		 * in md_arr where the entries have variable width (for compactness).
		 * md_rowLength have fixed-width entries so addressing is
		 * straightforward (based on a neuron/delay pair) */
		boost::shared_ptr<outgoing_addr_t> md_rowLength; // per neuron/delay pitch

		size_t m_allocated; // bytes

		size_t m_maxIncomingWarps;

		/*! Store the global parameters relating to this class */
		void setParameters(unsigned wpitch);
};

	} // end namespace cuda
} // end namespace nemo

#endif
