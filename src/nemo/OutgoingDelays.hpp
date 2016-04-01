#ifndef NEMO_OUTGOING_DELAYS_HPP
#define NEMO_OUTGOING_DELAYS_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <map>
#include <set>
#include <boost/unordered_map.hpp>
#include <sstream>
#include <nemo/internal_types.h>

namespace nemo {

/*! Per-neuron collection of outoing delays (accumulation) */
class OutgoingDelaysAcc
{
	public :

		OutgoingDelaysAcc() : m_maxDelay(0) { }

		/*! \param neuron global index 
		 *  \param delay
		 */
		void addDelay(nidx_t neuron, delay_t delay) {
			m_delays[neuron].insert(delay);
			m_maxDelay = std::max(m_maxDelay, delay);
		}

		delay_t maxDelay() const { return m_maxDelay; }

		void clear() { m_delays.clear() ; }
	
	private :

		friend class OutgoingDelays;

		std::map<nidx_t, std::set<delay_t> > m_delays;

		delay_t m_maxDelay;
};



/*! Per-neuron collection of outgoing delays (run-time) */
class OutgoingDelays
{
	public :

		OutgoingDelays();

		/*
		 * \param maxIdx
		 * 		max source neuron index which will be used for run-time queries
		 */
		OutgoingDelays(const OutgoingDelaysAcc&);

		void init(const OutgoingDelaysAcc&);

		delay_t maxDelay() const { return m_maxDelay; }

		typedef std::vector<delay_t>::const_iterator const_iterator;

		/*! \param neuron
		 * 		global neuron index
		 *  \return
		 *  	iterator pointing to first delay for the \a neuron
		 */
		const_iterator begin(nidx_t neuron) const;

		/*! \param neuron
		 * 		global neuron index
		 *  \return
		 *  	iterator pointing beyond the last delay for the \a neuron
		 */
		const_iterator end(nidx_t neuron) const;

		/*! \return a bitwise representation of the delays for a single source.
		 * The least significant bit corresponds to a delay of 1 */
		uint64_t delayBits(nidx_t source) const;
		
	private :

		/*! \note Did some experimentation with replacing std::vector with raw
		 * arrays but this did not improve performance appreciably. The main
		 * cost of using this data structure is in calling find on the hash
		 * table */
		boost::unordered_map<nidx_t, std::vector<delay_t> > m_data;

		delay_t m_maxDelay;

		OutgoingDelays(const OutgoingDelays& );
		OutgoingDelays& operator=(const OutgoingDelays&);

		bool hasSynapses(nidx_t source) const;
};

}

#endif
