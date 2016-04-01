#include <algorithm>
#include <cassert>

#include "OutgoingDelays.hpp"
#include "types.hpp"
#include "exception.hpp"

namespace nemo {


OutgoingDelays::OutgoingDelays() :
	m_maxDelay(0)
{
	;
}


OutgoingDelays::OutgoingDelays(const OutgoingDelaysAcc& acc) :
	m_maxDelay(0)
{
	init(acc);
}


void
OutgoingDelays::init(const OutgoingDelaysAcc& acc)
{
	m_maxDelay = acc.maxDelay();

	typedef std::map<unsigned, std::set<unsigned> >::const_iterator it;
	for(it i = acc.m_delays.begin(), i_end = acc.m_delays.end(); i != i_end; ++i) {
		const std::set<unsigned>& delays = i->second;
		unsigned neuron = i->first;
		m_data[neuron] = std::vector<delay_t>(delays.begin(), delays.end());
	}
}



OutgoingDelays::const_iterator
OutgoingDelays::begin(nidx_t source) const
{
	boost::unordered_map<nidx_t, std::vector<delay_t> >::const_iterator found = m_data.find(source);
	if(found == m_data.end()) {
		std::ostringstream oss;
		oss << "OutgoingDelays::begin - Invalid source neuron for source = " << source;
		throw nemo::exception(NEMO_INVALID_INPUT, oss.str());
	}
	return found->second.begin();
}



OutgoingDelays::const_iterator
OutgoingDelays::end(nidx_t source) const
{
	boost::unordered_map<nidx_t, std::vector<delay_t> >::const_iterator found = m_data.find(source);
	if(found == m_data.end()) {
		std::ostringstream oss;
		oss << "OutgoingDelays::end - Invalid source neuron for source = " << source;
		throw nemo::exception(NEMO_INVALID_INPUT, oss.str());
	}
	return found->second.end();
}


bool
OutgoingDelays::hasSynapses(nidx_t source) const
{
	return m_data.find(source) != m_data.end();
}


uint64_t
OutgoingDelays::delayBits(nidx_t source) const
{
	uint64_t bits = 0;
	if(hasSynapses(source)) {
		for(const_iterator d = begin(source), d_end = end(source); d != d_end; ++d) {
			bits = bits | (uint64_t(0x1) << uint64_t(*d - 1));
		}
	}
	return bits;
}

}
