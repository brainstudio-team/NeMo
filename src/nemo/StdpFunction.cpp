#include <boost/format.hpp>

#include "StdpFunction.hpp"
#include "exception.hpp"


namespace nemo {

const unsigned StdpFunction::MAX_FIRING_HISTORY = 64;


StdpFunction::StdpFunction(
		const std::vector<float>& prefire,
		const std::vector<float>& postfire,
		float minExcitatoryWeight, float maxExcitatoryWeight,
		float minInhibitoryWeight, float maxInhibitoryWeight) :
	m_prefire(prefire),
	m_postfire(postfire),
	m_minExcitatoryWeight(minExcitatoryWeight),
	m_maxExcitatoryWeight(maxExcitatoryWeight),
	m_minInhibitoryWeight(minInhibitoryWeight),
	m_maxInhibitoryWeight(maxInhibitoryWeight)
{ 
	if(m_maxInhibitoryWeight > 0.0f || m_minInhibitoryWeight > 0.0f) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"STDP function should not have positive limits for inhibitory synapses");
	}

	if(m_maxExcitatoryWeight < 0.0f || m_minExcitatoryWeight < 0.0f) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"STDP function should not have negative limits for inhibitory synapses");
	}

	if(m_maxExcitatoryWeight < m_minExcitatoryWeight) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"STDP should have maximum excitatory weight >= minimum excitatory weight");
	}

	if(m_maxInhibitoryWeight > m_minInhibitoryWeight) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"STDP should have abs(maximum inhibitory weight) >= abs(minimum inhibitory weight)");
	}

	if(m_prefire.size() + m_postfire.size() > MAX_FIRING_HISTORY) {
		throw nemo::exception(NEMO_INVALID_INPUT, "size of STDP window too large");
	}
}



void
StdpFunction::verifyDynamicWindowLength(unsigned d_max) const
{
	using boost::format;
	if(d_max + m_postfire.size() > MAX_FIRING_HISTORY) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Max delay + postfire part of STDP function must be <= %s")
					% MAX_FIRING_HISTORY));
	}
}



void
setBit(size_t bit, uint64_t& word)
{
	word = word | (uint64_t(1) << bit);
}



uint64_t
StdpFunction::getBits(bool (*pred)(float)) const
{
	uint64_t bits = 0;
	int n = 0;
	for(std::vector<float>::const_reverse_iterator f = m_postfire.rbegin();
			f != m_postfire.rend(); ++f, ++n) {
		if(pred(*f)) {
			setBit(n, bits);
		}
	}
	for(std::vector<float>::const_iterator f = m_prefire.begin(); 
			f != m_prefire.end(); ++f, ++n) {
		if(pred(*f)) {
			setBit(n, bits);
		}
	}
	return bits;
}


bool potentiation(float x ) { return x > 0.0f; }
bool depression(float x ) { return x < 0.0f; }


uint64_t
StdpFunction::potentiationBits() const
{
	return getBits(potentiation);
}


uint64_t
StdpFunction::depressionBits() const
{
	return getBits(depression);
}

}
