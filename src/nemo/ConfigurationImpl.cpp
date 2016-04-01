/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "ConfigurationImpl.hpp"

#include <boost/format.hpp>

#include "exception.hpp"


namespace nemo {

ConfigurationImpl::ConfigurationImpl() :
	m_logging(false),
	m_writeOnlySynapses(false),
	m_fractionalBits(20),
	m_cudaPartitionSize(0),
	m_cudaDevice(~0U),
	m_backend(~0U), // the wrapper class will set this
	m_backendDescription("No backend specified"),
	m_stdpEnabled(false)
{
	;
}



void
ConfigurationImpl::setStdpFunction(
		const std::vector<float>& prefire,
		const std::vector<float>& postfire,
		float minE, float maxE,
		float minI, float maxI)
{
	m_stdpFn = StdpFunction(prefire, postfire, minE, maxE, minI, maxI);
}



/*! \note this code is partially dead, in that this functionality is not
 * exposed in the API. Rather we have fixed the fixed-point format. Left the
 * code in anticipation of adding the functionality back (along with a more
 * complex auto-configuration). */
void
ConfigurationImpl::setFractionalBits(unsigned bits)
{
	using boost::format;

	const unsigned max_bits = 31;
	if(bits > max_bits) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid number of fractional bits (%u) specified. Max is %u")
					% bits % max_bits));
	}

	m_fractionalBits = static_cast<int>(bits);
}



unsigned
ConfigurationImpl::fractionalBits() const
{
	if(!fractionalBitsSet()) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "Fractional bits requested but never set");
	}
	return static_cast<unsigned>(m_fractionalBits);
}



bool
ConfigurationImpl::fractionalBitsSet() const
{
	return m_fractionalBits != s_defaultFractionalBits;
}



void
ConfigurationImpl::setBackend(backend_t backend)
{
	switch(backend) {
		case NEMO_BACKEND_CUDA :
		case NEMO_BACKEND_CPU :
			m_backend = backend;
			break;
		default :
			throw std::runtime_error("Invalid backend selected");
	}
}

void ConfigurationImpl::setStdpPeriod(unsigned period ) { m_stdpPeriod = period; }

unsigned ConfigurationImpl::stdpPeriod() const {return m_stdpPeriod; }

void ConfigurationImpl::setStdpReward(float reward) { m_stdpReward = reward; }

float ConfigurationImpl::stdpReward() const {return m_stdpReward; }

void ConfigurationImpl::setStdpEnabled(bool isEnabled) { m_stdpEnabled = isEnabled; }

bool ConfigurationImpl::stdpEnabled() const { return m_stdpEnabled; }

void
ConfigurationImpl::verifyStdp(unsigned d_max) const
{
	if(m_stdpFn) {
		m_stdpFn->verifyDynamicWindowLength(d_max);
	}
}


} // namespace nemo


std::ostream& operator<<(std::ostream& o, nemo::ConfigurationImpl const& conf)
{
	return o
		<< "STDP: " << conf.stdpFunction() << ", "
		<< "cuda_ps: " << conf.cudaPartitionSize() << ", "
		<< "device: " << conf.backendDescription();
	//! \todo print more info about STDP
}
