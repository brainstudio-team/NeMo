#ifndef NEMO_CONFIGURATION_IMPL_HPP
#define NEMO_CONFIGURATION_IMPL_HPP

//! \file ConfigurationImpl.hpp

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <ostream>
#include <boost/optional.hpp>

#include <nemo/config.h>
#include "StdpFunction.hpp"
#include "types.h"

#ifdef NEMO_MPI_ENABLED

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/optional.hpp>

namespace boost {
	namespace serialization {
		class access;
	}
}

#endif

namespace nemo {

class NEMO_BASE_DLL_PUBLIC ConfigurationImpl
{
public:

	ConfigurationImpl();

	/*! \copydoc nemo::Configuration::enableLogging */
	void enableLogging() {m_logging = true;}

	/*! \copydoc nemo::Configuration::disableLogging */
	void disableLogging() {m_logging = false;}

	/*! \copydoc nemo::Configuration::loggingEnabled */
	bool loggingEnabled() const {return m_logging;}

	void setCudaPartitionSize(unsigned ps) {m_cudaPartitionSize = ps;}
	unsigned cudaPartitionSize() const {return m_cudaPartitionSize;}

	void setCudaDevice(unsigned device) {m_cudaDevice = device;}
	unsigned cudaDevice() const {return m_cudaDevice;}

	/*! \copydoc nemo::Configuration::setStdpFunction */
	void setStdpFunction(
			const std::vector<float>& prefire,
			const std::vector<float>& postfire,
			float minExcitatoryWeight,
			float maxExcitatoryWeight,
			float minInhibitoryWeight,
			float maxInhibitoryWeight);

	const boost::optional<StdpFunction>& stdpFunction() const {return m_stdpFn;}

	/*! \copydoc nemo::Configuration::setWriteOnlySynapses */
	void setWriteOnlySynapses() {m_writeOnlySynapses = true;}

	/*! \copydoc nemo::Configuration::writeOnlySynapses */
	bool writeOnlySynapses() const {return m_writeOnlySynapses;}

	void setFractionalBits(unsigned bits);

	/*! \return the number of fractional bits. If the user has not
	 * specified this (\see fractionalBitsSet) the return value is
	 * undefined */
	unsigned fractionalBits() const;

	bool fractionalBitsSet() const;

	void setBackend(backend_t backend);
	backend_t backend() const {return m_backend;}

	void setBackendDescription(const char* descr) {m_backendDescription.assign(descr);}
	const char* backendDescription() const {return m_backendDescription.c_str();}

	void setStdpPeriod(unsigned period);

	unsigned stdpPeriod() const;

	void setStdpReward(float reward);

	float stdpReward() const;

	void setStdpEnabled(bool isEnabled);

	bool stdpEnabled() const;

	/*! Verify that the STDP configuration is valid
	 *
	 * \param d_max maximum delay in network
	 */
	void verifyStdp(unsigned d_max) const;

private:

	bool m_logging;
	boost::optional<StdpFunction> m_stdpFn;

	bool m_writeOnlySynapses;

	int m_fractionalBits;
	static const int s_defaultFractionalBits = -1;

	/* CUDA-specific */
	unsigned m_cudaPartitionSize;

	unsigned m_cudaDevice;

	friend void check_close(const ConfigurationImpl& lhs, const ConfigurationImpl& rhs);

	backend_t m_backend;

	std::string m_backendDescription;

	bool m_stdpEnabled;
	unsigned m_stdpPeriod;
	float m_stdpReward;

#ifdef NEMO_MPI_ENABLED
	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & m_logging;
		ar & m_stdpFn;
		ar & m_fractionalBits;
		ar & m_cudaPartitionSize;
		ar & m_backend;
		ar & m_backendDescription;
	}
#endif
};

}

NEMO_BASE_DLL_PUBLIC
std::ostream& operator<<(std::ostream& o, nemo::ConfigurationImpl const& conf);

#endif
