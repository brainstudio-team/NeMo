#ifndef NEMO_STDP_FUNCTION
#define NEMO_STDP_FUNCTION

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>

#include <nemo/config.h>
#include <nemo/internal_types.h>

#ifdef NEMO_MPI_ENABLED

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

namespace boost {
	namespace serialization {
		class access;
	}
}

#endif

namespace nemo {


/*! \brief User-configurable STDP function */
class NEMO_BASE_DLL_PUBLIC StdpFunction
{
	public :

		StdpFunction() :
			m_minExcitatoryWeight(0.0),
			m_maxExcitatoryWeight(0.0),
			m_minInhibitoryWeight(0.0),
			m_maxInhibitoryWeight(0.0) { }

		/*! Create an STDP function
		 *
		 * Excitatory synapses are allowed to vary in the range
		 * [\a minExcitatoryWeight, \a maxExcitatoryWeight].
		 * Inhibitory synapses are allowed to vary in the range
		 * [\a minInhibitoryWeight \a maxInhibitoryWeight].
		 */
		StdpFunction(const std::vector<float>& prefire,
				const std::vector<float>& postfire,
				float minExcitatoryWeight, float maxExcitatoryWeight,
				float minInhibitoryWeight, float maxInhibitoryWeight);

		/*! Verify that the STDP window length is not too long, considering the
		 * longest network delay */
		void verifyDynamicWindowLength(unsigned d_max) const;

		/* pre-fire part of STDP function, from dt=-1 and down */
		const std::vector<float>& prefire() const { return m_prefire; }

		/* pre-fire part of STDP function, from dt=+1 and up */
		const std::vector<float>& postfire() const { return m_postfire; }

		float minInhibitoryWeight() const { return m_minInhibitoryWeight; }
		float maxInhibitoryWeight() const { return m_maxInhibitoryWeight; }

		float minExcitatoryWeight() const { return m_minExcitatoryWeight; }
		float maxExcitatoryWeight() const { return m_maxExcitatoryWeight; }

		/*! \return bit mask indicating which cycles correspond to
		 * potentiation.  LSB = end of STDP window. */
		uint64_t potentiationBits() const;

		/*! \return bit mask indicating which cycles correspond to depression.
		 * LSB = end of STDP window. */
		uint64_t depressionBits() const;

	private :

		std::vector<float> m_prefire;

		std::vector<float> m_postfire;

		float m_minExcitatoryWeight;
		float m_maxExcitatoryWeight;

		float m_minInhibitoryWeight;
		float m_maxInhibitoryWeight;

		uint64_t getBits(bool (*pred)(float)) const;

		static const unsigned MAX_FIRING_HISTORY;

#ifdef NEMO_MPI_ENABLED
		friend class boost::serialization::access;

		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & m_prefire;
			ar & m_postfire;
			ar & m_minExcitatoryWeight;
			ar & m_maxExcitatoryWeight;
			ar & m_minInhibitoryWeight;
			ar & m_maxInhibitoryWeight;
		}
#endif
};

}

#endif
