#ifndef NEMO_CONSTRUCTION_RCM_HPP
#define NEMO_CONSTRUCTION_RCM_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>

#include <boost/tuple/tuple.hpp>
#include <boost/unordered_map.hpp>
#include <nemo/types.hpp>


namespace nemo {

	class ConfigurationImpl;

	namespace network {
		class Generator;
	}

	namespace runtime {
		class RCM;
	}

	namespace cuda {
		namespace runtime {
			class RCM;
		}
	}

	namespace construction {


/*! \brief Construction-time reverse connectivity matrix
 *
 * At construction-time the reverse connectivity matrix (RCM) is stored as a
 * structure-of-arrays (SoA).
 *
 * The RCM always contains a default payload containing target neuron and
 * delay. Other fields may or may not be included, depending on
 *
 * 1. STDP configuration
 * 2. Neuron type requirements
 *
 * Each (target) neuron has a row in the RCM. The row is always rounded up to
 * the nearest \a Width boundary. Rows are placed back-to-back within each
 * vector in the SoA. Lookups into the SoA is done via an index.
 *
 * At the end of construction, the data has the format to be used at run-time,
 * whereas the index may have to be converted (depending on the backend).
 *
 * \tparam Key type of the neuron index used at run time
 * \tparam Data type of the default data containing source neuron and delay
 * \tparam Width width of each row
 *
 * \see nemo::cuda::runtime::RCM
 */
template<class Key, class Data, size_t Width>
class RCM
{
	public :

		/*! Initialise an empty reverse connectivity matrix
		 *
		 * \param padding synapse data value to use for padding
		 */
		RCM(const nemo::ConfigurationImpl& conf,
				const nemo::network::Generator&,
				const Data& padding);

		/*! Add a new synapse to the reverse connectivity matrix
		 *
		 * \param target index of target neuron on device
		 * \param data packed format for default payload containing source and delay
		 * \param synapse full synapse
		 * \param f_addr word address of this synapse in the forward matrix
		 */
		void addSynapse(const Key& target,
				const Data& data,
				const Synapse& synapse,
				size_t f_addr);

		size_t synapseCount() const { return m_synapseCount; }

		/*! Number of words allocated in any enabled RCM fields
		 *
		 * The class maintains the invariant that all RCM fields are either of
		 * this size (if enabled) or \a Width (if disbled). Furthermore, the
		 * size is always a multiple of \a Width.
		 */
		size_t size() const;

	private :

		typedef boost::unordered_map<Key, std::vector<size_t> > warp_map;

		const Data& m_paddingData;

		size_t m_synapseCount;

		warp_map m_warps;

		/*! In order to keep track of when we need to start a new warp, store
		 * the number of synapses in each row */
		boost::unordered_map<Key, unsigned> m_dataRowLength;

		size_t m_nextFreeWarp;

		/*! Main reverse synapse data: source partition, source neuron, delay */
		std::vector<Data> m_data;
		bool m_useData;

		/*! Forward addressing, word offset into the FCM for each synapse */
		std::vector<uint32_t> m_forward;
		bool m_useForward;

		/*! The weights are \em optionally stored in the reverse format as
		 * well. This is normally not done as the weights are normally used
		 * only in spike delivery which uses the forward form. However, special
		 * neuron type plugins may require this. */
		std::vector<float> m_weights;
		bool m_useWeights;

		/*! Is the RCM in use at all? */
		bool m_enabled;

		/*! If the RCM is in use, do we only keep plastic synapses? See notes
		 * in constructor. */
		bool m_stdpEnabled;

		/*! Allocate space for a new RCM synapse for the given (target) neuron.
		 *
		 * \return
		 * 		word offset for the synapse. This is the same for all the different
		 * 		planes of data.
		 */
		size_t allocateSynapse(const Key& target);

		friend class cuda::runtime::RCM;
		friend class runtime::RCM;
};


	} // end namespace construction
} // end namespace nemo


#include "RCM.cpp"

#endif
