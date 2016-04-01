#ifndef NEMO_AXON_HPP
#define NEMO_AXON_HPP

#include <vector>

#include "types.hpp"
#include "synapse_indices.hpp"

namespace nemo {

/*! \brief Group of synapses sharing the same source neuron
 *
 * The data is stored as a structure-of-arrays, with each \i type of synapse
 * data stored in a separate vector. This is meant to facilitate more generic
 * synapse types.
 *
 * All methods maintain the invariant that all vectors are the same length. 
 *
 * \todo add (construction-time) variable number of parameters here. 
 */
class Axon
{
	public :

		/* default ctor is fine here */

		/*! Add a synapse 
		 *
		 * \return id of the newly added synapse
		 * \pre all internal vectors have the same length
		 * \post all internal vectors have the same length
		 */
		id32_t addSynapse(unsigned target,
				unsigned delay, float weight, bool plastic);

		Synapse getSynapse(nidx_t source, id32_t id) const;

		unsigned getTarget(id32_t id) const;

		unsigned getDelay(id32_t id) const;

		float getWeight(id32_t id) const;

		bool getPlastic(id32_t id) const;

		size_t size() const;

		/*! Populate vector with all synapse ids contained in this axon */
		void setSynapseIds(id32_t source, std::vector<synapse_id>&) const;

	private :

		std::vector<unsigned> m_targets;
		std::vector<unsigned> m_delays;
		std::vector<float> m_weights;
		std::vector<bool> m_plastic;
};

}

#endif
