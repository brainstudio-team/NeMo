#ifndef NEMO_READABLE_NETWORK_HPP
#define NEMO_READABLE_NETWORK_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <nemo/config.h>
#include <nemo/types.h>

namespace nemo {

/*! Network whose neurons and synapses can be queried
 *
 * Abstract base class
 */
class NEMO_BASE_DLL_PUBLIC ReadableNetwork
{
	public :

		virtual ~ReadableNetwork() { }

		/*! \return source neuron id for a synapse */
		unsigned getSynapseSource(const synapse_id& id) const;

		/*! \return target neuron id for a synapse */
		virtual unsigned getSynapseTarget(const synapse_id&) const = 0;

		/*! \return conductance delay for a synapse */
		virtual unsigned getSynapseDelay(const synapse_id&) const = 0;

		/*! \return weight for a synapse */
		virtual float getSynapseWeight(const synapse_id&) const = 0;

		/*! \return plasticity status for a synapse */
		virtual unsigned char getSynapsePlastic(const synapse_id&) const = 0;

		/*! \return
		 * 		vector of synapse ids for all synapses with the given source
		 * 		neuron
		 *
		 * The vector reference is valid until the next call to this function.
		 */
		virtual const std::vector<synapse_id>& getSynapsesFrom(unsigned neuron) = 0;
};

}

#endif
