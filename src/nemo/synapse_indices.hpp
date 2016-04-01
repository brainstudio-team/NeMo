#ifndef NEMO_SYNAPSE_INDICES_HPP
#define NEMO_SYNAPSE_INDICES_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <nemo/internal_types.h>

/* Each synapse is given a unique id which can be used when querying the
 * synapse state at run-time. From the user's point of view this is just a
 * 64-bit value. However, internally the encoding is such that the id contains
 *
 * 1. the source neuron
 * 2. a per-neuron synapse index
 *
 * The per-neuron synapse index should form a contigous range starting from 0.
 */

namespace nemo {

/*! \return the neuron index part of the global synapse id */
inline
nidx_t
neuronIndex(synapse_id id)
{
	return id >> 32;
}


/*! \return the per-neuron synapse id part of the global synapse id */
inline
id32_t
synapseIndex(synapse_id id)
{
	return id32_t(id & 0xffffffff);
}



inline
synapse_id
make_synapse_id(nidx_t neuron, id32_t synapse)
{
	//! \todo do a range check here
	return (id64_t(neuron) << 32) | id64_t(synapse);
}

}


#endif
