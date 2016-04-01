#include "ReadableNetwork.hpp"

#include <nemo/synapse_indices.hpp>

namespace nemo {

unsigned
ReadableNetwork::getSynapseSource(const synapse_id& id) const
{
	return neuronIndex(id);
}

}
