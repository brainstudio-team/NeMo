#include "Axon.hpp"

#include "exception.hpp"

namespace nemo {



id32_t
Axon::addSynapse(unsigned target,
				unsigned delay, float weight, bool plastic)
{
	id32_t id = id32_t(m_targets.size());
	m_targets.push_back(target);
	m_delays.push_back(delay);
	m_weights.push_back(weight);
	m_plastic.push_back(plastic);
	return id;
}


Synapse
Axon::getSynapse(nidx_t source, id32_t id) const
{
	if(id >= size()) {
		throw nemo::exception(NEMO_INVALID_INPUT, "synapse id out or range");
	}
	return Synapse(source, m_delays[id],
			AxonTerminal(id, m_targets[id], m_weights[id], m_plastic[id]));
}



unsigned
Axon::getTarget(id32_t id) const
{
	return m_targets.at(id);
}



unsigned
Axon::getDelay(id32_t id) const
{
	return m_delays.at(id);
}



float
Axon::getWeight(id32_t id) const
{
	return m_weights.at(id);
}



bool
Axon::getPlastic(id32_t id) const
{
	return m_plastic.at(id);
}



size_t
Axon::size() const
{
	/* all vectors have the same size due to class invariant. */
	return m_targets.size();
}



void
Axon::setSynapseIds(id32_t source, std::vector<synapse_id>& ids) const
{
	size_t nSynapses = size();
	ids.resize(nSynapses);
	for(size_t iSynapse = 0; iSynapse < nSynapses; ++iSynapse) {
		ids[iSynapse] = make_synapse_id(source, iSynapse);
	}
}

}
