/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Network.hpp"

#include "NetworkImpl.hpp"
#include "synapse_indices.hpp"

namespace nemo {

Network::Network() :
	m_impl(new network::NetworkImpl()),
	iz_type(~0U)
{
	;
}


Network::~Network()
{
	delete m_impl;
}


unsigned
Network::addNeuronType(const std::string& name)
{
	return m_impl->addNeuronType(name);
}


void
Network::addNeuron(unsigned type, unsigned idx,
		unsigned nargs, const float args[])
{
	m_impl->addNeuron(type, idx, nargs, args);
}


void
Network::addNeuron(unsigned idx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	if(iz_type == ~0U) {
		iz_type = m_impl->addNeuronType("Izhikevich");
	}
	float args[7] = {a, b, c, d, sigma, u, v};
	m_impl->addNeuron(iz_type, idx, 7, args);
}


void
Network::setNeuron(unsigned idx, unsigned nargs, const float args[])
{
	m_impl->setNeuron(idx, nargs, args);
}


void
Network::setNeuron(unsigned idx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	float args[7] = {a, b, c, d, sigma, u, v};
	m_impl->setNeuron(idx, 7, args);
}



float
Network::getNeuronState(unsigned neuron, unsigned var) const
{
	return m_impl->getNeuronState(neuron, var);
}



float
Network::getNeuronParameter(unsigned neuron, unsigned parameter) const
{
	return m_impl->getNeuronParameter(neuron, parameter);
}



void
Network::setNeuronState(unsigned neuron, unsigned var, float val)
{
	return m_impl->setNeuronState(neuron, var, val);
}



void
Network::setNeuronParameter(unsigned neuron, unsigned parameter, float val)
{
	return m_impl->setNeuronParameter(neuron, parameter, val);
}



synapse_id
Network::addSynapse(
		unsigned source,
		unsigned target,
		unsigned delay,
		float weight,
		unsigned char plastic)
{
	return m_impl->addSynapse(source, target, delay, weight, plastic);
}



unsigned
Network::getSynapseTarget(const synapse_id& id) const
{
	return m_impl->getSynapseTarget(id);
}



unsigned
Network::getSynapseDelay(const synapse_id& id) const
{
	return m_impl->getSynapseDelay(id);
}



float
Network::getSynapseWeight(const synapse_id& id) const
{
	return m_impl->getSynapseWeight(id);
}



unsigned char
Network::getSynapsePlastic(const synapse_id& id) const
{
	return m_impl->getSynapsePlastic(id);
}



const std::vector<synapse_id>&
Network::getSynapsesFrom(unsigned neuron)
{
	return m_impl->getSynapsesFrom(neuron);
}

unsigned
Network::maxDelay() const 
{
	return m_impl->maxDelay(); 
}



float
Network::maxWeight() const
{ 
	return m_impl->maxWeight();
}



float
Network::minWeight() const
{ 
	return m_impl->minWeight(); 
}



unsigned 
Network::neuronCount() const
{
	return m_impl->neuronCount();
}

long unsigned Network::synapseCount() const{
	return m_impl->synapseCount();
};


} // end namespace nemo
