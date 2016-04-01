#ifndef NEMO_NETWORK_IMPL_HPP
#define NEMO_NETWORK_IMPL_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with NeMo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <map>
#include <string>
#include <vector>

#include <nemo/config.h>
#include <nemo/network/Generator.hpp>
#include "Axon.hpp"
#include "Neurons.hpp"
#include "ReadableNetwork.hpp"
#include "RandomMapper.hpp"

#ifdef NEMO_MPI_ENABLED
namespace nemo {
	namespace mpi {
		class Master;
	}
}
#endif
namespace nemo {

namespace network {

namespace programmatic {
class synapse_iterator;
}

class
//NEMO_BASE_DLL_PUBLIC
NetworkImpl: public Generator, public ReadableNetwork {
public:

	NetworkImpl();

	/*! \copydoc nemo::Network::addNeuronType */
	unsigned addNeuronType(const std::string& name);

	int getNeuronTypeId(const std::string& name);

	/*! \copydoc nemo::Network::addNeuron */
	void addNeuron(unsigned type, unsigned idx, unsigned nargs, const float args[]);

#ifdef NEMO_MPI_ENABLED
	void addNeuronMpi(const unsigned type_id, const unsigned& g_idx, const size_t& nargs, const float args[]);
#endif

	/*! \copydoc nemo::Network::setNeuron */
	void setNeuron(unsigned idx, unsigned nargs, const float args[]);

	/*! \copydoc nemo::Network::addSynapse */
	synapse_id addSynapse(unsigned source, unsigned target, unsigned delay, float weight, unsigned char plastic);

	/*! \copydoc nemo::Network::getNeuronState */
	float getNeuronState(unsigned neuron, unsigned var) const;

	/*! \copydoc nemo::Network::getNeuronParameter */
	float getNeuronParameter(unsigned neuron, unsigned parameter) const;

	/*! \copydoc nemo::Network::setNeuronState */
	void setNeuronParameter(unsigned neuron, unsigned var, float val);

	/*! \copydoc nemo::Network::setNeuronParameter */
	void setNeuronState(unsigned neuron, unsigned var, float val);

	/*! \copydoc nemo::Network::getSynapseTarget */
	unsigned getSynapseTarget(const synapse_id&) const;

	/*! \copydoc nemo::Network::getSynapseDelay */
	unsigned getSynapseDelay(const synapse_id&) const;

	/*! \copydoc nemo::Network::getSynapseWeight */
	float getSynapseWeight(const synapse_id&) const;

	/*! \copydoc nemo::Network::getSynapsePlastic */
	unsigned char getSynapsePlastic(const synapse_id&) const;

	/*! \copydoc nemo::Network::getSynapsesFrom */
	const std::vector<synapse_id>& getSynapsesFrom(unsigned neuron);

	/* pre: network is not empty */
	nidx_t minNeuronIndex() const;

	/* pre: network is not empty */
	nidx_t maxNeuronIndex() const;

	delay_t maxDelay() const {
		return m_maxDelay;
	}
	float maxWeight() const {
		return m_maxWeight;
	}
	float minWeight() const {
		return m_minWeight;
	}

	unsigned neuronCount() const;

	long unsigned synapseCount() const;

	unsigned neuronCount(unsigned type_id) const;

	/*! \copydoc nemo::Network::Generator::neuronTypeCount */
	unsigned neuronTypeCount() const;

	/*! \copydoc nemo::Network::Generator::neuron_begin */
	neuron_iterator neuron_begin(unsigned type) const;

	/*! \copydoc nemo::Network::Generator::neuron_end */
	neuron_iterator neuron_end(unsigned type) const;

	synapse_iterator synapse_begin() const;
	synapse_iterator synapse_end() const;

	/*! \copydoc nemo::network::Generator::neuronType */
	const NeuronType& neuronType(unsigned) const;

private:

	#ifdef NEMO_MPI_ENABLED
	friend class nemo::mpi::Master;
	#endif

	/* Neurons are grouped by neuron type */
	std::vector<Neurons> m_neurons;

	/* Users keep access neuron type groups by via indices (returned by \a
	 * addNeuronType). For error-detecting purposes, keep the type name ->
	 * index mapping */
	std::map<std::string, unsigned> m_typeIds;

	const Neurons& neuronCollection(unsigned type_id) const;
	Neurons& neuronCollection(unsigned type_id);

	/* could use a separate type here, but kept it simple while we use this
	 * type in the neuron_iterator class */
	typedef std::pair<unsigned, unsigned> NeuronAddress;

	nemo::RandomMapper<NeuronAddress> m_mapper;

	/*! \todo consider using unordered here instead, esp. after removing
	 * iterator interface. Currently we need rbegin, which is not found in
	 * unordered_map */
	typedef std::map<nidx_t, Axon> fcm_t;

	fcm_t m_fcm;

	int m_minIdx;
	int m_maxIdx;
	delay_t m_maxDelay;
	float m_minWeight;
	float m_maxWeight;

	friend class programmatic::synapse_iterator;

	/*! Internal buffer for synapse queries */
	std::vector<synapse_id> m_queriedSynapseIds;

	const Axon& axon(nidx_t source) const;

};

} // end namespace network
} // end namespace nemo

#endif
