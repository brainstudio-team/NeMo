#ifndef NEMO_NETWORK_GENERATOR_HPP
#define NEMO_NETWORK_GENERATOR_HPP

#include <nemo/config.h>
#include <nemo/types.hpp>
#include <nemo/network/iterator.hpp>
#include <nemo/NeuronType.hpp>


namespace nemo {
	namespace network {

/* A network generator is simply a class which can produce a sequence of
 * neurons and a sequence of synapses. Network generators are expected to
 * provide all neurons first, then all synapses. Furthermore neurons are
 * accessed via separate iterators for each neuron type.*/
class NEMO_BASE_DLL_PUBLIC Generator
{
	public : 

		virtual ~Generator() { }

		typedef std::pair<nidx_t, Neuron> neuron;
		typedef Synapse synapse;
		
		/*! \return iterator to beginning of the ith neuron collection
		 *
		 * \pre 0 <= i < neuronTypeCount
		 */
		virtual neuron_iterator neuron_begin(unsigned i) const = 0;

		/*! \return iterator to end of the ith neuron collection
		 * \pre 0 <= i < neuronTypeCount
		 */
		virtual neuron_iterator neuron_end(unsigned i) const = 0;

		virtual synapse_iterator synapse_begin() const = 0;
		virtual synapse_iterator synapse_end() const = 0;

		/*! \return number of neurons in the network */
		virtual unsigned neuronCount() const = 0;

		/*! \return number of neurons of a particular type */
		virtual unsigned neuronCount(unsigned type) const = 0;

		/*! \return maximum delay (in time steps) in the network */
		virtual unsigned maxDelay() const = 0;

		virtual unsigned minNeuronIndex() const = 0;
		virtual unsigned maxNeuronIndex() const = 0;

		/*! \return the number of unique neuron types in the network */
		virtual unsigned neuronTypeCount() const = 0;

		/*! \return the neuron type found for the \a i th neuron collection
		 *
		 * \pre network is not empty
		 * \pre 0 < i < neuronTypeCount()
		 */
		virtual const class NeuronType& neuronType(unsigned i) const = 0;
};


	} // end namespace network
} // end namespace nemo


#endif
