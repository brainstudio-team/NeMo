#ifndef NEMO_TEST_UTILS_HPP
#define NEMO_TEST_UTILS_HPP

#include <nemo.hpp>

/* Run simulation for given length and return result in output vector */
void
runSimulation(
		const nemo::Network* net,
		nemo::Configuration conf,
		unsigned seconds,
		std::vector<unsigned>* fcycles,
		std::vector<unsigned>* fnidx,
		bool stdp,
		std::vector<unsigned> initFiring = std::vector<unsigned>(),
		std::vector< std::pair<unsigned, float> > initCurrent =
			std::vector< std::pair<unsigned, float> >()
		);

void
compareSimulationResults(
		const std::vector<unsigned>& cycles1,
		const std::vector<unsigned>& nidx1,
		const std::vector<unsigned>& cycles2,
		const std::vector<unsigned>& nidx2);


void
setBackend(backend_t, nemo::Configuration& conf);

nemo::Configuration
configuration(bool stdp, unsigned partitionSize,
#ifdef NEMO_CUDA_ENABLED
		backend_t backend = NEMO_BACKEND_CUDA
#else
		backend_t backend = NEMO_BACKEND_CPU
#endif
		);

/* Add a 'standard' excitatory neuron with fixed parameters */
void
addExcitatoryNeuron(unsigned nidx, nemo::Network& net, float sigma=0.0f);



/* Simple ring network.
 *
 * This is useful for basic testing as the exact firing pattern is known in
 * advance. Every cycle a single neuron fires. Each neuron connected to only
 * the next neuron (in global index space) with an abnormally strong synapse,
 * so the result is the firing propagating around the ring.
 *
 * \param n0 first neuron index
 * \param nstep number of indices between neurons
 */
nemo::Network*
createRing(unsigned ncount, unsigned n0=0, bool plastic=false, unsigned nstep=1, unsigned delay=1);


/*! Add a ring to an existing network */
void
createRing(nemo::Network*, unsigned ncount, unsigned n0=0, bool plastic=false, unsigned nstep=1, unsigned delay=1);


#endif
