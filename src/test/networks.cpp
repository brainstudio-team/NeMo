/* Various simple networks used for testing of serious errors (e.g.
 * out-of-bounds errors at run-time */

#include <cassert>
#include <iostream>
#include <boost/test/unit_test.hpp>
#include <boost/scoped_ptr.hpp>

#include <nemo.hpp>
#include "utils.hpp"


namespace networks {
	namespace no_outgoing {

unsigned
neuronIndex(unsigned i, bool contigous)
{
	return contigous ? i : 50 + 3 * i;
}


/* Stimulate some (8) neurons every cycle */
std::vector<unsigned>
firingStimulus(unsigned t, unsigned ncount, bool contigous)
{
	std::vector<unsigned> fstim;
	for(unsigned i = 0; i < ncount; ++i) {
		unsigned neuron = neuronIndex(i, contigous);
		if(neuron / 8 == t) {
			fstim.push_back(neuron);
		}
	}
	return fstim;
}


void
runOne( backend_t backend,
		unsigned ncount,
		unsigned firstUnconnected,
		unsigned lastUnconnected,
		bool contigous,
		bool stdp)
{
	assert(0 <= firstUnconnected);
	assert(firstUnconnected <= lastUnconnected);
	assert(lastUnconnected <= ncount);

	nemo::Network net;

	for(unsigned i=0; i < ncount; ++i) {
		unsigned source = neuronIndex(i, contigous);
		addExcitatoryNeuron(source, net);
		if(!(firstUnconnected <= i && i < lastUnconnected)) {
			for(unsigned j=0; j < ncount; ++j) {
				unsigned target = neuronIndex(j, contigous);
				net.addSynapse(source, target, 1, 2.0f, stdp);
			}
		}
	}

	nemo::Configuration conf = configuration(stdp, 1024, backend);
	boost::scoped_ptr<nemo::Simulation> sim;
	BOOST_REQUIRE_NO_THROW(sim.reset(nemo::simulation(net, conf)));

	for(unsigned t=0; t < 20; ++t) {
		BOOST_REQUIRE_NO_THROW(
			sim->step(firingStimulus(t, ncount, contigous))
		);
	}
}


void
runRange(backend_t backend, unsigned ncount, unsigned first, unsigned last)
{
	runOne(backend, ncount, first, last, false, false);
	runOne(backend, ncount, first, last, true, false);
	runOne(backend, ncount, first, last, false, true);
	runOne(backend, ncount, first, last, true, true);
}


void
run(backend_t backend)
{
	runRange(backend, 100, 0, 10);
	runRange(backend, 100, 45, 55);
	runRange(backend, 100, 90, 100);
}

} // end namespace no_outgoing


namespace invalid_targets {


void
run(backend_t backend)
{
	unsigned ncount = 100;
	bool stdp = false;

	nemo::Network net;

	for(unsigned source=0; source < ncount; ++source) {
		addExcitatoryNeuron(source, net);
		net.addSynapse(source, source + ncount, 1, 2.0f, stdp);
	}

	nemo::Configuration conf = configuration(stdp, 1024, backend);

	boost::scoped_ptr<nemo::Simulation> sim;
	/* Creating the simulation should fail due to presence of invalid target
	 * neurons */
	BOOST_REQUIRE_THROW(sim.reset(nemo::simulation(net, conf)), nemo::exception);
}


	} // end namespace invalid_targets
} // end namespace networks
