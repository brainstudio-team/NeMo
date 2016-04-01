/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with NeMo. If not, see <http://www.gnu.org/licenses/>.
 */

/* Tests for STDP functionality */

#include <cmath>
#include <iostream>
#include <boost/scoped_ptr.hpp>
#include <boost/test/unit_test.hpp>

#include <nemo.hpp>
#include "utils.hpp"

/* The test network consists of two groups of the same size. Connections
 * between these groups are one-way and are organised in a one-to-one fashion.
 */
static const unsigned groupSize = 2 * 768;

/*The initial weight should be too low to induce firing
 * based on just a single spike and still allow a few depressions before
 * reaching zero. */
static const float initWeight = 5.0f; 

static const unsigned nepochs = 4;
static const unsigned preFire = 10; // ms within epoch
static const unsigned postFireDelay = 10; // ms within epoch


/* Return prefire part of standard STDP function */
std::vector<float>
standardStdpPre(float alpha, unsigned tau)
{
	std::vector<float> fn(tau);
	for(unsigned dt = 0; dt < tau; ++dt) {
		fn.at(dt) = alpha * expf(-float(dt) / float(tau));
	}
	return fn;
}



std::vector<float>
standardStdpPost(float alpha, unsigned tau)
{
	std::vector<float> fn(tau);
	for(unsigned dt = 0; dt < tau; ++dt) {
		fn.at(dt) = alpha * expf(-float(dt) / float(tau));
	}
	return fn;
}



float
dwPre(int dt)
{
	assert(dt <= 0);
	return 1.0f * expf(float(dt) / 20.0f);
}



float
dwPost(int dt)
{
	assert(dt >= 0.0f);
	return -0.8f * expf(float(-dt) / 20.0f);
}


nemo::Configuration
configuration(backend_t backend)
{
	nemo::Configuration conf;

	std::vector<float> pre(20);
	std::vector<float> post(20);
	for(unsigned i = 0; i < 20; ++i) {
		int dt = i;
		pre.at(i) = dwPre(-dt);
		post.at(i) = dwPost(dt);
	}
	/* don't allow negative synapses to go much more negative.
	 * This is to avoid having large negative input currents,
	 * which will result in extra firing (by forcing 'u' to
	 * go highly negative) */
	conf.setStdpFunction(pre, post, -0.5, 2*initWeight);
	setBackend(backend, conf);

	return conf;
}



unsigned
globalIdx(unsigned group, unsigned local)
{
	return group * groupSize + local;
}


unsigned
localIdx(unsigned global)
{
	return global % groupSize;
}


/* The synaptic delays between neurons in the two groups depend only on the
 * index of the second neuron */
unsigned
delay(unsigned local)
{
	return 1 + (local % 20);
}


/* Return number of synapses per neuron */
unsigned
construct(nemo::Network& net, bool noiseConnections)
{
	/* Neurons in the two groups have standard parameters and no spontaneous
	 * firing */
	for(unsigned group=0; group < 2; ++group) {
		for(unsigned local=0; local < groupSize; ++local) {
			float r = 0.5;
			float b = 0.25f - 0.05f * r;
			float v = -65.0;
			net.addNeuron(globalIdx(group, local),
					0.02f + 0.08f * r, b, v, 2.0f, b*v, v, 0.0f);
		}
	}

	/* The plastic synapses  are one-way, from group 0 to group 1. The delay
	 * varies depending on the target neuron. The weights are set that a single
	 * spike is enough to induce firing in the postsynaptic neuron. */
	for(unsigned local=0; local < groupSize; ++local) {
		net.addSynapse(
				globalIdx(0, local),
				globalIdx(1, local),
				delay(local),
				initWeight, 1);
	}
	
	/* To complicate spike delivery and STDP computation, add a number of
	 * connections with very low negative weights. Even if potentiated, these
	 * will not lead to additional firing. Use a mix of plastic and static
	 * synapses. */
	if(noiseConnections) {
		for(unsigned lsrc=0; lsrc < groupSize; ++lsrc) 
		for(unsigned ltgt=0; ltgt < groupSize; ++ltgt) {
			if(lsrc != ltgt) {
				net.addSynapse(
						globalIdx(0, lsrc),
						globalIdx(1, ltgt),
						delay(ltgt + lsrc),
						-0.00001f,
						 ltgt & 0x1);
			}
		}
	}

	return noiseConnections ? groupSize : 1;
}



void
stimulateGroup(unsigned group, std::vector<unsigned>& fstim)
{
	for(unsigned local=0; local < groupSize; ++local) {
		fstim.push_back(globalIdx(group, local));
	}
}


/* Neurons are only stimulated at specific times */
const std::vector<unsigned>&
stimulus(unsigned ms, std::vector<unsigned>& fstim)
{
	if(ms == preFire) {
		stimulateGroup(0, fstim);
	} else if(ms == preFire + postFireDelay) {
		stimulateGroup(1, fstim);
	}
	return fstim;
}


void
verifyWeightChange(unsigned epoch, nemo::Simulation* sim, unsigned m, float reward)
{
	unsigned checked = 0; 

	for(unsigned local = 0; local < groupSize; ++local) {

		const std::vector<synapse_id>& synapses = sim->getSynapsesFrom(globalIdx(0, local));

		for(std::vector<synapse_id>::const_iterator id = synapses.begin();
				id != synapses.end(); ++id) {

			unsigned target = sim->getSynapseTarget(*id);

			if(local != localIdx(target))
				continue;

			unsigned actualDelay = sim->getSynapseDelay(*id);
			BOOST_REQUIRE_EQUAL(delay(localIdx(target)), actualDelay);
			BOOST_REQUIRE(sim->getSynapsePlastic(*id));

			/* dt is positive for pre-post pair, and negative for post-pre
			 * pairs */ 
			int dt = -(int(postFireDelay - actualDelay));

			float dw_expected = 0.0f; 
			if(dt > 0) {
				dw_expected = dwPost(dt-1);
			} else if(dt <= 0) {
				dw_expected = dwPre(dt);
			}

			float expectedWeight = initWeight + epoch * reward * dw_expected;
			float actualWeight = sim->getSynapseWeight(*id);

			const float tolerance = 0.001f; // percent
			BOOST_REQUIRE_CLOSE(expectedWeight, actualWeight, tolerance);

			checked += 1;
		}
	}

	std::cout << "Epoch " << epoch << ": checked " << checked << " synapses\n";
}


void
testStdp(backend_t backend, bool noiseConnections, float reward)
{
	nemo::Network net;
	unsigned m = construct(net, noiseConnections);
	nemo::Configuration conf = configuration(backend);

	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(net, conf));

	verifyWeightChange(0, sim.get(), m, reward);

	for(unsigned epoch = 1; epoch <= nepochs; ++epoch) {
		for(unsigned ms = 0; ms < 100; ++ms) {
			std::vector<unsigned> fstim;
			sim->step(stimulus(ms, fstim));
		}
		/* During the preceding epoch each synapse should have
		 * been updated according to the STDP rule exactly
		 * once. The magnitude of the weight change will vary
		 * between synapses according to their delay */
		sim->applyStdp(reward);
		verifyWeightChange(epoch, sim.get(), m, reward);
	}
}


void
simpleStdpRun(const nemo::Network& net, const nemo::Configuration& conf)
{
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(net, conf));
	sim->step();
	sim->applyStdp(1.0);
}



/* Make sure that calling applyStdp gives an error */
void
testInvalidStdpUsage(backend_t backend)
{
	boost::scoped_ptr<nemo::Network> net(createRing(10, 0, true));
	nemo::Configuration conf;
	setBackend(backend, conf);
	BOOST_REQUIRE_THROW(simpleStdpRun(*net, conf), nemo::exception);
}




/* Mixing up excitatory and inhibitory limits should throw */
void
testInvalidBounds()
{
	nemo::Configuration conf;
	std::vector<float> pre  = standardStdpPre(1.0, 20);
	std::vector<float> post = standardStdpPost(1.0, 20);

	// negative excitatory
	BOOST_REQUIRE_THROW(conf.setStdpFunction(pre, post, -0.01f,  1.0f, -0.01f, -1.0f), nemo::exception);
	BOOST_REQUIRE_THROW(conf.setStdpFunction(pre, post,  0.01f, -1.0f, -0.01f, -1.0f), nemo::exception);
	BOOST_REQUIRE_THROW(conf.setStdpFunction(pre, post, -0.01f, -1.0f, -0.01f, -1.0f), nemo::exception);

	// positive inhibitory
	BOOST_REQUIRE_THROW(conf.setStdpFunction(pre, post, 0.01f, 1.0f,  0.01f, -1.0f), nemo::exception);
	BOOST_REQUIRE_THROW(conf.setStdpFunction(pre, post, 0.01f, 1.0f, -0.01f,  1.0f), nemo::exception);
	BOOST_REQUIRE_THROW(conf.setStdpFunction(pre, post, 0.01f, 1.0f,  0.01f,  1.0f), nemo::exception);

	// incorrect order of excitatory
	BOOST_REQUIRE_THROW(conf.setStdpFunction(pre, post, 1.0f, 0.01f, -0.01f, -1.0), nemo::exception);

	// incorrect order of inhibitory
	BOOST_REQUIRE_THROW(conf.setStdpFunction(pre, post, 0.01f, 1.0f, -1.0f, -0.01f), nemo::exception);
}



/* Too long STDP window (on its own) should throw */
void
testInvalidStaticLength()
{
	nemo::Configuration conf;
	std::vector<float> pre  = standardStdpPre(1.0, 35);
	std::vector<float> post = standardStdpPost(1.0, 35);
	BOOST_REQUIRE_THROW(conf.setStdpFunction(pre, post, 0.01f,  1.0f, -0.01f, -1.0f), nemo::exception);
}



/* Too long STDP window (considering max network delay) should throw */
void
testInvalidDynamicLength(bool stdp)
{
	nemo::Configuration conf;
	if(stdp) {
		std::vector<float> pre  = standardStdpPre(1.0, 31);
		std::vector<float> post = standardStdpPost(1.0, 31);
		conf.setStdpFunction(pre, post, 0.01f,  1.0f, -0.01f, -1.0f);
	}

	nemo::Network net;
	unsigned iz = net.addNeuronType("Izhikevich");
	float param[7] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	net.addNeuron(iz, 0, 7, param);
	net.addNeuron(iz, 1, 7, param);
	net.addSynapse(0, 1, 34, 1.0, stdp);

	boost::scoped_ptr<nemo::Simulation> sim;

	if(stdp) {
		BOOST_REQUIRE_THROW(sim.reset(nemo::simulation(net, conf)), nemo::exception);
	} else {
		BOOST_REQUIRE_NO_THROW(sim.reset(nemo::simulation(net, conf)));
	}
}
