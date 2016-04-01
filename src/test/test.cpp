#define BOOST_TEST_MODULE nemo test

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with NeMo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <cmath>
#include <ctime>
#include <iostream>

#include <boost/math/special_functions/fpclassify.hpp> // isnan
#include <boost/test/unit_test.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/random.hpp>

#include <nemo.hpp>
#include <nemo/fixedpoint.hpp>
#include <examples.hpp>

#include "test.hpp"
#include "utils.hpp"
#include "rtest.hpp"
#include "c_api.hpp"


/* For a number of tests, we want to run both CUDA and CPU versions with the
 * same parameters. */
#ifdef NEMO_CUDA_ENABLED

// unary function
#define TEST_ALL_BACKENDS(name, fn)                                           \
    BOOST_AUTO_TEST_SUITE(name)                                               \
    BOOST_AUTO_TEST_CASE(cpu) { fn(NEMO_BACKEND_CPU); }                       \
    BOOST_AUTO_TEST_CASE(cuda) { fn(NEMO_BACKEND_CUDA); }                     \
    BOOST_AUTO_TEST_SUITE_END()

// n-ary function for n >= 2
#define TEST_ALL_BACKENDS_N(name, fn,...)                                     \
    BOOST_AUTO_TEST_SUITE(name)                                               \
    BOOST_AUTO_TEST_CASE(cpu) { fn(NEMO_BACKEND_CPU, __VA_ARGS__); }          \
    BOOST_AUTO_TEST_CASE(cuda) { fn(NEMO_BACKEND_CUDA, __VA_ARGS__); }        \
    BOOST_AUTO_TEST_SUITE_END()

#else

#define TEST_ALL_BACKENDS(name, fn)                                           \
    BOOST_AUTO_TEST_SUITE(name)                                               \
    BOOST_AUTO_TEST_CASE(cpu) { fn(NEMO_BACKEND_CPU); }                       \
    BOOST_AUTO_TEST_SUITE_END()

#define TEST_ALL_BACKENDS_N(name, fn,...)                                     \
    BOOST_AUTO_TEST_SUITE(name)                                               \
    BOOST_AUTO_TEST_CASE(cpu) { fn(NEMO_BACKEND_CPU, __VA_ARGS__); }          \
    BOOST_AUTO_TEST_SUITE_END()

#endif

typedef boost::mt19937 rng_t;
typedef boost::variate_generator<rng_t&, boost::bernoulli_distribution<double> > brng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_real<double> > urng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_int<> > uirng_t;


/* Run one simulation after another and make sure their firing output match
 * exactly */
void
compareSimulations(
		const nemo::Network* net1,
		const nemo::Configuration& conf1,
		const nemo::Network* net2,
		const nemo::Configuration& conf2,
		unsigned duration,
		bool stdp)
{
	// PEDRO: comment this out. We do we want << ?
	/*
	std::cout << "Comparing " << conf1 << "\n"
	          << "      and " << conf2 << "\n";
	*/
	std::vector<unsigned> cycles1, cycles2, nidx1, nidx2;
	runSimulation(net1, conf1, duration, &cycles1, &nidx1, stdp);
	runSimulation(net2, conf2, duration, &cycles2, &nidx2, stdp);
	compareSimulationResults(cycles1, nidx1, cycles2, nidx2);
}



template<typename T>
void
// pass by value here since the simulation data cannot be modified
sortAndCompare(std::vector<T> a, std::vector<T> b)
{
	std::sort(a.begin(), a.end());
	std::sort(b.begin(), b.end());
	BOOST_REQUIRE(a.size() == b.size());
	for(size_t i = 0; i < a.size(); ++i) {
		BOOST_REQUIRE(a[i] == b[i]);
	}
}


void
runComparisions(nemo::Network* net)
{
	/* No need to always test all these, so just skip some random proportion of
	 * these */
	rng_t rng;
	rng.seed(uint32_t(std::time(0)));
	brng_t skip(rng, boost::bernoulli_distribution<double>(0.8));
	unsigned duration = 2;

	/* simulations should produce repeatable results both with the same
	 * partition size and with different ones. */
	{
		bool stdp_conf[2] = { false, true };
		unsigned psize_conf[3] = { 1024, 512, 256 };

		for(unsigned si=0; si < 2; ++si)
		for(unsigned pi1=0; pi1 < 3; ++pi1) 
		for(unsigned pi2=0; pi2 < 3; ++pi2) {
			if(skip()) {
				continue;
			}
			nemo::Configuration conf1 = configuration(stdp_conf[si], psize_conf[pi1]);
			nemo::Configuration conf2 = configuration(stdp_conf[si], psize_conf[pi2]);
			compareSimulations(net, conf1, net, conf2, duration, stdp_conf[si]);
		}
	}

}



//! \todo migrate to networks.cpp
void
runSimple(unsigned startNeuron, unsigned neuronCount)
{
	nemo::Network net;
	for(int nidx = 0; nidx < 4; ++nidx) {
		net.addNeuron(nidx, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f);
	}
	nemo::Configuration conf;
	boost::scoped_ptr<nemo::Simulation> sim;
	BOOST_REQUIRE_NO_THROW(sim.reset(nemo::simulation(net, conf)));
	BOOST_REQUIRE_NO_THROW(sim->step());
}


/* Verify that neuron count actually counts the number of neurons.
 *
 * Create a fixed number of neurons with indices at fixed intervals starting
 * from \a startNeuron */
void
testNeuronCount(unsigned startNeuron, unsigned step)
{
	nemo::Network net;
	unsigned ncount = 2000U;
	for(unsigned i = 0; i < ncount; ++i) {
		unsigned nidx = startNeuron + i * step;
		net.addNeuron(nidx, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f);
	}
	BOOST_REQUIRE_EQUAL(ncount, net.neuronCount());
}



BOOST_AUTO_TEST_SUITE(neuron_count)
	BOOST_AUTO_TEST_CASE(cont_zero) { testNeuronCount(0, 1); }
	BOOST_AUTO_TEST_CASE(cont_nonzero) { testNeuronCount(1000, 1); }
	BOOST_AUTO_TEST_CASE(noncont_zero) { testNeuronCount(0, 5); }
	BOOST_AUTO_TEST_CASE(noncont_nonzero) { testNeuronCount(1000, 5); }
BOOST_AUTO_TEST_SUITE_END()



BOOST_AUTO_TEST_CASE(simulation_unary_network)
{
	runSimple(0, 1);
}


BOOST_AUTO_TEST_SUITE(networks)
	TEST_ALL_BACKENDS(no_outgoing, no_outgoing::run)
	TEST_ALL_BACKENDS(invalid_targets, invalid_targets::run)
BOOST_AUTO_TEST_SUITE_END()


/* It should be possible to create a network without any synapses */
BOOST_AUTO_TEST_CASE(simulation_without_synapses)
{
	runSimple(0, 4);
}


/* We should be able to deal with networs with neuron indices not starting at
 * zero */
BOOST_AUTO_TEST_CASE(simulation_one_based_indices)
{
	runSimple(1, 4);
}


void
testFiringStimulus(backend_t backend)
{
	unsigned ncount = 3000; // make sure to cross partition boundaries
	unsigned cycles = 1000;
	unsigned firing = 10;   // every cycle
	double p_fire = double(firing) / double(ncount);

	nemo::Network net;
	for(unsigned nidx = 0; nidx < ncount; ++nidx) {
		addExcitatoryNeuron(nidx, net);
	}

	nemo::Configuration conf;
	setBackend(backend, conf);
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(net, conf));

	rng_t rng;
	urng_t random(rng, boost::uniform_real<double>(0, 1));

	for(unsigned t = 0; t < cycles; ++t) {
		std::vector<unsigned> fstim;

		for(unsigned n = 0; n < ncount; ++n) {
			if(random() < p_fire) {
				fstim.push_back(n);
			}
		}

		const std::vector<unsigned>& fired = sim->step(fstim);

		/* The neurons which just fired should be exactly the ones we just stimulated */
		sortAndCompare(fstim, fired);
	}
}


TEST_ALL_BACKENDS(fstim, testFiringStimulus)


void
testCurrentStimulus(backend_t backend)
{
	unsigned ncount = 1500;
	unsigned duration = ncount * 2;

	nemo::Configuration conf = configuration(false, 1024, backend);
	boost::scoped_ptr<nemo::Network> net(createRing(ncount));
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));

	nemo::Simulation::current_stimulus istim;
	// add some noise before and after
	istim.push_back(std::make_pair(5U, 0.001f));
	istim.push_back(std::make_pair(8U, 0.001f));
	istim.push_back(std::make_pair(0U, 1000.0f));
	istim.push_back(std::make_pair(100U, 0.001f));
	istim.push_back(std::make_pair(1U, 0.001f));

	/* Simulate a single neuron to get the ring going */
	sim->step(istim);

	for(unsigned ms=1; ms < duration; ++ms) {
		const std::vector<unsigned>& fired = sim->step();
		BOOST_CHECK_EQUAL(fired.size(), 1U);
		BOOST_REQUIRE_EQUAL(fired.front(), ms % ncount);
	}
}


void
testInvalidCurrentStimulus(backend_t backend)
{
	unsigned ncount = 1000U;
	nemo::Configuration conf = configuration(false, 1024, backend);
	boost::scoped_ptr<nemo::Network> net(createRing(ncount));
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));

	sim->step();
	sim->step();

	nemo::Simulation::current_stimulus istim;
	istim.push_back(std::make_pair(ncount+1, 0.5));
	BOOST_REQUIRE_THROW(sim->step(istim), nemo::exception);
}



BOOST_AUTO_TEST_SUITE(istim)
	TEST_ALL_BACKENDS(single_injection, testCurrentStimulus)
	TEST_ALL_BACKENDS(invalid_injection, testInvalidCurrentStimulus)
BOOST_AUTO_TEST_SUITE_END()



void
runRing(backend_t backend, unsigned ncount, unsigned delay)
{
	/* Make sure we go around the ring at least a couple of times */
	const unsigned duration = ncount * 5 / 2;

	nemo::Configuration conf = configuration(false, 1024);
	setBackend(backend, conf);
	boost::scoped_ptr<nemo::Network> net(createRing(ncount, 0, false, 1, delay));
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));

	/* Stimulate a single neuron to get the ring going */
	sim->step(std::vector<unsigned>(1, 0));

	for(unsigned ms=1; ms < duration; ++ms) {
		const std::vector<unsigned>& fired = sim->step();
		if(delay == 1) {
			BOOST_CHECK_EQUAL(fired.size(), 1U);
			BOOST_REQUIRE_EQUAL(fired.front(), ms % ncount);
		} else if(ms % delay == 0) {
			BOOST_CHECK_EQUAL(fired.size(), 1U);
			BOOST_REQUIRE_EQUAL(fired.front(), (ms / delay) % ncount);
		} else {
			BOOST_CHECK_EQUAL(fired.size(), 0U);
		}
	}
}


/* Run two small non-overlapping rings with different delays */
void
runDoubleRing(backend_t backend)
{
	/* Make sure we go around the ring at least a couple of times */
	const unsigned ncount = 512;
	const unsigned duration = ncount * 5 / 2;

	nemo::Configuration conf = configuration(false, 1024);
	setBackend(backend, conf);

	boost::scoped_ptr<nemo::Network> net(new nemo::Network);

	createRing(net.get(), ncount, 0, false, 1, 1);
	createRing(net.get(), ncount, ncount, false, 1, 2);

	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));

	/* Stimulate a single neuron in each ring to get them going */
	std::vector<unsigned> fstim;
	fstim.push_back(0);
	fstim.push_back(ncount);

	sim->step(fstim);

	for(unsigned ms=1; ms < duration; ++ms) {
		const std::vector<unsigned>& fired = sim->step();
		if(ms % 2 == 0) {
			BOOST_CHECK_EQUAL(fired.size(), 2U);
			BOOST_REQUIRE_EQUAL(fired[0], ms % ncount);
			BOOST_REQUIRE_EQUAL(fired[1], ncount + (ms / 2) % ncount);
		} else {
			BOOST_CHECK_EQUAL(fired.size(), 1U);
			BOOST_REQUIRE_EQUAL(fired.front(), ms % ncount);
		}
	}
}




/* Run a regular ring network test, but with an additional variable-sized
 * population of unconnected neurons of a different type.
 *
 * The additional Poisson source neurons should not have any effect on the
 * simulation but should expose errors related to mixing local/global partition
 * indices.
 */
void
testNeuronTypeMixture(backend_t backend, unsigned szOthers, bool izFirst)
{
	const unsigned szRing = 1024;
	boost::scoped_ptr<nemo::Network> net(new nemo::Network());
	if(izFirst) {
		createRing(net.get(), szRing);
	}
	unsigned poisson = net->addNeuronType("PoissonSource");
	float p = 0.001f;
	for(unsigned n=szRing; n<szRing+szOthers; ++n) {
		net->addNeuron(poisson, n, 1, &p);
	}
	if(!izFirst) {
		createRing(net.get(), szRing);
	}

	nemo::Configuration conf = configuration(false, 1024, backend);
	boost::scoped_ptr<nemo::Simulation> sim;
	BOOST_REQUIRE_NO_THROW(sim.reset(nemo::simulation(*net, conf)));

	/* Stimulate a single neuron to get the ring going */
	sim->step(std::vector<unsigned>(1, 0));

	const unsigned duration = 1000;
	for(unsigned ms=1; ms < duration; ++ms) {
		std::vector<unsigned> fired = sim->step();
		BOOST_REQUIRE(fired.size() > 0);
		std::sort(fired.begin(), fired.end());
		BOOST_REQUIRE_EQUAL(fired[0], ms % szRing);
		if(fired.size() > 1) {
			BOOST_REQUIRE(fired[1] >= szRing);
		}
	}
}



void
testFixedPointSaturation(backend_t backend)
{
	nemo::Network net;

	unsigned iz = net.addNeuronType("Izhikevich");

	enum { INPUT0, INPUT1, OUTPUT_SAT, OUTPUT_REF, OUTPUT_WEAK, NCOUNT };

	float params[7] = { 0.02f, 0.2f, -65.0f, 8.0f, 0.0f, -13.0f, -65.0f }; // RS neurons

	for(unsigned n=0; n < NCOUNT; ++n) {
		net.addNeuron(iz, n, 7, params);
	}

	nemo::Configuration conf = configuration(false, 1024, backend);

	net.addSynapse(INPUT0, OUTPUT_REF, 1, 1024.0f, false);
	net.addSynapse(INPUT1, OUTPUT_REF, 1, 1024.0f, false);
	net.addSynapse(INPUT0, OUTPUT_WEAK, 1, 1000.0f, false);
	net.addSynapse(INPUT1, OUTPUT_WEAK, 1, 1000.0f, false);
	net.addSynapse(INPUT0, OUTPUT_SAT, 1, 1100.0f, false);
	net.addSynapse(INPUT1, OUTPUT_SAT, 1, 1100.0f, false);

	/* Now, stimulating both the input neurons should produce nearly the same
	 * result in the reference and saturating output neurons, as the stronger
	 * weights should saturate the very nearly the sum total of the 'middle
	 * strength' weights. The output with weak weights should produce a
	 * different result */

	boost::scoped_ptr<nemo::Simulation> sim;
	sim.reset(nemo::simulation(net, conf));

	std::vector<unsigned> fstim;
	fstim.push_back(INPUT0);
	fstim.push_back(INPUT1);

	std::vector<unsigned> fired;

	fired = sim->step(fstim);
	BOOST_REQUIRE_EQUAL(fired.size(), 2); // sanity checking
	fired = sim->step();
	BOOST_REQUIRE_EQUAL(fired.size(), 3); // sanity checking

	/* Compare u here since it has a dependency on v before the firing, whereas
	 * v does not */
	float u_ref  = sim->getNeuronState(OUTPUT_REF, 0);
	float u_weak = sim->getNeuronState(OUTPUT_WEAK, 0);
	float u_sat  = sim->getNeuronState(OUTPUT_SAT, 0);

	float eps = 0.00001f;
	BOOST_REQUIRE(fabs(u_ref - u_weak) > eps);
	BOOST_REQUIRE(fabs(u_ref - u_sat) < eps);
}



BOOST_AUTO_TEST_SUITE(ring_tests)
	// less than a single partition on CUDA backend
	TEST_ALL_BACKENDS_N(n1000, runRing, 1000, 1);

	// exactly one partition on CUDA backend
	TEST_ALL_BACKENDS_N(n1024, runRing, 1024, 1);

	// multiple partitions on CUDA backend
	TEST_ALL_BACKENDS_N(n2000, runRing, 2000, 1);

	TEST_ALL_BACKENDS_N(n4000, runRing, 4000, 1); // ditto
	TEST_ALL_BACKENDS_N(n2000d20, runRing, 2000, 20); // ditto
	// PEDRO: delay=80ms is not supported by CPU (max 64). Why is this test here?
	// TEST_ALL_BACKENDS_N(n2000d80, runRing, 2000, 80); // ditto
	TEST_ALL_BACKENDS_N(n2000d80, runRing, 2000, 63); // ditto
	TEST_ALL_BACKENDS(delays, runDoubleRing);
BOOST_AUTO_TEST_SUITE_END()



#ifdef NEMO_CUDA_ENABLED
BOOST_AUTO_TEST_CASE(mapping_tests_random)
{
	// only need to create the network once
	boost::scoped_ptr<nemo::Network> net(nemo::random::construct(1000, 1000, 1, true));
	runComparisions(net.get());
}
#endif



#ifdef NEMO_CUDA_ENABLED
BOOST_AUTO_TEST_CASE(mapping_tests_torus)
{
	//! \todo run for larger networks as well
	unsigned pcount = 1;
	unsigned m = 1000;
	unsigned sigma = 16;
	bool logging = false;

	// only need to create the network once
	boost::scoped_ptr<nemo::Network> net(nemo::torus::construct(pcount, m, true, sigma, logging));

	runComparisions(net.get());
}
#endif


void
testNonContigousNeuronIndices(backend_t backend, unsigned n0, unsigned nstep)
{
	unsigned ncount = 1000;
	bool stdp = false;

	boost::scoped_ptr<nemo::Network> net0(createRing(ncount, 0, false, nstep));
	boost::scoped_ptr<nemo::Network> net1(createRing(ncount, n0, false, nstep));

	std::vector<unsigned> cycles0, cycles1;
	std::vector<unsigned> fired0, fired1;

	unsigned seconds = 2;
	nemo::Configuration conf = configuration(false, 1024, backend);

	runSimulation(net0.get(), conf, seconds, &cycles0, &fired0, stdp, std::vector<unsigned>(1, 0));
	runSimulation(net1.get(), conf, seconds, &cycles1, &fired1, stdp, std::vector<unsigned>(1, n0));

	/* The results should be the same, except firing indices
	 * should have the same offset. */
	BOOST_REQUIRE_EQUAL(cycles0.size(), cycles1.size());
	BOOST_REQUIRE_EQUAL(fired0.size(), seconds*ncount);
	BOOST_REQUIRE_EQUAL(fired1.size(), seconds*ncount);

	for(unsigned i = 0; i < cycles0.size(); ++i) {
		BOOST_REQUIRE_EQUAL(cycles0.at(i), cycles1.at(i));
		BOOST_REQUIRE_EQUAL(fired0.at(i), fired1.at(i) - n0);
	}

	//! \todo also add ring networks with different steps.
}


BOOST_AUTO_TEST_SUITE(non_contigous_indices)
	TEST_ALL_BACKENDS_N(contigous_low, testNonContigousNeuronIndices, 1, 1)
	TEST_ALL_BACKENDS_N(contigous_high, testNonContigousNeuronIndices, 1000000, 1)
	TEST_ALL_BACKENDS_N(non_contigous_low, testNonContigousNeuronIndices, 1, 4)
	TEST_ALL_BACKENDS_N(non_contigous_high, testNonContigousNeuronIndices, 1000000, 4)
BOOST_AUTO_TEST_SUITE_END()



/* Create simulation and verify that the simulation data contains the same
 * synapses as the input network. Neurons are assumed to lie in a contigous
 * range of indices starting at n0. */
void
testGetSynapses(nemo::Network& net,
		nemo::Configuration& conf,
		unsigned n0,
		unsigned m)
{
	unsigned fbits = 20;
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(net, conf));

	for(unsigned src = n0, src_end = n0 + net.neuronCount(); src < src_end; ++src) {

		const std::vector<synapse_id>& s_ids = sim->getSynapsesFrom(src);
		const std::vector<synapse_id>& n_ids = net.getSynapsesFrom(src);

		BOOST_REQUIRE_EQUAL(s_ids.size(), n_ids.size());

		for(std::vector<synapse_id>::const_iterator i = s_ids.begin(); i != s_ids.end(); ++i) {
			BOOST_REQUIRE_EQUAL(sim->getSynapseTarget(*i), net.getSynapseTarget(*i));
			BOOST_REQUIRE_EQUAL(sim->getSynapseDelay(*i), net.getSynapseDelay(*i));
			BOOST_REQUIRE_EQUAL(sim->getSynapsePlastic(*i), net.getSynapsePlastic(*i));
			BOOST_REQUIRE_EQUAL(sim->getSynapsePlastic(*i),
					fx_toFloat(fx_toFix(net.getSynapsePlastic(*i), fbits), fbits));

		}
	}
}


void
testGetSynapses(backend_t backend, bool stdp)
{
	nemo::Configuration conf = configuration(stdp, 1024, backend);

	unsigned m = 1000;
	boost::scoped_ptr<nemo::Network> net1(nemo::torus::construct(4, m, stdp, 32, false));
	testGetSynapses(*net1, conf, 0, m);

	unsigned n0 = 1000000U;
	boost::scoped_ptr<nemo::Network> net2(createRing(1500, n0));
	testGetSynapses(*net2, conf, n0, 1);
}


void
testWriteOnlySynapses(backend_t backend)
{
	bool stdp = false;
	nemo::Configuration conf = configuration(stdp, 1024, backend);
	conf.setWriteOnlySynapses();
	boost::scoped_ptr<nemo::Network> net(createRing(10, 0, true));
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));
	sim->step();
	BOOST_REQUIRE_THROW(const std::vector<synapse_id> ids = sim->getSynapsesFrom(0), nemo::exception);

	/* so, we cant use getSynapsesFrom, but create a synapse id anyway */
	nidx_t neuron = 1;
	unsigned synapse = 0;
	synapse_id id = (uint64_t(neuron) << 32) | uint64_t(synapse);

	BOOST_REQUIRE_THROW(sim->getSynapseWeight(id), nemo::exception);
}



void
testGetSynapsesFromUnconnectedNeuron(backend_t backend)
{
	nemo::Network net;
	for(int nidx = 0; nidx < 4; ++nidx) {
		net.addNeuron(nidx, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f);
	}
	nemo::Configuration conf = configuration(false, 1024, backend);
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(net, conf));
	sim->step();

	/* If neuron is invalid we should throw */
	BOOST_REQUIRE_THROW(sim->getSynapsesFrom(4), nemo::exception);

	/* However, if neuron is unconnected, we should get an empty list */
	std::vector<synapse_id> ids;
	BOOST_REQUIRE_NO_THROW(ids = sim->getSynapsesFrom(3));
	BOOST_REQUIRE(ids.size() == 0);
}


/* The network should contain the same synapses before and after setting up the
 * simulation. The order of the synapses may differ, though. */
BOOST_AUTO_TEST_SUITE(get_synapses);
	TEST_ALL_BACKENDS_N(nostdp, testGetSynapses, false)
	TEST_ALL_BACKENDS_N(stdp, testGetSynapses, true)
	TEST_ALL_BACKENDS(write_only, testWriteOnlySynapses)
	TEST_ALL_BACKENDS(from_unconnected, testGetSynapsesFromUnconnectedNeuron)
BOOST_AUTO_TEST_SUITE_END();


void testStdp(backend_t backend, bool noiseConnections, float reward);
void testInvalidStdpUsage(backend_t);


void
testStdpWithAllStatic(backend_t backend)
{
	boost::scoped_ptr<nemo::Network> net(nemo::random::constructUniformRandom(1000, 1000, 1, false));
	nemo::Configuration conf = configuration(true, 1024, backend);
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));
	for(unsigned s=0; s<4; ++s) {
		for(unsigned ms=0; ms<1000; ++ms) {
			sim->step();
		}
		sim->applyStdp(1.0);
	}
}


void testInvalidBounds();
void testInvalidStaticLength();
void testInvalidDynamicLength(bool stdp);


BOOST_AUTO_TEST_SUITE(stdp);
	TEST_ALL_BACKENDS_N(simple, testStdp, false, 1.0)
	TEST_ALL_BACKENDS_N(noisy, testStdp, true, 1.0)
	TEST_ALL_BACKENDS_N(simple_fractional_reward, testStdp, false, 0.8f)
	TEST_ALL_BACKENDS_N(noise_fractional_reward, testStdp, true, 0.9f)
	TEST_ALL_BACKENDS(invalid, testInvalidStdpUsage)
	TEST_ALL_BACKENDS(all_static, testStdpWithAllStatic)
	BOOST_AUTO_TEST_SUITE(configuration)
		BOOST_AUTO_TEST_CASE(limits) { testInvalidBounds(); }
		BOOST_AUTO_TEST_CASE(dlength) { testInvalidStaticLength(); }
		BOOST_AUTO_TEST_CASE(slength_on) { testInvalidDynamicLength(true); }
		BOOST_AUTO_TEST_CASE(slength_off) { testInvalidDynamicLength(false); }
	BOOST_AUTO_TEST_SUITE_END();
BOOST_AUTO_TEST_SUITE_END();


void
testHighFiring(backend_t backend, bool stdp)
{
	//! \todo run for larger networks as well
	unsigned pcount = 1;
	unsigned m = 1000;
	unsigned sigma = 16;
	bool logging = false;


	boost::scoped_ptr<nemo::Network> net(nemo::torus::construct(pcount, m, stdp, sigma, logging));
	nemo::Configuration conf = configuration(stdp, 1024, backend);
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));

	std::vector<unsigned> fstim;
	for(unsigned n = 0; n < net->neuronCount(); ++n) {
		fstim.push_back(n);
	}

	for(unsigned ms = 0; ms < 1000; ++ms) {
		// BOOST_REQUIRE_NO_THROW(sim->step(fstim));
		sim->step(fstim);
	}
}


/* The firing queue should be able to handle all firing rates. It might be best
 * to enable device assertions for this test.  */
#ifdef NEMO_CUDA_ENABLED
BOOST_AUTO_TEST_SUITE(high_firing)
	BOOST_AUTO_TEST_CASE(cuda) {
		testHighFiring(NEMO_BACKEND_CUDA, false);
	}
BOOST_AUTO_TEST_SUITE_END()
#endif




/* create basic network with a single neuron and verify that membrane potential
 * is set correctly initially */
void
testVProbe(backend_t backend)
{
	nemo::Network net;
	float v0 = -65.0;
	net.addNeuron(0, 0.02f, 0.2f, -65.0f+15.0f*0.25f, 8.0f-6.0f*0.25f, 0.2f*-65.0f, v0, 5.0f);

	nemo::Configuration conf = configuration(false, 1024, backend);

	boost::scoped_ptr<nemo::Simulation> sim(simulation(net, conf));
	BOOST_REQUIRE_EQUAL(v0, sim->getMembranePotential(0));
}


TEST_ALL_BACKENDS(vprobe, testVProbe)


BOOST_AUTO_TEST_CASE(add_neuron)
{
	nemo::Network net;
	unsigned iz = net.addNeuronType("Izhikevich");
	std::vector<float> args(7, 0.0f);
	/* Duplicate neuron indices should report an error */
	net.addNeuron(iz, 0, args.size(), &args[0]);
	BOOST_REQUIRE_THROW(net.addNeuron(iz, 0, args.size(), &args[0]), nemo::exception);
}



/* Both the simulation and network classes have neuron setters. Here we perform
 * the same test for both. */
void
testSetNeuron(backend_t backend)
{
	float a = 0.02f;
	float b = 0.2f;
	float c = -65.0f+15.0f*0.25f;
	float d = 8.0f-6.0f*0.25f;
	float v = -65.0f;
	float u = b * v;
	float sigma = 5.0f;

	/* Create a minimal network with a single neuron */
	nemo::Network net;

	/* setNeuron should only succeed for existing neurons */
	BOOST_REQUIRE_THROW(net.setNeuron(0, a, b, c, d, u, v, sigma), nemo::exception);

	net.addNeuron(0, a, b, c-0.1f, d, u, v-1.0f, sigma);

	/* Invalid neuron */
	BOOST_REQUIRE_THROW(net.getNeuronParameter(1, 0), nemo::exception);
	BOOST_REQUIRE_THROW(net.getNeuronState(1, 0), nemo::exception);

	/* Invalid parameter */
	BOOST_REQUIRE_THROW(net.getNeuronParameter(0, 5), nemo::exception);
	BOOST_REQUIRE_THROW(net.getNeuronState(0, 2), nemo::exception);

	float e = 0.1f;
	BOOST_REQUIRE_NO_THROW(net.setNeuron(0, a-e, b-e, c-e, d-e, u-e, v-e, sigma-e));
	BOOST_REQUIRE_EQUAL(net.getNeuronParameter(0, 0), a-e);
	BOOST_REQUIRE_EQUAL(net.getNeuronParameter(0, 1), b-e);
	BOOST_REQUIRE_EQUAL(net.getNeuronParameter(0, 2), c-e);
	BOOST_REQUIRE_EQUAL(net.getNeuronParameter(0, 3), d-e);
	BOOST_REQUIRE_EQUAL(net.getNeuronParameter(0, 4), sigma-e);
	BOOST_REQUIRE_EQUAL(net.getNeuronState(0, 0), u-e);
	BOOST_REQUIRE_EQUAL(net.getNeuronState(0, 1), v-e);

	/* Try setting individual parameters during construction */

	net.setNeuronParameter(0, 0, a);
	BOOST_REQUIRE_EQUAL(net.getNeuronParameter(0, 0), a);

	net.setNeuronParameter(0, 1, b);
	BOOST_REQUIRE_EQUAL(net.getNeuronParameter(0, 1), b);

	net.setNeuronParameter(0, 2, c);
	BOOST_REQUIRE_EQUAL(net.getNeuronParameter(0, 2), c);

	net.setNeuronParameter(0, 3, d);
	BOOST_REQUIRE_EQUAL(net.getNeuronParameter(0, 3), d);

	net.setNeuronParameter(0, 4, sigma);
	BOOST_REQUIRE_EQUAL(net.getNeuronParameter(0, 4), sigma);

	net.setNeuronState(0, 0, u);
	BOOST_REQUIRE_EQUAL(net.getNeuronState(0, 0), u);

	net.setNeuronState(0, 1, v);
	BOOST_REQUIRE_EQUAL(net.getNeuronState(0, 1), v);

	/* Invalid neuron */
	BOOST_REQUIRE_THROW(net.setNeuronParameter(1, 0, 0.0f), nemo::exception);
	BOOST_REQUIRE_THROW(net.setNeuronState(1, 0, 0.0f), nemo::exception);

	/* Invalid parameter */
	BOOST_REQUIRE_THROW(net.setNeuronParameter(0, 5, 0.0f), nemo::exception);
	BOOST_REQUIRE_THROW(net.setNeuronState(0, 2, 0.0f), nemo::exception);

	nemo::Configuration conf = configuration(false, 1024, backend);

	/* Try setting individual parameters during simulation */
	{
		boost::scoped_ptr<nemo::Simulation> sim(simulation(net, conf));

		sim->step();

		sim->setNeuronState(0, 0, u-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronState(0, 0), u-e);

		sim->setNeuronState(0, 1, v-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronState(0, 1), v-e);

		/* Get the data back to later verify that it does in fact change during
		 * simulation, rather than being overwritten again on subsequent
		 * simulation steps */
		float u0 = sim->getNeuronState(0, 0);
		float v0 = sim->getNeuronState(0, 1);

		sim->setNeuronParameter(0, 0, a-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 0), a-e);

		sim->setNeuronParameter(0, 1, b-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 1), b-e);

		sim->setNeuronParameter(0, 2, c-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 2), c-e);

		sim->setNeuronParameter(0, 3, d-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 3), d-e);

		sim->setNeuronParameter(0, 4, sigma-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 4), sigma-e);

		sim->step();

		/* After simulating one more step all the parameter should remain the
		 * same, whereas all the state variables should have changed */
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 0), a-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 1), b-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 2), c-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 3), d-e);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 4), sigma-e);

		BOOST_REQUIRE(sim->getNeuronState(0, 0) != u0);
		BOOST_REQUIRE(sim->getNeuronState(0, 1) != v0);


		/* Invalid neuron */
		BOOST_REQUIRE_THROW(sim->setNeuronParameter(1, 0, 0.0f), nemo::exception);
		BOOST_REQUIRE_THROW(sim->setNeuronState(1, 0, 0.0f), nemo::exception);
		BOOST_REQUIRE_THROW(sim->getNeuronParameter(1, 0), nemo::exception);
		BOOST_REQUIRE_THROW(sim->getNeuronState(1, 0), nemo::exception);

		/* Invalid parameter */
		BOOST_REQUIRE_THROW(sim->setNeuronParameter(0, 5, 0.0f), nemo::exception);
		BOOST_REQUIRE_THROW(sim->setNeuronState(0, 2, 0.0f), nemo::exception);
		BOOST_REQUIRE_THROW(sim->getNeuronParameter(0, 5), nemo::exception);
		BOOST_REQUIRE_THROW(sim->getNeuronState(0, 2), nemo::exception);
	}

	float v0 = 0.0f;
	{
		boost::scoped_ptr<nemo::Simulation> sim(simulation(net, conf));
		sim->step();
		v0 = sim->getMembranePotential(0);
	}

	{
		boost::scoped_ptr<nemo::Simulation> sim(simulation(net, conf));
		BOOST_REQUIRE_EQUAL(sim->getNeuronState(0, 0), u);
		BOOST_REQUIRE_EQUAL(sim->getNeuronState(0, 1), v);
		/* Marginally change the 'c' parameter. This is only used if the neuron
		 * fires (which it shouldn't do this cycle). This modification
		 * therefore should not affect the simulation result (here measured via
		 * the membrane potential) */
		sim->setNeuron(0, a, b, c+1.0f, d, u, v, sigma);

		sim->step();

		BOOST_REQUIRE_EQUAL(v0, sim->getMembranePotential(0));

		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 0), a);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 1), b);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 2), c+1.0f);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 3), d);
		BOOST_REQUIRE_EQUAL(sim->getNeuronParameter(0, 4), sigma);
	}

	{
		/* Ensure that when setting the state variable, it is not copied
		 * multiple times */
		nemo::Network net0;
		net0.addNeuron(0, a, b, c, d, u, v, 0.0f);

		boost::scoped_ptr<nemo::Simulation> sim0(simulation(net0, conf));
		sim0->step();
		sim0->step();
		float v0 = sim0->getMembranePotential(0);

		boost::scoped_ptr<nemo::Simulation> sim1(simulation(net0, conf));
		sim1->step();
		sim1->setNeuron(0, a, b, c, d, u, v, 0.0f);
		sim1->step();
		sim1->step();
		float v1 = sim1->getMembranePotential(0);

		BOOST_REQUIRE_EQUAL(v0, v1);
	}

	{
		/* Modify membrane potential after simulation has been created.
		 * Again the result should be the same */
		nemo::Network net1;
		net1.addNeuron(0, a, b, c, d, u, v-1.0f, sigma);
		boost::scoped_ptr<nemo::Simulation> sim(simulation(net1, conf));
		sim->setNeuron(0, a, b, c, d, u, v, sigma);
		BOOST_REQUIRE_EQUAL(v, sim->getMembranePotential(0));
		sim->step();
		BOOST_REQUIRE_EQUAL(v0, sim->getMembranePotential(0));
	}
}



TEST_ALL_BACKENDS(set_neuron, testSetNeuron)


void
testInvalidNeuronType()
{
	nemo::Network net;
	unsigned iz = net.addNeuronType("Izhikevich");
	float args[7] = {0, 0, 0, 0, 0, 0, 0};
	for(unsigned n=0; n<100; ++n) {
		if(n != iz) {
			// incorrect parameter order for n and iz.
			BOOST_REQUIRE_THROW(net.addNeuron(n, iz, 7, args), nemo::exception);
		}
	}
}



void
testNoParamNeuronType()
{
	nemo::Network net;
	unsigned rs = net.addNeuronType("IzhikevichRS");
	float params[2] = { -13.0f, -65.0f };
	for(unsigned n=0; n<1000; ++n) {
		net.addNeuron(rs, n, 2, params);
	}
	// no synapses
	nemo::Configuration conf;
	conf.setCudaBackend();
	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(net, conf));
	for(unsigned t=0; t<100; ++t) {
		sim->step();
		const float u = sim->getNeuronState(100, 0);
		const float v = sim->getNeuronState(100, 1);
		BOOST_REQUIRE(!boost::math::isnan(u));
		BOOST_REQUIRE(!boost::math::isnan(v));
	}
}


BOOST_AUTO_TEST_SUITE(plugins)
	BOOST_AUTO_TEST_CASE(invalid_type) { testInvalidNeuronType(); }
#ifdef NEMO_CUDA_ENABLED
	BOOST_AUTO_TEST_CASE(no_parameters) { testNoParamNeuronType(); }
#endif
BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(regression)
	BOOST_AUTO_TEST_CASE(torus)    { runTorus(false);    }
	BOOST_AUTO_TEST_CASE(kuramoto) { runKuramoto(false); }
BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(c_api)

	BOOST_AUTO_TEST_SUITE(comparison)
		BOOST_AUTO_TEST_CASE(nostim) { nemo::test::c_api::compareWithCpp(false, false); }
		BOOST_AUTO_TEST_CASE(fstim) { nemo::test::c_api::compareWithCpp(true, false); }
		BOOST_AUTO_TEST_CASE(istim) { nemo::test::c_api::compareWithCpp(false, true); }
	BOOST_AUTO_TEST_SUITE_END()

	BOOST_AUTO_TEST_CASE(synapse_ids) { nemo::test::c_api::testSynapseId(); }
	BOOST_AUTO_TEST_CASE(set_neuron) { nemo::test::c_api::testSetNeuron(); }

	BOOST_AUTO_TEST_SUITE(get_synapse)
		TEST_ALL_BACKENDS_N(n0, nemo::test::c_api::testGetSynapses, 0)
		TEST_ALL_BACKENDS_N(n1000, nemo::test::c_api::testGetSynapses, 1000)
	BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(mix)
	TEST_ALL_BACKENDS_N(IP, testNeuronTypeMixture, 1024, true)
	TEST_ALL_BACKENDS_N(PI, testNeuronTypeMixture, 1024, false)
	/* Verify that it's possible to add a neuron type and then simply ignore
	 * it, with no ill effect */
	TEST_ALL_BACKENDS_N(IP0, testNeuronTypeMixture, 0, true)
	TEST_ALL_BACKENDS_N(PI0, testNeuronTypeMixture, 0, false)
BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(fixedpoint)
#ifdef NEMO_WEIGHT_FIXED_POINT_SATURATION
	TEST_ALL_BACKENDS(saturation, testFixedPointSaturation)
#endif
BOOST_AUTO_TEST_SUITE_END()

/* Neuron-type specific tests */

#include "PoissonSource.cpp"
#include "Input.cpp"
#include "Kuramoto.cpp"
#include "IF_curr_exp.cpp"

