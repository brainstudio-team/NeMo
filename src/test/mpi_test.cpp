#define BOOST_TEST_MODULE nemo_mpi test

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/scoped_ptr.hpp>

#include <nemo.hpp>
#include <mpi/nemo_mpi.hpp>
#include <examples/examples.hpp>
#include "utils.hpp"


/* ! \note if using this code elsewhere, factor out. It's used
 * in test.cpp as well. */
void
runRing(unsigned ncount, 
		boost::mpi::environment& env,
		boost::mpi::communicator& world)
{
	/* Make sure we go around the ring at least a couple of times */
	const unsigned duration = ncount * 5 / 2;
	// const unsigned duration = ncount * 3 / 2;

	nemo::Network net;
	for(unsigned source=0; source < ncount; ++source) {
		float v = -65.0f;
		float b = 0.2f;
		float r = 0.5f;
		float r2 = r * r;
		net.addNeuron(source, 0.02f, b, v+15.0f*r2, 8.0f-6.0f*r2, b*v, v, 0.0f);
		net.addSynapse(source, (source + 1) % ncount, 1, 1000.0f, false);
	}

	nemo::Configuration conf;
	conf.disableLogging();
	conf.setCudaPartitionSize(1024);

	nemo::mpi::Master sim(env, world, net, conf);

	/* Simulate a single neuron to get the ring going */
	sim.step(std::vector<unsigned>(1, 0));

	sim.readFiring();

	for(unsigned ms=1; ms < duration; ++ms) {
		sim.step();
		const std::vector<unsigned>& fired = sim.readFiring();
		BOOST_REQUIRE_EQUAL(fired.size(), 1U);
		BOOST_REQUIRE_EQUAL(fired.front(), ms % ncount);
	}
}



void
ring_mpi(boost::mpi::environment& env,
		boost::mpi::communicator& world,
		unsigned ncount)
{
	if(world.rank() == nemo::mpi::MASTER) {
		runRing(ncount, env, world);
	} else {
		nemo::mpi::runWorker(env, world);
	}
}



BOOST_AUTO_TEST_CASE(ring_tests)
{
	boost::mpi::environment env;
	boost::mpi::communicator world;

	ring_mpi(env, world, 512);  // less than a single partition on CUDA backend
	ring_mpi(env, world, 1024); // exactly one partition on CUDA backend
	ring_mpi(env, world, 2000); // multiple partitions on CUDA backend
}



void
runMpiSimulation(
		boost::mpi::environment& env,
		boost::mpi::communicator& world,
		const nemo::Network& net,
		nemo::Configuration& conf,
		unsigned seconds,
		std::vector<unsigned>& cycles,
		std::vector<unsigned>& neurons)
{
	nemo::mpi::Master sim(env, world, net, conf);

	cycles.clear();
	neurons.clear();

	for(unsigned s = 0; s < seconds; ++s)
	for(unsigned ms = 0; ms < 1000; ++ms) {
		sim.step();
		const std::vector<unsigned>& fired = sim.readFiring();
		std::copy(fired.begin(), fired.end(), back_inserter(neurons));
		std::fill_n(back_inserter(cycles), fired.size(), s*1000 + ms);
	}
}




BOOST_AUTO_TEST_CASE(comparison)
{
	boost::mpi::environment env;
	boost::mpi::communicator world;

	if(world.rank() == nemo::mpi::MASTER) {
		unsigned duration = 2;
		unsigned pcount = 4;
		bool stdp = false;
		nemo::Configuration conf = configuration(stdp, 1024);
		std::cout << "Non-mpi simulation using " << conf.backendDescription() << std::endl;
		std::vector<unsigned> cycles1, cycles2, nidx1, nidx2;
		boost::scoped_ptr<nemo::Network> net(nemo::torus::construct(pcount, 1000, stdp, 32, false));
		runMpiSimulation(env, world, *net, conf, duration, cycles1, nidx1);
		runSimulation(net.get(), conf, duration, &cycles2, &nidx2, stdp);
		std::cerr << "Ran MPI and non-MPI simulations resulting in "
			<< nidx1.size() << "/" << nidx2.size() << " firings\n";
		compareSimulationResults(cycles1, nidx1, cycles2, nidx2);
	} else {
		nemo::mpi::runWorker(env, world);
	}
}
