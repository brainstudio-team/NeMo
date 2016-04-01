/* Test that we get the same result as previous runs */

#include <iostream>
#include <fstream>

#include <boost/test/unit_test.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/random.hpp>

#include <nemo.hpp>
#include <examples.hpp>

#include "utils.hpp"
#include "rtest.hpp"


typedef boost::mt19937 rng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_int<> > uirng_t;


void
run(nemo::Network* net, 
	backend_t backend,
	unsigned seconds,
	const std::string& filename,
	bool stdp,
	bool stimulus, // add both current and firing stimulus
	bool creating) // are we comparing against existing data or creating fresh data
{
	using namespace std;

	rng_t rng;
	// assume contigous 0-based indices
	uirng_t randomNeuron(rng, boost::uniform_int<>(0, net->neuronCount()-1));

	nemo::Configuration conf = configuration(stdp, 1024, backend);
	std::cerr << "running test on " << conf.backendDescription()
		<< " with stdp=" << stdp
		<< " and stimulus=" << stimulus << std::endl;


	fstream file;
	//! \todo determine canonical filename based on configuration
	file.open(filename.c_str(), creating ? ios::out : ios::in);
	if(!file.is_open()) {
		std::cerr << "Failed to open file " << filename << std::endl;
		BOOST_REQUIRE(false);
	}

	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));

	unsigned ce, ne; // expexted values

	nemo::Simulation::firing_stimulus firingStimulus;
	nemo::Simulation::current_stimulus currentStimulus;

	for(unsigned s = 0; s < seconds; ++s) {
		for(unsigned ms = 0; ms < 1000; ++ms) {

			if(stimulus) {
				firingStimulus.resize(1);
				firingStimulus[0] = randomNeuron();
				currentStimulus.resize(1);
				currentStimulus[0] = std::make_pair(randomNeuron(), 0.1f);
			}

			const std::vector<unsigned>& fired = sim->step(firingStimulus, currentStimulus);
			for(std::vector<unsigned>::const_iterator ni = fired.begin(); ni != fired.end(); ++ni) {
				unsigned c = s * 1000 + ms;
				unsigned n = *ni;
				if(creating) {
					file << c << "\t" << n << "\n";
				} else {
					BOOST_REQUIRE(!file.eof());
					file >> ce >> ne;
					BOOST_REQUIRE_EQUAL(c, ce);
					BOOST_REQUIRE_EQUAL(n, ne);
				}
			}
		}
		if(stdp) {
			sim->applyStdp(1.0);
		}
	}

	if(!creating) {
		/* Read one more word to read off the end of the file. We need to make
		 * sure that we're at the end of the file, as otherwise the test will
		 * pass if the simulation produces no firing */
		file >> ce >> ne;
		BOOST_REQUIRE(file.eof());
	}
}



void
runTorus(bool creating)
{
	{
		bool stdp = false;
		boost::scoped_ptr<nemo::Network> torus(nemo::torus::construct(4, 1000, stdp, 64, false));
#ifdef NEMO_CUDA_ENABLED
		run(torus.get(), NEMO_BACKEND_CUDA, 4, "test-cuda.dat", stdp, false, creating);
		run(torus.get(), NEMO_BACKEND_CUDA, 4, "test-cuda-stim.dat", stdp, true, creating);
#endif
		run(torus.get(), NEMO_BACKEND_CPU, 4, "test-cpu.dat", stdp, false, creating);
		run(torus.get(), NEMO_BACKEND_CPU, 4, "test-cpu-stim.dat", stdp, true, creating);
	}

	{
		bool stdp = true;
		boost::scoped_ptr<nemo::Network> torus(nemo::torus::construct(4, 1000, stdp, 64, false));
#ifdef NEMO_CUDA_ENABLED
		run(torus.get(), NEMO_BACKEND_CUDA, 4, "test-cuda-stdp.dat", stdp, false, creating);
		run(torus.get(), NEMO_BACKEND_CUDA, 4, "test-cuda-stdp-stim.dat", stdp, true, creating);
#endif
		run(torus.get(), NEMO_BACKEND_CPU, 4, "test-cpu-stdp.dat", stdp, false, creating);
		run(torus.get(), NEMO_BACKEND_CPU, 4, "test-cpu-stdp-stim.dat", stdp, true, creating);
	}
}



void
runKuramoto(bool creating)
{
#ifdef NEMO_CUDA_ENABLED
	using namespace std;

	boost::scoped_ptr<nemo::Network> net(nemo::kuramoto::construct(2048, 32));
	const std::string filename = "test-cuda-kuramoto.dat";

	nemo::Configuration conf;
	conf.setCudaBackend();
	std::cerr << "running kuramoto test on " << conf.backendDescription() << std::endl;

	fstream file;
	file.open(filename.c_str(), creating ? ios::out : ios::in);
	if(!file.is_open()) {
		std::cerr << "Failed to open file " << filename << std::endl;
		BOOST_REQUIRE(false);
	}

	boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));

	/* Test for just a few values */
	std::vector<unsigned> sample;
	sample.push_back(0);
	sample.push_back(100);
	sample.push_back(500);
	sample.push_back(1500);

	// expexted values
	unsigned ce, ne;
	float pe;

	unsigned seconds = 4;
	for(unsigned s = 0; s < seconds; ++s) {
		for(unsigned ms = 0; ms < 1000; ++ms) {

			sim->step();

			for(std::vector<unsigned>::const_iterator ni = sample.begin(); ni != sample.end(); ++ni) {
				unsigned c = s * 1000 + ms;
				unsigned n = *ni;
				float p = sim->getMembranePotential(n);
				if(creating) {
					file << c << "\t" << n << "\t" << p << "\n";
				} else {
					BOOST_REQUIRE(!file.eof());
					file >> ce >> ne >> pe;
					BOOST_REQUIRE(c == ce);
					BOOST_REQUIRE(n == ne);
					BOOST_REQUIRE_CLOSE(p, pe, 0.001f);
				}
			}
		}
	}

	if(!creating) {
		/* Read one more word to read off the end of the file. We need to make
		 * sure that we're at the end of the file, as otherwise the test will
		 * pass if the simulation produces no firing */
		file >> ce >> ne >> pe;
		BOOST_REQUIRE(file.eof());
	}
#endif
}
