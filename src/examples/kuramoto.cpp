#include <vector>

#ifdef USING_MAIN
#	include <string>
#	include <iostream>
#	include <fstream>
#	include <boost/program_options.hpp>
#	include <boost/scoped_ptr.hpp>
#	include <examples/common.hpp>
#endif

#include <boost/random.hpp>
#include <nemo.hpp>
#include <nemo/util.h>

typedef boost::mt19937 rng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_real<double> > urng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_int<> > uirng_t;

namespace nemo {
	namespace kuramoto {

class OscillatorNetwork : public nemo::Network
{
	public :

		OscillatorNetwork() {
			m_type = addNeuronType("Kuramoto");
		}

		void add(unsigned idx, double frequency, double phase) {
			static float args[2];
			args[0] = float(frequency);
			args[1] = float(phase);
			addNeuron(m_type, idx, 2, args);
		}

		void connect(unsigned source, unsigned target,
				unsigned lag, float strength) {
			addSynapse(source, target, lag, strength, false);
		}

	private :

		unsigned m_type;
};




nemo::Network*
construct(unsigned ncount, unsigned scount)
{
	rng_t rng;
	/* Neuron parameters and weights are partially randomised */
	urng_t randomFrequency(rng, boost::uniform_real<double>(0, 0.2));
	urng_t randomPhase(rng, boost::uniform_real<double>(0, 2*M_PI));
	uirng_t randomTarget(rng, boost::uniform_int<>(0, ncount-1));

	OscillatorNetwork* net = new OscillatorNetwork();

	for(unsigned nidx=0; nidx < ncount; ++nidx) {
		net->add(nidx, randomFrequency(), randomPhase());
		for(unsigned s = 0; s < scount; ++s) {
			net->connect(nidx, randomTarget(), 1U, 0.001f);
		}
	}
	return net;
}

	} // namespace kuramoto
} // namespace nemo


#ifdef USING_MAIN


#define LOG(cond, ...) if(cond) { fprintf(stdout, __VA_ARGS__); fprintf(stdout, "\n"); }


int
main(int argc, char* argv[])
{
	namespace po = boost::program_options;

	try {

		po::options_description desc = commonOptions();
		desc.add_options()
			("oscillators,n", po::value<unsigned>()->default_value(1000), "number of oscillators")
			("couplings,m", po::value<unsigned>()->default_value(1000), "number of couplings (out-degree) per oscillators")
		;

		po::variables_map vm = processOptions(argc, argv, desc);

		unsigned ncount = vm["oscillators"].as<unsigned>();
		unsigned scount = vm["couplings"].as<unsigned>();
		unsigned verbose = vm["verbose"].as<unsigned>();

		LOG(verbose, "Constructing network");
		boost::scoped_ptr<nemo::Network> net(nemo::kuramoto::construct(ncount, scount));
		LOG(verbose, "Creating configuration");
		nemo::Configuration conf = configuration(vm);
		conf.setCudaBackend();
		conf.setCudaPartitionSize(256);
		LOG(verbose, "Simulation will run on %s", conf.backendDescription());
		LOG(verbose, "Creating simulation");
		boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));
		LOG(verbose, "Running simulation");
		unsigned seconds = 10;
		uint64_t elapsed = benchmark(sim.get(), ncount, scount, vm, seconds); // ms
		std::cout << "Elapsed: " << elapsed << "ms\n";
		uint64_t oscUpdates = uint64_t(seconds)*1000000/elapsed;
		std::cout << "Updates/second: " << oscUpdates << std::endl;
		uint64_t cplUpdatesM = uint64_t(seconds*ncount*scount)/elapsed; // a bunch of factors cancel out
		std::cout << "Couplings processed/second: " << cplUpdatesM << std::endl;
		LOG(verbose, "Simulation complete");
		return 0;
	} catch(std::exception& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	} catch(...) {
		std::cerr << "random: An unknown error occurred\n";
		return -1;
	}

}

#endif
