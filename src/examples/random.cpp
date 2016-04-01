/* Simple network with 1000 neurons with all-to-all connections with random
 * weights.

 * Author: Andreas K. Fidjeland <andreas.fidjeland@imperial.ac.uk>
 * Date: April 2010
 */
#ifdef USING_MAIN
#	include <string>
#	include <iostream>
#	include <fstream>
#	include <boost/program_options.hpp>
#	include <boost/scoped_ptr.hpp>
#	include <examples/common.hpp>
#endif

#include "randomNetUtils.hpp"

#ifdef USING_MAIN


#define LOG(cond, ...) if(cond) { fprintf(stdout, __VA_ARGS__); fprintf(stdout, "\n"); }


int
main(int argc, char* argv[])
{
	namespace po = boost::program_options;

	try {

		po::options_description desc = commonOptions();
		desc.add_options()
			("neurons,n", po::value<unsigned>()->default_value(1000), "number of neurons")
			("synapses,m", po::value<unsigned>()->default_value(1000), "number of synapses per neuron")
			("dmax,d", po::value<unsigned>()->default_value(1), "maximum excitatory delay,  where delays are uniform in range [1, dmax]")
		;

		po::variables_map vm = processOptions(argc, argv, desc);

		unsigned ncount = vm["neurons"].as<unsigned>();
		unsigned scount = vm["synapses"].as<unsigned>();
		unsigned dmax = vm["dmax"].as<unsigned>();
		unsigned duration = vm["duration"].as<unsigned>();
		unsigned stdp = vm["stdp-period"].as<unsigned>();
		unsigned verbose = vm["verbose"].as<unsigned>();
		bool runBenchmark = vm.count("benchmark") != 0;

		std::ofstream file;
		std::string filename;

		if(vm.count("output-file")) {
			filename = vm["output-file"].as<std::string>();
			file.open(filename.c_str()); // closes on destructor
		}

		std::ostream& out = filename.empty() ? std::cout : file;

		LOG(verbose, "Constructing network");
		boost::scoped_ptr<nemo::Network> net(nemo::random::constructUniformRandom(ncount, scount, dmax, stdp != 0));
		LOG(verbose, "Creating configuration");
		nemo::Configuration conf = configuration(vm);
		LOG(verbose, "Simulation will run on %s", conf.backendDescription());
		LOG(verbose, "Creating simulation");
		boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));
		LOG(verbose, "Running simulation");
		if(runBenchmark) {
			benchmark(sim.get(), ncount, scount, vm);
		} else {
			simulate(sim.get(), duration, stdp, out);
		}
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
