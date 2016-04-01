#include <iostream>
#include <fstream>
#include <cmath>

#include <boost/random.hpp>

#include "common.hpp"

typedef boost::mt19937 rng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_int<> > uirng_t;


long
benchmark(nemo::Simulation* sim, unsigned n, unsigned m,
				boost::program_options::variables_map& vm, unsigned seconds)
{

	unsigned stdpPeriod = vm["stdp-period"].as<unsigned>();
	float stdpReward = vm["stdp-reward"].as<float>();
	bool csv = vm.count("csv") != 0;
	bool verbose = !csv;
	bool provideFiringStimulus = vm.count("fstim") != 0;
	bool provideCurrentStimulus = vm.count("istim") != 0;
	bool vprobe = vm.count("vprobe") != 0;

	const unsigned MS_PER_SECOND = 1000;

	rng_t rng;
	uirng_t randomNeuron(rng, boost::uniform_int<>(0, n-1));

	sim->resetTimer();

	unsigned t = 0;

	/* Run for a few seconds to warm up the network */
	if(verbose)
		std::cout << "Running simulation (warming up)...";
	for(unsigned s=0; s < 5; ++s) {
		for(unsigned ms = 0; ms < MS_PER_SECOND; ++ms, ++t) {
			sim->step();
			if(stdpPeriod && t % stdpPeriod == 0) {
				sim->applyStdp(stdpReward);
			}
		}
	}
	if(verbose)
		std::cout << "[" << sim->elapsedWallclock() << "ms elapsed]";
	sim->resetTimer();

	if(verbose) {
		std::cout << std::endl;
		std::cout << "Running simulation (gathering performance data)...";
	}

	nemo::Simulation::firing_stimulus firingStimulus;
	nemo::Simulation::current_stimulus currentStimulus;

	unsigned long nfired = 0;
	for(unsigned s=0; s < seconds; ++s) {
		if(verbose)
			std::cout << s << " ";
		for(unsigned ms=0; ms<1000; ++ms, ++t) {

			if(provideFiringStimulus) {
				firingStimulus.resize(1);
				firingStimulus[0] = randomNeuron();
			}

			if(provideCurrentStimulus) {
				currentStimulus.resize(1);
				currentStimulus[0] = std::make_pair(randomNeuron(), 0.001f);
			}

			const std::vector<unsigned>& fired = sim->step(firingStimulus, currentStimulus);
			nfired += fired.size();

			if(vprobe) {
				/* Just read a single value. The whole neuron population will be synced */
				sim->getMembranePotential(0);
			}

			if(stdpPeriod && t % stdpPeriod == 0) {
				sim->applyStdp(stdpReward);
			}
		}
	}
	long int elapsedWallclock = sim->elapsedWallclock();
	if(verbose)
		std::cout << "[" << elapsedWallclock << "ms elapsed]";
	if(verbose)
		std::cout << std::endl;

	unsigned long narrivals = nfired * m;
	double f = (double(nfired) / n) / double(seconds);

	/* Throughput is measured in terms of the number of spike arrivals per
	 * wall-clock second */
	unsigned long throughput = MS_PER_SECOND * narrivals / elapsedWallclock;
	double speedup = double(seconds*MS_PER_SECOND)/elapsedWallclock;

	if(verbose) {
		std::cout << "Total firings: " << nfired << std::endl;
		std::cout << "Avg. firing rate: " << f << "Hz\n";
		std::cout << "Spike arrivals: " << narrivals << std::endl;
		std::cout << "Approx. throughput: " << throughput/1000000
				<< "Ma/s (million spike arrivals per second)\n";
		std::cout << "Speedup wrt real-time: " << speedup << std::endl;
	}

	if(csv) {
		std::string sep = ", ";
		// raw data
		std::cout << n << sep << m << sep << seconds*MS_PER_SECOND
			<< sep << elapsedWallclock << sep << stdpPeriod << sep << nfired;
		// derived data. Not strictly needed, at least if out-degree is fixed.
		std::cout << sep << narrivals << sep << f << sep << speedup
			<< sep << throughput/1000000 << std::endl;
	}

	return elapsedWallclock;
}



void
simulate(nemo::Simulation* sim, unsigned time_ms, unsigned stdp, std::ostream& out)
{
	for(unsigned ms=0; ms<time_ms; ) {
		const std::vector<unsigned>& fired = sim->step();
		for(std::vector<unsigned>::const_iterator fi = fired.begin(); fi != fired.end(); ++fi) {
			out << ms << " " << *fi << "\n";
		}
		ms += 1;
		if(stdp != 0 && ms % stdp == 0) {
			sim->applyStdp(1.0);
		}
	}
}



void
simulateToFile(nemo::Simulation* sim, unsigned time_ms, unsigned stdp, const char* firingFile)
{
	std::ofstream file;
	file.open(firingFile);
	simulate(sim, time_ms, stdp, file);
	file.close();
}



void
setStandardStdpFunction(nemo::Configuration& conf)
{
	std::vector<float> pre(20);
	std::vector<float> post(20);
	for(unsigned i = 0; i < 20; ++i) {
		float dt = float(i + 1);
		pre.at(i) = 0.1f * expf(-dt / 20.0f);
		post.at(i) = -0.08f * expf(-dt / 20.0f);
	}
	conf.setStdpFunction(pre, post, -1.0f, 1.0f);
}



nemo::Configuration
configuration(boost::program_options::variables_map& opts)
{
	bool stdp = opts["stdp-period"].as<unsigned>() != 0;
	bool log = opts["verbose"].as<unsigned>() >= 2;
	bool cpu = opts.count("cpu") != 0;
	bool cuda = opts.count("cuda") != 0;

	if(cpu && cuda) {
		std::cerr << "Multiple backends selected on command line\n";
		exit(-1);
	}

	nemo::Configuration conf;
	conf.setWriteOnlySynapses();

	if(log) {
		conf.enableLogging();
	}

	if(stdp) {
		setStandardStdpFunction(conf);
	}

	if(cpu) {
		conf.setCpuBackend();
	} else if(cuda) {
		conf.setCudaBackend();
	}
	// otherwise just go with the default backend

	return conf;
}



boost::program_options::options_description
commonOptions()
{
	namespace po = boost::program_options;

	po::options_description desc("Allowed options");
	desc.add_options()
		("help,h", "print this message")
		("duration,t", po::value<unsigned>()->default_value(1000), "duration of simulation (ms)")
		("stdp-period", po::value<unsigned>()->default_value(0), "STDP application period (ms). If 0 do not use STDP")
		("stdp-reward", po::value<float>()->default_value(1.0), "STDP reward")
		("cpu", "Use the CPU backend")
		("cuda", "Use the CUDA backend (default device)")
		("verbose,v", po::value<unsigned>()->default_value(0), "Set verbosity level")
		("output-file,o", po::value<std::string>(), "output file for firing data")
		("benchmark", "report performance results instead of returning firing")
		("csv", "when benchmarking, output a compact CSV format with the following fields: neurons, synapses, simulation time (ms), wallclock time (ms), STDP enabled, fired neurons, PSPs generated, average firing rate, speedup wrt real-time, throughput (million PSPs/second")
		("fstim", "provide (very weak) firing stimulus, for benchmarking purposes")
		("istim", "provide (very weak) current stimulus, for benchmarking purposes")
		("vprobe", "probe membrane potential every cycle")
	;

	return desc;
}



boost::program_options::variables_map
processOptions(int argc, char* argv[],
		const boost::program_options::options_description& desc)
{
	namespace po = boost::program_options;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if(vm.count("help")) {
		std::cout << "Usage:\n\t" << argv[0] << " [OPTIONS]\n\n";
		std::cout << desc << std::endl;
		exit(1);
	}

	return vm;
}
