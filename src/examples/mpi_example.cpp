#include <iostream>
#include <fstream>
#include <iterator>
#include <boost/scoped_ptr.hpp>

#include <nemo.hpp>
#include <nemo/exception.hpp>
#include <examples.hpp>

#include "MpiTypes.hpp"
#include "mpi/Master.hpp"
#include "mpi/Worker.hpp"
#include "randomNetUtils.hpp"

#include <boost/filesystem.hpp>

typedef boost::mt19937 rng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_int<> > uirng_t;

void writeStats(const char* filename, unsigned spncount, unsigned elapsedSimulation, unsigned elapsedWallClock,
		const nemo::Network& net, nemo::Configuration& conf) {

	std::ofstream file(filename, std::fstream::out | std::fstream::app);

	file << "--------------------------------------------------" << std::endl;
	file << "Neurons: " << net.neuronCount() << std::endl;
	file << "Total number of synapses: " << net.synapseCount() << std::endl;
	file << "Synapses per neuron (average): " << net.synapseCount() / net.neuronCount() << std::endl;
	file << "Max delay: " << net.maxDelay() << std::endl;
	file << "Max weight: " << net.maxWeight() << std::endl;
	file << "Min weight: " << net.minWeight() << std::endl;
	file << std::endl;
	file << "Simulated: " << elapsedSimulation << " ms." << std::endl;
	file << "Wall clock time: " << elapsedWallClock << "ms." << std::endl;
	file << std::endl;
}

void setStandardStdpFunction(nemo::Configuration& conf) {
	std::vector<float> pre(20);
	std::vector<float> post(20);
	for ( unsigned i = 0; i < 20; ++i ) {
		float dt = float(i + 1);
		pre.at(i) = 0.1f * expf(-dt / 20.0f);
		post.at(i) = -0.08f * expf(-dt / 20.0f);
	}
	conf.setStdpFunction(pre, post, -1.0f, 1.0f);
}

int run(boost::mpi::environment& env, boost::scoped_ptr<nemo::Network>& net, unsigned ncount, unsigned scount,
		unsigned duration, unsigned nType, unsigned mType, const char* outfile, const char* configFile, const char* logDir) {

	boost::mpi::communicator world;

	boost::filesystem::path logDirPath(logDir);
	boost::filesystem::create_directory(logDirPath);

	std::ostringstream simDir;
	simDir << logDir << "/simulation-p" << world.size() << "-n" << ncount << "-s" << scount << "-nt" << nType << "-mt" << mType;
	boost::filesystem::path simDirPath(simDir.str());
	boost::filesystem::create_directory(simDirPath);

	try {
		if ( world.rank() == nemo::mpi::MASTER ) {
			nemo::Configuration conf(world.rank(), configFile);
			std::cout << "Constructing master..." << std::endl;
			nemo::mpi::Master master(env, world, *net, mType, conf, logDir);

			std::ostringstream filename;
			filename << simDir.str() << "/" << outfile;
			std::ofstream file(filename.str().c_str());

			rng_t rng;
			uirng_t randomNeuron(rng, boost::uniform_int<>(0, ncount - 1));

			master.resetTimer();
			nemo::Simulation::firing_stimulus firingStimulus(10);

			for ( unsigned ms = 0; ms < duration; ++ms ) {
				for ( unsigned i = 0; i < 10; i++ )
					firingStimulus[i] = randomNeuron();

				master.step(firingStimulus);
				const std::vector<unsigned>& firing = master.readFiring();
				file << ms << ": ";
				std::copy(firing.begin(), firing.end(), std::ostream_iterator<unsigned>(file, " "));
				file << std::endl;
			}

			std::ostringstream oss;
			oss << "Simulated " << master.elapsedSimulation() << " ms wall clock time: " << master.elapsedWallclock() << "ms"
					<< std::endl;
			std::cout << oss.str() << std::endl;

			/* Write general stats to simulation folder */
			filename.str("");
			filename.clear();
			filename << simDir.str() << "/general-stats.txt";
			writeStats(filename.str().c_str(), scount, master.elapsedSimulation(), master.elapsedWallclock(), *net, conf);
			master.appendMpiStats(filename.str().c_str());

			/* Write firing counters to simulation folder */
			filename.str("");
			filename.clear();
			filename << simDir.str() << "/firing-counters.txt";
			master.writeFiringCounters(filename.str().c_str());
		}
		else {
			std::cout << "Starting worker " << world.rank() << "..." << std::endl;
			nemo::Configuration conf(world.rank(), configFile);
			if ( conf.stdpEnabled() ) {
				std::cout << "STDP enabled" << std::endl;
				setStandardStdpFunction(conf);
			}
			nemo::mpi::runWorker(env, world, conf, simDir.str().c_str());
		}
	}
	catch (nemo::exception& e) {
		std::cerr << world.rank() << ":" << e.what() << std::endl;
		env.abort(e.errorNumber());
	}
	catch (boost::mpi::exception& e) {
		std::cerr << world.rank() << ": " << e.what() << std::endl;
		env.abort(-1);
	}

	return 0;
}

/*! \note when Master is a proper subclass of Simulation, we can share code
 * between the two run functions. */
int runNoMPI(unsigned ncount, unsigned scount, unsigned duration, unsigned mType, const char* filename,
		const char* logDir) {
	std::cout << "Start simulation." << std::endl;

	nemo::Network* net;
//nemo::Network* net = simpleNet(15, 64, false);

	if ( mType == 0 )
		net = nemo::random::constructUniformRandom(ncount, scount, 40, false);
	else if ( mType == 1 )
		net = nemo::random::constructWattsStrogatz(20, 0.1, ncount, 40, false);
	else if ( mType == 43 )
		net = nemo::random::simpleNet(15, 64, false);
	else
		exit(1);

	nemo::Configuration conf;

	conf.setCudaBackend();
	//conf.setCpuBackend();

	conf.setStdpEnabled(false);
	conf.setStdpPeriod(10);
	conf.setStdpReward(1);
	if ( conf.stdpEnabled() ) {
		std::cout << "STDP enabled" << std::endl;
		setStandardStdpFunction(conf);
	}

	nemo::Simulation* sim = nemo::simulation(*net, conf);

	std::ofstream file(filename);

	nemo::NeuronType neuronType("Izhikevich");

	std::ostringstream oss;
	for ( unsigned n = 0; n < 15; n++ ) {
		oss << "Neuron " << n << " --- Parameters: ";

		for ( unsigned i = 0; i < neuronType.parameterCount(); i++ )
			oss << "  " << net->getNeuronParameter(n, i);

		oss << " --- State: ";
		for ( unsigned i = 0; i < neuronType.stateVarCount(); i++ )
			oss << "  " << net->getNeuronState(n, i);

		oss << std::endl;
	}

	std::cout << oss.str() << std::endl;

	sim->resetTimer();
	std::vector<unsigned> stim;

	for ( unsigned ms = 0; ms < duration; ++ms ) {
		if ( ms % 2 == 0 ) {
			stim.push_back(0);
		}
		else {
			stim.clear();
		}
		const std::vector<unsigned>& fired = sim->step(stim);
		file << ms << ": ";
		std::copy(fired.begin(), fired.end(), std::ostream_iterator<unsigned>(file, " "));
		file << std::endl;

		if ( conf.stdpEnabled() && sim->elapsedSimulation() % conf.stdpPeriod() == 0 ) {
			sim->applyStdp(conf.stdpReward());
		}
	}

	oss.str("");
	oss.clear();
	oss << "Simulated " << sim->elapsedSimulation() << " ms wall clock time: " << sim->elapsedWallclock() << "ms"
			<< std::endl;
	std::cout << oss.str() << std::endl;

	oss.str("");
	oss.clear();
	oss << logDir << "/stats_global-n" << ncount << "-s" << scount << "-st" << sim->elapsedSimulation() << ".txt";
	writeStats(oss.str().c_str(), scount, sim->elapsedSimulation(), sim->elapsedWallclock(), *net, conf);

	delete net;

	return 0;
}

int main(int argc, char* argv[]) {
	if ( argc != 9 && argc != 10 ) {
		std::cerr
				<< "Usage: example <ncount> <scount> <duration> <nettype> <mappertype> <outfile> <configfile> <logdir> [--nompi]\n";
		std::cerr << "Use logdir = logConsole for redirecting logging to terminal." << std::endl;
		return -1;
	}

	unsigned ncount = atoi(argv[1]);
	unsigned scount = atoi(argv[2]);
	unsigned duration = atoi(argv[3]);
	unsigned nettype = atoi(argv[4]);
	unsigned mappertype = atoi(argv[5]);
	char* outfile = argv[6];
	char* configFile = argv[7];
	char* logDir = argv[8];
	bool usingMpi = true;

	if ( argc == 9 && strcmp(argv[8], "--nompi") == 0 ) {
		usingMpi = false;
	}

	if ( usingMpi ) {
		boost::mpi::environment env(argc, argv);
		boost::mpi::communicator world;

		boost::scoped_ptr<nemo::Network> net;
		if ( world.rank() == nemo::mpi::MASTER ) {
			std::cout << "Constructing network..." << std::endl;
			if ( nettype == 0 )
				net.reset(nemo::random::constructUniformRandom(ncount, scount, 40, false));
			else if ( nettype == 1 )
				net.reset(nemo::random::constructSemiRandom(ncount, scount, 40, false, world.size() - 1, 0.05));
			else if ( nettype == 2 )
				net.reset(nemo::random::constructWattsStrogatz(scount, 0.1, ncount, 40, false));
			else
				exit(1);
		}
		run(env, net, ncount, scount, duration, nettype, mappertype, outfile, configFile, logDir);
		return 0;
	}
	else {
		return runNoMPI(ncount, scount, duration, mappertype, outfile, logDir);
	}
}
