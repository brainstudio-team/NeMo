/* 
 * Toroidal network where the torus is constructed from a variable number of
 * 32x32-sized patches of neurons. The connectivity is such that the distance
 * between pre- and postsynaptic neurons is normally distributed (2D euclidian
 * distance along the torus surface) with the tails of the distributions capped
 * at 20x32.  Conductance delays are linearly dependent on distance and ranges
 * from 1ms to 20ms.
 *
 * This network shows usage of libnemo and can be used for benchmarking
 * purposes.
 *
 * Author: Andreas K. Fidjeland <andreas.fidjeland@imperial.ac.uk>
 * Date: March 2010
 */ 

#include <cmath>
#include <vector>
#include <boost/random.hpp>

#ifdef USING_MAIN
#	include <string>
#	include <fstream>
#	include <boost/scoped_ptr.hpp>
#	include <examples/common.hpp>
#endif

#include <nemo.hpp>


#define PATCH_WIDTH 32
#define PATCH_HEIGHT 32
#define PATCH_SIZE ((PATCH_WIDTH) * (PATCH_HEIGHT))

#define MAX_DELAY 20U

#define PI 3.14159265358979323846264338327

typedef unsigned char uchar;

/* Random number generators */
typedef boost::mt19937 rng_t;
typedef boost::variate_generator<rng_t&, boost::normal_distribution<double> > grng_t;
typedef boost::variate_generator<rng_t&, boost::uniform_real<double> > urng_t;


namespace nemo {
	namespace torus {

/* Global neuron index and distance */
typedef std::pair<unsigned, double> target_t;




/* Return global neuron index given location on torus */
unsigned
neuronIndex(unsigned patch, unsigned x, unsigned y)
{
	assert(x >= 0);
	assert(x < PATCH_WIDTH);
	assert(y >= 0);
	assert(y < PATCH_HEIGHT);
	return patch * PATCH_SIZE + y * PATCH_WIDTH + x; 
}



void
addExcitatoryNeuron(nemo::Network* net, unsigned nidx, urng_t& param)
{
	float v = -65.0f;
	float a = 0.02f;
	float b = 0.2f;
	float r1 = float(param());
	float r2 = float(param());
	float c = v + 15.0f * r1 * r1;
	float d = 8.0f - 6.0f * r2 * r2;
	float u = b * v;
	float sigma = 5.0f;
	net->addNeuron(nidx, a, b, c, d, u, v, sigma);
}



void
addInhibitoryNeuron(nemo::Network* net, unsigned nidx, urng_t& param)
{
	float v = -65.0f;
	float r1 = float(param());
	float a = 0.02f + 0.08f * r1;
	float r2 = float(param());
	float b = 0.25f - 0.05f * r2;
	float c = v; 
	float d = 2.0f;
	float u = b * v;
	float sigma = 2.0f;
	net->addNeuron(nidx, a, b, c, d, u, v, sigma);
}




/* Round to nearest integer away from zero */
inline
double
round(double r) {
	return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
	//return (r > 0.0) ? ceil(r) : floor(r);
}




target_t
targetNeuron(
		unsigned sourcePartition,
		unsigned sourceX,
		unsigned sourceY,
		unsigned pcount,
		grng_t& distance,
		urng_t& angle)
{
	//! \todo should we set an upper limit to the distance? 
	/* Make sure we don't connect back to self with (near) 0-distance connection.
	 * Perhaps better to simply reject very short distances? */
	//! \todo use fabs here. May need to update rtests
	double dist = 1.0 + abs(distance());
	double theta = angle();

	double distX = dist * cos(theta);
	double distY = dist * sin(theta);

	/* global x,y-coordinates */
	int globalX = sourcePartition * PATCH_WIDTH + sourceX + int(round(distX));
	int globalY = sourceY + int(round(distY));

	int targetY = globalY % PATCH_HEIGHT;
	if(targetY < 0)
		targetY += PATCH_HEIGHT;

	int torusSize = PATCH_WIDTH * pcount;
	globalX = globalX % torusSize;
	if(globalX < 0)
		globalX += torusSize;

	/* We only cross partition boundaries in the X direction */
	// deal with negative numbers here
	int targetPatch = globalX / PATCH_WIDTH;
	int targetX = globalX % PATCH_WIDTH;

	/* Don't connect to self unless we wrap around torus */
	assert(!(targetX == int(sourceX) && targetY == int(sourceY) && dist < PATCH_HEIGHT-1));

	return std::make_pair<unsigned, double>(neuronIndex(targetPatch, targetX, targetY), dist);
}




unsigned
delay(unsigned distance)
{
	if(distance >= MAX_DELAY*PATCH_WIDTH) {
		return MAX_DELAY;
	} else {
		unsigned d = 1 + distance / PATCH_WIDTH;
		assert(d <= MAX_DELAY);
		return d;
	}
}



void
addExcitatorySynapses(
		nemo::Network* net,
		unsigned patch, unsigned x, unsigned y,
		unsigned pcount, unsigned m,
		bool stdp,
		grng_t& distance,
		urng_t& angle,
		urng_t& rweight)
{
	unsigned source = neuronIndex(patch, x, y);
	for(unsigned sidx = 0; sidx < m; ++sidx) {
		//! \todo add dependence of delay on distance
		target_t target = targetNeuron(patch, x, y, pcount, distance, angle);
		float weight = 0.5f * float(rweight());
		net->addSynapse(source, target.first, delay(unsigned(target.second)), weight, stdp);
	}
}


void
addInhibitorySynapses(
		nemo::Network* net,
		unsigned patch, unsigned x, unsigned y,
		unsigned pcount, unsigned m,
		bool stdp,
		grng_t& distance,
		urng_t& angle,
		urng_t& rweight)
{
	unsigned source = neuronIndex(patch, x, y);
	for(unsigned sidx = 0; sidx < m; ++sidx) {
		//! \todo add dependence of delay on distance
		target_t target = targetNeuron(patch, x, y, pcount, distance, angle);
		float weight = float(-rweight());
		net->addSynapse(source, target.first, delay(unsigned(target.second)), weight, stdp);
	}
}



nemo::Network*
construct(unsigned pcount, unsigned m, bool stdp, double sigma, bool logging=true)
{
	nemo::Network* net = new nemo::Network();

	/* The network is a torus which consists of pcount rectangular patches,
	 * each with dimensions height * width. The size of each patch is the same
	 * as the partition size on the device. */
	const unsigned height = PATCH_HEIGHT;
	const unsigned width = PATCH_WIDTH;
	//! \todo check that this matches partition size
	
	rng_t rng;

	/* 80% of neurons are excitatory, 20% inhibitory. The spatial distribution
	 * of excitatory and inhibitory neurons is uniformly random. */
	boost::variate_generator<rng_t&, boost::bernoulli_distribution<double> >
		isExcitatory(rng, boost::bernoulli_distribution<double>(0.8));

	/* Postsynaptic neurons have a gaussian distribution of distance from
	 * presynaptic. Target neurons are in practice drawn from a 2D laplacian.
	 */ 

	/* Most inhibitory synapses are local. 95% fall within a patch. */
	double sigmaIn = width/2;
	grng_t distanceIn(rng, boost::normal_distribution<double>(0, sigmaIn));
	
	/* The user can control the distribution of the exitatory synapses */
	assert(sigma >= sigmaIn);
	grng_t distanceEx(rng, boost::normal_distribution<double>(0, sigma));

	urng_t angle(rng, boost::uniform_real<double>(0, 2*PI));

	/* Neuron parameters and weights are partially randomised */
	urng_t randomParameter(rng, boost::uniform_real<double>(0, 1));

	unsigned exCount = 0;
	unsigned inCount = 0;

	for(unsigned p = 0; p < pcount; ++p) {
		if(logging) {
			std::cout << "Partition " << p << std::endl;
		}
		for(unsigned y = 0; y < height; ++y) {
			for(unsigned x = 0; x < width; ++x) {
				unsigned nidx = neuronIndex(p, x, y);
				if(isExcitatory()) {
					addExcitatoryNeuron(net, nidx, randomParameter);
					addExcitatorySynapses(net, p, x, y, pcount, m, stdp,
							distanceEx, angle, randomParameter);
					exCount++;
				} else {
					addInhibitoryNeuron(net, nidx, randomParameter);
					addInhibitorySynapses(net, p, x, y, pcount, m, false,
							distanceIn, angle, randomParameter);
					inCount++;
				}
			}
		}
	}

	if(logging) {
		std::cout << "Constructed network with " << exCount + inCount << " neurons\n"
			<< "\t" << exCount << " excitatory\n"		
			<< "\t" << inCount << " inhibitory\n";
		//! \todo report connectivity stats as well
	}

	return net;
}

	} // namespace torus
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
			("pcount,n", po::value<unsigned>()->default_value(1), "number of 1024-sized partitions")
			("synapses,m", po::value<unsigned>()->default_value(1000), "number of synapses per neuron")
			("sigma,s", po::value<unsigned>()->default_value(32), "standard deviation in connectivity probability");
		;

		po::variables_map vm = processOptions(argc, argv, desc);

		unsigned pcount = vm["pcount"].as<unsigned>();
		unsigned sigma = vm["sigma"].as<unsigned>();
		unsigned m = vm["synapses"].as<unsigned>();
		unsigned stdp = vm["stdp-period"].as<unsigned>();
		unsigned duration = vm["duration"].as<unsigned>();
		unsigned verbose = vm["verbose"].as<unsigned>();
		bool runBenchmark = vm.count("benchmark") != 0;

		assert(sigma >= PATCH_WIDTH/2);

		std::ofstream file;
		std::string filename;

		if(vm.count("output-file")) {
			filename = vm["output-file"].as<std::string>();
			file.open(filename.c_str()); // closes on destructor
		}

		std::ostream& out = filename.empty() ? std::cout : file;

		//! \todo get RNG seed option from command line
		//! \todo otherwise seed from system time
	
		LOG(verbose, "Constructing network");
		boost::scoped_ptr<nemo::Network> net(nemo::torus::construct(pcount, m, stdp != 0, sigma, verbose >= 1));
		LOG(verbose, "Creating configuration");
		nemo::Configuration conf = configuration(vm);
		LOG(verbose, "Simulation will run on %s", conf.backendDescription());
		LOG(verbose, "Creating simulation\n");
		boost::scoped_ptr<nemo::Simulation> sim(nemo::simulation(*net, conf));
		LOG(verbose, "Running simulation");

		if(runBenchmark) {
			benchmark(sim.get(), pcount*PATCH_SIZE, m, vm);
		} else {
			simulate(sim.get(), duration, stdp, out);
		}
		LOG(verbose, "Simulation complete");
		return 0;
	} catch(std::exception& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	} catch(...) {
		std::cerr << "torus: An unknown error occurred\n";
		return -1;
	}
}

#endif

