#include "randomNetUtils.hpp"
#include <iostream>
namespace nemo {
namespace random {

void addExcitatoryNeuron(nemo::Network* net, unsigned nidx, urng_t& param) {
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

void addInhibitoryNeuron(nemo::Network* net, unsigned nidx, urng_t& param) {
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

nemo::Network* constructUniformRandom(unsigned ncount, unsigned scount, unsigned dmax, bool stdp) {
	rng_t rng;
	/* Neuron parameters and weights are partially randomised */
	urng_t randomParameter(rng, boost::uniform_real<double>(0, 1));
	uirng_t randomTarget(rng, boost::uniform_int<>(0, ncount - 1));
	uirng_t randomDelay(rng, boost::uniform_int<>(1, dmax));

	nemo::Network* net = new nemo::Network();

	for ( unsigned nidx = 0; nidx < ncount; ++nidx ) {
		if ( nidx < (ncount * 4) / 5 ) { // excitatory
			addExcitatoryNeuron(net, nidx, randomParameter);
			for ( unsigned s = 0; s < scount; ++s ) {
				net->addSynapse(nidx, randomTarget(), randomDelay(), 0.5f * float(randomParameter()), stdp);
			}
		}
		else { // inhibitory
			addInhibitoryNeuron(net, nidx, randomParameter);
			for ( unsigned s = 0; s < scount; ++s ) {
				net->addSynapse(nidx, randomTarget(), 1U, float(-randomParameter()), 0);
			}
		}
	}
	return net;
}

/* Creates a ring lattice with N nodes and neighbourhood size k */
void networkRingLattice(std::vector<std::vector<unsigned> >& matrix, unsigned N, unsigned k) {

	for ( unsigned i = 0; i < N; i++ ) {
		for ( unsigned j = 0; j < N; j++ ) {
			matrix[i][j] = 0;
		}
	}

	for ( unsigned i = 0; i < N; i++ ) {
		for ( unsigned j = i + 1; j < N; j++ ) {
			unsigned diff = j - i;
			if ( std::min(diff, N - diff) <= (k / 2) ) {
				matrix[i][j] = 1;
				matrix[j][i] = 1;
			}
		}
	}
}

/* Creates a ring lattice with N nodes and neighbourhood size k, then
 rewires it according to the Watts-Strogatz procedure with probability p
 */
void networkWattsStrogatz(std::vector<std::vector<unsigned> >& matrix, unsigned N, unsigned k, double p) {
	networkRingLattice(matrix, N, k);

	for ( unsigned i = 0; i < N; i++ ) {
		for ( unsigned j = i; j < N; j++ ) {
			double rand = ((double) std::rand() / (RAND_MAX));
			if ( matrix[i][j] == 1 && rand < p ) {
				matrix[i][j] = 0;
				matrix[j][i] = 0;
				unsigned idx = static_cast<unsigned>(i + std::ceil(rand * (N - 1)) - 1) % N;
				matrix[i][idx] = 0;
				matrix[idx][i] = 0;
			}
		}
	}
}

nemo::Network* constructWattsStrogatz(unsigned k, unsigned p, unsigned ncount, unsigned dmax, bool stdp) {
	rng_t rng;
	/* Neuron parameters and weights are partially randomised */
	urng_t randomParameter(rng, boost::uniform_real<double>(0, 1));
	uirng_t randomDelay(rng, boost::uniform_int<>(1, dmax));

	std::vector<std::vector<unsigned> > matrix(ncount, std::vector<unsigned>(ncount));
	networkWattsStrogatz(matrix, ncount, k, p);

	nemo::Network* net = new nemo::Network();

	for ( unsigned nidx = 0; nidx < ncount; ++nidx ) {
		if ( nidx < (ncount * 4) / 5 ) { // excitatory
			addExcitatoryNeuron(net, nidx, randomParameter);
			for ( unsigned j = 0; j < ncount; ++j ) {
				if ( matrix[nidx][j] == 1 )
					net->addSynapse(nidx, j, randomDelay(), 0.5f * float(randomParameter()), stdp);
			}
		}
		else { // inhibitory
			addInhibitoryNeuron(net, nidx, randomParameter);
			for ( unsigned j = 0; j < ncount; ++j ) {
				if ( matrix[nidx][j] == 1 )
					net->addSynapse(nidx, j, 1U, float(-randomParameter()), 0);
			}
		}
	}
	return net;
}

/* Used for debugging */
nemo::Network* simpleNet(unsigned ncount, unsigned dmax, bool stdp) {
	rng_t rng;
	/* Neuron parameters and weights are partially randomised */
	urng_t randomParameter(rng, boost::uniform_real<double>(0, 1));
	uirng_t randomDelay(rng, boost::uniform_int<>(1, dmax));
	nemo::Network* net = new nemo::Network();
	for ( unsigned nidx = 0; nidx < ncount; ++nidx ) {
		if ( nidx < (ncount * 4) / 5 ) { // excitatory
			nemo::random::addExcitatoryNeuron(net, nidx, randomParameter);
			if ( nidx == 0 ) {
				net->addSynapse(nidx, 1, randomDelay(), 0.5f * float(randomParameter()), stdp);
				net->addSynapse(nidx, 6, randomDelay(), 0.5f * float(randomParameter()), stdp);
				net->addSynapse(nidx, 11, randomDelay(), 0.5f * float(randomParameter()), stdp);
				net->addSynapse(2, nidx, randomDelay(), 0.5f * float(randomParameter()), stdp);
				net->addSynapse(3, nidx, randomDelay(), 0.5f * float(randomParameter()), stdp);
				net->addSynapse(4, nidx, randomDelay(), 0.5f * float(randomParameter()), stdp);
			}
			if ( nidx == 8 ) {
				net->addSynapse(nidx, 2, randomDelay(), 0.5f * float(randomParameter()), stdp);
				net->addSynapse(nidx, 7, randomDelay(), 0.5f * float(randomParameter()), stdp);
				net->addSynapse(nidx, 14, randomDelay(), 0.5f * float(randomParameter()), stdp);
			}
		}
		else { // inhibitory
			nemo::random::addInhibitoryNeuron(net, nidx, randomParameter);
			if ( nidx == 13 ) {
				net->addSynapse(nidx, 3, randomDelay(), -0.5f * float(randomParameter()), stdp);
				net->addSynapse(nidx, 8, randomDelay(), -0.5f * float(randomParameter()), stdp);
				net->addSynapse(nidx, 14, randomDelay(), -0.5f * float(randomParameter()), stdp);
			}
		}
	}
	return net;
}

unsigned getRandom(int start, int end) {
	return start + (std::rand() % (end - start + 1));
}

/* Used to demonstrate the optimal network for MPI simulations */
nemo::Network* constructSemiRandom(unsigned ncount, unsigned scount, unsigned dmax, bool stdp, unsigned workers, float ratio) {
	rng_t rng;
	/* Neuron parameters and weights are partially randomised */
	urng_t randomParameter(rng, boost::uniform_real<double>(0, 1));
	uirng_t randomDelay(rng, boost::uniform_int<>(1, dmax));

	nemo::Network* net = new nemo::Network();

	/* Firstly, add ncount neurons to the network.
	 * Properties: 80% excitatory, 20% inhibitory.
	 */
	for ( unsigned nidx = 0; nidx < ncount; ++nidx ) {
		if ( nidx < (ncount * 4) / 5 ) { // excitatory
			addExcitatoryNeuron(net, nidx, randomParameter);
		}
		else { // inhibitory
			addInhibitoryNeuron(net, nidx, randomParameter);
		}
	}

	/* Initialize partition counters */
	float avgNeurons = ceil(((float) ncount) / ((float) workers));
	std::vector<unsigned> counters(workers, avgNeurons);
	counters.back() = ncount - (workers - 1) * avgNeurons;

	std::vector<std::pair<nidx_t, nidx_t> > ranges;

	/* Make index ranges for each partition */
	for ( unsigned r = 0; r < workers; r++ ) {
		unsigned start = r * avgNeurons;
		std::pair<unsigned, unsigned> range(start, start + counters[r] - 1);
		ranges.push_back(range);
	}

	for ( unsigned i = 0; i < workers; i++ ) {
		uirng_t randomLocalNeuron(rng, boost::uniform_int<>(ranges[i].first, ranges[i].second));
		unsigned partitionSynapses = counters[i] * scount;
		unsigned globalSynapses = ratio * partitionSynapses;
		unsigned localSynapses = partitionSynapses - globalSynapses;

//		std::cout << "partitionSynapses: " << partitionSynapses << std::endl;
//		std::cout << "localSynapses: " << localSynapses << std::endl;
//		std::cout << "globalSynapses: " << globalSynapses << std::endl;
//		std::cout << std::endl;

		for ( unsigned j = 0; j < localSynapses; j++ ) {
			unsigned source;
			unsigned target;
			while (true) {
				source = randomLocalNeuron();
				target = randomLocalNeuron();
				if ( source != target )
					break;
			}

			if ( (unsigned) randomLocalNeuron() < (ncount * 4) / 5 )
				net->addSynapse(source, target, randomDelay(), 0.5f * float(randomParameter()), stdp);
			else
				net->addSynapse(source, target, 1U, float(-randomParameter()), stdp);
		}

		for ( unsigned j = 0; j < globalSynapses; j++ ) {

			uirng_t randomWorker(rng, boost::uniform_int<>(0, workers - 1));
			unsigned randomNeighbour;
			while (true) {
				randomNeighbour = randomWorker();
				if ( randomNeighbour != i )
					break;
			}

			uirng_t randomGlobalNeuron(rng,
					boost::uniform_int<>(ranges[randomNeighbour].first, ranges[randomNeighbour].second));

			if ( (unsigned) randomLocalNeuron() < (ncount * 4) / 5 )
				net->addSynapse(randomLocalNeuron(), randomGlobalNeuron(), randomDelay(), 0.5f * float(randomParameter()), stdp);
			else
				net->addSynapse(randomLocalNeuron(), randomGlobalNeuron(), 1U, float(-randomParameter()), stdp);

		}
	}

//	std::cout << "net->neuronCount: " << net->neuronCount() << std::endl;
//	std::cout << "net->synapseCount: " << net->synapseCount() << std::endl;
	return net;
}

} // namespace random
} // namespace nemo
