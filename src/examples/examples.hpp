#ifndef NEMO_EXAMPLES_HPP
#define NEMO_EXAMPLES_HPP

namespace nemo {

	/*! Construct a network with ncount neurons each of which has scount
	 * synapses. The synapses are given uniformly random targets from the whole
	 * population. 80% of synapses are excitatory and 20% are inhibitory, with
	 * weights chosen as in Izhikevich' reference implementation. All
	 * inhibitory delays are 1ms. Excitatory delays are uniformly random in the
	 * range [1, dmax]
	 *
	 * If 'stdp' is true, all excitatory synapses are marked as plastic, while
	 * inhibitory synapses are marked as static.
	 */
	namespace random {
		nemo::Network* constructUniformRandom(unsigned ncount, unsigned scount, unsigned dmax, bool stdp);
	}

	namespace torus {
		nemo::Network* construct(unsigned pcount, unsigned m, bool stdp, double sigma, bool logging);
	}

	namespace kuramoto {
		nemo::Network* construct(unsigned ncount, unsigned scount);
	}
}

#endif
