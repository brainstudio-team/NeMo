#ifndef NEMO_TEST_HPP
#define NEMO_TEST_HPP

/* Network where some neurons have no outgoing synapses */
namespace networks {
	namespace no_outgoing {
		void run(backend_t backend);
	}

	namespace invalid_targets {
		void run(backend_t backend);
	}
}

#endif
