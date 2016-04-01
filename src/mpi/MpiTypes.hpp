#ifndef NEMO_MPI_TYPES_HPP
#define NEMO_MPI_TYPES_HPP

#include <nemo/config.h>
#include <nemo/types.hpp>
#include <nemo/network/Generator.hpp>
#include <nemo/Network.hpp>
#include <nemo/NetworkImpl.hpp>
#include <nemo/Configuration.hpp>
#include <nemo/ConfigurationImpl.hpp>

#include <boost/scoped_ptr.hpp>

#include <boost/mpi.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/utility.hpp>

#ifdef NEMO_MPI_DEBUG_TIMING
#	include "MpiTimer.hpp"
#endif

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sources/logger.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/attributes/attribute.hpp>

#include <sstream>

namespace nemo {
namespace mpi {

typedef std::pair<nidx_t, delay_t> syn_key;
typedef std::pair<nidx_t, float> syn_value;

enum Ranks {
	MASTER = 0
};

enum CommTag {
	NEURON_VECTOR, NEURONS_END, SYNAPSE_VECTOR, SYNAPSES_END, MASTER_STEP, WORKER_STEP
};


/* Every cycle the master synchronises with each worker. */
class SimulationStep {
	public:

		SimulationStep() :
				terminate(false) {
		}

		SimulationStep(bool terminate, std::vector<unsigned> fstim) :
				terminate(terminate), fstim(fstim) {
		}

		/* Add neuron to list of neurons which should be
		 * forced to fire */
		void forceFiring(nidx_t neuron) {
			fstim.push_back(neuron);
		}

		bool terminate;
		std::vector<unsigned> fstim;

	private:

		friend class boost::serialization::access;

		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & terminate;
			ar & fstim;
		}
};

}

}

#endif
