#ifndef NEMO_MPI_WORKER_LOGGER_HPP
#define NEMO_MPI_WORKER_LOGGER_HPP

#include <MpiTypes.hpp>
#include "mpiUtils.hpp"
#include "Worker.hpp"
#include <fstream>
#include <iostream>

using namespace boost::log::trivial;

namespace nemo {
namespace mpi {

class Worker;

class WorkerLogger {

	friend class Worker;
public:
	//! \todo move this type to nemo::FiringBuffer instead perhaps typedefed as Fired::neuron_list
	typedef std::vector<nidx_t> nidxVec;
	typedef std::vector<boost::mpi::request> mpiReqVec;
	typedef std::map<rank_t, nidxVec> rankIndexesMap;

	WorkerLogger(const Worker * worker, const char *);

	void writeStats();

	void logNeurons();

	void logNeuron(nidx_t n);

	void logNeuronIndexes(const RandomMapper<nidx_t>& localMapper);

	void logNeuronIndexes(const nemo::cuda::Mapper& localMapper);

	void logPreStepData();

	void logIncomingSynapses();

	/*! log std::map<rank_t, nidxVec> data */
	void logMap(rankIndexesMap& map, std::string prefix);

	/*! Iterate over current queue in SpikeQueue */
	void logCurrentQueue();

	void logFiringStimulus();

	void logCurrentStimulus();

	void logFiredNeurons(nemo::Simulation::firing_output& firedNeurons);

#ifdef NEMO_MPI_COMMUNICATION_COUNTERS
	std::string reportCommunicationCounters();
#endif

private:
	const Worker * w;
	const char * m_logDir;
	boost::log::sources::severity_logger<severity_level> m_lg;
};

}
}

#endif
