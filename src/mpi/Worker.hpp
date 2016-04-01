#ifndef NEMO_MPI_WORKER_HPP
#define NEMO_MPI_WORKER_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */
#include <vector>
#include <deque>
#include <set>
#include <map>
#include <algorithm>
#include <boost/shared_ptr.hpp>

#include <nemo/internals.hpp>
#include <nemo/ConnectivityMatrix.hpp>
#include "nemo/RandomMapper.hpp"
#include "nemo/cpu/Simulation.hpp"
#include "nemo/cuda/Simulation.hpp"

#include "MpiTypes.hpp"
#include "Mapper.hpp"
#include "SpikeQueue.hpp"
#include "mpiUtils.hpp"
#include "WorkerLogger.hpp"

using namespace boost::log::trivial;

namespace nemo {

namespace cuda {
class ConnectivityMatrix;
}

namespace network {
class NetworkImpl;
}
//! \todo use consistent interface here
class ConfigurationImpl;
class Configuration;
class ConnectivityMatrix;

namespace mpi {

class WorkerLogger;
class Mapper;
class SpikeQueue;

void
runWorker(boost::mpi::environment& env, boost::mpi::communicator& world, const Configuration& conf, const char * logDir);

/*
 * prefixes:
 * 	l/g distinguishes local/global
 * 	i/o distinguishes input/output (for global)
 */
class Worker {
	friend class WorkerLogger;

public:

	Worker(boost::mpi::communicator& world, boost::mpi::environment& env, const Configuration& conf, const char * logDir);
	~Worker();

	void runSimulation();

	//! \todo move this type to nemo::FiringBuffer instead perhaps typedefed as Fired::neuron_list
	typedef std::vector<nidx_t> nidxVec;
	typedef std::vector<boost::mpi::request> mpiReqVec;
	typedef std::map<rank_t, nidxVec> rankIndexesMap;

private:
	/* While most synapses are likely to be local to a single simulation
	 * (i.e. both the source and target neurons are found on the same
	 * node), there will also be a large number of /global/ synapses, which
	 * cross node boundaries. Most of the associated synapse data is stored
	 * at the node containing the target neuron. On the node with the source,
	 * we only need to store a mapping from the neuron (in local indices) to the target nodes (rank id). */
	std::map<nidx_t, std::set<rank_t> > m_fcmOut;

	/* Worker's network */
	network::NetworkImpl m_netImpl;

	/* Initialized before the first simulation step */
	nemo::SimulationBackend * m_sim;

	boost::mpi::communicator m_world;

	/* MPI rank of the worker */
	rank_t m_rank;

	Configuration m_configuration;

	/* Synapses with source neuron located on some other node, but target neuron is local. */
	std::map<syn_key, syn_value> m_synapses;

	/*! nemo::mpi::mapper */
	Mapper m_mapper;

	nemo::Simulation::current_stimulus m_currentStimulus;

	/*! Contains one queue with neuron indexes per delay */
	SpikeQueue m_spikeQueue;

	/* Incoming master request */
	boost::mpi::request m_MasterReq;
	SimulationStep m_SimStep;

	/* Incoming peer requests */
	rankIndexesMap m_inBufs;

	mpiReqVec m_inReqs;

	/*! Outgoing peer requests. Not all peers are necessarily potential targets
	 * for local neurons, so the output buffer could in principle be smaller.
	 * Allocated it with potentially unused entries so that insertion (which is
	 * frequent) is faster */
	rankIndexesMap m_outBufs;
	mpiReqVec m_outReqs;

	/*! All the peers to which this worker should send firing data every
	 * simulation cycle. */
	std::set<rank_t> mg_targetNodes;

	/*! All the peers from which this worker should receive firing data
	 * every simulation cycle */
	std::set<rank_t> mg_sourceNodes;

#ifdef NEMO_MPI_COMMUNICATION_COUNTERS
	uint64_t m_packetsSent;
	uint64_t m_packetsReceived;
	uint64_t m_bytesSent;
	uint64_t m_bytesReceived;

	void reportCommunicationCounters() const;
#endif

	unsigned ml_scount;
	unsigned mgi_scount;
	unsigned mgo_scount;
	unsigned m_ncount;
	long unsigned m_globalSpikesCount;
	WorkerLogger * m_wlogger;

#ifdef NEMO_MPI_DEBUG_TIMING
	/* For basic profiling, time the different stages of the main step loop.
	 * Note that the MPI timers we use here are wallclock-timers, and are thus
	 * sensitive to OS effects */
	MpiTimer m_timer;
#endif

	void loadMapper();

	/* Gets a subset of pairs of <neuron indexes, Neuron object> */
	void loadNeurons();

	/* Loads a subset of pairs of <neuron indexes, Neuron object>. Sent from master process. */
	void loadSynapses();

	/* Loads a set of local and global synapses. Sent from master process. */
	void addSynapse(const Synapse& s);

	void initRecvReqs();

	void bufferFiringData(const nidxVec& fired);

	void initSendReqs(mpiReqVec& oreqs);

	void waitAllRequests();

	void waitForPrevFirings();

	void enqueAllIncoming(const rankIndexesMap& bufs);

	void enqueueIncoming(const Worker::nidxVec& fired);

	void gatherCurrent();

	/* Utilities */
	bool isCpuSimulation();

	bool isCudaSimulation();
};

}
}

#endif
