/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Worker.hpp"

using std::endl;

namespace nemo {
namespace mpi {

void runWorker(boost::mpi::environment& env, boost::mpi::communicator& world, const Configuration& conf,
		const char * logDir) {

	try {
		Worker worker(world, env, conf, logDir);
		worker.runSimulation();
	}
	catch (nemo::exception& e) {
		std::cerr << world.rank() << ":" << e.what() << endl;
		env.abort(e.errorNumber());
	}
	catch (std::exception& e) {
		std::cerr << world.rank() << ": " << e.what() << endl;
		env.abort(-1);
	}
}

Worker::Worker(boost::mpi::communicator& world, boost::mpi::environment& env, const Configuration& conf,
		const char * logDir) :
		m_world(world), m_rank(world.rank()), m_configuration(conf), m_spikeQueue(64),
#ifdef NEMO_MPI_COMMUNICATION_COUNTERS
				m_packetsSent(0),
				m_packetsReceived(0),
				m_bytesSent(0),
				m_bytesReceived(0),
#endif
				ml_scount(0), mgi_scount(0), mgo_scount(0), m_ncount(0), m_globalSpikesCount(0), m_wlogger(new WorkerLogger(this, logDir)) {

	if ( m_configuration.loggingEnabled() ) {
		BOOST_LOG_SEV(m_wlogger->m_lg, debug)<< m_rank << " - Constructing worker "
		<< world.rank() << " on " << env.processor_name() << endl;
		BOOST_LOG_SEV(m_wlogger->m_lg, debug) << m_rank << " - Using "
		<< m_configuration.backendDescription() << endl;
		BOOST_LOG_SEV(m_wlogger->m_lg, debug) << m_rank << " - Using "
		<< m_netImpl.maxDelay();
	}

	loadMapper();
	loadNeurons();
	loadSynapses();

	if ( m_configuration.loggingEnabled() )
	m_wlogger->logPreStepData();
}

Worker::~Worker() {
	m_wlogger->writeStats();
#ifdef NEMO_MPI_DEBUG_TIMING
	m_timer.report(m_rank);
#endif
}

void Worker::loadMapper() {
	BOOST_LOG_SEV(m_wlogger->m_lg, debug)<< m_rank << " - " << "Load mapper." << endl;
	broadcast(m_world, m_mapper, MASTER);
	m_mapper.setRank(m_rank);
}

void Worker::loadNeurons() {
	BOOST_LOG_SEV(m_wlogger->m_lg, debug)<< m_rank << " - " << "Load neurons." << endl;
	typedef std::vector<network::Generator::neuron> genNeuronVec;
	genNeuronVec neurons;
	while (true) {
		int tag;
		broadcast(m_world, tag, MASTER);

		BOOST_LOG_SEV(m_wlogger->m_lg, debug) << m_rank << " - " << "broadcast: received tag " << tag << endl;
		if (tag == NEURON_VECTOR) {
			scatter(m_world, neurons, MASTER);
			BOOST_LOG_SEV(m_wlogger->m_lg, debug) << m_rank << " - "
			<< "scatter: neurons size " << neurons.size() << endl;

			std::string neuronTypeName;
			broadcast(m_world, neuronTypeName, MASTER);
			NeuronType neuronType(neuronTypeName);

			BOOST_LOG_SEV(m_wlogger->m_lg, debug) << m_rank << " - "
			<< "Received neuronTypeName " << neuronTypeName << endl;

			unsigned typeId = m_netImpl.addNeuronType(neuronTypeName);

			BOOST_LOG_SEV(m_wlogger->m_lg, debug) << m_rank << " - " << "type ID is "
			<< typeId << endl;

			for (genNeuronVec::const_iterator n = neurons.begin(); n != neurons.end();
					++n) {
				unsigned parCount = neuronType.parameterCount();
				unsigned stateCount = neuronType.stateVarCount();
				float args[parCount + stateCount];
				std::copy((n->second).getParameters(),
						(n->second).getParameters() + parCount, args);
				std::copy((n->second).getState(), (n->second).getState() + stateCount,
						args + parCount);

				m_netImpl.addNeuronMpi(typeId, n->first, parCount + stateCount, args);

				if ( m_configuration.loggingEnabled() )
				m_wlogger->logNeuron(n->first);
				m_ncount++;
			}
		}
		else if (tag == NEURONS_END) {
			break;
		}
		else {
			throw nemo::exception(NEMO_MPI_ERROR,
					"Unknown tag received during neuron scatter");
		}
	}
}

void Worker::loadSynapses() {
	BOOST_LOG_SEV(m_wlogger->m_lg, debug)<< m_rank << " - " << "Load synapses." << endl;

	while (true) {
		int tag;
		broadcast(m_world, tag, MASTER);

		if (tag == SYNAPSE_VECTOR) {
			std::vector<Synapse> ss;
			scatter(m_world, ss, MASTER);

			for (std::vector<Synapse>::const_iterator s = ss.begin(); s != ss.end();
					++s) {
				addSynapse(*s);
			}
		}
		else if (tag == SYNAPSES_END) {
			BOOST_LOG_SEV(m_wlogger->m_lg, debug) << m_rank << " - "
			<< "Received tag SYNAPSES_END." << endl;
			break;
		}
		else {
			throw nemo::exception(
					NEMO_MPI_ERROR, "Unknown tag received during synapse scatter");
		}
	}

	/* Send total number of local synapses to master */
	gather(m_world, ml_scount, MASTER);

	/* Send total number of global synapses to master */
	gather(m_world, mgo_scount, MASTER);
}

void Worker::addSynapse(const Synapse& s) {
	const int sourceRank = m_mapper.rankOf(s.source);
	const int targetRank = m_mapper.rankOf(s.target());

	if ( sourceRank == targetRank ) {
		/* Most neurons should be purely local neurons */
		assert(sourceRank == m_rank); // see how master performs sending
		m_netImpl.addSynapse(s.source, s.target(), s.delay, s.weight(), s.plastic());
		ml_scount++;
	}
	else if ( sourceRank == m_rank ) {
		/* Source neuron is found on this node, but target is on some other node */
		m_fcmOut[s.source].insert(targetRank);
		mg_targetNodes.insert(targetRank);
		mgo_scount++;
	}
	else if ( targetRank == m_rank ) {

		/* Source neuron is found on some other node, but target neuron is
		 * found here. Incoming spikes are handled by the worker, before being
		 * handed over to the workers underlying simulation */
		mgi_scount++;
		mg_sourceNodes.insert(sourceRank);

		syn_key cur_key = std::make_pair(s.source, s.delay);
		syn_value cur_val = std::make_pair(s.target(), s.weight());
		m_synapses[cur_key] = cur_val;
	}

	if ( m_configuration.loggingEnabled() )
		m_wlogger->logIncomingSynapses();
}

bool Worker::isCpuSimulation() {
	return (m_configuration.backend() == NEMO_BACKEND_CPU);
}

bool Worker::isCudaSimulation() {
	return (m_configuration.backend() == NEMO_BACKEND_CUDA);
}

/* This method initialises and executes simulation steps in a loop until master node requests termination. */
void Worker::runSimulation() {

	BOOST_LOG_SEV(m_wlogger->m_lg, debug)<< m_rank << " - " << "Init simulation."
	<< endl;

	std::cout << m_rank << " - " << "Starts simulation." << endl;

	/* Initialization of map with empty vectors */
	for (std::set<rank_t>::const_iterator i = mg_sourceNodes.begin();
			i != mg_sourceNodes.end(); ++i) {
		m_inBufs[*i] = nidxVec();
	}

	/* Initialization of map with empty vectors */
	for (std::set<rank_t>::const_iterator i = mg_targetNodes.begin();
			i != mg_targetNodes.end(); ++i) {
		m_outBufs[*i] = nidxVec();
	}

	/* Everyone should have set up the local simulation now */
	m_world.barrier();

	/* Scatter empty firing packages to start with */
	initSendReqs(m_outReqs);

	ConfigurationImpl& conf = *m_configuration.m_impl;

	/* Delegate to specific simulation */
	if (isCpuSimulation()) {
		BOOST_LOG_SEV(m_wlogger->m_lg, debug) << m_rank << " - "
		<< "Starting CPU simulation." << endl;

		m_sim = new nemo::cpu::Simulation(m_netImpl, *m_configuration.m_impl);

		const RandomMapper<nidx_t>& localMapper =
		dynamic_cast<nemo::cpu::Simulation*>(m_sim)->m_mapper;

		if ( m_configuration.loggingEnabled() )
		m_wlogger->logNeuronIndexes(localMapper);
	}
	else if (isCudaSimulation()) {
		m_sim = new nemo::cuda::Simulation(m_netImpl, *m_configuration.m_impl);

		BOOST_LOG_SEV(m_wlogger->m_lg, debug) << m_rank << " - "
		<< "Starting CUDA simulation." << endl;

		const nemo::cuda::Mapper localMapper =
		dynamic_cast<nemo::cuda::Simulation*>(m_sim)->getMapper();
		nemo::cuda::ConnectivityMatrix * g_fcmIn_cuda = new nemo::cuda::ConnectivityMatrix(m_netImpl, conf, localMapper);

		if ( m_configuration.loggingEnabled() )
		m_wlogger->logNeuronIndexes(localMapper);
	}
	else {
		throw nemo::exception(NEMO_INVALID_INPUT, "Configuration backend type is wrong.");
	}

	while (true) {
		BOOST_LOG_SEV(m_wlogger->m_lg, debug) << m_rank << " - "
		<< "Elapsed simulation (STEP): " << m_sim->elapsedSimulation() << endl;

		m_MasterReq = m_world.irecv(MASTER, MASTER_STEP, m_SimStep);

		initRecvReqs();
		waitAllRequests();
		waitForPrevFirings();

		enqueAllIncoming(m_inBufs);
		gatherCurrent();

		m_MasterReq.wait();
		if (m_SimStep.terminate) {
			break;
		}

		if ( m_configuration.loggingEnabled() ) {
			m_wlogger->logFiringStimulus();
			m_wlogger->logCurrentStimulus();
		}

		nemo::Simulation::firing_output firedNeurons = m_sim->step(m_SimStep.fstim,
				m_currentStimulus);

		if ( m_configuration.loggingEnabled() )
		m_wlogger->logFiredNeurons(firedNeurons);

		bufferFiringData(firedNeurons);
		initSendReqs(m_outReqs);
		gather(m_world, firedNeurons, MASTER); //MPI gather
		m_spikeQueue.step();

//		if (conf.stdpEnabled() && m_sim->elapsedSimulation() % conf.stdpPeriod() == 0) {
//			m_sim->applyStdp(conf.stdpReward());
//		}

#ifdef NEMO_MPI_DEBUG_TIMING
		m_timer.step();
#endif
	}

}

/* Initialize asynchronous MPI receive requests for each connected worker */
void Worker::initRecvReqs() {
	BOOST_LOG_SEV(m_wlogger->m_lg, debug)<< m_rank << " - " << "Init global gather";
	assert(mg_sourceNodes.size() == m_inBufs.size());
	unsigned sid = 0;
	typedef std::set<rank_t>::const_iterator it;
	for (it source = mg_sourceNodes.begin(); source != mg_sourceNodes.end();
			++source, ++sid) {
		assert(m_inBufs.find(*source) != m_inBufs.end());

		nidxVec& incoming = m_inBufs[*source];
		incoming.clear();
		m_inReqs.push_back(m_world.irecv(*source, WORKER_STEP, incoming));
	}
}

/* Wait for all incoming firings sent during the previous cycle. Add these
 * firings to the queue */
void Worker::waitForPrevFirings() {
	BOOST_LOG_SEV(m_wlogger->m_lg, debug)<< m_rank << " - " << "Wait global gather";

	typedef std::pair<boost::mpi::status, mpiReqVec::iterator> resultType;
	unsigned nreqs = m_inReqs.size();
	for (unsigned r = 0; r < nreqs; ++r) {
		resultType result = wait_any(m_inReqs.begin(), m_inReqs.end());
		m_inReqs.erase(result.second);
	}

	if ( m_configuration.loggingEnabled() )
	m_wlogger->logMap(m_inBufs, "Incoming");

#ifdef NEMO_MPI_COMMUNICATION_COUNTERS
	m_packetsReceived += m_inReqs.size();
#endif
}

void Worker::enqueAllIncoming(const rankIndexesMap& ibufs) {
	//for each worker
	for ( rankIndexesMap::const_iterator i = ibufs.begin(); i != ibufs.end(); ++i ) {
		const nidxVec& firedNeurons = i->second;

		if ( firedNeurons.size() == 0 )
			continue;

		rank_t source = i->first;
		BOOST_LOG_SEV(m_wlogger->m_lg, debug)<< m_rank << " - "
		<< "Enqueue all incoming";
		BOOST_LOG_SEV(m_wlogger->m_lg, debug)<< m_rank << " - " << "Received "
		<< firedNeurons.size() << " firings from source = " << source;

#ifdef NEMO_MPI_COMMUNICATION_COUNTERS
		m_bytesReceived += sizeof(unsigned) * firedNeurons.size();
#endif
		enqueueIncoming(firedNeurons);	//for each fired neuron
	}

	if ( m_configuration.loggingEnabled() )
		m_wlogger->logCurrentQueue();
}

/* Incoming spike/delay pairs to spike queue */
void Worker::enqueueIncoming(const nidxVec& firedNeurons) {
	for ( nidxVec::const_iterator i_source = firedNeurons.begin(); i_source != firedNeurons.end(); ++i_source ) {
		nidx_t source = *i_source;

		for ( std::map<syn_key, syn_value>::const_iterator cur_syn = m_synapses.begin(); cur_syn != m_synapses.end();
				++cur_syn ) {
			if ( source == cur_syn->first.first ) {
				m_spikeQueue.enqueue(source, cur_syn->first.second, 1); // -1 since spike has already been in flight for a cycle
			}
		}
	}
}

void Worker::gatherCurrent() {
	BOOST_LOG_SEV(m_wlogger->m_lg, debug)<< m_rank << " - " << "Gather current" << endl;

	m_currentStimulus.clear();

	SpikeQueue::const_iterator arrival_end = m_spikeQueue.current_end();
	for (SpikeQueue::const_iterator arrival = m_spikeQueue.current_begin();
			arrival != arrival_end; ++arrival) {
		syn_key cur_key = std::make_pair(arrival->source(), arrival->delay());
		m_currentStimulus.push_back(m_synapses[cur_key]);
	}
}

/* Sort outgoing firing data into per-node buffers
 *
 * \param fired
 * 		Firing generated this cycle in the local simulation
 * \param obuf
 * 		Per-rank buffer of firing.
 */
void Worker::bufferFiringData(const nidxVec& fired) {
	BOOST_LOG_SEV(m_wlogger->m_lg, debug)<< m_rank << " - " << "Buffer scatter data";
	for (rankIndexesMap::iterator i = m_outBufs.begin(); i != m_outBufs.end(); ++i) {
		i->second.clear();
	}

	/* Each local firing may be sent to zero or more peers */
	for (std::vector<unsigned>::const_iterator source = fired.begin();
			source != fired.end(); ++source) {
		const std::set<rank_t>& targets = m_fcmOut[*source];
		for (std::set<rank_t>::const_iterator target = targets.begin();
				target != targets.end(); ++target) {
			rank_t targetRank = *target;
			assert(mg_targetNodes.count(targetRank) == 1);
			m_outBufs[targetRank].push_back(*source);
		}
	}

	if ( m_configuration.loggingEnabled() )
	m_wlogger->logMap(m_outBufs, "Outgoing");
}

/* Initialise asynchronous send of firing to all neighbours.
 *
 * \param oreqs
 * 		List of requests which will be populated by this function. Any existing
 * 		contents will be cleared.
 * \param obuf
 * 		Per-rank buffer of firing.
 */
void Worker::initSendReqs(mpiReqVec& oreqs) {
	BOOST_LOG_SEV(m_wlogger->m_lg, debug)<< m_rank << " - "
	<< "initSendReqs: Sending firing to " << mg_targetNodes.size()
	<< " peers." << endl;

#ifdef NEMO_MPI_COMMUNICATION_COUNTERS
	m_packetsSent += mg_targetNodes.size();
#endif

	oreqs.clear();

	for (std::set<rank_t>::const_iterator target = mg_targetNodes.begin();
			target != mg_targetNodes.end(); ++target) {
		rank_t targetRank = *target;
		oreqs.push_back(m_world.isend(targetRank, WORKER_STEP, m_outBufs[targetRank]));

		m_globalSpikesCount += m_outBufs[targetRank].size();
#ifdef NEMO_MPI_COMMUNICATION_COUNTERS
		m_bytesSent += sizeof(unsigned) * m_outBufs[targetRank].size();
#endif

	}
}

void Worker::waitAllRequests() {
	BOOST_LOG_SEV(m_wlogger->m_lg, debug)<<m_rank << " - " << "Wait global scatter";
	boost::mpi::wait_all(m_outReqs.begin(), m_outReqs.end());
}

}
// end namespace mpi
}// end namespace nemo
