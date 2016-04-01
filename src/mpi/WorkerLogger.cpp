#include "WorkerLogger.hpp"

using std::endl;

namespace nemo {
namespace mpi {

WorkerLogger::WorkerLogger(const Worker * worker, const char * logDir) : m_logDir(logDir) {
	w = worker;

	if ( w->m_configuration.loggingEnabled() ) {
		utils::logInit(m_lg, w->m_rank, logDir);
	}
	else {
		boost::shared_ptr<boost::log::core> core = boost::log::core::get();
		core->set_logging_enabled(false);
	}

}

void WorkerLogger::writeStats() {
	std::ostringstream oss;
	oss << m_logDir << "/w" << w->m_rank << "-stats.txt";
	std::ofstream file(oss.str().c_str(), std::fstream::out | std::fstream::app);

	file << "--------------------------------------------------" << endl;
	file << "Worker: " << w->m_rank << endl;
	file << "Backend: " << w->m_configuration.backendDescription() << endl;
	file << "STDP enabled: " << w->m_configuration.stdpEnabled() << endl;
	file << "Partitioning type: " << w->m_mapper.mapperTypeDescr() << endl;
	file << endl;
	file << "Neurons count = " << w->m_ncount << endl;
	file << "Local synapses = " << w->ml_scount << endl;
	file << "Global Synapses (in) = " << w->mgi_scount << endl;
	file << "Global Synapses (out) = " << w->mgo_scount << endl;
	file << "Global spikes (out) = " << w->m_globalSpikesCount << endl;
	file << endl;

#ifdef NEMO_MPI_COMMUNICATION_COUNTERS
	file << reportCommunicationCounters();
#endif
}

void WorkerLogger::logNeurons() {
	for ( network::neuron_iterator n = w->m_netImpl.neuron_begin(0); n != w->m_netImpl.neuron_end(0); ++n )
		logNeuron(n->first);
}

void WorkerLogger::logNeuron(nidx_t n) {
	NeuronType neuronType = w->m_netImpl.neuronType(0); //Izhi
	std::ostringstream oss;
	oss << w->m_rank << " - Neuron " << n << " --- Parameters: ";
	for ( unsigned i = 0; i < neuronType.parameterCount(); i++ )
		oss << "  " << w->m_netImpl.getNeuronParameter(n, i);

	oss << " --- State: ";
	for ( unsigned i = 0; i < neuronType.stateVarCount(); i++ )
		oss << "  " << w->m_netImpl.getNeuronState(n, i);

	BOOST_LOG_SEV(m_lg, debug)<< oss.str() << endl;
}

void WorkerLogger::logNeuronIndexes(const RandomMapper<nidx_t>& localMapper) {
	for ( RandomMapper<nidx_t>::const_iterator i = localMapper.begin(); i != localMapper.end(); i++ ) {
		BOOST_LOG_SEV(m_lg, debug)<< w->m_rank << " - Global = " << i->first << " --> Local = "
		<< localMapper.localIdx(i->first) << endl;
	}
}

void WorkerLogger::logNeuronIndexes(const nemo::cuda::Mapper& localMapper) {
	for ( network::neuron_iterator n = w->m_netImpl.neuron_begin(0); n != w->m_netImpl.neuron_end(0); ++n ) {
		BOOST_LOG_SEV(m_lg, debug)<< w->m_rank << " - Global = " << n->first << " --> Local = "
		<< localMapper.localIdx1D(n->first) << endl;
	}
}

void WorkerLogger::logPreStepData() {
	BOOST_LOG_SEV(m_lg, debug)<< w->m_rank << " - Neurons count = " << w->m_ncount << endl;

	std::ostringstream oss;
	oss << "Neuron indexes: ";
	for (network::neuron_iterator n = w->m_netImpl.neuron_begin(0); n != w->m_netImpl.neuron_end(0); ++n) {
		oss << n->first << " ";
	}
	BOOST_LOG_SEV(m_lg, info) << oss.str() << endl;

	BOOST_LOG_SEV(m_lg, info) << w->m_rank << " - Local synapses = " << w->ml_scount << endl;
	BOOST_LOG_SEV(m_lg, info) << w->m_rank << " - Global Synapses (in) = " << w->mgi_scount << endl;
	BOOST_LOG_SEV(m_lg, info) << w->m_rank << " - Global Synapses (out) = " << w->mgo_scount << endl;

	oss.str("");
	oss.clear();
	for (unsigned n = 0; n < w->m_mapper.neuronCount(); n++) {
		oss << "Neuron " << n << " --> Worker " << w->m_mapper.getRankMap()[n] << endl;
	}
	BOOST_LOG_SEV(m_lg, info) << oss.str() << endl;

	oss.str("");
	oss.clear();
	typedef std::map<nidx_t, std::set<rank_t> >::const_iterator mapIt;
	typedef std::set<rank_t>::const_iterator setIt;
	for (mapIt mapIterator = w->m_fcmOut.begin(); mapIterator != w->m_fcmOut.end(); mapIterator++) {
		oss << w->m_rank << " - Source " << mapIterator->first << " has global synapse with workers: ";
		for (setIt setIterator = mapIterator->second.begin(); setIterator != mapIterator->second.end(); setIterator++) {
			oss << " " << *setIterator;
		}
		oss << endl;
	}
	BOOST_LOG_SEV(m_lg, info) << oss.str() << endl;
}

void WorkerLogger::logIncomingSynapses() {
	std::ostringstream oss;
	oss << w->m_rank << " - Incoming synapses < source, target, delay > :" << endl;

	for ( std::map<syn_key, syn_value>::const_iterator it = w->m_synapses.begin(); it != w->m_synapses.end(); ++it ) {
		oss << it->first.first << ", " << it->second.first << ", " << it->first.second << endl;
	}
	BOOST_LOG_SEV(m_lg, info)<< oss.str() << endl;
}

void WorkerLogger::logFiringStimulus() {
	std::ostringstream oss;
	oss << "STEP " << w->m_sim->elapsedSimulation() << " - Pre-step firing stimulus: ";
	std::copy(w->m_SimStep.fstim.begin(), w->m_SimStep.fstim.end(), std::ostream_iterator<unsigned>(oss, " "));
	BOOST_LOG_SEV(m_lg, debug)<< oss.str() << endl;
}

void WorkerLogger::logCurrentStimulus() {
	std::ostringstream oss;
	oss << "STEP " << w->m_sim->elapsedSimulation() << " - Pre-step current stimulus pairs: ";

	for ( nemo::Simulation::current_stimulus::const_iterator it = w->m_currentStimulus.begin();
			it != w->m_currentStimulus.end(); it++ ) {
		oss << "<" << it->first << "," << it->second << "> ";
	}
	BOOST_LOG_SEV(m_lg, debug)<< oss.str() << endl;
}

void WorkerLogger::logFiredNeurons(nemo::Simulation::firing_output& firedNeurons) {
	std::ostringstream oss;
	oss << "STEP " << w->m_sim->elapsedSimulation() << " - Fired neurons: ";
	std::copy(firedNeurons.begin(), firedNeurons.end(), std::ostream_iterator<unsigned>(oss, " "));
	BOOST_LOG_SEV(m_lg, debug)<< oss.str() << endl;
}

void WorkerLogger::logCurrentQueue() {
	std::ostringstream oss;
	oss << w->m_rank << " - " << "SpikeQueue contains pairs: ";
	SpikeQueue::const_iterator arrival_end = w->m_spikeQueue.current_end();
	for ( SpikeQueue::const_iterator arrival = w->m_spikeQueue.current_begin(); arrival != arrival_end; ++arrival ) {
		oss << "<" << arrival->source() << "," << arrival->delay() << "> ";
	}
	BOOST_LOG_SEV(m_lg, debug)<< oss.str() << endl;
}

void WorkerLogger::logMap(rankIndexesMap& map, std::string prefix) {
	typedef rankIndexesMap::const_iterator mapIt;
	typedef nidxVec::const_iterator vecIt;

	std::ostringstream oss;
	oss << w->m_rank << " - " << prefix << " spikes (neuron indexes): " << endl;
	for ( mapIt mapIterator = map.begin(); mapIterator != map.end(); mapIterator++ ) {
		oss << "rank = " << mapIterator->first << ": nidxVec: ";
		for ( vecIt vecIterator = mapIterator->second.begin(); vecIterator != mapIterator->second.end(); vecIterator++ ) {
			oss << " " << *vecIterator;
		}
		oss << endl;
	}
	BOOST_LOG_SEV(m_lg, debug)<< oss.str() << endl;

}

#ifdef NEMO_MPI_COMMUNICATION_COUNTERS
std::string WorkerLogger::reportCommunicationCounters() {
	std::ostringstream oss;
	oss << "Packets sent: " << w->m_packetsSent << endl;
	oss << "Bytes sent: " << w->m_bytesSent << endl;
	oss << "Average bytes per sent packet: " << (w->m_packetsSent ? w->m_bytesSent / w->m_packetsSent : 0) << endl;
	oss << "Packets received: " << w->m_packetsReceived << endl;
	oss << "Bytes received: " << w->m_bytesReceived << endl;
	oss << "Average bytes per received packet: " << (w->m_packetsReceived ? w->m_bytesReceived / w->m_packetsReceived : 0) << endl;
	return oss.str();
}
#endif

}
}
