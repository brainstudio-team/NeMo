/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Master.hpp"

using std::endl;

namespace nemo {
namespace mpi {

typedef std::vector<std::pair<nidx_t, nemo::Neuron> > nvector;
typedef std::vector<nemo::Synapse> svector;

Master::Master(boost::mpi::environment& env, boost::mpi::communicator& world, const Network& net, mapperType mType,
		const Configuration& conf, const char * logDir) :
		m_world(world), m_mapper(net.neuronCount(), world, net, mType), m_firingCounters(net.neuronCount()), m_workerNeurons(
				world.size()), m_workerSynapses(world.size()) {
	if ( conf.loggingEnabled() ) {
		utils::logInit(m_lg, world.rank(), logDir);
	}
	else {
		boost::shared_ptr<boost::log::core> core = boost::log::core::get();
		core->set_logging_enabled(false);
	}

	/* Need a dummy entry, to pop on first call to readFiring */
	m_firing.push_back(std::vector<unsigned>());

	std::cout << "Master distributes mapper." << endl;
	distributeMapper(*net.m_impl);
	std::cout << "Master distributes neurons." << endl;
	distributeNeurons(*net.m_impl);
	std::cout << "Master distributes synapses." << endl;
	distributeSynapses(*net.m_impl);

	/* The workers now set up the local simulations. This could take some time. */
	m_world.barrier();

	/* We're now ready to run the simulation. The caller does this using class
	 * methods. */
	m_timer.reset();

#ifdef NEMO_MPI_DEBUG_TIMING
	m_mpiTimer.reset();
#endif
}

Master::~Master() {
#ifdef NEMO_MPI_DEBUG_TIMING
	m_mpiTimer.report(0);
#endif
	terminate();
}

unsigned Master::workers() const {
	return m_world.size() - 1;
}

template<class T>
void flushBuffer(int tag, std::vector<T>& input, std::vector<std::vector<T> >& output,
		boost::mpi::communicator& world) {
	broadcast(world, tag, MASTER);
	scatter(world, output, input, MASTER);
	for ( typename std::vector<std::vector<T> >::iterator i = output.begin(); i != output.end(); ++i ) {
		i->clear();
	}
}

void Master::distributeMapper(const nemo::network::NetworkImpl& gen) {
	BOOST_LOG_SEV(m_lg, info)<< "Master broadcasts mapper to workers." << endl;
	boost::mpi::broadcast(m_world, m_mapper, MASTER);
}

void Master::distributeNeurons(const nemo::network::NetworkImpl& gen) {
	BOOST_LOG_SEV(m_lg, info)<< "Master distributes neurons" << endl;
	nvector input; // dummy
	std::vector<nvector> output(m_world.size());

	//TODO handle case where we have multiple collections
	std::ostringstream oss;
	for (network::neuron_iterator n = gen.neuron_begin(0); n != gen.neuron_end(0); ++n) {
		rank_t r = m_mapper.rankOf(n->first);
		oss << " n->first = " << n->first << " --- rank = " << r << endl;
		output.at(r).push_back(*n);
	}

	for(unsigned i=0; i < output.size(); i++) {
		m_workerNeurons[i] = output[i].size();
	}

	BOOST_LOG_SEV(m_lg, info) << oss.str() << endl;

	BOOST_LOG_SEV(m_lg, info) << "Master flushes NEURON_VECTOR" << endl;

	flushBuffer(NEURON_VECTOR, input, output, m_world);

	BOOST_LOG_SEV(m_lg, info) << "Master broadcasts neuron type" << endl;

	std::string neuronTypeName = gen.neuronType(0).name();
	broadcast(m_world, neuronTypeName, MASTER);

	int tag = NEURONS_END;
	broadcast(m_world, tag, MASTER);
}

void Master::distributeSynapses(const network::Generator& net) {
	BOOST_LOG_SEV(m_lg, info)<< "Master distributes synapses" << endl;
	svector input; // dummy
	std::vector<svector> output(m_world.size());

	long unsigned scount = 0;
	for (network::synapse_iterator s = net.synapse_begin(); s != net.synapse_end(); ++s, ++scount) {
		int sourceRank = m_mapper.rankOf(s->source);
		int targetRank = m_mapper.rankOf(s->target());
		BOOST_LOG_SEV(m_lg, info) << "s->source = " << s->source << " --- s->target() = " << s->target()
		<< " --- sourceRank = " << sourceRank << " - targetRank = " << targetRank;
		output.at(sourceRank).push_back(*s);
		if ( sourceRank != targetRank ) {
			output.at(targetRank).push_back(*s);
		}
	}

	BOOST_LOG_SEV(m_lg, info) << "Master - global synapses = " << scount << endl;
	std::ostringstream oss;
	for (rank_t i = 0; i < m_world.size(); i++) {
		oss << "Master - SYNAPSE_VECTOR for worker " << i << ": ";
		for (svector::const_iterator it = output[i].begin(); it != output[i].end(); ++it) {
			oss << "<" << it->source << ", " << it->delay << ">  ";
		}
		oss << endl;
	}
	BOOST_LOG_SEV(m_lg, debug) << oss.str() << endl;

	flushBuffer(SYNAPSE_VECTOR, input, output, m_world);
	int tag = SYNAPSES_END;
	broadcast(m_world, tag, MASTER);

	std::vector<unsigned> localSynapses(m_world.size());
	std::vector<unsigned> globalSynapses(m_world.size());

	/* Gather all fired global #synapses from workers */
	unsigned dummy = 0;
	gather(m_world, dummy, localSynapses, MASTER);

	m_workerSynapses[0].first = scount;
	for(unsigned i=1; i < localSynapses.size(); i++) {
		m_workerSynapses[i].first = localSynapses[i];
	}

	gather(m_world, dummy, globalSynapses, MASTER);
	for(unsigned i=0; i < globalSynapses.size(); i++) {
		m_workerSynapses[i].second = globalSynapses[i];
	}
}

/*! Request force firing of neurons in the appropriate worker. */
void distributeFiringStimulus(const Mapper& mapper, const std::vector<unsigned>& fstim,
		std::vector<SimulationStep>& reqs) {
	for ( std::vector<unsigned>::const_iterator i = fstim.begin(); i != fstim.end(); ++i ) {
		nidx_t neuron = nidx_t(*i);
		assert(unsigned(mapper.rankOf(neuron) - 1) < reqs.size());
		reqs.at(mapper.rankOf(neuron) - 1).forceFiring(neuron);
	}
}

/*! Top level entry point for the mpi simulation */
void Master::step(const std::vector<unsigned>& fstim) {
	m_timer.step();

#ifdef NEMO_MPI_DEBUG_TIMING
	m_mpiTimer.reset();
#endif

	unsigned wcount = workers();

	std::vector<SimulationStep> oreqData(wcount);
	std::vector<boost::mpi::request> oreqs(wcount);

	distributeFiringStimulus(m_mapper, fstim, oreqData);

#ifdef NEMO_MPI_DEBUG_TIMING
	m_mpiTimer.substep();
#endif

	/* Send asynchronous request with force fire neuron list to each worker */
	for ( unsigned r = 0; r < wcount; ++r ) {
		oreqs.at(r) = m_world.isend(r + 1, MASTER_STEP, oreqData.at(r));
	}

#ifdef NEMO_MPI_DEBUG_TIMING
	m_mpiTimer.substep();
#endif

	boost::mpi::wait_all(oreqs.begin(), oreqs.end());

#ifdef NEMO_MPI_DEBUG_TIMING
	m_mpiTimer.substep();
#endif

	m_firing.push_back(std::vector<unsigned>());

	std::vector<unsigned> dummy_fired;
	std::vector<std::vector<unsigned> > fired;

	/* Gather all fired neuron global indexes from workers */
	gather(m_world, dummy_fired, fired, MASTER);

	/* If neurons are allocated to nodes ordered by neuron index, we can get a
	 * sorted list by just concatenating the per-node lists in rank order */
	for ( unsigned r = 0; r < wcount; ++r ) {
		const std::vector<unsigned>& node_fired = fired.at(r + 1);
		std::copy(node_fired.begin(), node_fired.end(), std::back_inserter(m_firing.back()));

		for ( unsigned i = 0; i < node_fired.size(); i++ )
			m_firingCounters[node_fired[i]]++;
		BOOST_LOG_SEV(m_lg, info)<< "Master received " << node_fired.size() << " firings from worker: " << r + 1;
	}

	std::ostringstream oss;
	std::copy(m_firing.back().begin(), m_firing.back().end(), std::ostream_iterator<unsigned>(oss, ", "));
	BOOST_LOG_SEV(m_lg, info)<< oss.str() << endl;

#ifdef NEMO_MPI_DEBUG_TIMING
	m_mpiTimer.substep();
	m_mpiTimer.step();
#endif
}

/* Send simulation step with terminate set to false */
void Master::terminate() {
	unsigned wcount = workers();
	SimulationStep data(true, std::vector<unsigned>());
	std::vector<boost::mpi::request> reqs(wcount);
	for ( unsigned r = 0; r < wcount; ++r ) {
		reqs[r] = m_world.isend(r + 1, MASTER_STEP, data);
	}
	boost::mpi::wait_all(reqs.begin(), reqs.end());

	BOOST_LOG_SEV(m_lg, debug)<< "Simulated " << elapsedSimulation() << " ms wall clock time: " << elapsedWallclock() << "ms" << endl;
}

/*! User can read the list of neurons that fired after the execution of a step */
const std::vector<unsigned>&
Master::readFiring() {
	//! \todo deal with underflow here
	m_firing.pop_front();
	return m_firing.front();
}

unsigned long Master::elapsedWallclock() const {
	return m_timer.elapsedWallclock();
}

unsigned long Master::elapsedSimulation() const {
	return m_timer.elapsedSimulation();
}

void Master::resetTimer() {
	m_timer.reset();
}

const std::vector<unsigned>& Master::getFiringsCounters() const {
	return m_firingCounters;
}

void Master::appendMpiStats(const char* filename) const {

	std::ofstream file(filename, std::fstream::out | std::fstream::app);

	file << "MPI Stats:" << std::endl;
	file << "  Nodes: " << m_world.size() << std::endl;
	file << "  Partitioning: " << m_mapper.mapperTypeDescr() << std::endl;

	std::vector<rank_t> rankMap = m_mapper.getRankMap();
	std::vector<unsigned> counters(m_world.size());

	for(unsigned i=1; i < rankMap.size(); i++)
		counters[rankMap[i]] ++;

	file << "  Partition sizes: " << std::endl;
	for(unsigned i=1; i < counters.size(); i++) {
		file << "    Worker " << i << ": " << counters[i] << std::endl;
	}

	long unsigned sum = 0;
	for ( unsigned i = 0; i < m_firingCounters.size(); i++ ) {
		sum += m_firingCounters[i];
	}
	file << "  Total number of firings: " << sum << std::endl;

	file << std::endl;

	file << "  Local/total neurons ratio: " << std::endl;
	for ( unsigned i = 1; i < m_workerNeurons.size(); i++ ) {
		float ratio = ((float) m_workerNeurons[i] / (float) m_mapper.neuronCount());
		file << "    Worker " << i << " ratio: " << ratio << std::endl;
	}
	file << std::endl;

	file << "  Local - Global synapses per worker: " << std::endl;
	for ( unsigned i = 1; i < m_workerSynapses.size(); i++ ) {
		file << "    Worker " << i << ": " << m_workerSynapses[i].first << " - " << m_workerSynapses[i].second << std::endl;
	}
	file << std::endl;

	file << "  Global/local synapses ratio: " << std::endl;
	for ( unsigned i = 1; i < m_workerSynapses.size(); i++ ) {
		float ratio = ((float) m_workerSynapses[i].second / (float) m_workerSynapses[i].first);
		file << "    Worker " << i << ": " << ratio << std::endl;
	}
	file << std::endl;
}

void Master::writeFiringCounters(const char * filename) const {
	std::ofstream file(filename);
	std::ostringstream oss;

	for ( unsigned i = 0; i < m_firingCounters.size(); i++ ) {
		oss << "Neuron " << i << ": " << m_firingCounters[i] << std::endl;
	}
	file << oss.str();
}

} // end namespace mpi
} // end namespace nemo
