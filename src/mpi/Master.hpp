#ifndef NEMO_MPI_MASTER_HPP
#define NEMO_MPI_MASTER_HPP

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
#include <iterator>

#include <nemo/Timer.hpp>

#include "Mapper.hpp"
#include "MpiTypes.hpp"
#include "mpiUtils.hpp"

using namespace boost::log::trivial;

namespace nemo {

class Network;
class Configuration;
class ConfigurationImpl;
namespace network {
class NetworkImpl;
}

namespace mpi {

class Master {
public:

	Master(boost::mpi::environment& env, boost::mpi::communicator& world, const Network&, mapperType mType, const Configuration & conf,
			const char * logDir);

	~Master();

	void step(const std::vector<unsigned>& fstim = std::vector<unsigned>());

	/* Return reference to first buffered cycle's worth of firing. The
	 * reference is invalidated by any further calls to readFiring, or to
	 * step. */
	const std::vector<unsigned>& readFiring();

	/*! \copydoc nemo::Simulation::elapsedWallclock */
	unsigned long elapsedWallclock() const;

	/*! \copydoc nemo::Simulation::elapsedSimulation */
	unsigned long elapsedSimulation() const;

	/*! \copydoc nemo::Simulation::resetTimer */
	void resetTimer();

	const std::vector<unsigned>& getFiringsCounters() const;

	void appendMpiStats(const char* filename) const;

	void writeFiringCounters(const char * filename) const;

protected:
	boost::log::sources::severity_logger<severity_level> m_lg;

private:

	boost::mpi::communicator m_world;

	Mapper m_mapper;

	unsigned workers() const;

	void terminate();

	//! \todo use FiringBuffer here instead
	std::deque<std::vector<unsigned> > m_firing;

	void distributeMapper(const nemo::network::NetworkImpl& net);

	void distributeNeurons(const nemo::network::NetworkImpl& net);

	void distributeSynapses(const network::Generator& net);

	Timer m_timer;

	std::vector<unsigned> m_firingCounters;

	std::vector< unsigned > m_workerNeurons;

	/* Local - global #synapses pair per worker */
	std::vector< std::pair<unsigned, unsigned> > m_workerSynapses;
#ifdef NEMO_MPI_DEBUG_TIMING
	MpiTimer m_mpiTimer;
#endif
};

} // end namespace mpi
} // end namespace nemo

#endif
