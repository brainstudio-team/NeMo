#ifndef TIMER_HPP
#define TIMER_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/numeric/conversion/cast.hpp>

namespace nemo {

/*! \brief Simple timer measuring both simulated and wall-clock time.
 *
 * The main simulation code keeps track of the current cycle as well, but this
 * timer can be reset.
 */
class Timer 
{
	public:

		Timer() { reset(); }

		/*! Update internal counters. Should be called for every simulation
		 * step. */
		void step() { m_simCycles++ ; }

		/*! \return elapsed wall-clock time in milliseconds */
		unsigned long elapsedWallclock() const;

		/*! \return elapsed simulation time in milliseconds */
		unsigned long elapsedSimulation() const;

		/*! Reset internal counters. */
		void reset();

	private:

		boost::posix_time::ptime m_start;

		unsigned long m_simCycles;
};



inline
unsigned long
Timer::elapsedWallclock() const
{
	using namespace boost::posix_time;

	time_duration elapsed = ptime(microsec_clock::local_time()) - m_start;
	return boost::numeric_cast<unsigned long, time_duration::tick_type>(elapsed.total_milliseconds());
}



inline
unsigned long
Timer::elapsedSimulation() const
{
	return m_simCycles;
}



inline
void
Timer::reset()
{
	using namespace boost::posix_time;
	m_start = ptime(microsec_clock::local_time());
	m_simCycles = 0;
}

}

#endif
