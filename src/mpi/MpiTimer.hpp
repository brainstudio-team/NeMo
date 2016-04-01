#ifndef NEMO_MPI_TIMER_HPP
#define NEMO_MPI_TIMER_HPP

#include <vector>
#include <boost/mpi/timer.hpp>
#include <stdio.h>

namespace nemo {
	namespace mpi {

class MpiTimer
{
	public :

		MpiTimer();

		void reset();

		void substep();

		void step();

		//! \todo provide the names here
		void report(unsigned rank) const;

	private :

		std::vector<double> m_bins;

		unsigned m_current;

		boost::mpi::timer m_timer;
};

	}
}

#endif
