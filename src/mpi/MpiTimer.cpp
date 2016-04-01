#include "MpiTimer.hpp"


namespace nemo {
	namespace mpi {

MpiTimer::MpiTimer() :
	m_current(0)
{
	;
}



void
MpiTimer::reset()
{
	m_timer.restart();
}



void
MpiTimer::substep()
{
	if(m_bins.size() <= m_current) {
		m_bins.resize(m_current+1, 0);
	}
	m_bins.at(m_current) += m_timer.elapsed();
	m_current += 1;
	m_timer.restart();
}



void
MpiTimer::step()
{
	m_current = 0;
}



void
MpiTimer::report(unsigned rank) const
{
	double total = 0.0;
	for(std::vector<double>::const_iterator i = m_bins.begin();
			i != m_bins.end(); ++i) {
		total += *i;
	}

	for(std::vector<double>::const_iterator i = m_bins.begin();
			i != m_bins.end(); ++i) {
		fprintf(stderr, "Worker %u bin %lu: %fs (%f%%)\n", rank, i - m_bins.begin(), *i, 100.0*(*i)/total);
	}
}



	}
}
