#include "SpikeQueue.hpp"

#include <cassert>
#include <stdexcept>

namespace nemo {
	namespace mpi {


SpikeQueue::SpikeQueue(delay_t maxDelay) :
	m_queue(maxDelay+1),
	m_current(0)
{
	;
}

unsigned SpikeQueue::size() const {
	return m_queue.size();
}

unsigned
SpikeQueue::slot(unsigned delay) const
{
	unsigned sz = m_queue.size();
	//! \todo throw here instead
	//assert(delay < sz);
	unsigned entry = m_current + delay;
	return entry >= sz ? entry - sz : entry;
}


void
SpikeQueue::enqueue(nidx_t source, delay_t delay, delay_t elapsed)
{
	unsigned index = slot(delay-elapsed);
	if(index < 0 || index >= m_queue.size())
		throw std::runtime_error("SpikeQueue index out of bounds.");

	m_queue.at(index).push_back(arrival(source, delay));
}


void
SpikeQueue::step()
{
	m_queue.at(m_current).clear();
	m_current = slot(1);
}


SpikeQueue::const_iterator
SpikeQueue::current_begin() const
{
	return m_queue.at(m_current).begin();
}


SpikeQueue::const_iterator
SpikeQueue::current_end() const
{
	return m_queue.at(m_current).end();
}

}	}
