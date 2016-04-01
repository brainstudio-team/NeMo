#include "FiringBuffer.hpp"
#include "exception.hpp"

namespace nemo {


FiringBuffer::FiringBuffer() :
	/* Need a dummy entry, to pop on first call to readFiring */
	m_fired(1),
	m_oldestCycle(~0U)
{
}


void
FiringBuffer::addFiredNeuron(unsigned neuron)
{
	m_fired.back().push_back(neuron);
}




void
FiringBuffer::enqueueCycle()
{
	m_oldestCycle += 1;
	m_fired.push_back(std::vector<unsigned>());
}



FiredList
FiringBuffer::dequeueCycle()
{
	if(m_fired.size() < 2) {
		throw nemo::exception(NEMO_BUFFER_UNDERFLOW, "Firing buffer underflow");
	}
	m_fired.pop_front();
	return FiredList(m_oldestCycle, m_fired.front());
}


}
