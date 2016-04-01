#ifndef NEMO_FIRING_BUFFER_HPP
#define NEMO_FIRING_BUFFER_HPP

#include <deque>
#include <vector>

#include <nemo/config.h>
#include <nemo/types.h>

namespace nemo {


/* One cycle's worth of fired neurons. Note that we store a reference to the
 * vector of neurons, so the user needs to be aware of the lifetime of this and
 * copy the contents if appropriate. */
struct NEMO_BASE_DLL_PUBLIC FiredList
{
	cycle_t cycle;
	std::vector<unsigned>& neurons;

	FiredList(cycle_t cycle, std::vector<unsigned>& neurons) :
		cycle(cycle), neurons(neurons) { }
};


/*! \brief Firing buffer containing a FIFO of firing data with one entry for each cycle. */
class NEMO_BASE_DLL_PUBLIC FiringBuffer
{
	public :

		FiringBuffer();

		/*! Add a new cycle's firing vector (empty) at the end of the FIFO.
		 * The caller can fill this in by calling addFiredNeuron */
		void enqueueCycle();

		void addFiredNeuron(unsigned neuron);

		/*! Discard the current oldest cycle's data and return reference to the
		 * new oldest cycle's data. The data referenced in the returned list of
		 * firings is valid until the next call to \a read or \a dequeue. */
		FiredList dequeueCycle();

	private :

		std::deque< std::vector<unsigned> > m_fired;

		cycle_t m_oldestCycle;
};

}

#endif
