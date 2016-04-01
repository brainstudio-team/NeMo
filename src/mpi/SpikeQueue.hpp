#ifndef NEMO_MPI_SPIKE_QUEUE
#define NEMO_MPI_SPIKE_QUEUE

#include <vector>
#include <nemo/types.hpp>

namespace nemo {
	namespace mpi {


/* This could potentially be compressed to 32b instead of 64b */
struct arrival
{
	public :

		arrival(nidx_t source, delay_t delay) :
			m_source(source), m_delay(delay) { }

		nidx_t source() const { return m_source; }
		delay_t delay() const { return m_delay; }

	private :

		nidx_t m_source;
		delay_t m_delay;
};



/* Usage:
 * 		queue.enqueue	<possibly repeatedly>
 * 		foo(queue.current_begin(), queue.current_end());
 * 		queue.step();
 */
class SpikeQueue 
{
	public :

		SpikeQueue(delay_t maxDelay);

		unsigned size() const;

		/*! Enqueue an arrival entry for a spike from the given source neuron
		 * with the given conductance delay.
		 *
		 * \param elapsedDelay
		 * 		Length of time which this spike has already been in flight.
		 */
		void enqueue(nidx_t source, delay_t delay, delay_t elapsedDelay=0);

		/* Rotate queue one slot, discarding current cycles' arrivals. Any
		 * existing iterators returned by current_begin/current_end will be
		 * invalidated. */
		void step();

		/* Iterator of arrivals */
		typedef std::vector<arrival>::const_iterator const_iterator;

		/*! \return iterator pointing to the first neuron whose spikes should
		 * be delivered *this* cycle */
		const_iterator current_begin() const;

		/*! \return iterator pointing to the beyond the last neuron whose
		 * spikes should be delivered *this* cycle */
		const_iterator current_end() const;

	private :

		/* Vectors of <source, delay> pairs. Each vector is indexed by the delay */
		std::vector< std::vector<arrival> > m_queue;

		/* Offset into m_queue corresponding to current cycle */
		unsigned m_current;

		unsigned slot(unsigned delay) const;
};
	
}	}

#endif
