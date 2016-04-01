#ifndef NEMO_SIMULATION_HPP
#define NEMO_SIMULATION_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with NeMo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <utility>
#include <cstddef>
#include <nemo/config.h>
#include <nemo/types.h>
#include <nemo/ReadableNetwork.hpp>

namespace nemo {

class Network;
class Configuration;


/*! \class Simulation
 *
 * \brief Simulation of a single network
 *
 * Concrete instances are created using the \a nemo::simulation factory
 * function.
 *
 * Internal errors are signaled by exceptions. Thrown exceptions are all
 * of the type \a nemo::exception which in turn subclass std::exception.
 *
 * \ingroup cpp-api
 */
class NEMO_BASE_DLL_PUBLIC Simulation : public ReadableNetwork
{
	public :

		virtual ~Simulation();

		/*! Vector of neurons which fired during a particular simulation step */
		typedef std::vector<unsigned> firing_output;

		/*! Vector of neurons which should be forced to fire during a particular simulation step */
		typedef std::vector<unsigned> firing_stimulus;

		/*! Current stimulus specified as pairs of neuron index and input current */
		typedef std::vector< std::pair<unsigned, float> > current_stimulus;

		/*! Run simulation for a single cycle (1ms) without external stimulus */
		virtual const firing_output& step() = 0;

		/*! Run simulation for a single cycle (1ms) with firing stimulus
		 *
		 * \param fstim
		 * 		An list of neurons, which will be forced to fire this cycle.
		 * \return
		 * 		List of neurons which fired this cycle. The referenced data is
		 * 		valid until the next call to step.
		 */
		virtual const firing_output& step(const firing_stimulus& fstim) = 0;

		/*! Run simulation for a single cycle (1ms) with current stimulus
		 *
		 * \param istim
		 * 		Optional per-neuron vector specifying externally provided input
		 * 		current for this cycle.
		 * \return
		 * 		List of neurons which fired this cycle. The referenced data is
		 * 		valid until the next call to step.
		 */
		virtual const firing_output& step(const current_stimulus& istim) = 0;

		/*! Run simulation for a single cycle (1ms) with both firing stimulus
		 * and current stimulus
		 *
		 * \param fstim
		 * 		An list of neurons, which will be forced to fire this cycle.
		 * \param istim
		 * 		Optional per-neuron vector specifying externally provided input
		 * 		current for this cycle.
		 * \return
		 * 		List of neurons which fired this cycle. The referenced data is
		 * 		valid until the next call to step.
		 */
		virtual const firing_output& step(
					const firing_stimulus& fstim,
					const current_stimulus& istim) = 0;

#ifdef NEMO_BRIAN_ENABLED
		/* Propagate spikes on given some firing
		 *
		 * This function is intended for integration with Brian.
		 * The neuron state is unaffected by calls to \a propagate
		 *
		 * \param fired pointer (wrapped in size_t) which is either a
		 * 		non-compact array of fired neurons stored on a CUDA device with
		 * 		on uint32_t per neuron or a pointer ot a compact array of fired
		 * 		neurons on the host.
		 * \param nfired if \a fired points to host memory \a nfired is the
		 * 		number of neurons in that array. If \a fired points to CUDA device
		 * 		memory this argument is ignored, as the length of \a fired is
		 * 		the implicitly the length of the index space used by NeMo.
		 * \return pointers to per-neuron accumulated weights, the first one
		 * 		for excitatory, the second for inhbitiory weights. The pointer
		 * 		is either to device or host memory, depending on the type of \a
		 * 		fired.
		 */
		std::pair<size_t, size_t> propagate(size_t fired, int nfired);
#endif


		/*! \name Modifying the network
		 *
		 * Neuron parameters and state variables can be modified during
		 * simulation. However, synapses can not be modified during simulation
		 * in the current version of NeMo
		 */

		/*! Set an existing neuron
		 *
		 * \param idx neuron index
		 * \param nargs number of parameters and state variables
		 * \param args parameters and state variables (in that order) of the neuron
		 *
		 * \pre The parameter and state array must have the dimensions
		 * 		matching the neuron type specified when the neuron was first
		 * 		added.
		 */
		virtual void setNeuron(unsigned idx, unsigned nargs, const float args[]) = 0;

		/*! Change an existing Izhikevich neuron.
		 *
		 * \see nemo::Network::addNeuron for parameters
		 */
		void setNeuron(unsigned idx,
				float a, float b, float c, float d,
				float u, float v, float sigma);

		/*! \copydoc nemo::Network::setNeuronState */
		virtual void setNeuronState(unsigned neuron, unsigned var, float value) = 0;

		/*! \copydoc nemo::Network::setNeuronParameter */
		virtual void setNeuronParameter(unsigned neuron, unsigned parameter, float value) = 0;

		/*! Update synapse weights using the accumulated STDP statistics
		 *
		 * \param reward
		 * 		Multiplier for the accumulated weight change
		 */
		virtual void applyStdp(float reward) = 0;

		/*! \name Queries
		 *
		 * Neuron and synapse state is availble at run-time.
		 *
		 * The synapse state can be read back at run-time by specifiying a list
		 * of synpase ids (see \a addSynapse). The weights may change at
		 * run-time, while the other synapse data is static.
		 *
		 * \{ */

		/*! \copydoc nemo::Network::getNeuronState */
		virtual float getNeuronState(unsigned neuron, unsigned var) const = 0;

		/*! \copydoc nemo::Network::getNeuronParameter */
		virtual float getNeuronParameter(unsigned neuron, unsigned parameter) const = 0;

		/*! \return
		 * 		membrane potential of the specified neuron
		 */
		virtual float getMembranePotential(unsigned neuron) const = 0;

		/*! \copydoc nemo::ReadableNetwork::getSynapsesFrom */
		virtual const std::vector<synapse_id>& getSynapsesFrom(unsigned neuron) = 0;

		/* \} */ // end simulation (queries) section

		/*! \name Simulation (timing)
		 *
		 * The simulation has two internal timers which keep track of the
		 * elapsed \e simulated time and \e wallclock time. Both timers measure
		 * from the first simulation step, or from the last timer reset,
		 * whichever comes last.
		 *
		 * \{ */

		/*! \return number of milliseconds of wall-clock time elapsed since
		 * first simulation step (or last timer reset). */
		virtual unsigned long elapsedWallclock() const = 0;

		/*! \return number of milliseconds of simulated time elapsed since first
		 * simulation step (or last timer reset) */
		virtual unsigned long elapsedSimulation() const = 0;

		/*! Reset both wall-clock and simulation timer */
		virtual void resetTimer() = 0;

		/* \} */ // end simulation (timing) section

	protected :

		Simulation() { };

	private :

		/* Disallow copying of Simulation object */
		Simulation(const Simulation&);
		Simulation& operator=(const Simulation&);

#ifdef NEMO_BRIAN_ENABLED
		virtual std::pair<float*, float*> propagate_raw(uint32_t* fired, int nfired) = 0;
#endif

};

};

#endif
