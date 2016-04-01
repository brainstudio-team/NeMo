#ifndef NEMO_SIMULATION_BACKEND_HPP
#define NEMO_SIMULATION_BACKEND_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

//! \file SimulationBackend.hpp

#include "Simulation.hpp"
#include "internal_types.h"
#include "FiringBuffer.hpp"
#include "Mapper.hpp"

namespace nemo {

class Network;
class Configuration;

#ifdef NEMO_MPI_ENABLED
		namespace mpi {
			class Worker;
		}
#endif

/*! \class SimulationBackend
 *
 * \brief Abstract low-level interface for simulation classes
 *
 * This is the lower-level interface that concrete backends should implement.
 * This interface is not exposed in the public API, which instead uses the
 * \a nemo::Simulation base class interface.
 */
class NEMO_BASE_DLL_PUBLIC SimulationBackend : public Simulation
{
	public :

		virtual ~SimulationBackend();

		virtual unsigned getFractionalBits() const = 0;

		/*! Set firing stimulus for the next simulation step.
		 *
		 * The behaviour is undefined if this function is called multiple times
		 * between two calls to \a step */
		virtual void setFiringStimulus(const std::vector<unsigned>& nidx) = 0;

		/* CURRENT STIMULUS
		 *
		 * This class provides two interfaces for setting the current stimulus.
		 * First callers can use a low-level interface where each stimulus pair
		 * is set separately and is specified using global neuron indices:
		 *
		 * 	sim->initCurrentStimulus();
		 * 	for each pair
		 * 		sim->addCurrentStimulus(pair);
		 *  sim->setCurrentStimulus
		 *
		 * Alternatively, call the setCurrentStimulus method where neurons are
		 * specified using the internal neuron indexing and the current is
		 * already converted to the correct fixed-point format. This can be
		 * quite a bit faster, especially if there is input current for *every*
		 * neuron, and is used in the MPI backend.
		 *
		 * Only one of these interfaces should be used.
		 */

		/*! Perform any required internal initialisation of input current
		 * buffers, assuming that \a count stimuli will be provided */
		virtual void initCurrentStimulus(size_t count) = 0;

		/*! Add a single neuron/current stimlulus pair */
		virtual void addCurrentStimulus(unsigned, float) = 0;

		/*! Perform any finalisation of input current stimulus buffers. */
		virtual void finalizeCurrentStimulus(size_t count) = 0;

		/*! Set per-neuron input current on the device and set the relevant
		 * member variable containing the device pointer. If there is no input
		 * the device pointer is NULL.
		 *
		 * This function should only be called once per cycle and should not be
		 * called in the same cycle as the floating-point function with the
		 * same name.
		 *
		 * \param current
		 * 		Input current vector using the same fixed-point format as the
		 * 		backend. The current vector indices should have a one-to-one
		 * 		mapping to the valid indices in the simulation. In other words,
		 * 		the input current vector must have been constructed filled
		 * 		taking into account the backend's own mapper.
		 */
		virtual void setCurrentStimulus(const std::vector<float>& current) = 0;

		/*! Perform the first of three parts of the simulation step. This includes
		 *
		 * 1. any setup
		 * 2. compute incoming current for each neuron
		 *
		 * For the CUDA backend this call is asynchronous.
		 */
		virtual void prefire() = 0;

		/*! Perform the 'fire' part of the simulation step, i.e. compute the
		 * next neuron state and determine what neurons fired. This may use
		 * user-provided stimuli provided since the last call to 'fire'.
		 *
		 * For the CUDA backend this also performs host-to-device copy for
		 * input stimulus */
		virtual void fire() = 0;

		/*! Perform the third of three parts of the simulation step. This
		 * includes:
		 *
		 * 1. 'scatter', i.e. distributing spikes from fired neurons
		 * 2. update STDP statistics
		 * 3. copy firing data from simulation
		 *
		 * For the CUDA backend this call is partially asyncrhonous. The first
		 * two parts are run in the background and the function returns once
		 * the device-to-host copy is completed.
		 */
		virtual void postfire() = 0;

		/*! \copydoc nemo::Simulation::step */
		const firing_output& step();

		/*! \copydoc nemo::Simulation::step */
		const firing_output& step(const firing_stimulus&);

		/*! \copydoc nemo::Simulation::step */
		const firing_output& step(const current_stimulus&);

		/*! \copydoc nemo::Simulation::step */
		const firing_output& step(const firing_stimulus&, const current_stimulus&);

		/*! \copydoc nemo::Simulation::applyStdp */
		virtual void applyStdp(float reward) = 0;

		/*! \return tuple oldest buffered cycle's worth of firing data and the
		 * associated cycle number. */
		virtual FiredList readFiring() = 0;

	protected :

		SimulationBackend() { };

	private :

#ifdef NEMO_MPI_ENABLED
friend class nemo::mpi::Worker;
#endif

		/* Disallow copying of SimulationBackend object */
		SimulationBackend(const Simulation&);
		SimulationBackend& operator=(const Simulation&);

};

};

#endif
