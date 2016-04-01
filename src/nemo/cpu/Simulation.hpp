#ifndef NEMO_CPU_SIMULATION_HPP
#define NEMO_CPU_SIMULATION_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>
#include <boost/shared_ptr.hpp>

#include <nemo/config.h>
#include <nemo/internal_types.h>
#include <nemo/internals.hpp>
#include <nemo/ConnectivityMatrix.hpp>
#include <nemo/FiringBuffer.hpp>
#include <nemo/Neurons.hpp>
#include <nemo/RandomMapper.hpp>
#include <nemo/Timer.hpp>

#include "Neurons.hpp"


namespace nemo {

#ifdef NEMO_MPI_ENABLED
		namespace mpi {
			class Worker;
		}
#endif

	namespace cpu {

class NEMO_CPU_DLL_PUBLIC Simulation : public nemo::SimulationBackend
{
	public:

		Simulation(const network::Generator&, const nemo::ConfigurationImpl&);

		unsigned getFractionalBits() const;

		/*! \copydoc nemo::SimulationBackend::setFiringStimulus
		 *
		 * \pre the internal firing stimulus buffer (\a m_fstim) is all false
		 */
		void setFiringStimulus(const std::vector<unsigned>& fstim);

		/*! \copydoc nemo::SimulationBackend::setCurrentStimulus */
		void setCurrentStimulus(const std::vector<float>& current);

		/*! \copydoc nemo::SimulationBackend::initCurrentStimulus */
		void initCurrentStimulus(size_t count);

		/*! \copydoc nemo::SimulationBackend::addCurrentStimulus */
		void addCurrentStimulus(nidx_t neuron, float current);

		/*! \copydoc nemo::SimulationBackend::finalizeCurrentStimulus */
		void finalizeCurrentStimulus(size_t count);

		/*! \copydoc nemo::SimulationBackend::prefire */
		void prefire() { }

		/*! \copydoc nemo::SimulationBackend::fire */
		void fire();

		/*! \copydoc nemo::SimulationBackend::postfire */
		void postfire() { }

#ifdef NEMO_BRIAN_ENABLED
		/*! \copydoc nemo::Simulation::propagate_raw */
		std::pair<float*, float*> propagate_raw(uint32_t*, int nfired);
#endif

		/*! \copydoc nemo::SimulationBackend::readFiring */
		FiredList readFiring();

		/*! \copydoc nemo::SimulationBackend::applyStdp */
		void applyStdp(float reward);

		/*! \copydoc nemo::SimulationBackend::setNeuron */
		void setNeuron(unsigned idx, unsigned nargs, const float args[]);

		/*! \copydoc nemo::Simulation::setNeuronState */
		void setNeuronState(unsigned neuron, unsigned var, float val);

		/*! \copydoc nemo::Simulation::setNeuronParameter */
		void setNeuronParameter(unsigned neuron, unsigned parameter, float val);

		/*! \copydoc nemo::Simulation::getNeuronState */
		float getNeuronState(unsigned neuron, unsigned var) const;

		/*! \copydoc nemo::Simulation::getNeuronParameter */
		float getNeuronParameter(unsigned neuron, unsigned param) const;

		/*! \copydoc nemo::Simulation::getMembranePotential */
		float getMembranePotential(unsigned neuron) const;

		/*! \copydoc nemo::Simulation::getSynapsesFrom */
		const std::vector<synapse_id>& getSynapsesFrom(unsigned neuron);

		/*! \copydoc nemo::Simulation::getSynapseTarget */
		unsigned getSynapseTarget(const synapse_id& synapse) const;

		/*! \copydoc nemo::Simulation::getSynapseDelay */
		unsigned getSynapseDelay(const synapse_id& synapse) const;

		/*! \copydoc nemo::Simulation::getSynapseWeight */
		float getSynapseWeight(const synapse_id& synapse) const;

		/*! \copydoc nemo::Simulation::getSynapsePlastic */
		unsigned char getSynapsePlastic(const synapse_id& synapse) const;

		/*! \copydoc nemo::Simulation::elapsedWallclock */
		unsigned long elapsedWallclock() const;

		/*! \copydoc nemo::Simulation::elapsedSimulation */
		unsigned long elapsedSimulation() const;

		/*! \copydoc nemo::Simulation::resetTimer */
		void resetTimer();

	private:

		#ifdef NEMO_MPI_ENABLED
		friend class nemo::mpi::Worker;
		#endif

		typedef std::vector< boost::shared_ptr<Neurons> > neuron_groups;
		neuron_groups m_neurons;

		RandomMapper<nidx_t> m_mapper;

		typedef std::vector<fix_t> current_vector_t;

		//! \todo can we get rid of this?
		size_t m_neuronCount;

		/* last cycles firing, one entry per neuron */
		std::vector<unsigned> m_fired;

		/* last 64 cycles worth of firing, one entry per neuron */
		std::vector<uint64_t> m_recentFiring;

		/* bit-mask containing delays at which neuron has *any* outoing
		 * synapses */
		std::vector<uint64_t> m_delays;

		boost::shared_ptr<nemo::ConnectivityMatrix> m_cm;

		/* Per-neuron accumulated current from EPSPs */
		std::vector<wfix_t> mfx_currentE;
		std::vector<float> m_currentE;

		/* Per-neuron accumulated current from IPSPs */
		std::vector<wfix_t> mfx_currentI;
		std::vector<float> m_currentI;

		/* Per-neuron user-provided input current */
		std::vector<float> m_currentExt;

		/*! firing stimulus (for a single cycle).
		 *
		 * This is really a boolean vector, but use unsigned to support
		 * parallelisation
		 */
		std::vector<unsigned> m_fstim;

		/*! Deliver spikes due for delivery.
		 *
		 * Updates m_currentE and m_currentI
		 */
		void deliverSpikes();

		void setFiring();

		FiringBuffer m_firingBuffer;

		void deliverSpikesOne(nidx_t source, delay_t delay);

		Timer m_timer;

		nidx_t validLocalIndex(unsigned g_idx) const;

};



NEMO_CPU_DLL_PUBLIC
const char* deviceDescription();


	} // namespace cpu
} // namespace nemo


#endif
