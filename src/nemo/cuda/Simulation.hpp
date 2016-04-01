#ifndef NEMO_CUDA_SIMULATION_HPP
#define NEMO_CUDA_SIMULATION_HPP

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

#include <boost/optional.hpp>

#include <nemo/config.h>
#include <nemo/Timer.hpp>
#include <nemo/internal_types.h>
#include <nemo/ConfigurationImpl.hpp>
#include <nemo/SimulationBackend.hpp>

#include "Mapper.hpp"
#include "NVector.hpp"
#include "ConnectivityMatrix.hpp"
#include "DeviceAssertions.hpp"
#include "FiringStimulus.hpp"
#include "FiringBuffer.hpp"
#include "Neurons.hpp"
#include "LocalQueue.hpp"

#include "parameters.cu_h"

namespace nemo {

	namespace network {
		class Generator;
	}

#ifdef NEMO_MPI_ENABLED
		namespace mpi {
			class Worker;
		}
#endif
	namespace cuda {

/*! \namespace nemo::cuda
 *
 * \brief CUDA simulation backend
 *
 * This is documented more fully in the \ref cuda_backend "Cuda backend" section
 */

/*! \page cuda_backend Cuda backend
 *
 * \section cuda_code Code organisation
 *
 * The CUDA simulation backend contains both host-side and device-side code.
 * The host code is spread over a number of C++ classes found inside the
 * \ref nemo::cuda namespace. The kernel functions exist in the global
 * namespace.  The global device functions are called via the wrapper functions
 * in \ref kernel.cu, the most important of which are \ref step and \ref applyStdp.
 *
 * Typically the code relating to some data structure used in the simulation is
 * spread over three or four files. First, a .hpp/.cpp pair contains a class
 * for the host side code which performs allocation, syncing, etc. Second, a
 * .cu file contains device code or shared host/device code for manipulating
 * the data on the device. Third, a .cu_h file may provide prototypes for \c
 * __host__ functions used by the .cpp file but declared in the .cu file.
 *
 * \section cuda_coding_conventions Coding conventions
 *
 * Various prefixes are used to distinguish between different types of data:
 *
 * - The prefixes \c d_ or \c h_ is used to denote device and host data respectively
 * - In device code, the prefixes \c g_, \c s_, and \c c_ denote data in
 *   repectively global memory, shared memory and constant memory.
 * - The prefixes \c i_ and \c f_ denote integer and floating point data.
 * - The prefixes \c f_ and \c r_ denote forward and reverse connectivity data.
 * - The prefixes \c w_ and \c b_ denote word and byte sizes respectivly
 *
 * Prefixes are combined where applicable, such that the prefix \c df denotes
 * device data for forwared connectivity.
 *
 * The prefixes \c i, \c b, and \c n are sometimes used for loop variables.
 * What would have been inner loops if the code was sequential are typically
 * unrolled such that threads in a block deal with different iterations. In
 * such code, if processing vector \c X the loop index is generally named \c iX
 * and is based on a loop base index \c bX and the thread index. \c nX denotes
 * the number of data in \c X. A typical loop over a vector \c X thus looks
 * like:
 *
 * \code
 * for(unsigned bX = 0; bX < nX; bX += THREADS_PER_BLOCK) {
 *     unsigned iX = bX + threadIdx.x;
 *     s_X[iX] = foo(g_X[iX]);
 *     ...
 * }
 * \endcode
 *
 * Groups of functions relating to the same data structure are often given a
 * common prefix. For example
 *
 * - \c bv_ for functions manipulating bit vectors
 * - \c lq_ for functions manipulating the local queue
 * - \c fx_ for fixed point functionality
 *
 * Adherence to these guidelines is patchy at best.
 *
 * \section cuda_neuron_indices Neuron indices and network partitioning
 *
 * Neurons are divided into groups of at most \ref MAX_PARTITION_SIZE neurons.
 * Individual neurons are indexed using a 2D index specifying both a partition
 * index and a neuron index in the range [0:MAX_PARTITION_SIZE).
 *
 * Generally processing is done on a per-partition basis, i.e. a thread block
 * deals primarily with a single partition, and a thread deals with one or more
 * neurons.
 *
 * \section cuda_number_format Number format
 *
 * Synapse weights are stored using a 32-bit fixed-point format. The number of
 * fractional bits can be configured when the simulation is set up by calling
 * \ref fx_setFormat. Other functions for manipulating fixed-point data are
 * found in \ref fixedpoint.cu.
 *
 * Non-weight data is stored as (single-precision) floats.
 *
 * \section cuda_data Data structures
 *
 * Data can be broadly divided into
 *
 * - neuron data (\ref Neurons)
 * - forward connectivity data (\ref nemo::cuda::ConnectivityMatrix, \ref nemo::cuda::Outgoing)
 * - reverse connectivity data (\ref nemo::cuda::runtime::RCM)
 * - runtime queues (\ref nemo::cuda::LocalQueue, \ref nemo::cuda::GlobalQueue)
 *
 * The forward connectivity data and runtime queues are used for \ref
 * cuda_delivery "spike delivery".
 *
 * \section cuda_delivery Spike delivery
 *
 * The spikes resulting from neuron firing have to be delivered to targets
 * which are disparate both spatially (different neuron indices) and temporally
 * (different conduction delays). The spike delivery takes place in three
 * stages: local scatter, global scatter, and gather. Each of these involve
 * several data structures and device functions. An overview of the full
 * process is provided here, with additional information available in th
 * documentation for the relevant classes and device functions.
 *
 * sort spikes temporally and spatially respectively. Each step has a separate
 * queue. This organisation is chosen in order to
 *
 * \subsection cuda_local_delivery Local scatter
 *
 * The local scatter step sorts spikes temporally, but does so for each
 * partition indivdually, hence \e local. Pairs of neuron/delay are stored into
 * a rotating queue with one entry for each possible delay. This requires a
 * mapping from source neurons to the list of all delays for which this neuron
 * has outgoing synapses. This mapping is the \ref
 * nemo::cuda::ConnectivityMatrix::m_delays "delay bits" data structure. The
 * queue structure itself, the \e local queue, is found in \ref
 * nemo::cuda::LocalQueue with the corresponding device functions found in \ref
 * localQueue.cu. The kernel function \ref scatterLocal performs the local
 * scatter step.
 *
 * \subsection cuda_global_delivery Global spike delivery
 *
 * The global scatter step (\ref scatterGlobal) exchanges spikes between different partitions.
 * Since spikes (in the form of neuron/delay pairs) are sorted temporally in
 * the local step, only one cycles' worth of spikes needs to be popped from the
 * local queue and exchanged here.
 *
 * The global queue is in fact a 2D grid of queues with one queue for each
 * source/target partition pair. The queue data structure itself is found in
 * \ref nemo::cuda::GlobalQueue with the corresponding device functions found
 * in \ref globalQueue.cu.
 *
 * The data that is added to the global queue in this step is the address of
 * the synapse warp, i.e. a reference to up to 32 synapses sharing the same
 * source partition/target partition and delay. Each neuron/delay pair found in
 * the local queue may have a number of synapse warps. The mapping from
 * neuron/delay to this synapse warp list is called the 'outgoing' data and is
 * found in \ref nemo::cuda::Outgoing, with the device functions found in \ref
 * outgoing.cu.
 *
 * \subsection cuda_gather Spike gather
 *
 * Finally, the actual synapse data (e.g. weights) are loaded and processed in
 * the \ref gather step. For each partition this reads (and resets) the
 * relevant entry in the global queue. The data thus loaded are lists of
 * synapse warp numbers. These are used to load the synapse data from the
 * forward connectivity matrix (\ref nemo::cuda::ConnectivityMatrix).
 */


/*! \brief Top-level simulation class for CUDA simulation */
class Simulation : public nemo::SimulationBackend
{
	public :

		~Simulation();

		/* CONFIGURATION */

		unsigned getFractionalBits() const;

		/* SIMULATION */

		/*! \copydoc nemo::SimulationBackend::setFiringStimulus */
		void setFiringStimulus(const std::vector<unsigned>& nidx);

		/*! \copydoc nemo::SimulationBackend::setCurrentStimulus */
		void setCurrentStimulus(const std::vector<float>& current);

		/*! \copydoc nemo::SimulationBackend::initCurrentStimulus */
		void initCurrentStimulus(size_t count);

		/*! \copydoc nemo::SimulationBackend::addCurrentStimulus */
		void addCurrentStimulus(nidx_t neuron, float current);

		/*! \copydoc nemo::SimulationBackend::finalizeCurrentStimulus */
		void finalizeCurrentStimulus(size_t count);

		/*! \copydoc nemo::SimulationBackend::prefire */
		void prefire();

		/*! \copydoc nemo::SimulationBackend::fire */
		void fire();

		/*! \copydoc nemo::SimulationBackend::postfire */
		void postfire();

#ifdef NEMO_BRIAN_ENABLED
		/*! \copydoc nemo::Simulation::propagate */
		std::pair<float*, float*> propagate_raw(uint32_t*, int nfired);
#endif

		/*! \copydoc nemo::SimulationBackend::readFiring */
		FiredList readFiring();

		/*! \copydoc nemo::Simulation::getNeuronState */
		float getNeuronState(unsigned neuron, unsigned var) const;

		/*! \copydoc nemo::Simulation::getNeuronState */
		float getNeuronParameter(unsigned neuron, unsigned parameter) const;

		/*! \copydoc nemo::Simulation::getMembranePotential */
		float getMembranePotential(unsigned neuron) const;

		/*! \copydoc nemo::Simulation::applyStdp */
		void applyStdp(float reward);

		/*! \copydoc nemo::Simulation::setNeuron */
		void setNeuron(unsigned idx, unsigned nargs, const float args[]);

		/*! \copydoc nemo::Simulation::setNeuronState */
		void setNeuronState(unsigned neuron, unsigned var, float val);

		/*! \copydoc nemo::Simulation::setNeuronParameter */
		void setNeuronParameter(unsigned neuron, unsigned parameter, float val);

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

		Mapper getMapper() const;

		void finishSimulation();

		/* TIMING */

		/*! \copydoc nemo::Simulation::elapsedWallclock */
		unsigned long elapsedWallclock() const;

		/*! \copydoc nemo::Simulation::elapsedSimulation */
		unsigned long elapsedSimulation() const;

		/*! \copydoc nemo::Simulation::resetTimer */
		void resetTimer();

	private :

#ifdef NEMO_MPI_ENABLED
		friend class nemo::mpi::Worker;
#endif

		/* Use factory method for generating objects */
		Simulation(const network::Generator&, const nemo::ConfigurationImpl&);

		friend SimulationBackend* simulation(const network::Generator& net, const ConfigurationImpl& conf);

		Mapper m_mapper;

		nemo::ConfigurationImpl m_conf;

		//! \todo add this to logging output
		/*! \return
		 * 		number of bytes allocated on the device
		 *
		 * It seems that cudaMalloc*** does not fail properly when running out
		 * of memory, so this value could be useful for diagnostic purposes */
		size_t d_allocated() const;

		typedef std::vector< boost::shared_ptr<Neurons> > neuron_groups;
		neuron_groups m_neurons;

		ConnectivityMatrix m_cm;

		LocalQueue m_lq;

		NVector<uint64_t> m_recentFiring;

		FiringStimulus m_firingStimulus;

		NVector<float> m_currentStimulus; // user-provided
		NVector<float> m_current;         // driven by simulation

		/* The firing buffer keeps data for a certain duration. One bit is
		 * required per neuron (regardless of whether or not it's firing */
		FiringBuffer m_firingBuffer;

		/* The simulation also needs to store sparse firing data between kernel
		 * calls */
		NVector<nidx_dt> m_fired;
		boost::shared_array<unsigned> md_nFired;

		boost::shared_ptr<param_t> md_params;

		/* Initialise the simulation-wide parameters on the device
		 *
		 * All kernels use a single pitch for all 64-, 32-, and 1-bit
		 * per-neuron data This function sets these common pitches and also
		 * checks that all relevant arrays have the same pitch.
		 *
		 * \param pitch1 pitch of 1-bit per-neuron data
		 * \param pitch32 pitch of 32-bit per-neuron data
		 * \param maxDelay maximum delay found in the network
		 *
		 * \return device pointer to parameters. The device memory is handled
		 * 		by this class rather than the caller.
		 */
		param_t* setParameters(size_t pitch1, size_t pitch32, unsigned maxDelay);

		/* Size of each partition, stored on the device in a single array. */
		boost::shared_array<unsigned> md_partitionSize;

		DeviceAssertions m_deviceAssertions;

		boost::optional<StdpFunction> m_stdp;

		void configureStdp();

		Timer m_timer;

		/* Device pointers to simulation stimulus. The stimulus may be set
		 * separately from the step, hence member variables */
		float* md_istim;

		void runKernel(cudaError_t);

		cudaStream_t m_streamCompute;
		cudaStream_t m_streamCopy;

		cudaEvent_t m_eventFireDone;
		cudaEvent_t m_firingStimulusDone;
		cudaEvent_t m_currentStimulusDone;
};

	} // end namespace cuda
} // end namespace nemo

#endif
