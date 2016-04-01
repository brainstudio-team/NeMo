#ifndef NEMO_CUDA_NEURONS_HPP
#define NEMO_CUDA_NEURONS_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <map>

#include <boost/utility.hpp>

#include <nemo/cuda/plugins/neuron_model.h>
#include <nemo/NeuronType.hpp>
#include <nemo/Plugin.hpp>

#include "Mapper.hpp"
#include "NVector.hpp"
#include "Bitvector.hpp"
#include "kernel.cu_h"
#include "parameters.cu_h"
#include "types.h"

namespace nemo {

	namespace network {
		class Generator;
	}

	namespace cuda {

	class Mapper;

/*! Per-neuron device data
 *
 * Per-neuron data is split into parameters and state variables. The former are
 * fixed at run-time while the latter is modififed by the simulation.
 *
 * Additionally we split data into floating-point data and (unsigned) integer
 * data, which are indicated with prefixes 'f' and 'u' respectively.
 *
 * In the current implementation only floating point parameters and state
 * variables can be read or written. The need for doing this with integer data
 * does not arise when using Izhikevich neurons.
 */
class Neurons : boost::noncopyable
{
	public:

		Neurons(const nemo::network::Generator&, unsigned type_id, const Mapper&);

		/*! Initialise the state of all neurons */
		cudaError_t initHistory(
				unsigned globalPartitionCount,
				param_t* d_params,
				unsigned* d_psize);

		/*! Update the state of all neurons */
		cudaError_t update(
				cudaStream_t stream,
				cycle_t cycle,
				unsigned globalPartitionCount,
				param_t* d_params,
				unsigned* d_psize,
				uint32_t* d_fstim,
				float* d_istim,
				float* d_current,
				uint32_t* d_fout,
				unsigned* d_nFired,
				nidx_dt* d_fired,
				rcm_dt*);

		/*! \return number of bytes allocated on the device */
		size_t d_allocated() const;

		/*! \return the word pitch of each per-partition row of data (for a
		 * single parameter/state variable) */
		size_t wordPitch32() const;

		/*! \return the word pitch of bitvectors */
		size_t wordPitch1() const { return m_valid.wordPitch(); }

		/*! \copydoc nemo::Network::setNeuron */
		void setNeuron(const DeviceIdx&, unsigned nargs, const float args[]);

		/*! Get a single parameter for a single neuron
		 *
		 * \param neuron neuron index
		 * \param parameter parameter index
		 * \return parameter with index \a parameter.
		 *
		 * For the Izhikevich model the parameter indices are 0=a, 1=b, 2=c, 3=d, 4=sigma.
		 *
		 * The first call to this function in any given cycle may take some
		 * time, since synchronisation is needed between the host and the
		 * device.
		 */
		float getParameter(const DeviceIdx& neuron, unsigned parameter) const;

		/*! Change a single parameter for an existing neuron
		 *
		 * \param neuron neuron index
		 * \param parameter parameter index
		 * \param value new value of the state variable
		 *
		 * For the Izhikevich model the parameter indices are 0=a, 1=b, 2=c, 3=d, 4=sigma.
		 *
		 * The first call to this function in any given cycle may take some
		 * time, since synchronisation is needed between the host and the
		 * device. Additionaly, the next simulation step will involve copying
		 * data from the host to the device.
		 */
		void setParameter(const DeviceIdx& idx, unsigned parameter, float value);

		/*! Get a single state variable for a single neuron
		 *
		 * \param neuron neuron index
		 * \param var variable index
		 * \return state variable \a n.
		 *
		 * For the Izhikevich model the variable indices are 0=u, 1=v.
		 *
		 * The first call to this function in any given cycle may take some
		 * time, since synchronisation is needed between the host and the
		 * device.
		 */
		float getState(const DeviceIdx& neuron, unsigned var) const;

		float getMembranePotential(const DeviceIdx&) const;

		/*! Change a single state variable for an existing neuron
		 *
		 * \param neuron neuron index
		 * \param var state variable index
		 * \param value new value of the state variable
		 *
		 * For the Izhikevich model variable indices 0=u, 1=v.
		 *
		 * The first call to this function in any given cycle may take some
		 * time, since synchronisation is needed between the host and the
		 * device. Additionaly, the next simulation step will involve copying
		 * data from the host to the device.
		 */
		void setState(const DeviceIdx& neuron, unsigned var, float value);

		/*! \return iterator to beginning of partition size vector */
		std::vector<unsigned>::const_iterator psize_begin() const {
			return mh_partitionSize.begin();
		}

		/*! \return iterator to end of partition size vector */
		std::vector<unsigned>::const_iterator psize_end() const {
			return mh_partitionSize.end();
		}

	private:

		NeuronType m_type;

		size_t parameterCount() const { return m_type.parameterCount(); }
		size_t stateVarCount() const { return m_type.stateVarCount(); }

		/* Neuron parameters do not change at run-time (unless the user
		 * specifically does it through \a setParameter) */
		NVector<float> m_param;

		/* Neuron state variables are updated during simulation. */
		mutable NVector<float> m_state;

		/* Index of state buffer corresponding to most recent state */
		unsigned m_stateCurrent;

		/*! \return offset (in terms of 'planes') to the up-to-date data for variable \a var */
		size_t currentStateVariable(unsigned var) const;

		/*! Normal RNG state */
		NVector<unsigned> m_nrngState;
		nrng_t m_nrng;

		/* In the translation from global neuron indices to device indices,
		 * there may be 'holes' left in the index space. The valid bitvector
		 * specifies which neurons are valid/existing, so that the kernel can
		 * ignore these. Of course, the ideal situation is that the index space
		 * is contigous, so that warp divergence is avoided on the device */
		Bitvector m_valid;

		cycle_t m_cycle;
		mutable cycle_t m_lastSync;

		bool m_paramDirty;
		bool m_stateDirty;

		/* Each neuron type has a contigous range of partition indices starting
		 * from \a m_basePartition */
		unsigned m_basePartition;

		/* Size of each partition */
		std::vector<unsigned> mh_partitionSize;

		/* Number of partitions of \i this type */
		unsigned localPartitionCount() const;

		/*! Read the neuron state from the device, if it the device data is not
		 * already cached on the host */
		void readStateFromDevice() const; // conceptually const, this is just caching

		/*! Perform any required synchronisation between host and device data.
		 * Such synchronisation may be required if the user has requested that
		 * the data should be updated. The sync function should be called for
		 * every simulation cycle. */
		void syncToDevice();

		/* The update function itself is found in a plugin which is loaded
		 * dynamically */
		Plugin m_plugin;
		cuda_update_neurons_t* m_update_neurons;
};


	} // end namespace cuda
} // end namespace nemo

#endif
