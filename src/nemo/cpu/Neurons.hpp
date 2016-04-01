#ifndef NEMO_CPU_NEURONS_HPP
#define NEMO_CPU_NEURONS_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/multi_array.hpp>

#include <nemo/RandomMapper.hpp>
#include <nemo/Plugin.hpp>
#include <nemo/RNG.hpp>
#include <nemo/network/Generator.hpp>
#include <nemo/cpu/plugins/neuron_model.h>

namespace nemo {
	namespace cpu {


/*! Neuron population for CPU backend
 *
 * The neurons are stored internally in dense structure-of-arrays with
 * contigous local indices starting from zero.
 *
 * \todo deal with multi-threading inside this class
 */
class Neurons
{
	public :

		/*! Set up local storage for all neurons with the generator neuron type id.
		 *
		 * As a side effect, the mapper is updated to contain mappings between
		 * global and local indices for the relevant neurons, as well as
		 * mappings between type_id and local neuron index.
		 */
		Neurons(const network::Generator& net,
				unsigned type_id,
				RandomMapper<nidx_t>& mapper);

		/*! Update the state of all neurons
		 *
		 * \param currentEPSP input current due to EPSPs
		 * \param currentIPSP input current due to IPSPs
		 * \param currentExternal externally (user-provided input current)
		 *
		 * \post the input current vector is set to all zero.
		 * \post the firing stimulus buffer (\a fstim) is set to all false.
		 */
		void update(unsigned cycle, unsigned fbits,
			float currentEPSP[],
			float currentIPSP[],
			float currentExternal[],
			unsigned fstim[], uint64_t recentFiring[],
			unsigned fired[], void* rcm);

		/*! Get a single state variable for a single neuron
		 *
		 * \param l_idx local neuron index
		 * \param var variable index
		 * \return state variable with index \a var.
		 */
		float getState(unsigned l_idx, unsigned var) const;

		/*! \copydoc nemo::Network::getNeuronParameter */
		/*! Get a single parameter for a single neuron
		 *
		 * \param l_idx local neuron index
		 * \param parameter parameter index
		 * \return parameter with index \a parameter.
		 */
		float getParameter(unsigned l_idx, unsigned param) const;

		float getMembranePotential(unsigned l_idx) const {
			return getState(l_idx, m_type.membranePotential());
		}

		/*! \copydoc nemo::Network::setNeuron */
		void set(unsigned l_idx, unsigned nargs, const float args[]);

		/*! Change a single state variable for an existing neuron
		 *
		 * \param l_idx local neuron index
		 * \param var state variable index
		 * \param value new value of the state variable
		 */
		void setState(unsigned l_idx, unsigned var, float val);

		/*! Change a single parameter for an existing neuron
		 *
		 * \param l_idx local neuron index
		 * \param parameter parameter index
		 * \param value new value of the state variable
		 */
		void setParameter(unsigned l_idx, unsigned param, float val);

		/*! \return number of neurons in this collection */
		size_t size() const { return m_size; }

	private :

		unsigned m_base;

		/*! Common type for all neurons in this collection */
		NeuronType m_type;

		/* Number of parameters and state variables */
		const unsigned m_nParam;
		const unsigned m_nState;

		/*! Neurons are stored in several structure-of-arrays, supporting
		 * arbitrary neuron types. Functions modifying these maintain the
		 * invariant that the shapes are the same.
		 *
		 * The indices here are:
		 *
		 * 1. (outer) parameter index
		 * 2. (inner) neuron index
		 */
		typedef boost::multi_array<float, 2> param_type;
		param_type m_param;

		/*! Neuron state is stored in a structure-of-arrays format, supporting
		 * arbitrary neuron types. Functions modifying these maintain the
		 *
		 * The indices here are:
		 *
		 * 1. (outer) history index
		 * 2.         variable index
		 * 3. (inner) neuron index
		 */
		typedef boost::multi_array<float, 3> state_type;
		state_type m_state;

		/* History index corresponding to most recent state */
		unsigned m_stateCurrent;

		/*! Set neuron, like \a cpu::Neurons::set, but with fewer checks */
		void setUnsafe(unsigned l_idx, const float param[], const float state[]);

		/*! Number of neurons in this collection */
		size_t m_size;

		/*! \return parameter index after checking its validity */
		unsigned parameterIndex(unsigned i) const;

		/*! \return state variable index after checking its validity */
		unsigned stateIndex(unsigned i) const;

		/*! RNG with separate state for each neuron */
		std::vector<RNG> m_rng;

		//! \todo maintain firing buffer etc. here instead?

		/* The update function itself is found in a plugin which is loaded
		 * dynamically */
		Plugin m_plugin;
		cpu_update_neurons_t* m_update_neurons;
};


	}
}

#endif
