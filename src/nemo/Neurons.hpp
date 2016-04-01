#ifndef NEMO_NEURONS_HPP
#define NEMO_NEURONS_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>

#include <nemo/config.h>
#include "NeuronType.hpp"

namespace nemo {

namespace network {
class NetworkImpl;
}

/*! \brief collection of neurons of the same type
 *
 * A network consisting of neurons of multiple types can be created from
 * several instances of this class.
 */
class NEMO_BASE_DLL_PUBLIC Neurons
{
public :

	Neurons(const NeuronType&);

	/*! Add a new neuron
	 *
	 * \param gidx global neuron index
	 * \param nargs number of parameters and state variables
	 * \param args all parameters and state variables (in that order)
	 *
	 * \return local index (wihtin this class) of the newly added neuron
	 *
	 * \pre the input arguments must have the lengths that was specified by
	 * 		the neuron type used when this object was created.
	 */
	size_t add(unsigned gidx, unsigned nargs, const float args[]);

	/*! Modify an existing neuron
	 *
	 * \param nargs number of parameters and state variables
	 * \param args all parameters and state variables (in that order)
	 *
	 * \pre nidx refers to a valid neuron in this collection
	 * \pre the input array have the lengths specified by the neuron type
	 * 		used when this object was created.
	 */
	void set(size_t nidx, unsigned nargs, const float args[]);

	/*! \copydoc NetworkImpl::getNeuronParameter */
	float getParameter(size_t nidx, unsigned pidx) const;

	/*! \copydoc NetworkImpl::getNeuronState */
	float getState(size_t nidx, unsigned sidx) const;

	/*! \copydoc NetworkImpl::getMembranePotential */
	float getMembranePotential(size_t nidx) const;

	/*! \copydoc NetworkImpl::setNeuronParameter */
	void setParameter(size_t nidx, unsigned pidx, float val);

	/*! \copydoc NetworkImpl::setNeuronState */
	void setState(size_t nidx, unsigned sidx, float val);

	const std::vector<unsigned>& getIndexes() const {return m_gidx;};

	/*! \return number of neurons in this collection */
	size_t size() const {return m_size;}

	/*! \return neuron type common to all neurons in this collection */
	const NeuronType& type() const {return m_type;}

private :

	/* Neurons are stored in several Structure-of-arrays, supporting
	 * arbitrary neuron types. Functions modifying these maintain the
	 * invariant that the shapes are the same. */
	std::vector< std::vector<float> > m_param;
	std::vector< std::vector<float> > m_state;

	/* We store the global neuron indices as well, in order to avoid
	 * reverse lookup in the mapper, when getting data out of the
	 * simulation */
	std::vector<unsigned> m_gidx;

	size_t m_size;

	NeuronType m_type;

	/*! \return parameter index after checking its validity */
	unsigned parameterIndex(unsigned i) const;

	/*! \return state variable index after checking its validity */
	unsigned stateIndex(unsigned i) const;

	friend class nemo::network::NetworkImpl;
};

}

#endif
