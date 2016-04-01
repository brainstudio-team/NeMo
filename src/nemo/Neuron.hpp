#ifndef NEMO_NEURON_HPP
#define NEMO_NEURON_HPP

#include <cstddef>
#include <vector>

#include <nemo/config.h>
#include "NeuronType.hpp"

#ifdef NEMO_MPI_ENABLED
#include <boost/serialization/vector.hpp>
#endif

namespace nemo {

class NEMO_BASE_DLL_PUBLIC Neuron
{
	public :

	Neuron() {}

	explicit Neuron(const NeuronType&);

	Neuron(const NeuronType&, float param[], float state[]);

	const std::vector<float>& getParametersVec() const{
		return m_param;
	}

	/*! \return all parameters of neuron (or NULL if there are none) */
	const float* getParameters() const {
		return m_param.empty() ? NULL : &m_param[0];
	}

	/*! \return all state variables of neuron (or NULL if there are none) */
	const float* getState() const {
		return m_state.empty() ? NULL : &m_state[0];
	}

	/*! \return i'th parameter of neuron */
	float getParameter(size_t i) const;

	/*! \return i'th state variable of neuron */
	float getState(size_t i) const;

	/*! set i'th parameter of neuron */
	void setParameter(size_t i, float val);

	/*! set i'th state variable of neuron */
	void setState(size_t i, float val);

	private :

	void init(const NeuronType& type);

	void set(float param[], float state[]);

	std::vector<float> m_param;
	std::vector<float> m_state;

	const float& paramRef(size_t i) const;
	const float& stateRef(size_t i) const;

#ifdef NEMO_MPI_ENABLED

	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & m_param;
		ar & m_state;
	}
#endif

};

}

#endif
