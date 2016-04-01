#include "Neuron.hpp"

#include <algorithm>
#include <boost/format.hpp>

#include "exception.hpp"

namespace nemo {


Neuron::Neuron(const NeuronType& type) :
	m_param(type.parameterCount(), 0.0f),
	m_state(type.stateVarCount(), 0.0f)
{ }


Neuron::Neuron(const NeuronType& type, float param[], float state[]) :
	m_param(param, param + type.parameterCount()),
	m_state(state, state + type.stateVarCount())
{ }



void
Neuron::set(float param[], float state[])
{
	std::copy(param, param + m_param.size(), m_param.begin());
	std::copy(state, state + m_state.size(), m_state.begin());
}

float
Neuron::getParameter(size_t i) const
{
	return paramRef(i);
}



float
Neuron::getState(size_t i) const
{
	return stateRef(i);
}



const float&
Neuron::paramRef(size_t i) const
{
	using boost::format;
	if(i >= m_param.size()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid neuron parameter index (%u)") % i));
	}
	return m_param[i];
}




const float&
Neuron::stateRef(size_t i) const
{
	using boost::format;
	if(i >= m_state.size()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid neuron state variable index (%u)") % i));
	}
	return m_state[i];
}



void
Neuron::setParameter(size_t i, float val)
{
	const_cast<float&>(paramRef(i)) = val;
}



void
Neuron::setState(size_t i, float val)
{
	const_cast<float&>(stateRef(i)) = val;
}

}
