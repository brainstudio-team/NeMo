#include "Neurons.hpp"

namespace nemo {
	namespace cpu {


Neurons::Neurons(const nemo::network::Generator& net,
				unsigned type_id,
				RandomMapper<nidx_t>& mapper) :
	m_base(mapper.typeBase(type_id)),
	m_type(net.neuronType(type_id)),
	m_nParam(m_type.parameterCount()),
	m_nState(m_type.stateVarCount()),
	m_param(boost::extents[m_nParam][net.neuronCount(type_id)]),
	m_state(boost::extents[m_type.stateHistory()][m_nState][net.neuronCount(type_id)]),
	m_stateCurrent(0),
	m_size(0),
	m_rng(net.neuronCount(type_id)),
	m_plugin(m_type.pluginDir() / "cpu", m_type.name()),
	m_update_neurons((cpu_update_neurons_t*) m_plugin.function("cpu_update_neurons"))
{
	using namespace nemo::network;

	std::fill(m_state.data(), m_state.data() + m_state.num_elements(), 0.0f);

	for(neuron_iterator i = net.neuron_begin(type_id), i_end = net.neuron_end(type_id);
			i != i_end; ++i) {

		unsigned userIdx = i->first;
		unsigned localIdx = m_size;
		unsigned simIdx = m_base + m_size;
		mapper.insert(userIdx, simIdx);
		mapper.insertTypeMapping(simIdx, type_id);

		const Neuron& n = i->second;
		setUnsafe(localIdx, n.getParameters(), n.getState());

		m_size++;
	}

	nemo::initialiseRng(m_base, m_base+m_size-1, m_rng);

	cpu_init_neurons_t* init_neurons = (cpu_init_neurons_t*) m_plugin.function("cpu_init_neurons");
	init_neurons(m_base, m_base + size(),
			m_param.data(), m_param.strides()[0],
			m_state.data(), m_state.strides()[0], m_state.strides()[1],
			&m_rng[0]);
}



void
Neurons::set(unsigned l_idx, unsigned nargs, const float args[])
{
	using boost::format;

	if(nargs != m_nParam + m_nState) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Unexpected number of parameters/state variables when modifying neuron. Expected %u, found %u")
						% (m_nParam + m_nState) % nargs));
	}

	setUnsafe(l_idx, args, args+m_nParam);
}


void
Neurons::setUnsafe(unsigned l_idx, const float param[], const float state[])
{
	for(unsigned i=0; i < m_nParam; ++i) {
		m_param[i][l_idx] = param[i];
	}
	for(unsigned i=0; i < m_nState; ++i) {
		m_state[0][i][l_idx] = state[i];
	}
}



float
Neurons::getState(unsigned l_idx, unsigned var) const
{
	return m_state[m_stateCurrent][stateIndex(var)][l_idx];
}



void
Neurons::setState(unsigned l_idx, unsigned var, float val)
{
	m_state[m_stateCurrent][stateIndex(var)][l_idx] = val;
}



void
Neurons::setParameter(unsigned l_idx, unsigned param, float val)
{
	m_param[parameterIndex(param)][l_idx] = val;
}



float
Neurons::getParameter(unsigned l_idx, unsigned param) const
{
	return m_param[parameterIndex(param)][l_idx];
}


void
Neurons::update(
		unsigned cycle,
		unsigned fbits,
		float currentEPSP[],
		float currentIPSP[],
		float currentExternal[],
		unsigned fstim[],
		uint64_t recentFiring[],
		unsigned fired[],
		void* rcm)
{
	m_stateCurrent = (cycle+1) % m_type.stateHistory();

	m_update_neurons(m_base, m_base + size(), cycle,
			m_param.data(), m_param.strides()[0],
			m_state.data(), m_state.strides()[0], m_state.strides()[1],
			fbits,
			fstim,
			&m_rng[0],
			currentEPSP,
			currentIPSP,
			currentExternal,
			recentFiring,
			fired,
			rcm);
}



unsigned
Neurons::stateIndex(unsigned i) const
{
	using boost::format;
	if(i >= m_nState) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid state variable index %u") % i));
	}
	return i;
}



unsigned
Neurons::parameterIndex(unsigned i) const
{
	using boost::format;
	if(i >= m_nParam) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid parameter index %u") % i));
	}
	return i;
}


	} // end namespace cpu
} // end namespace nemo
