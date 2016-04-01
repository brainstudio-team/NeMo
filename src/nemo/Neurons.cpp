/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Neurons.hpp"

#include <boost/format.hpp>

#include "exception.hpp"

namespace nemo {

Neurons::Neurons(const NeuronType& type) :
	m_param(type.parameterCount()),
	m_state(type.stateVarCount()),
	m_size(0),
	m_type(type)
{
	;
}



size_t
Neurons::add(unsigned gidx, unsigned nargs, const float args[])
{
	using boost::format;

	if(nargs != m_param.size() + m_state.size()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Unexpected number of parameters/state variables when adding neuron. Expected %u, found %u")
						% (m_param.size() + m_state.size()) % nargs));
	}

	for(unsigned i=0; i < m_param.size(); ++i) {
		m_param[i].push_back(*args++);
	}
	for(unsigned i=0; i < m_state.size(); ++i) {
		m_state[i].push_back(*args++);
	}
	m_gidx.push_back(gidx);
	return m_size++;
}



unsigned
Neurons::parameterIndex(unsigned i) const
{
	using boost::format;
	if(i >= m_param.size()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid parameter index %u") % i));
	}
	return i;
}



unsigned
Neurons::stateIndex(unsigned i) const
{
	using boost::format;
	if(i >= m_state.size()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid state variable index %u") % i));
	}
	return i;
}



void
Neurons::set(size_t n, unsigned nargs, const float args[])
{
	using boost::format;

	if(nargs != m_param.size() + m_state.size()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Unexpected number of parameters/state variables when modifying neuron. Expected %u, found %u")
						% (m_param.size() + m_state.size()) % nargs));
	}

	for(unsigned i=0; i < m_param.size(); ++i) {
		m_param[i][n] = *args++;
	}
	for(unsigned i=0; i < m_state.size(); ++i) {
		m_state[i][n] = *args++;
	}
}



float
Neurons::getParameter(size_t nidx, unsigned pidx) const
{
	return m_param[parameterIndex(pidx)][nidx];
}

float
Neurons::getState(size_t nidx, unsigned sidx) const
{
	return m_state[stateIndex(sidx)][nidx];
}



float
Neurons::getMembranePotential(size_t nidx) const
{
	return getState(nidx, m_type.membranePotential());
}



void
Neurons::setParameter(size_t nidx, unsigned pidx, float val)
{
	m_param[parameterIndex(pidx)][nidx] = val;
}


void
Neurons::setState(size_t nidx, unsigned sidx, float val)
{
	m_state[stateIndex(sidx)][nidx] = val;
}


}
