/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Neurons.hpp"

#include <vector>

#include <boost/format.hpp>

#include <nemo/config.h>
#include <nemo/network/Generator.hpp>
#include <nemo/RNG.hpp>

#include "types.h"
#include "exception.hpp"
#include "kernel.hpp"


namespace nemo {
	namespace cuda {


Neurons::Neurons(const network::Generator& net,
		unsigned type_id,
		const Mapper& mapper) :
	m_type(net.neuronType(type_id)),
	m_param(m_type.parameterCount(), mapper.partitionCount(type_id), mapper.partitionSize(), true, false),
	m_state(m_type.stateVarCount() * m_type.stateHistory(),
			mapper.partitionCount(type_id), mapper.partitionSize(), true, false),
	m_stateCurrent(0U),
	m_valid(mapper.partitionCount(type_id), true),
	m_cycle(~0),
	m_lastSync(~0),
	m_paramDirty(false),
	m_stateDirty(false),
	m_basePartition(mapper.typeBase(type_id)),
	m_plugin(m_type.pluginDir() / "cuda", m_type.name()),
	m_update_neurons((cuda_update_neurons_t*) m_plugin.function("cuda_update_neurons"))
{
	if(m_type.usesNormalRNG()) {
		m_nrngState = NVector<unsigned>(RNG_STATE_COUNT,
				mapper.partitionCount(), mapper.partitionSize(), true, false);
	}

	std::map<pidx_t, nidx_t> maxPartitionNeuron;

	/* Create all the RNG seeds */
	//! \todo seed this from configuration
	std::vector<RNG> rngs(mapper.maxHandledGlobalIdx() - mapper.minHandledGlobalIdx() + 1);
	initialiseRng(mapper.minHandledGlobalIdx(), mapper.maxHandledGlobalIdx(), rngs);

	for(network::neuron_iterator i = net.neuron_begin(type_id), i_end = net.neuron_end(type_id);
			i != i_end; ++i) {

		//! \todo insertion here, but make sure the usage is correct in the Simulation class
		DeviceIdx devGlobal = mapper.localIdx(i->first);
		DeviceIdx dev(devGlobal.partition - m_basePartition, devGlobal.neuron);
		const nemo::Neuron& n = i->second;

		for(unsigned i=0, i_end=parameterCount(); i < i_end; ++i) {
			m_param.setNeuron(dev.partition, dev.neuron, n.getParameter(i), i);
		}
		for(unsigned i=0, i_end=stateVarCount(); i < i_end; ++i) {
			// no need to offset based on time here, since this is beginning of the simulation.
			m_state.setNeuron(dev.partition, dev.neuron, n.getState(i), i);
		}

		m_valid.setNeuron(dev);

		if(m_type.usesNormalRNG()) {
			//! \todo avoid mapping back again
			nidx_t localIdx = mapper.globalIdx(devGlobal) - mapper.minHandledGlobalIdx();
			for(unsigned plane = 0; plane < 4; ++plane) {
				m_nrngState.setNeuron(devGlobal.partition, devGlobal.neuron, rngs[localIdx].state[plane], plane);
			}
		}

		maxPartitionNeuron[dev.partition] =
			std::max(maxPartitionNeuron[dev.partition], dev.neuron);
	}

	m_param.copyToDevice();
	m_state.replicateInitialPlanes(m_type.stateVarCount());
	m_state.copyToDevice();
	if(m_type.usesNormalRNG()) {
		m_nrngState.moveToDevice();
	}
	m_nrng.state = m_nrngState.deviceData();
	m_nrng.pitch = m_nrngState.wordPitch();
	m_valid.moveToDevice();

	/* The partitions should form a contigous range here. If not, there is a
	 * logic error in the mapping.  */
	mh_partitionSize.resize(maxPartitionNeuron.size(), 0);
	for(std::map<pidx_t, nidx_t>::const_iterator i = maxPartitionNeuron.begin();
			i != maxPartitionNeuron.end(); ++i) {
		mh_partitionSize.at(i->first) = i->second + 1;
	}
}



cudaError_t
Neurons::initHistory(
		unsigned globalPartitionCount,
		param_t* d_params,
		unsigned* d_psize)
{
	cuda_init_neurons_t* init_neurons = (cuda_init_neurons_t*) m_plugin.function("cuda_init_neurons");
	return init_neurons(globalPartitionCount,
			localPartitionCount(),
			m_basePartition,
			d_psize,
			d_params,
			m_param.deviceData(),
			m_state.deviceData(),
			m_nrng,
			m_valid.d_data());
}



unsigned
Neurons::localPartitionCount() const
{
	return mh_partitionSize.size();
}



size_t
Neurons::d_allocated() const
{
	return m_param.d_allocated()
		+ m_state.d_allocated()
		+ m_nrngState.d_allocated();
}



size_t
Neurons::wordPitch32() const
{
	size_t f_param_pitch = m_param.wordPitch();
	size_t f_state_pitch = m_state.wordPitch();

	if(f_param_pitch != f_state_pitch
			&& f_param_pitch != 0
			&& f_state_pitch != 0) {
		throw nemo::exception(NEMO_LOGIC_ERROR, "State and parameter data have different pitches");
	}
	/* Either pitch is the same or one or both are zero */
	return std::max(f_param_pitch, f_state_pitch);
}



cudaError_t
Neurons::update(
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
		rcm_dt* d_rcm)
{
	syncToDevice();
	m_cycle = cycle;
	m_stateCurrent = (cycle+1) % m_type.stateHistory();
	return m_update_neurons(stream,
			cycle,
			globalPartitionCount,
			localPartitionCount(),
			m_basePartition,
			d_psize,
			d_params,
			m_param.deviceData(),
			m_state.deviceData(),
			m_nrng,
			m_valid.d_data(),
			d_fstim, d_istim,
			d_current, d_fout, d_nFired, d_fired,
			d_rcm);
}



inline
void
verifyParameterIndex(unsigned parameter, unsigned maxParameter)
{
	using boost::format;
	if(parameter >= maxParameter) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid neuron parameter index (%u)") % parameter));
	}
}



float
Neurons::getParameter(const DeviceIdx& idx, unsigned parameter) const
{
	verifyParameterIndex(parameter, parameterCount());
	return m_param.getNeuron(idx.partition, idx.neuron, parameter);
}



size_t
Neurons::currentStateVariable(unsigned var) const
{
	return m_stateCurrent * stateVarCount() + var;
}



void
Neurons::setNeuron(const DeviceIdx& dev, unsigned nargs, const float args[])
{
	using boost::format;

	//! \todo pre-compute these
	unsigned nparam = parameterCount();
	unsigned nstate = stateVarCount();
	if(nargs != nparam + nstate) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Unexpected number of parameters/state variables when modifying neuron. Expected %u, found %u")
						% (nparam + nstate) % nargs));
	}

	readStateFromDevice();
	for(unsigned i=0; i < nparam; ++i) {
		m_param.setNeuron(dev.partition, dev.neuron, *args++, i);
	}
	m_paramDirty = true;
	for(unsigned i=0; i < nstate; ++i) {
		m_state.setNeuron(dev.partition, dev.neuron, *args++, currentStateVariable(i));
	}
	m_stateDirty = true;
}



void
Neurons::setParameter(const DeviceIdx& idx, unsigned parameter, float value)
{
	verifyParameterIndex(parameter, parameterCount());
	m_param.setNeuron(idx.partition, idx.neuron, value, parameter);
	m_paramDirty = true;
}



void
Neurons::readStateFromDevice() const
{
	if(m_lastSync != m_cycle) {
		//! \todo read only part of the data here
		m_state.copyFromDevice();
		m_lastSync = m_cycle;
	}
}



inline
void
verifyStateVariableIndex(unsigned var, unsigned maxVar)
{
	using boost::format;
	if(var >= maxVar) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid neuron state variable index (%u)") % var));
	}
}



float
Neurons::getMembranePotential(const DeviceIdx& neuron) const
{
	return getState(neuron, m_type.membranePotential());
}



float
Neurons::getState(const DeviceIdx& idx, unsigned var) const
{
	verifyStateVariableIndex(var, stateVarCount());
	readStateFromDevice();
	return m_state.getNeuron(idx.partition, idx.neuron, currentStateVariable(var));
}



void
Neurons::setState(const DeviceIdx& idx, unsigned var, float value)
{
	verifyStateVariableIndex(var, stateVarCount());
	readStateFromDevice();
	m_state.setNeuron(idx.partition, idx.neuron, value, currentStateVariable(var));
	m_stateDirty = true;
}



void
Neurons::syncToDevice()
{
	if(m_paramDirty) {
		m_param.copyToDevice();
		m_paramDirty = false;
	}
	if(m_stateDirty) {
		m_state.copyToDevice();
		m_stateDirty = false;
	}
}

	} // end namespace cuda
} // end namespace nemo
