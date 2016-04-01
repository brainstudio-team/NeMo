#include "Simulation.hpp"

#include <cmath>

#include <boost/format.hpp>
#include <boost/numeric/conversion/cast.hpp>

#ifdef NEMO_CPU_OPENMP_ENABLED
#include <omp.h>
#endif

#include <nemo/internals.hpp>
#include <nemo/exception.hpp>
#include <nemo/bitops.h>
#include <nemo/fixedpoint.hpp>
#include <nemo/ConnectivityMatrix.hpp>

#ifdef NEMO_CPU_DEBUG_TRACE

#include <cstdio>
#include <cstdlib>

#define LOG(...) fprintf(stdout, __VA_ARGS__);

#else

#define LOG(...)

#endif


namespace nemo {
	namespace cpu {


Simulation::Simulation(
		const nemo::network::Generator& net,
		const nemo::ConfigurationImpl& conf) :
	m_neuronCount(net.neuronCount()),
	m_fired(m_neuronCount, 0),
	m_recentFiring(m_neuronCount, 0),
	m_delays(m_neuronCount, 0),
	mfx_currentE(m_neuronCount, 0U),
	m_currentE(m_neuronCount, 0.0f),
	mfx_currentI(m_neuronCount, 0U),
	m_currentI(m_neuronCount, 0.0),
	m_currentExt(m_neuronCount, 0.0f),
	m_fstim(m_neuronCount, 0)
{
	using boost::format;

	if(net.maxDelay() > 64) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("The network has synapses with delay %ums. The CPU backend supports a maximum of 64 ms")
						% net.maxDelay()));
	}

	/* Contigous local neuron indices */
	nidx_t l_idx = 0;

	for(unsigned type_id=0, id_end=net.neuronTypeCount(); type_id < id_end; ++type_id) {

		/* Wrap in smart pointer to ensure the class is not copied */
		m_mapper.insertTypeBase(type_id, l_idx);

		if(net.neuronCount(type_id) == 0) {
			continue;
		}

		boost::shared_ptr<Neurons> ns(new Neurons(net, type_id, m_mapper));
		l_idx += ns->size();
		m_neurons.push_back(ns);
	}

	m_cm.reset(new nemo::ConnectivityMatrix(net, conf, m_mapper));

	for(size_t source=0; source < m_neuronCount; ++source) {
		m_delays[source] = m_cm->delayBits(source);
	}

	resetTimer();
}



unsigned
Simulation::getFractionalBits() const
{
	return m_cm->fractionalBits();
}



void
Simulation::fire()
{
	deliverSpikes();
	for(neuron_groups::const_iterator i = m_neurons.begin();
			i != m_neurons.end(); ++i) {
		(*i)->update(
			m_timer.elapsedSimulation(), getFractionalBits(),
			&m_currentE[0], &m_currentI[0], &m_currentExt[0],
			&m_fstim[0], &m_recentFiring[0], &m_fired[0],
			const_cast<void*>(static_cast<const void*>(m_cm->rcm())));
	}

	//! \todo do this in the postfire step
	m_cm->accumulateStdp(m_recentFiring);
	setFiring();
	m_timer.step();
}



#ifdef NEMO_BRIAN_ENABLED
std::pair<float*, float*>
Simulation::propagate_raw(uint32_t* fired, int nfired)
{
	//! \todo assert that STDP is not enabled

	/* convert the input firing to the format required by deliverSpikes */
#pragma omp parallel for default(shared)
	for(unsigned n=0; n <= m_mapper.maxGlobalIdx(); ++n) {
		m_recentFiring[n] <<= 1;
	}

#pragma omp parallel for default(shared)
	for(int i=0; i < nfired; ++i) {
		uint32_t n = fired[i];
		m_recentFiring[n] |= uint64_t(1);
	}
	deliverSpikes();
	m_timer.step();

	return std::make_pair<float*, float*>(&m_currentE[0], &m_currentI[0]);
}
#endif


void
Simulation::setFiringStimulus(const std::vector<unsigned>& fstim)
{
	for(std::vector<unsigned>::const_iterator i = fstim.begin();
			i != fstim.end(); ++i) {
		m_fstim.at(m_mapper.localIdx(*i)) = 1;
	}
}



void
Simulation::initCurrentStimulus(size_t count)
{
	/* The current is cleared after use, so no need to reset */
}



void
Simulation::addCurrentStimulus(nidx_t neuron, float current)
{
	m_currentExt[m_mapper.localIdx(neuron)] = current;
}



void
Simulation::finalizeCurrentStimulus(size_t count)
{
	/* The current is cleared after use, so no need to reset */
}



void
Simulation::setCurrentStimulus(const std::vector<float>& current)
{
	// TODO: In Andreas' NeMo this is m_current, not m_currentExt
	if(m_currentExt.empty()) {
		//! do we need to clear current?
		return;
	}
	/*! \todo We need to deal with the mapping from global to local neuron
	 * indices. Before doing this, we should probably change the interface
	 * here. Note that this function is only used internally (see mpi::Worker),
	 * so we might be able to use the existing interface, and make sure that we
	 * only use local indices. */
	throw nemo::exception(NEMO_API_UNSUPPORTED, "setting current stimulus vector not supported for CPU backend");
#if 0
	if(current.size() != m_currentExt.size()) {
		throw nemo::exception(NEMO_INVALID_INPUT, "current stimulus vector not of expected size");
	}
	m_currentExt = current;
#endif
}



//! \todo use per-thread buffers and just copy these in bulk
void
Simulation::setFiring()
{
	m_firingBuffer.enqueueCycle();
	for(unsigned n=0; n < m_neuronCount; ++n) {
		if(m_fired[n]) {
			m_firingBuffer.addFiredNeuron(m_mapper.globalIdx(n));
		}
	}
}



FiredList
Simulation::readFiring()
{
	return m_firingBuffer.dequeueCycle();
}



void
Simulation::setNeuron(unsigned g_idx, unsigned nargs, const float args[])
{
	unsigned l_idx = m_mapper.localIdx(g_idx);
	unsigned type = m_mapper.typeIdx(l_idx);
	m_neurons.at(type)->set(l_idx, nargs, args);
}



void
Simulation::setNeuronState(unsigned g_idx, unsigned var, float val)
{
	unsigned l_idx = m_mapper.localIdx(g_idx);
	unsigned type = m_mapper.typeIdx(l_idx);
	m_neurons.at(type)->setState(l_idx, var, val);
}



void
Simulation::setNeuronParameter(unsigned g_idx, unsigned parameter, float val)
{
	unsigned l_idx = m_mapper.localIdx(g_idx);
	unsigned type = m_mapper.typeIdx(l_idx);
	m_neurons.at(type)->setParameter(l_idx, parameter, val);
}



void
Simulation::applyStdp(float reward)
{
	m_cm->applyStdp(reward);
}



void
Simulation::deliverSpikes()
{
	/* Ignore spikes outside of max delay. We keep these older spikes as they
	 * may be needed for STDP */
	uint64_t validSpikes = ~(((uint64_t) (~0)) << m_cm->maxDelay());

	for(size_t source=0; source < m_neuronCount; ++source) {

		uint64_t f = m_recentFiring[source] & validSpikes & m_delays[source];

		int delay = 0;
		while(f) {
			int shift = 1 + ctz64(f);
			delay += shift;
			f = f >> shift;
			deliverSpikesOne(source, delay);
		}
	}

	/* convert current back to float */
	unsigned fbits = getFractionalBits();
	int ncount = boost::numeric_cast<int, unsigned>(m_neuronCount);
#pragma omp parallel for default(shared)
	for(int n=0; n < ncount; n++) {
		m_currentE[n] = wfx_toFloat(mfx_currentE[n], fbits);
		mfx_currentE[n] = 0U;
		m_currentI[n] = wfx_toFloat(mfx_currentI[n], fbits);
		mfx_currentI[n] = 0U;
	}
}



void
Simulation::deliverSpikesOne(nidx_t source, delay_t delay)
{
	const nemo::Row& row = m_cm->getRow(source, delay);

	for(unsigned s=0; s < row.len; ++s) {
		const FAxonTerminal& terminal = row[s];
		std::vector<wfix_t>& current = terminal.weight >= 0 ? mfx_currentE : mfx_currentI;
		current.at(terminal.target) += terminal.weight;
		LOG("c%lu: n%u -> n%u: %+f (delay %u)\n",
				elapsedSimulation(),
				m_mapper.globalIdx(source),
				m_mapper.globalIdx(terminal.target),
				fx_toFloat(terminal.weight, getFractionalBits()), delay);
	}
}



float
Simulation::getNeuronState(unsigned g_idx, unsigned var) const
{
	unsigned l_idx = m_mapper.localIdx(g_idx);
	unsigned type = m_mapper.typeIdx(l_idx);
	return m_neurons.at(type)->getState(l_idx, var);
}



float
Simulation::getNeuronParameter(unsigned g_idx, unsigned param) const
{
	unsigned l_idx = m_mapper.localIdx(g_idx);
	unsigned type = m_mapper.typeIdx(l_idx);
	return m_neurons.at(type)->getParameter(l_idx, param);
}


float
Simulation::getMembranePotential(unsigned g_idx) const
{
	unsigned l_idx = m_mapper.localIdx(g_idx);
	unsigned type = m_mapper.typeIdx(l_idx);
	return m_neurons.at(type)->getMembranePotential(l_idx);
}



const std::vector<synapse_id>&
Simulation::getSynapsesFrom(unsigned neuron)
{
	return m_cm->getSynapsesFrom(neuron);
}



unsigned
Simulation::getSynapseTarget(const synapse_id& synapse) const
{
	return m_cm->getTarget(synapse);
}



unsigned
Simulation::getSynapseDelay(const synapse_id& synapse) const
{
	return m_cm->getDelay(synapse);
}



float
Simulation::getSynapseWeight(const synapse_id& synapse) const
{
	return m_cm->getWeight(synapse);
}



unsigned char
Simulation::getSynapsePlastic(const synapse_id& synapse) const
{
	return m_cm->getPlastic(synapse);
}

unsigned long
Simulation::elapsedWallclock() const
{
	return m_timer.elapsedWallclock();
}



unsigned long
Simulation::elapsedSimulation() const
{
	return m_timer.elapsedSimulation();
}



void
Simulation::resetTimer()
{
	m_timer.reset();
}



const char*
deviceDescription()
{
	/* Store a static string here so we can safely pass a char* rather than a
	 * string object across DLL interface */
#ifdef NEMO_CPU_OPENMP_ENABLED
	using boost::format;
	static std::string descr = str(format("CPU backend (OpenMP, %u cores)") % omp_get_num_procs());
#else
	static std::string descr("CPU backend (single-threaded)");
#endif
	return descr.c_str();
}


	} // namespace cpu
} // namespace nemo
