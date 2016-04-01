/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Simulation.hpp"

#include <vector>

#include <boost/format.hpp>

#include <nemo/exception.hpp>
#include <nemo/NetworkImpl.hpp>
#include <nemo/fixedpoint.hpp>

#include "DeviceAssertions.hpp"
#include "exception.hpp"

#include "device_assert.cu_h"
#include "kernel.cu_h"
#include "kernel.hpp"

namespace nemo {
namespace cuda {

/*! Map global neuron indices to compact local indices
 *
 * Multiple neuron types are mapped such that
 *
 * 1. no partition contains more than one neuron type
 * 2. all neurons of a single type are found in a contigous range of partitions
 */
Mapper mapCompact(const nemo::network::Generator& net, unsigned partitionSize) {
	using namespace nemo::network;

	Mapper mapper(partitionSize);

	pidx_t pidx = 0;

	for (unsigned type_id = 0, id_end = net.neuronTypeCount(); type_id < id_end; ++type_id) {

		mapper.insertTypeBase(type_id, pidx);
		nidx_t nidx = 0;

		for (neuron_iterator i = net.neuron_begin(type_id), i_end = net.neuron_end(type_id);
				i != i_end; ++i) {

			if (nidx == 0) {
				/* First neuron in a new partition */
				mapper.insertTypeMapping(pidx, type_id);
			}

			unsigned g_idx = i->first;
			DeviceIdx l_idx(pidx, nidx);
			mapper.insert(g_idx, l_idx);
			nidx++;
			if (nidx == partitionSize) {
				nidx = 0;
				pidx++;
			}
		}

		/* neuron types should never cross partition boundaries */
		if (nidx != 0) {
			pidx++;
		}
	}
	return mapper;
}

/*! Verify device memory pitch
 *
 * On the device a number of arrays have exactly the same shape. These share a
 * common pitch parameter. This function verifies that the memory allocation
 * does what we expect.
 */
void checkPitch(size_t found, size_t expected) {
	using boost::format;
	if (found != 0 && expected != found) {
		throw nemo::exception(NEMO_CUDA_MEMORY_ERROR,
				str(
						format("Pitch mismatch in device memory allocation. Found %u, expected %u")
								% found % expected));
	}
}

Simulation::Simulation(const nemo::network::Generator& net, const nemo::ConfigurationImpl& conf) :
		m_mapper(mapCompact(net, conf.cudaPartitionSize())), m_conf(conf), m_cm(net, conf,
				m_mapper), m_lq(m_mapper.partitionCount(), m_mapper.partitionSize(),
				std::max(1U, net.maxDelay())), m_recentFiring(2, m_mapper.partitionCount(),
				m_mapper.partitionSize(), false, false), m_firingStimulus(
				m_mapper.partitionCount()), m_currentStimulus(1, m_mapper.partitionCount(),
				m_mapper.partitionSize(), true, true), m_current(2, m_mapper.partitionCount(),
				m_mapper.partitionSize(), false, false), m_firingBuffer(m_mapper), m_fired(1,
				m_mapper.partitionCount(), m_mapper.partitionSize(), false, false), md_nFired(
				d_array<unsigned>(m_mapper.partitionCount(), true, "Fired count")), md_partitionSize(
				d_array<unsigned>(MAX_PARTITION_COUNT, true, "partition size array")), m_deviceAssertions(
				m_mapper.partitionCount()), m_stdp(conf.stdpFunction()), md_istim(NULL), m_streamCompute(
				0), m_streamCopy(0) {
	using boost::format;

	size_t pitch1 = m_firingBuffer.wordPitch();
	size_t pitch32 = m_current.wordPitch();

	/* Populate all neuron collections */
	std::vector<unsigned> h_partitionSize;
	for (unsigned type_id = 0, id_end = net.neuronTypeCount(); type_id < id_end; ++type_id) {

		if (net.neuronCount(type_id) == 0) {
			continue;
		}

		//! \todo could do mapping here to avoid two passes over neurons
		/* Wrap in smart pointer to ensure the class is not copied */
		boost::shared_ptr<Neurons> ns(new Neurons(net, type_id, m_mapper));
		checkPitch(ns->wordPitch32(), pitch32);
		checkPitch(ns->wordPitch1(), pitch1);
		//! \todo verify contigous range
		std::copy(ns->psize_begin(), ns->psize_end(), std::back_inserter(h_partitionSize));
		m_neurons.push_back(ns);
	}
	h_partitionSize.resize(MAX_PARTITION_COUNT, 0); // extend
	memcpyToDevice(md_partitionSize.get(), h_partitionSize);

	if (m_stdp) {
		if (m_cm.maxDelay() > 64) {
			throw nemo::exception(NEMO_INVALID_INPUT,
					str(
							format(
									"The network has synapses with delay %ums. The CUDA backend supports a maximum of 64ms when STDP is used")
									% m_cm.maxDelay()));
		}
		configureStdp();
	}
	param_t* d_params = setParameters(pitch1, pitch32, net.maxDelay());
	resetTimer();

	CUDA_SAFE_CALL(cudaStreamCreate(&m_streamCompute));
	CUDA_SAFE_CALL(cudaStreamCreate(&m_streamCopy));
	CUDA_SAFE_CALL(cudaEventCreate(&m_eventFireDone));
	CUDA_SAFE_CALL(cudaEventCreate(&m_firingStimulusDone));
	CUDA_SAFE_CALL(cudaEventCreate(&m_currentStimulusDone));

	//! \todo do m_cm size reporting here as well
	if (conf.loggingEnabled()) {
		std::cout << "\tLocal queue: " << m_lq.allocated() / (1 << 20) << "MB\n";
	}

	for (neuron_groups::const_iterator i = m_neurons.begin(); i != m_neurons.end(); ++i) {
		runKernel((*i)->initHistory(m_mapper.partitionCount(), d_params, md_partitionSize.get()));
	}
}

Simulation::~Simulation() {
	finishSimulation();
}

void Simulation::configureStdp() {
	std::vector<float> flfn;

	std::copy(m_stdp->prefire().rbegin(), m_stdp->prefire().rend(), std::back_inserter(flfn));
	std::copy(m_stdp->postfire().begin(), m_stdp->postfire().end(), std::back_inserter(flfn));

	std::vector<fix_t> fxfn(flfn.size());
	unsigned fb = m_cm.fractionalBits();
	for (unsigned i = 0; i < fxfn.size(); ++i) {
		fxfn.at(i) = fx_toFix(flfn[i], fb);
	}
	CUDA_SAFE_CALL(
			::configureStdp(m_stdp->prefire().size(), m_stdp->postfire().size(),
					m_stdp->potentiationBits(), m_stdp->depressionBits(),
					const_cast<fix_t*>(&fxfn[0])));
}

void Simulation::setFiringStimulus(const std::vector<unsigned>& nidx) {
	m_firingStimulus.set(m_mapper, nidx, m_streamCopy);
	CUDA_SAFE_CALL(cudaEventRecord(m_firingStimulusDone, m_streamCopy));
}

void Simulation::initCurrentStimulus(size_t count) {
	if (count > 0) {
		m_currentStimulus.fill(0);
	}
}

void Simulation::addCurrentStimulus(nidx_t neuron, float current) {
	DeviceIdx dev = m_mapper.deviceIdx(neuron);
	m_currentStimulus.setNeuron(dev.partition, dev.neuron, current);
}

void Simulation::finalizeCurrentStimulus(size_t count) {
	if (count > 0) {
		m_currentStimulus.copyToDeviceAsync(m_streamCopy);
		md_istim = m_currentStimulus.deviceData();
		CUDA_SAFE_CALL(cudaEventRecord(m_currentStimulusDone, m_streamCopy));
	}
	else {
		md_istim = NULL;
	}
}

void Simulation::setCurrentStimulus(const std::vector<float>& current) {
	if (current.empty()) {
		md_istim = NULL;
		return;
	}
	m_currentStimulus.set(current);
	m_currentStimulus.copyToDeviceAsync(m_streamCopy);
	md_istim = m_currentStimulus.deviceData();
	CUDA_SAFE_CALL(cudaEventRecord(m_currentStimulusDone, m_streamCopy));
}

size_t Simulation::d_allocated() const {
	size_t nsz = 0;
	for (neuron_groups::const_iterator i = m_neurons.begin(); i != m_neurons.end(); ++i) {
		nsz += (*i)->d_allocated();
	}

	return m_firingStimulus.d_allocated() + m_currentStimulus.d_allocated()
			+ m_recentFiring.d_allocated() + nsz + m_firingBuffer.d_allocated() + m_cm.d_allocated();
}

param_t*
Simulation::setParameters(size_t pitch1, size_t pitch32, unsigned maxDelay) {
	param_t params;

	/* Need a max of at least 1 in order for local queue to be non-empty */
	params.maxDelay = std::max(1U, maxDelay);
	params.pitch1 = pitch1;
	params.pitch32 = pitch32;
	params.pitch64 = m_recentFiring.wordPitch();
	checkPitch(m_currentStimulus.wordPitch(), params.pitch32);
	checkPitch(m_firingBuffer.wordPitch(), params.pitch1);
	checkPitch(m_firingStimulus.wordPitch(), params.pitch1);
	;

	unsigned fbits = m_cm.fractionalBits();
	params.fixedPointScale = 1 << fbits;
	params.fixedPointFractionalBits = fbits;

	m_cm.setParameters(&params);

	void* d_ptr;
	d_malloc(&d_ptr, sizeof(param_t), "Global parameters");
	md_params = boost::shared_ptr<param_t>(static_cast<param_t*>(d_ptr), d_free);
	memcpyBytesToDevice(d_ptr, &params, sizeof(param_t));
	return md_params.get();
}

void Simulation::runKernel(cudaError_t status) {
	using boost::format;

	/* Check device assertions before /reporting/ errors. If we have an
	 * assertion failure we're likely to also have an error, but we'd like to
	 * know what the cause of it was. */
	m_deviceAssertions.check(m_timer.elapsedSimulation());

	if (status != cudaSuccess) {
		throw nemo::exception(NEMO_CUDA_INVOCATION_ERROR,
				str(
						format("Cuda error in cycle %u: %s") % m_timer.elapsedSimulation()
								% cudaGetErrorString(status)));
	}
}

void Simulation::prefire() {
	initLog();

	runKernel(
			::gather(m_streamCompute, m_timer.elapsedSimulation(), m_mapper.partitionCount(),
					md_partitionSize.get(), md_params.get(), m_current.deviceData(), m_cm.d_fcm(),
					m_cm.d_gqData(), m_cm.d_gqFill()));
}

void Simulation::fire() {
	CUDA_SAFE_CALL(cudaEventSynchronize(m_firingStimulusDone));
	CUDA_SAFE_CALL(cudaEventSynchronize(m_currentStimulusDone));
	/*! \todo if we separate input neurons from others, we could run input
	 * neurons in advance of gather completion */
	for (neuron_groups::const_iterator i = m_neurons.begin(); i != m_neurons.end(); ++i) {
		runKernel(
				(*i)->update(m_streamCompute, m_timer.elapsedSimulation(),
						m_mapper.partitionCount(), md_params.get(), md_partitionSize.get(),
						m_firingStimulus.d_buffer(), md_istim, m_current.deviceData(),
						m_firingBuffer.d_buffer(), md_nFired.get(), m_fired.deviceData(),
						m_cm.d_rcm()));
	}
	cudaEventRecord(m_eventFireDone, m_streamCompute);
}

void Simulation::postfire() {
	runKernel(
			::scatter(m_streamCompute, m_timer.elapsedSimulation(), m_mapper.partitionCount(),
					md_params.get(),
					// firing buffers
					md_nFired.get(), m_fired.deviceData(),
					// outgoing
					m_cm.d_outgoingAddr(), m_cm.d_outgoing(), m_cm.d_gqData(), m_cm.d_gqFill(),
					// local spike delivery
					m_lq.d_data(), m_lq.d_fill(), m_cm.d_ndData(), m_cm.d_ndFill()));

	if (m_stdp) {
		runKernel(
				::updateStdp(m_streamCompute, m_timer.elapsedSimulation(),
						m_mapper.partitionCount(), md_partitionSize.get(), md_params.get(),
						m_cm.d_rcm(), m_recentFiring.deviceData(), m_firingBuffer.d_buffer(),
						md_nFired.get(), m_fired.deviceData()));
	}

	cudaEventSynchronize(m_eventFireDone);
	m_firingBuffer.sync(m_streamCopy);

	/* Must clear stimulus pointers in case the low-level interface is used and
	 * the user does not provide any fresh stimulus */
	//! \todo make this a kind of step function instead?
	m_firingStimulus.reset();

	flushLog();
	endLog();

	m_timer.step();
}

#ifdef NEMO_BRIAN_ENABLED
std::pair<float*, float*>
Simulation::propagate_raw(uint32_t* d_fired, int nfired)
{
	assert_or_throw(!m_stdp, "Brian-specific function propagate only well-defined when STDP is not enabled");
	runKernel(::compact(m_streamCompute,
					md_partitionSize.get(),
					md_params.get(),
					m_mapper.partitionCount(),
					d_fired,
					md_nFired.get(),
					m_fired.deviceData()));
	postfire();
	prefire();
	float* acc = m_current.deviceData();
	return std::make_pair<float*, float*>(acc, acc+m_current.size());
	/* Brian does its own neuron update */
}
#endif

void Simulation::applyStdp(float reward) {
	using boost::format;

	if (!m_stdp) {
		throw exception(NEMO_LOGIC_ERROR, "applyStdp called, but no STDP model specified");
		return;
	}

	if (reward == 0.0f) {
		m_cm.clearStdpAccumulator();
	}
	else {
		initLog();

		unsigned fbits = m_cm.fractionalBits();
		if (fx_toFix(reward, fbits) == 0U && reward != 0) {
			throw nemo::exception(NEMO_INVALID_INPUT,
					str(
							format(
									"STDP reward rounded down to zero. The smallest valid reward is %f")
									% fx_toFloat(1U, fbits)));
		}

		::applyStdp(m_streamCompute, m_mapper.partitionCount(), md_partitionSize.get(), fbits,
				md_params.get(), m_cm.d_fcm(), m_cm.d_rcm(), m_stdp->minExcitatoryWeight(),
				m_stdp->maxExcitatoryWeight(), m_stdp->minInhibitoryWeight(),
				m_stdp->maxInhibitoryWeight(), reward);
		flushLog();
		endLog();
	}

	m_deviceAssertions.check(m_timer.elapsedSimulation());
}

void Simulation::setNeuron(unsigned h_neuron, unsigned nargs, const float args[]) {
	DeviceIdx d_neuron = m_mapper.deviceIdx(h_neuron);
	unsigned type = m_mapper.typeIdx(d_neuron.partition);
	m_neurons.at(type)->setNeuron(d_neuron, nargs, args);
}

const std::vector<synapse_id>&
Simulation::getSynapsesFrom(unsigned neuron) {
	return m_cm.getSynapsesFrom(neuron);
}

unsigned Simulation::getSynapseTarget(const synapse_id& synapse) const {
	return m_cm.getTarget(synapse);
}

unsigned Simulation::getSynapseDelay(const synapse_id& synapse) const {
	return m_cm.getDelay(synapse);
}

float Simulation::getSynapseWeight(const synapse_id& synapse) const {
	return m_cm.getWeight(elapsedSimulation(), synapse);
}

unsigned char Simulation::getSynapsePlastic(const synapse_id& synapse) const {
	return m_cm.getPlastic(synapse);
}

FiredList Simulation::readFiring() {
	return m_firingBuffer.readFiring();
}

void Simulation::setNeuronState(unsigned h_neuron, unsigned var, float val) {
	DeviceIdx d_neuron = m_mapper.deviceIdx(h_neuron);
	unsigned type = m_mapper.typeIdx(d_neuron.partition);
	return m_neurons.at(type)->setState(d_neuron, var, val);
}

void Simulation::setNeuronParameter(unsigned h_neuron, unsigned parameter, float val) {
	DeviceIdx d_neuron = m_mapper.deviceIdx(h_neuron);
	unsigned type = m_mapper.typeIdx(d_neuron.partition);
	return m_neurons.at(type)->setParameter(d_neuron, parameter, val);
}

float Simulation::getNeuronState(unsigned h_neuron, unsigned var) const {
	DeviceIdx d_neuron = m_mapper.deviceIdx(h_neuron);
	unsigned type = m_mapper.typeIdx(d_neuron.partition);
	return m_neurons.at(type)->getState(d_neuron, var);
}

float Simulation::getNeuronParameter(unsigned h_neuron, unsigned parameter) const {
	DeviceIdx d_neuron = m_mapper.deviceIdx(h_neuron);
	unsigned type = m_mapper.typeIdx(d_neuron.partition);
	return m_neurons.at(type)->getParameter(d_neuron, parameter);
}

float Simulation::getMembranePotential(unsigned h_neuron) const {
	DeviceIdx d_neuron = m_mapper.deviceIdx(h_neuron);
	unsigned type = m_mapper.typeIdx(d_neuron.partition);
	return m_neurons.at(type)->getMembranePotential(d_neuron);
}

void Simulation::finishSimulation() {
	cudaEventDestroy(m_eventFireDone);
	cudaEventDestroy(m_firingStimulusDone);
	cudaEventDestroy(m_currentStimulusDone);
	if (m_streamCompute)
		cudaStreamDestroy(m_streamCompute);
	if (m_streamCopy)
		cudaStreamDestroy(m_streamCopy);

	//! \todo perhaps clear device data here instead of in dtor
}

unsigned long Simulation::elapsedWallclock() const {
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	return m_timer.elapsedWallclock();
}

unsigned long Simulation::elapsedSimulation() const {
	return m_timer.elapsedSimulation();
}

void Simulation::resetTimer() {
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	m_timer.reset();
}

unsigned Simulation::getFractionalBits() const {
	return m_cm.fractionalBits();
}

Mapper Simulation::getMapper() const {
	return m_mapper;
}
} // end namespace cuda
} // end namespace nemo
