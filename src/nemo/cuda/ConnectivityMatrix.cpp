/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "ConnectivityMatrix.hpp"

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <cmath>

#include <boost/format.hpp>

#include <nemo/util.h>
#include <nemo/ConfigurationImpl.hpp>
#include <nemo/fixedpoint.hpp>
#include <nemo/synapse_indices.hpp>
#include <nemo/construction/RCM.hpp>
#include <nemo/cuda/construction/FcmIndex.hpp>
#include <nemo/cuda/runtime/Delays.hpp>

#include "exception.hpp"
#include "fcm.cu_h"
#include "kernel.hpp"
#include "device_memory.hpp"
#include "parameters.cu_h"


namespace nemo {
	namespace cuda {



ConnectivityMatrix::ConnectivityMatrix(
		const nemo::network::Generator& net,
		const nemo::ConfigurationImpl& conf,
		const Mapper& mapper) :
	m_mapper(mapper),
	m_maxDelay(0),
	mhf_weights(WARP_SIZE, 0),
	md_fcmPlaneSize(0),
	md_fcmAllocated(0),
	m_fractionalBits(conf.fractionalBits()),
	m_writeOnlySynapses(conf.writeOnlySynapses())
{
	//! \todo change synapse_t, perhaps to nidx_dt
	std::vector<synapse_t> hf_targets(WARP_SIZE, INVALID_FORWARD_SYNAPSE);
	construction::FcmIndex fcm_index;
	typedef runtime::RCM::key_t rcm_key;
	nemo::construction::RCM<rcm_key, uint32_t, WARP_SIZE> h_rcm(conf, net, INVALID_REVERSE_SYNAPSE);

	bool logging = conf.loggingEnabled();

	if(logging) {
		//! \todo log to correct output stream
		std::cout << "Using fixed point format Q"
			<< 31-m_fractionalBits << "." << m_fractionalBits << " for weights\n";
	}

	/*! \todo perhaps we should reserve a large chunk of memory for
	 * hf_targets/h_weights in advance? It's hard to know exactly how much is
	 * needed, though, due the organisation in warp-sized chunks. */

	size_t nextFreeWarp = 1; // leave space for null warp at beginning
	for(network::synapse_iterator si = net.synapse_begin();
			si != net.synapse_end(); ++si) {
		const Synapse& s = *si;
		setMaxDelay(s);
		DeviceIdx source = mapper.deviceIdx(s.source);
		DeviceIdx target = mapper.deviceIdx(s.target());
		size_t f_addr = addForward(s, source, target, nextFreeWarp, fcm_index, hf_targets, mhf_weights);
		rcm_key rkey(target.partition, target.neuron);
		rsynapse_t rsynapse = make_rsynapse(source.partition, source.neuron, s.delay);
		h_rcm.addSynapse(rkey, rsynapse, s, f_addr);
		if(!m_writeOnlySynapses) {
			addAuxillary(s, f_addr);
		}
	}

	verifySynapseTerminals(m_cmAux, mapper);

	moveFcmToDevice(nextFreeWarp, hf_targets, mhf_weights);
	hf_targets.clear();

	md_rcm = runtime::RCM(mapper.partitionCount(), h_rcm);

	md_delays.reset(new runtime::Delays(mapper.partitionCount(), fcm_index));

	m_outgoing = Outgoing(mapper.partitionCount(), fcm_index);
	m_gq.allocate(mapper.partitionCount(), m_outgoing.maxIncomingWarps(), 1.0);

	if(conf.loggingEnabled()) {
		printMemoryUsage(std::cout);
		// fcm_index.reportWarpSizeHistogram(std::cout);
	}
}



void
ConnectivityMatrix::setMaxDelay(const Synapse& s)
{
	using boost::format;

	m_maxDelay = std::max(m_maxDelay, s.delay);

	if(s.delay < 1) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Neuron %u has synapses with delay < 1 (%u)") % s.source % s.delay));
	}
	if(s.delay > MAX_DELAY) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Neuron %u has synapses with delay %ums. The CUDA backend supports a maximum of %ums")
						% s.source % s.delay % MAX_DELAY));
	}
}



size_t
ConnectivityMatrix::addForward(
		const Synapse& s,
		const DeviceIdx& d_source,
		const DeviceIdx& d_target,
		size_t& nextFreeWarp,
		construction::FcmIndex& index,
		std::vector<synapse_t>& h_targets,
		std::vector<weight_dt>& h_weights)
{
	SynapseAddress addr = index.addSynapse(d_source, d_target.partition, s.delay, nextFreeWarp);

	if(addr.synapse == 0 && addr.row == nextFreeWarp) {
		nextFreeWarp += 1;
		/* Resize host buffers to accomodate the new warp. This
		 * allocation scheme could potentially result in a
		 * large number of reallocations, so we might be better
		 * off allocating larger chunks here */
		h_targets.resize(nextFreeWarp * WARP_SIZE, INVALID_FORWARD_SYNAPSE);
		h_weights.resize(nextFreeWarp * WARP_SIZE, 0);
	}

	size_t f_addr = addr.row * WARP_SIZE + addr.synapse;
	//! \todo range check this address

	assert(d_target.neuron < MAX_PARTITION_SIZE);
	h_targets.at(f_addr) = d_target.neuron;
	h_weights.at(f_addr) = fx_toFix(s.weight(), m_fractionalBits);
	return f_addr;
}



/*! \note We could verify the synapse terminals during FCM construction. This
 * was found to be somewhat slower, however, as we then end up performing
 * around twice as many checks (since each source is tested many times).
 *
 * If synapses are configured to be write-only, this check will pass, since the
 * CM is empty.
 */
void
ConnectivityMatrix::verifySynapseTerminals(const aux_map& cm, const Mapper& mapper)
{
	using boost::format;

	for(aux_map::const_iterator ni = cm.begin(); ni != cm.end(); ++ni) {

		nidx_t source = ni->first;

		if(!mapper.existingGlobal(source)) {
			throw nemo::exception(NEMO_INVALID_INPUT,
					str(format("Invalid synapse source neuron %u") % source));
		}

		aux_row row = ni->second;

#ifndef NDEBUG
		assert(m_synapsesPerNeuron[source] == row.size());
#endif

		for(aux_row::const_iterator si = row.begin(); si != row.end(); ++si) {
			nidx_t target = si->target();
			if(!mapper.existingGlobal(target)) {
				throw nemo::exception(NEMO_INVALID_INPUT,
						str(format("Invalid synapse target neuron %u (source: %u)") % target % source));
			}
		}
	}
}



void
ConnectivityMatrix::moveFcmToDevice(size_t totalWarps,
		const std::vector<synapse_t>& h_targets,
		const std::vector<weight_dt>& h_weights)
{
	md_fcmPlaneSize = totalWarps * WARP_SIZE;
	size_t bytes = md_fcmPlaneSize * 2 * sizeof(synapse_t);

	void* d_fcm;
	d_malloc(&d_fcm, bytes, "fcm");
	md_fcm = boost::shared_ptr<synapse_t>(static_cast<synapse_t*>(d_fcm), d_free);
	md_fcmAllocated = bytes;

	memcpyToDevice(md_fcm.get() + md_fcmPlaneSize * FCM_ADDRESS, h_targets, md_fcmPlaneSize);
	memcpyToDevice(md_fcm.get() + md_fcmPlaneSize * FCM_WEIGHT,
			reinterpret_cast<const synapse_t*>(&h_weights[0]), md_fcmPlaneSize);
}



void
ConnectivityMatrix::printMemoryUsage(std::ostream& out) const
{
	const size_t MEGA = 1<<20;
	out << "Memory usage on device:\n";
	out << "\tforward matrix: " << (md_fcmAllocated / MEGA) << "MB\n";
	out << "\treverse matrix: " << (md_rcm.d_allocated() / MEGA) << "MB\n";
	out << "\tglobal queue: " << (m_gq.allocated() / MEGA) << "MB\n";
	out << "\tdelays: " << md_delays->allocated() / MEGA << "MB\n";
	out << "\toutgoing: " << (m_outgoing.allocated() / MEGA) << "MB\n" << std::endl;
}



/* Data used when user reads FCM back from device. These are indexed by
 * (global) synapse ids, and are thus filled in a random order. To populate
 * these in a single pass over the input, resize on insertion.  The synapse ids
 * are required to form a contigous range, so every element should be assigned
 * exactly once. */
void
ConnectivityMatrix::addAuxillary(const Synapse& s, size_t addr)
{
	id32_t id = s.id();
	aux_row& row= m_cmAux[s.source];
	if(id >= row.size()) {
		row.resize(id+1);
	}
	row.at(id) = AxonTerminalAux(s, addr);
#ifndef NDEBUG
	m_synapsesPerNeuron[s.source] += 1;
#endif
}



const std::vector<synapse_id>&
ConnectivityMatrix::getSynapsesFrom(unsigned source)
{
	using boost::format;

	if(m_writeOnlySynapses) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"Cannot read synapse state if simulation configured with write-only synapses");
	}

	/* The relevant data is stored in the auxillary synapse map, which is
	 * already indexed in global neuron indices. Therefore, no need to map into
	 * device ids */
	size_t nSynapses = 0;
	aux_map::const_iterator iRow = m_cmAux.find(source);
	if(iRow == m_cmAux.end()) {
		if(!m_mapper.existingGlobal(source)) {
			throw nemo::exception(NEMO_INVALID_INPUT,
					str(format("Non-existing source neuron id (%u) in synapse id query") % source));
		}
		/* else just leave nSynapses at zero */
	} else {
		/* Synapse ids are consecutive */
		nSynapses = iRow->second.size();
	}

	m_queriedSynapseIds.resize(nSynapses);

	for(size_t iSynapse = 0; iSynapse < nSynapses; ++iSynapse) {
		m_queriedSynapseIds[iSynapse] = make_synapse_id(source, iSynapse);
	}

	return m_queriedSynapseIds;
}



const std::vector<weight_dt>&
ConnectivityMatrix::syncWeights(cycle_t cycle) const
{
	if(cycle != m_lastWeightSync && !mhf_weights.empty()) {
		//! \todo refine this by only doing the minimal amount of copying
		memcpyFromDevice(reinterpret_cast<synapse_t*>(&mhf_weights[0]),
					md_fcm.get() + FCM_WEIGHT * md_fcmPlaneSize,
					md_fcmPlaneSize);
		m_lastWeightSync = cycle;
	}
	return mhf_weights;
}



const AxonTerminalAux&
ConnectivityMatrix::axonTerminalAux(const synapse_id& id) const
{
	using boost::format;

	if(m_writeOnlySynapses) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"Cannot read synapse state if simulation configured with write-only synapses");
	}

	aux_map::const_iterator it = m_cmAux.find(neuronIndex(id));
	if(it == m_cmAux.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Non-existing neuron id (%u) in synapse query") % neuronIndex(id)));
	}
	return (it->second)[synapseIndex(id)];
}



float
ConnectivityMatrix::getWeight(cycle_t cycle, const synapse_id& id) const
{
	size_t addr = axonTerminalAux(id).addr();
	const std::vector<weight_dt>& h_weights = syncWeights(cycle);
	return fx_toFloat(h_weights[addr], m_fractionalBits);;
}



unsigned
ConnectivityMatrix::getTarget(const synapse_id& id) const
{
	return axonTerminalAux(id).target();
}


unsigned
ConnectivityMatrix::getDelay(const synapse_id& id) const
{
	return axonTerminalAux(id).delay();
}


unsigned char
ConnectivityMatrix::getPlastic(const synapse_id& id) const
{
	return axonTerminalAux(id).plastic();
}


void
ConnectivityMatrix::clearStdpAccumulator()
{
	md_rcm.clearAccumulator();
}



size_t
ConnectivityMatrix::d_allocated() const
{
	return md_fcmAllocated
		+ md_rcm.d_allocated()
		+ m_gq.allocated()
		+ md_delays->allocated()
		+ m_outgoing.allocated();
}



void
ConnectivityMatrix::setParameters(param_t* params) const
{
	m_outgoing.setParameters(params);
	params->fcmPlaneSize = md_fcmPlaneSize;
}



	} // end namespace cuda
} // end namespace nemo
