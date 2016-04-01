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
#include <utility>
#include <stdlib.h>

#include <boost/tuple/tuple_comparison.hpp>
#include <boost/format.hpp>

#include <nemo/config.h>
#include <nemo/network/Generator.hpp>
#include "ConfigurationImpl.hpp"
#include "exception.hpp"
#include "fixedpoint.hpp"
#include "synapse_indices.hpp"


namespace nemo {


Row::Row(const std::vector<FAxonTerminal>& ss) :
	len(ss.size())
{
	void* ptr;
#ifdef HAVE_POSIX_MEMALIGN
	//! \todo factor out the memory aligned allocation
	int error = posix_memalign(&ptr,
			ASSUMED_CACHE_LINE_SIZE,
			ss.size()*sizeof(FAxonTerminal));
	if(error) {
		throw nemo::exception(NEMO_ALLOCATION_ERROR, "Failed to allocate CM row");
	}
#else
	ptr = malloc(ss.size()*sizeof(FAxonTerminal));
#endif
	FAxonTerminal* term = static_cast<FAxonTerminal*>(ptr);
	std::copy(ss.begin(), ss.end(), term);
	data = boost::shared_array<FAxonTerminal>(term, free);
}



/* Insert into vector, resizing if appropriate */
template<typename T>
void
insert(size_t idx, const T& val, std::vector<T>& vec)
{
	if(idx >= vec.size()) {
		vec.resize(idx+1);
	}
	vec.at(idx) = val;
}



ConnectivityMatrix::ConnectivityMatrix(
		const network::Generator& net,
		const ConfigurationImpl& conf,
		const mapper_t& mapper) :
	m_mapper(mapper),
	m_fractionalBits(conf.fractionalBits()),
	m_maxDelay(0),
	m_writeOnlySynapses(conf.writeOnlySynapses())
{
	if(conf.stdpFunction()) {
		m_stdp = StdpProcess(conf.stdpFunction().get(), m_fractionalBits);
	}

	construction::RCM<nidx_t, RSynapse, 32> m_racc(conf, net, RSynapse(~0U,0));
	network::synapse_iterator i = net.synapse_begin();
	network::synapse_iterator i_end = net.synapse_end();

	for( ; i != i_end; ++i) {
		nidx_t source = mapper.localIdx(i->source);
		nidx_t target = mapper.localIdx(i->target());
		sidx_t sidx = addSynapse(source, target, *i);
		m_racc.addSynapse(target, RSynapse(source, i->delay), *i, sidx);
	}

	//! \todo avoid two passes here
	bool verifySources = true;
	finalizeForward(mapper, verifySources);
	m_rcm.reset(new runtime::RCM(m_racc));
}

ConnectivityMatrix::ConnectivityMatrix(
		const network::Generator& net,
		const ConfigurationImpl& conf,
		const mapper_t& mapper,
		const bool verifySources) :
	m_mapper(mapper),
	m_fractionalBits(conf.fractionalBits()),
	m_maxDelay(0),
	m_writeOnlySynapses(conf.writeOnlySynapses())
{
	if(conf.stdpFunction()) {
		m_stdp = StdpProcess(conf.stdpFunction().get(), m_fractionalBits);
	}

	construction::RCM<nidx_t, RSynapse, 32> m_racc(conf, net, RSynapse(~0U,0));
	network::synapse_iterator i = net.synapse_begin();
	network::synapse_iterator i_end = net.synapse_end();

	for( ; i != i_end; ++i) {
		nidx_t source = mapper.localIdx(i->source);
		nidx_t target = mapper.localIdx(i->target());
		sidx_t sidx = addSynapse(source, target, *i);
		m_racc.addSynapse(target, RSynapse(source, i->delay), *i, sidx);
	}

	//! \todo avoid two passes here
	finalizeForward(mapper, verifySources);
	m_rcm.reset(new runtime::RCM(m_racc));
}



sidx_t
ConnectivityMatrix::addSynapse(nidx_t source, nidx_t target, const Synapse& s)
{
	delay_t delay = s.delay;
	fix_t weight = fx_toFix(s.weight(), m_fractionalBits);

	fidx_t fidx(source, delay);
	row_t& row = m_acc[fidx];
	sidx_t sidx = row.size();
	row.push_back(FAxonTerminal(target, weight));

	//! \todo could do this on finalize pass, since there are fewer steps there
	m_delaysAcc.addDelay(source, delay);

	if(!m_writeOnlySynapses) {
		/* The auxillary synapse maps always uses the global (user-specified)
		 * source and target neuron ids, since it's used for lookups basd on
		 * these global ids */
		aux_row& auxRow = m_cmAux[s.source];
		insert(s.id(), AxonTerminalAux(sidx, delay, s.plastic() != 0), auxRow);
	}
	return sidx;
}



/* The fast lookup is indexed by source and delay. */
void
ConnectivityMatrix::finalizeForward(const mapper_t& mapper, bool verifySources)
{
	m_maxDelay = m_delaysAcc.maxDelay();
	m_delays.init(m_delaysAcc);
	m_delaysAcc.clear();

	if(m_acc.empty()) {
		return;
	}

	/* This relies on lexicographical ordering of tuple */
	nidx_t maxSourceIdx = m_acc.rbegin()->first.get<0>();
	m_cm.resize((maxSourceIdx+1) * m_maxDelay);

	//! \todo change order here: default to Row() in all location, and then just iterate over map
	for(nidx_t n=0; n <= maxSourceIdx; ++n) {
		for(delay_t d=1; d <= m_maxDelay; ++d) {

#if 0
			if(d < 1) {
				//! \todo make sure to report global index again here
				throw nemo::exception(NEMO_INVALID_INPUT,
						str(format("Neuron %u has synapses with delay < 1 (%u)") % source % delay));
			}
#endif

			std::map<fidx_t, row_t>::const_iterator row = m_acc.find(fidx_t(n, d));
			if(row != m_acc.end()) {
				// TODO: the next line was included by Andreas, but commented out by Pavlos. What should we do?
				//verifySynapseTerminals(row->first, row->second, mapper, verifySources);
				m_cm.at(addressOf(n,d)) = Row(row->second);
			} else {
				/* Insertion into map does not invalidate existing iterators */
				m_cm.at(addressOf(n,d)) = Row(); // defaults to empty row
			}
			//! \todo can delete the map now
		}
	}
}



void
ConnectivityMatrix::verifySynapseTerminals(fidx_t idx,
		const row_t& row,
		const mapper_t& mapper,
		bool verifySource) const
{
	using boost::format;

	if(verifySource) {
		nidx_t source = idx.get<0>();
		if(!mapper.existingLocal(source)) {
			throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Invalid synapse source neuron %u") % source));
		}
	}

	for(size_t s=0; s < row.size(); ++s) {
		nidx_t target = row.at(s).target;
		if(!mapper.existingLocal(target)) {
			throw nemo::exception(NEMO_INVALID_INPUT,
					str(format("Invalid synapse target neuron %u (source: %u)") % target % idx.get<0>()));
		}
	}
}



void
ConnectivityMatrix::accumulateStdp(const std::vector<uint64_t>& recentFiring)
{
	if(!m_stdp) {
		return;
	}

	for(runtime::RCM::warp_iterator i = m_rcm->warp_begin();
			i != m_rcm->warp_end(); i++) {

		const nidx_t target = i->first;
		const std::vector<size_t>& warps = i->second;

		if(recentFiring[target] & m_stdp->postFireMask()) {

			size_t remaining = m_rcm->indegree(target);

			for(std::vector<size_t>::const_iterator wi = warps.begin();
					wi != warps.end(); ++wi) {

				const RSynapse* rdata_ptr = m_rcm->data(*wi);
				fix_t* accumulator = m_rcm->accumulator(*wi);

				for(unsigned s=0; s < m_rcm->WIDTH && remaining--; s++) {
					const RSynapse& rdata = rdata_ptr[s];
					uint64_t preFiring = recentFiring[rdata.source] >> rdata.delay;
					fix_t w_diff = m_stdp->weightChange(preFiring, rdata.source, target);
					if(w_diff != 0.0) {
						accumulator[s] += w_diff;
					}
				}
			}
		}
	}
}



fix_t*
ConnectivityMatrix::weight(const RSynapse& s, uint32_t sidx) const
{
	const Row& row = m_cm.at(addressOf(s.source, s.delay));
	assert(sidx < row.len);
	return &row.data[sidx].weight;
}



void
ConnectivityMatrix::applyStdp(float reward)
{
	using boost::format;

	if(!m_stdp) {
		throw exception(NEMO_LOGIC_ERROR, "applyStdp called, but no STDP model specified");
	}
	fix_t fx_reward = fx_toFix(reward, m_fractionalBits);

	if(fx_reward == 0U && reward != 0) {
		throw exception(NEMO_INVALID_INPUT,
				str(format("STDP reward rounded down to zero. The smallest valid reward is %f")
					% fx_toFloat(1U, m_fractionalBits)));
	}

	if(fx_reward == 0) {
		m_rcm->clearAccumulator();
	}

	for(runtime::RCM::warp_iterator i = m_rcm->warp_begin();
			i != m_rcm->warp_end(); i++) {

		const nidx_t target = i->first;
		const std::vector<size_t>& warps = i->second;
		size_t remaining = m_rcm->indegree(target);

		for(std::vector<size_t>::const_iterator wi = warps.begin();
				wi != warps.end(); ++wi) {

			const RSynapse* rdata_ptr = m_rcm->data(*wi);
			const uint32_t* forward = m_rcm->forward(*wi);
			fix_t* accumulator = m_rcm->accumulator(*wi);

			for(unsigned s=0; s < m_rcm->WIDTH && remaining--; s++) {

				const RSynapse& rsynapse = rdata_ptr[s];
				fix_t* w_old = weight(rsynapse, forward[s]);
				fix_t w_new = m_stdp->updatedWeight(*w_old, fx_mul(fx_reward, accumulator[s], m_fractionalBits));

				if(*w_old != w_new) {
#ifdef DEBUG_TRACE
					fprintf(stderr, "stdp (%u -> %u) %f %+f = %f\n",
							rsynapse.source, target, fx_toFloat(*w_old, m_fractionalBits),
							fx_toFloat(fx_mul(reward, accumulator[s], m_fractionalBits), m_fractionalBits),
							fx_toFloat(w_new, m_fractionalBits));
#endif
					*w_old = w_new;
				}
				accumulator[s] = 0;
			}
		}
	}
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
	 * local ids */
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



const Row&
ConnectivityMatrix::getRow(nidx_t source, delay_t delay) const
{
	return m_cm.at(addressOf(source, delay));
}



const AxonTerminalAux&
ConnectivityMatrix::axonTerminalAux(const synapse_id& id) const
{
	using boost::format;

	if(m_writeOnlySynapses) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"Cannot read synapse state if simulation configured with write-only synapses");
	}

	nidx_t neuron = neuronIndex(id);
	aux_map::const_iterator it = m_cmAux.find(neuron);
	if(it == m_cmAux.end()) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				str(format("Non-existing neuron id (%u) in synapse query") % neuron));
	}
	return it->second.at(synapseIndex(id));
}



unsigned
ConnectivityMatrix::getTarget(const synapse_id& id) const
{
	const AxonTerminalAux& s = axonTerminalAux(id);
	nidx_t l_source = m_mapper.localIdx(neuronIndex(id));
	nidx_t l_target = m_cm[addressOf(l_source, s.delay)].data[s.idx].target;
	return m_mapper.globalIdx(l_target);
}



float
ConnectivityMatrix::getWeight(const synapse_id& id) const
{
	const AxonTerminalAux& s = axonTerminalAux(id);
	nidx_t l_source = m_mapper.localIdx(neuronIndex(id));
	const Row& row = m_cm[addressOf(l_source, s.delay)];
	assert(s.idx < row.len);
	fix_t w = row.data[s.idx].weight;
	return fx_toFloat(w, m_fractionalBits);
}



unsigned
ConnectivityMatrix::getDelay(const synapse_id& id) const
{
	return axonTerminalAux(id).delay;
}



unsigned char
ConnectivityMatrix::getPlastic(const synapse_id& id) const
{
	return axonTerminalAux(id).plastic;
}



ConnectivityMatrix::delay_iterator
ConnectivityMatrix::delay_begin(nidx_t source) const
{
	return m_delays.begin(source);
}



ConnectivityMatrix::delay_iterator
ConnectivityMatrix::delay_end(nidx_t source) const
{
	return m_delays.end(source);
}


} // namespace nemo
