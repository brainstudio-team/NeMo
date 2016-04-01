/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

//! \file StdpProcess.cpp

#include "StdpProcess.hpp"
#include "StdpFunction.hpp"
#include "fixedpoint.hpp"
#include "bitops.h"

namespace nemo {


StdpProcess::StdpProcess(const StdpFunction& fn, unsigned fbits) :
	m_minExcitatoryWeight(fx_toFix(fn.minExcitatoryWeight(), fbits)),
	m_maxExcitatoryWeight(fx_toFix(fn.maxExcitatoryWeight(), fbits)),
	m_minInhibitoryWeight(fx_toFix(fn.minInhibitoryWeight(), fbits)),
	m_maxInhibitoryWeight(fx_toFix(fn.maxInhibitoryWeight(), fbits))
{
	for(std::vector<float>::const_iterator i = fn.prefire().begin();
			i != fn.prefire().end(); ++i) {
		m_fnPre.push_back(fx_toFix(*i, fbits));
	}

	for(std::vector<float>::const_iterator i = fn.postfire().begin();
			i != fn.postfire().end(); ++i) {
		m_fnPost.push_back(fx_toFix(*i, fbits));
	}

	m_potentiationBits = fn.potentiationBits(); 
	m_depressionBits = fn.depressionBits();

	int prePostWindow = fn.prefire().size();
	m_postPreWindow = fn.postfire().size();

	m_prePostBits = (~(uint64_t(~0) << uint64_t(prePostWindow))) << uint64_t(m_postPreWindow);
	m_postPreBits = ~(uint64_t(~0) << uint64_t(m_postPreWindow));

	m_postFireMask = uint64_t(1) << m_postPreWindow;
}



fix_t
StdpProcess::lookupPre(int dt) const
{
	return m_fnPre.at(dt);
}



fix_t
StdpProcess::lookupPost(int dt) const
{
	return m_fnPost.at(dt);
}





unsigned
StdpProcess::closestPreFire(uint64_t arrivals) const
{
	uint64_t validArrivals = arrivals & m_prePostBits;
	int dt =  ctz64(validArrivals >> m_postPreWindow);
	return validArrivals ? (unsigned) dt : STDP_NO_APPLICATION;
}


unsigned
StdpProcess::closestPostFire(uint64_t arrivals) const
{
	uint64_t validArrivals = arrivals & m_postPreBits;
	int dt = clz64(validArrivals << uint64_t(64 - m_postPreWindow));
	return validArrivals ? (unsigned) dt : STDP_NO_APPLICATION;
}



fix_t
StdpProcess::updateRegion(uint64_t arrivals, nidx_t source, nidx_t target) const
{
	/* The potentiation can happen on either side of the firing. We want to
	 * find the one closest to the firing. We therefore need to compute the
	 * prefire and postfire dt's separately. */
	fix_t w_diff = 0;

	if(arrivals) {

		unsigned dt_pre = closestPreFire(arrivals);
		unsigned dt_post = closestPostFire(arrivals);

		if(dt_pre < dt_post) {
			//! \todo inline
			w_diff = lookupPre(dt_pre);
			//LOG("c%u %s: %u -> %u %+f (dt=%d)\n",
			//		elapsedSimulation(), "ltp", source, target, w_diff, dt_pre);
		} else if(dt_post < dt_pre) {
			w_diff = lookupPost(dt_post);
			//LOG("c%u %s: %u -> %u %+f (dt=%d)\n",
			//		elapsedSimulation(), "ltd", source, target, w_diff, dt_post);
		}
		// if neither is applicable dt_post == dt_pre == STDP_NO_APPLICATION
	}
	return w_diff;
}


fix_t
StdpProcess::weightChange(uint64_t preFiring, nidx_t pre, nidx_t post) const
{
	uint64_t p_spikes = preFiring & m_potentiationBits;
	uint64_t d_spikes = preFiring & m_depressionBits;
	return 
		updateRegion(p_spikes, pre, post) +
		updateRegion(d_spikes, pre, post);
}



fix_t
StdpProcess::updatedWeight(fix_t w_old, fix_t w_diff) const
{
	fix_t w_new = 0;
	if(w_old > 0) {
		w_new = std::min(m_maxExcitatoryWeight, std::max(w_old + w_diff, m_minExcitatoryWeight));
	} else if(w_old < 0) {
		w_new = std::min(m_minInhibitoryWeight, std::max(w_old + w_diff, m_maxInhibitoryWeight));
	}
	return w_new;
}



} // end namespace nemo
