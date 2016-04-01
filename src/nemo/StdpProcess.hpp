#ifndef NEMO_STDP_PROCESS_HPP
#define NEMO_STDP_PROCESS_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */


#include <vector>

#include <nemo/config.h>
#include <nemo/internal_types.h>

namespace nemo {

class StdpFunction;


class NEMO_BASE_DLL_PUBLIC StdpProcess
{
	public:

		StdpProcess(const StdpFunction&, unsigned fractionalBits);

		/* Return weight change */
		fix_t weightChange(uint64_t spikes, nidx_t source, nidx_t target) const;

		/*! \return updated weight given the weight difference. This might be
		 * different than w_old + w_diff due saturation and disallowed sign
		 * change. */
		fix_t updatedWeight(fix_t w_old, fix_t w_diff) const;

		/* Bitmask indicating the position of the postsynaptic firing within
		 * the window. This is useful in determining when to compute STDP updates. */
		uint64_t postFireMask() const { return m_postFireMask; }

	private :

		fix_t updateRegion(uint64_t spikes, nidx_t source, nidx_t target) const;

		/*! \return value of STDP function at the given (negative) value of dt */
		fix_t lookupPre(int dt) const;

		/*! \return value of STDP function at the given (positive) value of dt */
		fix_t lookupPost(int dt) const;

		unsigned closestPreFire(uint64_t arrivals) const;
		unsigned closestPostFire(uint64_t arrivals) const;

		static const unsigned STDP_NO_APPLICATION = unsigned(~0);

	private:

		/* pre-fire part of STDP function, from dt=-1 and down */
		std::vector<fix_t> m_fnPre;

		/* pre-fire part of STDP function, from dt=+1 and up */
		std::vector<fix_t> m_fnPost;

		int m_postPreWindow;

		uint64_t m_potentiationBits;
		uint64_t m_depressionBits; 

		/* Bitmasks indicating the pre-post and post-pre regions of the window */
		uint64_t m_prePostBits;
		uint64_t m_postPreBits;

		uint64_t m_postFireMask;

		fix_t m_minExcitatoryWeight;
		fix_t m_maxExcitatoryWeight;
		fix_t m_minInhibitoryWeight;
		fix_t m_maxInhibitoryWeight;
};

} // namespace nemo


#endif
