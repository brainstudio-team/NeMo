#ifndef NEMO_RUNTIME_RCM_HPP
#define NEMO_RUNTIME_RCM_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/unordered_map.hpp>
#include <nemo/construction/RCM.hpp>
#include <nemo/types.hpp>

namespace nemo {

	namespace construction {
		template<class I, class D, size_t W> class RCM;
	}

	namespace runtime {

class NEMO_BASE_DLL_PUBLIC RCM
{
	private :

		typedef boost::unordered_map<nidx_t, std::vector<size_t> > warp_map;

	public :

		enum { WIDTH = 32 };

		typedef nemo::construction::RCM<nidx_t, RSynapse, WIDTH> construction_t;

		typedef warp_map::const_iterator warp_iterator;

		/*! Create a runtime RCM 
		 *
		 * The data in the constrution-time RCM are freed as a side effect. In
		 * effect this class takes ownership of the underlying data
		 */
		explicit RCM(construction_t& rcm);

		/*! Set accumulator field to all zero */
		void clearAccumulator();

		/*! \return iterator to beginning of all warp indices */
		warp_iterator warp_begin() const { return m_warps.begin(); }

		/*! \return iterator to end of all warp indices */
		warp_iterator warp_end() const { return m_warps.end(); }

		/*! \return reference to the warps of a specific target neuron */
		const std::vector<size_t>& warps(nidx_t target) const;

		/*! \return number of incoming synapses for the given target neuron */
		unsigned indegree(nidx_t target) const;

		/*! \return a single warp of reverse synape data */
		const RSynapse* data(size_t warp) const;

		/*! \return a single warp of accumulator data */
		fix_t* accumulator(size_t warp);

		/*! \return a single warp of FCM addresses */
		const uint32_t* forward(size_t warp) const; 

		/*! \return a single warp of weights */
		const float* weight(size_t warp) const;

	private :

		warp_map m_warps;

		boost::unordered_map<nidx_t, unsigned> m_indegree;

		/*! Main reverse synapse data: source partition, source neuron, delay */
		std::vector<RSynapse> m_data;

		/*! Forward addressing, word offset into the FCM for each synapse */
		std::vector<uint32_t> m_forward;

		/*! The weights are \em optionally stored in the reverse format as
		 * well. This is normally not done as the weights are normally used
		 * only in spike delivery which uses the forward form. However, special
		 * neuron type plugins may require this. */
		std::vector<float> m_weights;

		std::vector<fix_t> m_accumulator;
};

	} // end namespace runtime
} // end namespace nemo

#endif
