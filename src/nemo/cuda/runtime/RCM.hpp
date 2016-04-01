#ifndef NEMO_CUDA_RUNTIME_RCM_HPP
#define NEMO_CUDA_RUNTIME_RCM_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/shared_array.hpp>
#include <nemo/cuda/rcm.cu_h>
#include <nemo/cuda/kernel.cu_h>
//! \todo define DeviceIdx somewhere else
#include <nemo/cuda/Mapper.hpp>

namespace nemo {

	namespace construction {
		template<class I, class D, size_t W> class RCM;
	}

	namespace cuda {

		namespace runtime {


/*! \brief Runtime representation of the reverse connectivity matrix
 *
 * The main data is stored in a structure-of-arrays format. Lookups into this
 * data is done via an index.
 *
 * The index is logically a map from neuron to a list of warp numbers (row),
 * where the warp number is an offset into the reverse connectivity matrix.
 *
 * The length of the different rows may differ greatly. In order to save memory
 * the index itself is stored in a compact form where
 *
 * - each row is stored in a contiguous chunk of memory
 * - the extent of each row in the index (start and length) is stored in a
 *   separate fixed-size table
 *
 * \see nemo::construction::RCM
 */
class RCM
{
	public :

		typedef boost::tuple<pidx_t, nidx_t> key_t;
		typedef nemo::construction::RCM<key_t, uint32_t, WARP_SIZE> construction_t;

		RCM() : m_allocated(0), m_planeSize(0) {}

		/*! Create an RCM on the device.
		 *
		 * The (host) data in \a rcm is cleared as a side effect, rendering the
		 * object essentially void. We clear this data at the earliest possible
		 * moment since the data structures involved can be quite large.
		 */
		RCM(size_t partitionCount, construction_t& rcm);

		/*! \return number of bytes allocated on the device */
		size_t d_allocated() const { return m_allocated; }

		void clearAccumulator();

		/*! \return RCM device pointers */
		rcm_dt* d_rcm() { return &md_rcm; }

	private :

		boost::shared_array<rsynapse_t> md_data;
		boost::shared_array<uint32_t> md_forward;
		boost::shared_array<float> md_weights;
		boost::shared_array<weight_dt> md_accumulator;
		boost::shared_array<rcm_address_t> md_index;
		boost::shared_array<rcm_index_address_t> md_metaIndex;

		/* POD struct to pass to the kernel */
		rcm_dt md_rcm;

		/*! Bytes of allocated device memory */
		size_t m_allocated;

		/*! Size (words) of each plane of data in the RCM */
		size_t m_planeSize;
};

		}
	}
}

#endif
