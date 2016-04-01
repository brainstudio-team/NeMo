#ifndef NEMO_CUDA_FIRING_DELAYS
#define NEMO_CUDA_FIRING_DELAYS

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/shared_array.hpp>
#include <nemo/cuda/types.h>

namespace nemo {
	namespace cuda {

		namespace construction {
			class FcmIndex;
		}

		namespace runtime {


/*! Mapping from neurons to delays 
 *
 * The data structure is fixed-size with a size set based on the maximum delay
 * in the network. Addressing is thus trivial. There is one row for each
 * neuron, and the initial entries contains the relevant delays (zero-based)
 * while the final entries are padding (unless all delays are set). A separate
 * vector of row lengths indicate where padding begins.
 *
 * The size of this data structure is ncount x d_max x 2B. In large example
 * would be 128k neurons and 1024 delays which would require a 256MB.
 *
 * The size could be reduced by setting the width based on the actual max delay.
 *
 * \todo set the width based on actual max delay (put the pitch in the global parameters)
 *
 * \author Andreas K. Fidjeland
 */
class Delays
{
	public :

		Delays(unsigned partitionCount, const construction::FcmIndex& index); 

		/*! \return device pointer to delays data */
		delay_dt* d_data() const { return md_data.get(); }

		/*! \return device pointer to delays fill data */
		unsigned* d_fill() const { return md_fill.get(); }

		/*! \return bytes of allocated memory */
		size_t allocated() const { return mb_allocated; }

	private :

		boost::shared_array<delay_dt> md_data;
		boost::shared_array<unsigned> md_fill;
		size_t mb_allocated;

};


}	}	} // end namespace

#endif
