#ifndef NEMO_CUDA_MAPPER_HPP
#define NEMO_CUDA_MAPPER_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <algorithm>
#include <nemo/RandomMapper.hpp>

#include "types.h"

namespace nemo {
	namespace cuda {

/*! Neuron indices as used on CUDA devices
 *
 * The network is split into partitions when moved onto the device. Neurons on
 * the device are thus addressed using a two-level address. */
struct DeviceIdx
{
	public:

		pidx_t partition;
		nidx_t neuron;

		DeviceIdx(pidx_t p, nidx_t n) : partition(p), neuron(n) {}
};


inline
bool
operator<(const DeviceIdx& lhs, const DeviceIdx& rhs)
{
	return lhs.partition < rhs.partition ||
		(lhs.partition == rhs.partition && lhs.neuron < rhs.neuron);
}



/*! Maps between local and global indices.
 *
 * The local indices can be either 2D (partition/neuron) or 1D with
 * straightforward mappings between them.
 */
class Mapper : public nemo::RandomMapper<DeviceIdx>
{
	public :

		Mapper(unsigned partitionSize) :
			m_partitionSize(partitionSize),
			m_partitionCount(0) {}

		/* Add a new neuron to mapper
		 *
		 * \pre device indices are added in incremental order
		 */
		void insert(nidx_t g_idx, const DeviceIdx& l_idx) {
			//! \todo range-check device index
			m_partitionCount = std::max(m_partitionCount, l_idx.partition+1);
			nemo::RandomMapper<DeviceIdx>::insert(g_idx, l_idx);
		}

		/*! Convert from device index (2D) to local 1D index.
		 *
		 * The existence of the input device index is not checked.
		 */
		nidx_t localIdx1D(const DeviceIdx& d) const {
			return d.partition * m_partitionSize + d.neuron;
		}

		/*! \copydoc nemo::cuda::Mapper::localIdx1D */
		nidx_t localIdx1D(const nidx_t& global) const {
			return localIdx1D(localIdx(global));
		}

		/*! \copydoc nemo::RandomMapper::localIdx */
		DeviceIdx deviceIdx(nidx_t global) const {
			return nemo::RandomMapper<DeviceIdx>::localIdx(global);
		}

		unsigned partitionSize() const { return m_partitionSize; }

		unsigned partitionCount() const { return m_partitionCount; }

		//! \todo remove this method
		/*! \return minimum global indexed supported by this mapper */
		unsigned minHandledGlobalIdx() const { return minGlobalIdx(); }

		//! \todo remove this method
		/*! \return maximum global indexed supported by this mapper */
		unsigned maxHandledGlobalIdx() const { return maxGlobalIdx(); }

		/*! \return the number of partitions for the given neuron type */
		unsigned partitionCount(unsigned tidx) const {
			return std::count(m_typeIndex.begin(), m_typeIndex.end(), tidx);
		}

	private :

		unsigned m_partitionSize;

		unsigned m_partitionCount;
};

	} // end namespace cuda
} // end namespace nemo

#endif
