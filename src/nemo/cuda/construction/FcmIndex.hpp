#ifndef NEMO_CUDA_CONSTRUCTION_FCM_INDEX_HPP
#define NEMO_CUDA_CONSTRUCTION_FCM_INDEX_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <vector>

#include <boost/tuple/tuple.hpp>
#include <boost/unordered_map.hpp>
#include <boost/functional/hash.hpp>

#include <nemo/types.hpp>
//! \todo move DeviceIdx to types.hpp
#include <nemo/cuda/Mapper.hpp>
#include <nemo/cuda/types.h>


namespace nemo {
	namespace cuda {
		namespace construction {

/*! \brief Construction-time mapping from neuron/delay to synapse warp addresses
 *
 * This mapping is a temporary structure used during construction of the device
 * data for the forward connectivity matrix. Synapses are organised in groups,
 * where each group contains synapses sharing the same source neuron, target
 * partition and delay. Each group is in turn split into fixed-size warps, which
 * is the basic unit for spike delivery in NeMo.
 *
 * The warp address table serves two purposes:
 *
 * 1. allocating addresses (warp index + column index) for each synapse
 * 2. accumulate the addressing data that will be used at run-time.
 *
 * This class thus needs to refer to two types of data. The first is the
 * synapse data itself (referred to just as 'data' here), and the second is the
 * index into the data. The synapse data is only ever lookup up via the index,
 * but the data key specifies how synapses are grouped into rows.
 *
 * The warp address table is intended for incremental construction, by adding
 * individual synapses.
 *
 * At runtime this mapping is found in the data structure \ref Outgoing.
 *
 * The map structures used internally are boost::unordered_map rather than
 * std::map as this noticably speeds up the construction of the table.
 *
 * \see nemo::cuda::Outgoing
 */
class FcmIndex
{
	public :

		/* Synapses are grouped into 'rows' which share the same source neuron,
		 * target partition and delay
		 *                   source  source  target */
		typedef boost::tuple<pidx_t, nidx_t, pidx_t, delay_t> data_key;

		/* However, at run-time we get addresses only based on the source
		 * neuron and delay. */
		typedef boost::tuple<pidx_t, nidx_t, delay_t> index_key;

		/* Each row may be spread over a disparate set of warps. Each target
		 * partition may have synapses in several warps. */
		typedef boost::unordered_map<pidx_t, std::vector<size_t> > row_t;

	private :

		typedef boost::unordered_map<index_key, row_t> warp_map;

	public :

		/*
		 * \param nextFreeWarp
		 * 		The next unused warp in the host FCM.
		 *
		 * \return
		 * 		Address of this synapse in the form of a warp address and a
		 * 		within-warp address. This might refer to an existing warp or a
		 * 		new warp.
		 */
		SynapseAddress addSynapse(const DeviceIdx&, pidx_t, delay_t, size_t nextFreeWarp);

		typedef warp_map::const_iterator iterator;

		iterator begin() const { return m_warps.begin(); }
		iterator end() const { return m_warps.end(); }

		typedef row_t::const_iterator row_iterator;

		/*! \return
		 * 		length of a row in the lookup table, i.e. the number of warps
		 * 		in a row with the given key */
		unsigned indexRowLength(const index_key& k) const;

		/*! \return print histogram of sizes of each synapse
		 * warp to stdout */
		void reportWarpSizeHistogram(std::ostream& out) const;

	private :

		warp_map m_warps;

		/* In order to keep track of when we need to start a new warp, store
		 * the number of synapses in each row */
		boost::unordered_map<data_key, unsigned> m_dataRowLength;

		boost::unordered_map<index_key, unsigned> m_indexRowLength;
};



		} // end namespace construction
	} // end namespace cuda
} // end namespace nemo


#endif
