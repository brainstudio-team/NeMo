#ifndef NEMO_COMPACTING_MAPPER_HPP
#define NEMO_COMPACTING_MAPPER_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/bimap.hpp>
#include <boost/format.hpp>

#include <nemo/internal_types.h>
#include <nemo/Mapper.hpp>
#include <nemo/exception.hpp>

namespace nemo {

/*! Mapper between global neuron index space and another index space
 *
 * The user of this class is responsible for providing both indices
 */
template<class L>
class RandomMapper : public Mapper<nidx_t, L>
{
	private :

		typedef boost::bimap<nidx_t, L> bm_type;

	public :

		~RandomMapper() {}

		/*! Add a new global/local neuron index pair */
		virtual void insert(nidx_t gidx, const L& lidx) {
			using boost::format;
			std::pair<typename bm_type::iterator,bool> insertion =
				m_bm.insert(typename bm_type::value_type(gidx, lidx));
			if(!insertion.second) {
				throw nemo::exception(NEMO_INVALID_INPUT,
					str(format("Duplicate neuron index %u") % gidx));
			}
		}

		/*! \return local index corresponding to the global neuron index \a gidx 
		 *
		 * \throws nemo::exception if the global neuron does not exist
		 * \todo return a reference here instead
		 */
		L localIdx(const nidx_t& gidx) const {
			using boost::format;
			try {
				return m_bm.left.at(gidx);
			} catch(std::out_of_range) {
				throw nemo::exception(NEMO_INVALID_INPUT,
					str(format("Non-existing neuron index %u") % gidx));
			} 
		}

		nidx_t globalIdx(const L& lidx) const {
			try {
				return m_bm.right.at(lidx);
			} catch(std::out_of_range) {
				//! \todo print the local index as well here
				throw nemo::exception(NEMO_INVALID_INPUT,
						"Non-existing local neuron index");
			}
		}

		bool existingGlobal(const nidx_t& gidx) const {
			return m_bm.left.find(gidx) != m_bm.left.end();
		}

		bool existingLocal(const L& lidx) const {
			return m_bm.right.find(lidx) != m_bm.right.end();
		}

		L minLocalIdx() const {
			return m_bm.size() == 0 ? 0 : m_bm.right.begin()->first;
		}

		L maxLocalIdx() const {
			return m_bm.size() == 0 ? 0 : m_bm.right.rbegin()->first;
		}

		nidx_t minGlobalIdx() const {
			return m_bm.size() == 0 ? 0 : m_bm.left.begin()->first;
		}

		nidx_t maxGlobalIdx() const {
			return m_bm.size() == 0 ? 0 : m_bm.left.rbegin()->first;
		}

		/*! Iterator over <global,local> pairs */
		typedef typename bm_type::left_const_iterator const_iterator;

		const_iterator begin() const { return m_bm.left.begin(); }
		const_iterator end() const { return m_bm.left.end(); }

		/*! Add a new local 0-based contiguous index to mapper
		 *
		 * \pre local increases monotonically on subsequent calls to this function
		 * \pre function is called only once for each partition
		 */
		void insertTypeMapping(unsigned local, unsigned type_id) {
			if(local != m_typeIndex.size()) {
				throw nemo::exception(NEMO_LOGIC_ERROR,
						"Internal error: unexpected partition added to mapper");
			}
			m_typeIndex.push_back(type_id);
		}


		/*! \return type index of a given local index */
		unsigned typeIdx(unsigned local) const { return m_typeIndex.at(local); }

		/*! Add a new neuron type to mapper
		 *
		 * \param base
		 * 		lowest index (in some compact 0-based index) for the given neuron type
		 *
		 * \pre type_id increases monotonically on subsequent calls to this function
		 */
		void insertTypeBase(unsigned type_id, unsigned base) {
			if(type_id != m_typeBase.size()) {
				throw nemo::exception(NEMO_LOGIC_ERROR,
						"Internal error: unexpected neuron type added to mapper");
			}
			m_typeBase.push_back(base);
		}

		/*! \return the base partition index for a neuron type */
		unsigned typeBase(unsigned tidx) const { return m_typeBase.at(tidx); }


	protected :

		/* Start of a neuron type group
		 *
		 * All neurons belonging to a single neuron type are found in a
		 * contigous range of indices following this.
		 */
		std::vector<unsigned> m_typeBase;

		/* Mapping from some compact local 0-based index space (e.g. partition
		 * on CUDA, local neuron index on CPU) to neuron type (also compact and
		 * 0-based). */
		std::vector<unsigned> m_typeIndex;

	private :

		bm_type m_bm;
};


}

#endif
