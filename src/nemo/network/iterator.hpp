#ifndef NEMO_NETWORK_NEURON_ITERATOR_HPP
#define NEMO_NETWORK_NEURON_ITERATOR_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */


#include <nemo/types.hpp>

namespace nemo {
	namespace network {

template<class T>
class abstract_iterator
{
	public:

		typedef T value_type;
		
		virtual ~abstract_iterator() { }

		virtual const T& operator*() const = 0;

		virtual const T* operator->() const = 0;

		virtual abstract_iterator* clone() const = 0;

		virtual abstract_iterator& operator++() = 0;

		virtual bool operator==(const abstract_iterator&) const = 0;

		virtual bool operator!=(const abstract_iterator&) const = 0;
};


typedef abstract_iterator< std::pair<unsigned, nemo::Neuron> > abstract_neuron_iterator;
typedef abstract_iterator<Synapse> abstract_synapse_iterator;


template<class T>
class iterator
{
	public:

		typedef T value_type;

		explicit iterator(abstract_iterator<T>* base) :
			m_base(base) { }

		iterator(const iterator& other) :
			m_base(other.m_base->clone()) { }

		~iterator() {
			delete m_base;
		}

		iterator& operator=(const iterator& rhs) {
			if(this != &rhs) {
				delete m_base;
				m_base = rhs.m_base->clone();
			}
			return *this;
		}

		const T& operator*() const {
			return *(*m_base);
		}

		const abstract_iterator<T>& operator->() const {
			return *m_base;
		}

		iterator& operator++() {
			++(*m_base);
			return *this;
		}

		bool operator==(const iterator& rhs) const {
			return *(rhs.m_base) == *m_base;
		}

		bool operator!=(const iterator& rhs) const {
			return !(*this == rhs);
		}

	private :

		iterator() : m_base(NULL) { }

		abstract_iterator<T>* m_base;
};


typedef iterator<abstract_neuron_iterator::value_type> neuron_iterator;
typedef iterator<abstract_synapse_iterator::value_type> synapse_iterator;


}	}

#endif
