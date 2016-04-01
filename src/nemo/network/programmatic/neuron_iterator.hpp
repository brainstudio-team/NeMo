#ifndef NEMO_NETWORK_PROGRAMMATIC_NEURON_ITERATOR_HPP
#define NEMO_NETWORK_PROGRAMMATIC_NEURON_ITERATOR_HPP

#include <typeinfo>
#include <vector>

#include <nemo/network/iterator.hpp>
#include <nemo/NeuronType.hpp>


namespace nemo {
	namespace network {
		namespace programmatic {

//! \todo could probably use a boost iterator here
class NEMO_BASE_DLL_PUBLIC neuron_iterator : public abstract_neuron_iterator
{
	public :

		typedef std::vector<unsigned>::const_iterator base_iterator;

		neuron_iterator(base_iterator it,
			const std::vector< std::vector<float> >& param,
			const std::vector< std::vector<float> >& state,
			const NeuronType& type) :
			m_it(it), m_nt(type), m_param(param), m_state(state), m_lidx(0) {}

		void set_value() const {
			m_data.second = Neuron(m_nt);
			m_data.first = *m_it;
			for(size_t i=0; i < m_param.size(); ++i) {
				m_data.second.setParameter(i, m_param[i][m_lidx]);
			}
			for(size_t i=0; i < m_state.size(); ++i) {
				m_data.second.setState(i, m_state[i][m_lidx]);
			}
		}

		const value_type& operator*() const {
			set_value();
			return m_data;
		}

		const value_type* operator->() const {
			set_value();
			return &m_data;
		}

		nemo::network::abstract_neuron_iterator* clone() const {
			return new neuron_iterator(*this);
		}

		nemo::network::abstract_neuron_iterator& operator++() {
			++m_it;
			++m_lidx;
			return *this;
		}

		bool operator==(const abstract_neuron_iterator& rhs) const {
			return typeid(*this) == typeid(rhs) 
				&& m_it == static_cast<const neuron_iterator&>(rhs).m_it;
		}

		bool operator!=(const abstract_neuron_iterator& rhs) const {
			return !(*this == rhs);
		}

	private :

		base_iterator m_it;

		mutable value_type m_data;

		const NeuronType m_nt;

		const std::vector< std::vector<float> >& m_param;
		const std::vector< std::vector<float> >& m_state;

		/* Current local index in the underlying neuron collection. This is a
		 * contigous 0-based space of neurons */
		unsigned m_lidx;
};

}	}	}

#endif
