#ifndef NEMO_NETWORK_PROGRAMMATIC_SYNAPSE_ITERATOR_HPP
#define NEMO_NETWORK_PROGRAMMATIC_SYNAPSE_ITERATOR_HPP

#include <typeinfo>
#include <nemo/network/iterator.hpp>
#include <nemo/NetworkImpl.hpp>

namespace nemo {
	namespace network {

		class NetworkImpl;

		namespace programmatic {

class NEMO_BASE_DLL_PUBLIC synapse_iterator : public abstract_synapse_iterator
{
	public :

		synapse_iterator(
				NetworkImpl::fcm_t::const_iterator ni,
				NetworkImpl::fcm_t::const_iterator ni_end,
				id32_t gi,
				id32_t gi_end) :
			ni(ni), ni_end(ni_end), 
			gi(gi), gi_end(gi_end) { }

		const value_type& operator*() const {
			m_data = ni->second.getSynapse(ni->first, gi);
			return m_data;
		}

		const value_type* operator->() const {
			m_data = ni->second.getSynapse(ni->first, gi);
			return &m_data;
		}

		abstract_synapse_iterator* clone() const {
			return new synapse_iterator(*this);
		}

		abstract_synapse_iterator& operator++() {
			++gi;
			if(gi == gi_end) {
				++ni;
				if(ni == ni_end) {
					/* When reaching the end, all three iterators are at
					 * their respective ends. Further increments leads to
					 * undefined behaviour */
					return *this;
				}
				gi = 0;
				gi_end = ni->second.size();
			}
			return *this;
		 }

		bool operator==(const abstract_synapse_iterator& rhs_) const {
			if(typeid(*this) != typeid(rhs_)) {
				return false;
			}
			const synapse_iterator& rhs = static_cast<const synapse_iterator&>(rhs_);
			return ni == rhs.ni && ni_end == rhs.ni_end
				&& gi == rhs.gi && gi_end == rhs.gi_end;
		}

		bool operator!=(const abstract_synapse_iterator& rhs) const {
			return !(*this == rhs);
		}

	private :

		NetworkImpl::fcm_t::const_iterator ni;
		NetworkImpl::fcm_t::const_iterator ni_end;
		id32_t gi;
		id32_t gi_end;

		mutable value_type m_data;
};



}	}	}

#endif
