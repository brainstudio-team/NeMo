#ifndef NEMO_EXCEPTION_HPP
#define NEMO_EXCEPTION_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdexcept>
#include <string>

#include "types.h"

namespace nemo {

/* Minor extension of std::exception which adds return codes (for use in the C
 * API). The error codes are listed in errors.h. */
//! \todo should probably inherit virtually here
class exception : public std::runtime_error
{
	public :

		exception(int errorNumber, const std::string& msg) :
			std::runtime_error(msg),
			m_errno(errorNumber) {}

		~exception() throw () {}

		int errorNumber() const { return m_errno; }

	private :

		int m_errno;
};


/*! Assert condition and throw exception with the given message otherwise. Note
 * that unlike the standard assert this function is always executed, regardless
 * of the compilation flags */
inline
void
assert_or_throw(bool cond, const char* str)
{
	if(!cond) {
		throw nemo::exception(NEMO_LOGIC_ERROR, str);
	}
}



inline
void
assert_or_throw(bool cond, const std::string& str)
{
	if(!cond) {
		throw nemo::exception(NEMO_LOGIC_ERROR, str);
	}
}


} // end namespace nemo

#endif
