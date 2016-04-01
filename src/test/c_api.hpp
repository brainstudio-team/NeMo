#ifndef NEMO_TEST_C_API_HPP
#define NEMO_TEST_C_API_HPP

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with NeMo. If not, see <http://www.gnu.org/licenses/>.
 */

namespace nemo {
	namespace test {
		namespace c_api {

void compareWithCpp(bool useFstim, bool useIstim);
void testSynapseId();
void testSetNeuron();
void testGetSynapses(backend_t, unsigned n0);

}	}	}

#endif
