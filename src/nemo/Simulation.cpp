/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "Simulation.hpp"

namespace nemo {

Simulation::~Simulation()
{
	;
}


void
Simulation::setNeuron(unsigned idx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	const float args[7] = {a, b, c, d, sigma, u, v};
	setNeuron(idx, 7, args);
}




#ifdef NEMO_BRIAN_ENABLED

std::pair<size_t, size_t>
Simulation::propagate(size_t fired, int nfired)
{
	std::pair<float*, float*> ret = propagate_raw(reinterpret_cast<uint32_t*>(fired), nfired);
	return std::make_pair<size_t, size_t>(size_t(ret.first), size_t(ret.second));
}

#endif



} // end namespace nemo
