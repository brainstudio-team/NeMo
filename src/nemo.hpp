#ifndef NEMO_HPP
#define NEMO_HPP

//! \file nemo.hpp

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

/*! \file nemo.hpp C++ API
 *
 * Top-level include file for the C++ API. This file contains a few global
 * functions, but most functionality is found in the included classes.
 */

#include <nemo/Configuration.hpp>
#include <nemo/Network.hpp>
#include <nemo/Simulation.hpp>
#include <nemo/exception.hpp>
#include <nemo/types.h>


namespace nemo {

/*! Create a simulation using one of the available backends. Returns NULL if
 * unable to create simulation.
 *
 * Any missing/unspecified fields in the configuration are filled in */
NEMO_DLL_PUBLIC
Simulation* simulation(const Network& net, const Configuration& conf);


/*! \return Number of CUDA devices on this system */
NEMO_DLL_PUBLIC
unsigned
cudaDeviceCount();


/*! \return a textual description of a specific CUDA device
 *
 * \param device device number as allocated by NeMo. See \ref cuda_device_numbers
 */
NEMO_DLL_PUBLIC
const char*
cudaDeviceDescription(unsigned device);


/*! \return version number of the NeMo library */
NEMO_DLL_PUBLIC
const char*
version();



/*! \copydoc nemo::Plugin:addPath */
NEMO_DLL_PUBLIC
void
addPluginPath(const std::string& dir);


}

#endif
