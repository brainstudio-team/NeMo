#ifndef NEMO_CUDA_DEVICES_HPP
#define NEMO_CUDA_DEVICES_HPP

/*! \file devices.hpp Device enumeration */

/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <nemo/config.h>

namespace nemo {
	class ConfigurationImpl;
}

extern "C" {

typedef void cuda_set_configuration_t(nemo::ConfigurationImpl*, int);

/*! Choose the CUDA device to use and fill in the relevant field in the
 * configuration object. If dev = -1, have the backend choose a suitable
 * device, otherwise check that the user-selected device is a valid choice */
NEMO_CUDA_DLL_PUBLIC
void
cuda_set_configuration(nemo::ConfigurationImpl*, int dev);



typedef void cuda_test_device_t(unsigned device);

NEMO_CUDA_DLL_PUBLIC
void
cuda_test_device(unsigned device);



typedef unsigned cuda_device_count_t(void);

NEMO_CUDA_DLL_PUBLIC
unsigned
cuda_device_count(void);



typedef const char* cuda_device_description_t(unsigned device);

NEMO_CUDA_DLL_PUBLIC
const char*
cuda_device_description(unsigned device);

}


namespace nemo {
	namespace cuda {

/* Commit to using a specific device for this *thread*. If called multiple
 * times with different devices chose, this *may* raise an exception */
void setDevice(unsigned device);

}	}


#endif
