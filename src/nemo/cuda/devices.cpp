/*! \file devices.cpp Device enumeration */

/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

#include "devices.hpp"

#include <map>
#include <vector>
#include <boost/format.hpp>

#include <nemo/config.h>
#include <nemo/ConfigurationImpl.hpp>

#include "exception.hpp"
#include "kernel.cu_h"

namespace nemo {
	namespace cuda {

/*! \page cuda_device_numbers Cuda device numbers
 *
 * We want to present users of libnemo with a contiguous range of device
 * indices. Since the host system may have devices which are not suitable for
 * use by NeMo, the device indexing may differ between nemo and the CUDA
 * driver. We refer to our own ids as \e local ids, and the CUDA ids as \e
 * driver ids.
 *
 * Users of the library should use the top-level functions \ref
 * nemo::cudaDeviceCount and \ref nemo::cudaDeviceDescription for enumeration
 * of the devices.
 */


/*! \brief Mapping between driver and local device indices
 *
 * Use \ref instance to create a (unique global) instance of the mapping.
 *
 * \see \ref cuda_device_numbers
 */
class DeviceMap
{
	public :

		static DeviceMap* instance();

		typedef unsigned local_id;
		typedef int driver_id;

		unsigned deviceCount() const { return m_devices.size(); }

		/*! \post the return value is a valid index in the device list */
		driver_id driverId(local_id) const;

		void setConfiguration(ConfigurationImpl& conf, int device) const;

		const char* description(local_id device) const;

		const cudaDeviceProp& properties(local_id device) const;

	private :

		static DeviceMap* s_instance;

		DeviceMap();

		/*! All the suitable devices, indexed by driver id */
		std::map<driver_id, cudaDeviceProp> m_devices;

		/*! Full description of the device */
		std::map<driver_id, std::string> m_description;

		/*! Mapping from local ids to driver ids */
		std::vector<driver_id> m_driverIds;

		/*! Default device to use (local id) */
		local_id m_defaultDevice;

		bool deviceSuitable(driver_id device);

		void setDefaultDevice(const std::vector<int>& driverIds);

		bool validLocalId(local_id id) const { return id < m_driverIds.size(); }
};


DeviceMap* DeviceMap::s_instance = NULL;


DeviceMap*
DeviceMap::instance()
{
	if(s_instance == NULL) {
		s_instance = new DeviceMap();
	}
	return s_instance;
}




DeviceMap::DeviceMap() :
	m_defaultDevice(-1)
{
#ifdef __DEVICE_EMULATION__
	cudaDeviceProp prop;
	prop.major = 9999;
	prop.minor = 9999;
	driver_id dev;
	CUDA_SAFE_CALL(cudaChooseDevice(&dev, &prop));
	m_devices[0] = prop;
	m_description[0] = prop.name;
	m_defaultDevice = 0;
	m_driverIds.push_back(0);
#else
	using boost::format;

	driver_id dcount = 0;
	CUDA_SAFE_CALL(cudaGetDeviceCount(&dcount));
	for(driver_id device = 0; device < dcount; ++device) {
		if(deviceSuitable(device)) {
			cudaDeviceProp prop;
			CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, device));
			m_devices[device] = prop;
			m_description[device] = str(format("%s [%u.%u]")
					% prop.name % prop.major % prop.minor);
			m_driverIds.push_back(device);
		}
	}

	setDefaultDevice(m_driverIds);
#endif
}



bool
DeviceMap::deviceSuitable(driver_id device)
{
	cudaDeviceProp prop;
	CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, device));

	/* 9999.9999 is the 'emulation device' which is always present. Unless the
	 * library was built specifically for emulation mode, this should be
	 * considered an error. */
	if(prop.major == 9999 || prop.minor == 9999) {
		return false;
	}

	/* 1.2 required for shared memory atomics */
	if(prop.major <= 1 && prop.minor < 2) {
		return false;
	}

	return true;
}



void
DeviceMap::setDefaultDevice(const std::vector<int>& driverIds)
{
	if(driverIds.empty())
		return;

	cudaDeviceProp prop;
	prop.major = 1;
	prop.minor = 2;

	/* The warp-size requirement is due to assumptions made in the kernel code.
	 * It should be possible to lift this constraint */
	prop.warpSize = WARP_SIZE;

	int dev;
	CUDA_SAFE_CALL(cudaChooseDevice(&dev, &prop));

	/* The chosen device should already be present in the device map */
	for(unsigned localId = 0; localId < driverIds.size(); ++localId) {
		if(driverIds.at(localId) == dev) {
			m_defaultDevice = localId;
			return;
		}
	}
	throw nemo::exception(NEMO_LOGIC_ERROR, "Internal error: default device not in device map");
}



DeviceMap::driver_id
DeviceMap::driverId(local_id l_id) const
{
	using boost::format;

	if(l_id >= m_driverIds.size()) {
		throw nemo::exception(NEMO_CUDA_ERROR, str(format("Invalid device: %u") % l_id));
	}

	driver_id d_id = m_driverIds.at(l_id);
	if(m_devices.find(d_id) == m_devices.end()) {
		throw nemo::exception(NEMO_CUDA_ERROR, str(format("Invalid device: %u") % l_id));
	}
	return d_id;
}


const char*
DeviceMap::description(local_id device) const
{
	return m_description.find(m_driverIds.at(device))->second.c_str();
}



const cudaDeviceProp&
DeviceMap::properties(local_id device) const
{
	return m_devices.find(m_driverIds.at(device))->second;
}



void
DeviceMap::setConfiguration(ConfigurationImpl& conf, int device) const // local Id
{
	using boost::format;

	if(deviceCount() == 0) {
		throw nemo::exception(NEMO_CUDA_ERROR, "No CUDA devices available");
	}
	conf.setBackend(NEMO_BACKEND_CUDA);
	if(device < 0) { // Use default
		conf.setCudaDevice(m_defaultDevice);
	} else {
		if(!validLocalId(device)) {
			throw nemo::exception(NEMO_CUDA_ERROR, str(format("Invalid device: %u") % device));
		}
		conf.setCudaDevice(unsigned(device));
	}

	conf.setCudaPartitionSize(MAX_PARTITION_SIZE);
}


void
setDevice(DeviceMap::local_id device)
{
	using boost::format;

	int userDev = DeviceMap::instance()->driverId(device);
	int existingDev;
	CUDA_SAFE_CALL(cudaGetDevice(&existingDev));
	if(existingDev != userDev) {
		/* If the device has already been set, we'll get an error here */
		CUDA_SAFE_CALL(cudaSetDevice(userDev));
	}

#ifndef __DEVICE_EMULATION__
	/* The kernel relies on hard-coded warp size */
	//! \todo lift this requirement
	if(DeviceMap::instance()->properties(device).warpSize != WARP_SIZE) {
		throw nemo::exception(NEMO_LOGIC_ERROR,
				str(format( "CUDA device has warp size which is not %u") % WARP_SIZE));
	}
#endif
}



}	}


/* C API */


void
cuda_set_configuration(nemo::ConfigurationImpl* conf, int device)
{
	nemo::cuda::DeviceMap::instance()->setConfiguration(*conf, device);
}



unsigned
cuda_device_count()
{
	return nemo::cuda::DeviceMap::instance()->deviceCount();
}



const char*
cuda_device_description(unsigned device)
{
	return nemo::cuda::DeviceMap::instance()->description(device);
}
