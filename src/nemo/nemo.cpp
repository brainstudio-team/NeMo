/* This file contains factory methods only, both for the public and the
 * internal API. Most functionality is found in the respective C++ classes and
 * in the C API wrapper file nemo_c.cpp */

#include <nemo/config.h>

#ifdef NEMO_CUDA_ENABLED
#ifdef NEMO_CUDA_DYNAMIC_LOADING
#include <boost/scoped_ptr.hpp>
#include "Plugin.hpp"
#endif
#endif

#include <nemo/internals.hpp>
#include <nemo/exception.hpp>
#include <nemo/network/Generator.hpp>
#include <nemo/NetworkImpl.hpp>

#ifdef NEMO_CUDA_ENABLED
#include <nemo/cuda/create_simulation.hpp>
#include <nemo/cuda/devices.hpp>
#endif
#include <nemo/cpu/Simulation.hpp>

namespace nemo {

#ifdef NEMO_CUDA_ENABLED
#ifdef NEMO_CUDA_DYNAMIC_LOADING


boost::scoped_ptr<Plugin> libcuda;


/*! \return pointer to unique global handle for the CUDA simulator plugin */
const Plugin*
cudaPlugin()
{
	if(!libcuda) {
		libcuda.reset(new Plugin("nemo_cuda"));
	}
	return libcuda.get();
}

#endif
#endif



unsigned
cudaDeviceCount()
{
#ifdef NEMO_CUDA_ENABLED
	try {
#ifdef NEMO_CUDA_DYNAMIC_LOADING
		cuda_device_count_t* fn = (cuda_device_count_t*) cudaPlugin()->function("cuda_device_count");
		return fn();
#else
		return cuda_device_count();
#endif
	} catch(...) {
		return 0;
	}
#else // NEMO_CUDA_ENABLED
	throw nemo::exception(NEMO_API_UNSUPPORTED,
			"Cannot return CUDA device count: library compiled without CUDA support");
#endif // NEMO_CUDA_ENABLED
}


/* Throws on error */
const char*
cudaDeviceDescription(unsigned device)
{
#ifdef NEMO_CUDA_ENABLED
#ifdef NEMO_CUDA_DYNAMIC_LOADING
	cuda_device_description_t* fn =
		(cuda_device_description_t*) cudaPlugin()->function("cuda_device_description");
	return fn(device);
#else
	return cuda_device_description(device);
#endif
#else // NEMO_CUDA_ENABLED
	throw nemo::exception(NEMO_API_UNSUPPORTED,
			"libnemo compiled without CUDA support");
#endif // NEMO_CUDA_ENABLED
}



#ifdef NEMO_CUDA_ENABLED

SimulationBackend*
cudaSimulation(const network::Generator& net, const ConfigurationImpl& conf)
{
#ifdef NEMO_CUDA_DYNAMIC_LOADING
	cuda_simulation_t* ctor = (cuda_simulation_t*) cudaPlugin()->function("cuda_simulation");
	return ctor(&net, &conf);
#else
	return cuda_simulation(&net, &conf);
#endif
}

#endif


/* Sometimes using the slightly lower-level interface provided by NetworkImpl
 * makes sense (see e.g. nemo::mpi::Worker), so provide an overload of 'create'
 * that takes such an object directly. */
SimulationBackend*
simulationBackend(const network::Generator& net, const ConfigurationImpl& conf)
{
	if(net.neuronCount() == 0) {
		throw nemo::exception(NEMO_INVALID_INPUT,
				"Cannot create simulation from empty network");
		return NULL;
	}

	conf.verifyStdp(net.maxDelay());

	switch(conf.backend()) {
#ifdef NEMO_CUDA_ENABLED
		case NEMO_BACKEND_CUDA:
			return cudaSimulation(net, conf);
#else
		case NEMO_BACKEND_CUDA:
			throw nemo::exception(NEMO_API_UNSUPPORTED,
					"nemo was compiled without Cuda support. Cannot create simulation");
#endif
		case NEMO_BACKEND_CPU:
			return new cpu::Simulation(net, conf);
		default :
			throw nemo::exception(NEMO_LOGIC_ERROR, "unknown backend in configuration");
	}
}

SimulationBackend*
simulationBackend(const Network& net, const Configuration& conf)
{
	return simulationBackend(*net.m_impl, *conf.m_impl);
}


Simulation*
simulation(const Network& net, const Configuration& conf)
{
	return dynamic_cast<Simulation*>(simulationBackend(net, conf));
}




/* Set the default CUDA device if possible. Throws if anything goes wrong or if
 * there are no suitable devices. If device is -1, have the backend choose a
 * device. Otherwise, try to use the device provided by the user.  */
void
setCudaDeviceConfiguration(nemo::ConfigurationImpl& conf, int device)
{
#ifdef NEMO_CUDA_ENABLED
#ifdef NEMO_CUDA_DYNAMIC_LOADING
	cuda_set_configuration_t* fn = (cuda_set_configuration_t*) cudaPlugin()->function("cuda_set_configuration");
	fn(&conf, device);
#else
	cuda_set_configuration(&conf, device);
#endif
#else // NEMO_CUDA_ENABLED
	throw nemo::exception(NEMO_API_UNSUPPORTED,
			"libnemo compiled without CUDA support");
#endif // NEMO_CUDA_ENABLED
}




void
setDefaultHardware(nemo::ConfigurationImpl& conf)
{
#ifdef NEMO_CUDA_ENABLED
	try {
		setCudaDeviceConfiguration(conf, -1);
	} catch(...) {
		conf.setBackend(NEMO_BACKEND_CPU);
	}
#else
		conf.setBackend(NEMO_BACKEND_CPU);
#endif
}



const char*
version()
{
	return NEMO_VERSION;
}



void
addPluginPath(const std::string& dir)
{
	Plugin::addPath(dir);
}


}
