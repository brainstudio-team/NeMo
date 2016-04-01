/* Copyright 2010 Imperial College London
 *
 * This file is part of nemo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with nemo. If not, see <http://www.gnu.org/licenses/>.
 */

//! \file nemo_c.cpp

/*! C API for libnemo
 *
 * This simply wraps the C++ API
 */

#include <nemo.h>
#include "internals.hpp"
#include "exception.hpp"

/* We cannot propagate exceptions via the C API, so we catch all and convert to
 * error codes instead. Error descriptions are stored on a per-process basis. */

static std::string g_lastError;
static nemo_status_t g_lastCallStatus = NEMO_OK;

struct nemo_network_class : public nemo::Network {};
struct nemo_simulation_class : public nemo::SimulationBackend {};
struct nemo_configuration_class : public nemo::Configuration {};


void
setResult(const char* msg, nemo_status_t status) {
	g_lastError = msg;
	g_lastCallStatus = status;
}




/* Call method on wrapped object, and /set/ status and error */
#define CALL(call) {                                                          \
        g_lastCallStatus = NEMO_OK;                                           \
        try {                                                                 \
            call;                                                             \
        } catch (nemo::exception& e) {                                        \
            setResult(e.what(), e.errorNumber());                             \
        } catch (std::exception& e) {                                         \
            setResult(e.what(), NEMO_UNKNOWN_ERROR);                          \
        } catch (...) {                                                       \
            setResult("unknown exception", NEMO_UNKNOWN_ERROR);               \
        }                                                                     \
    }

/* Call method on wrapper object, and return status and error */
#define CATCH_(obj, call) {                                                   \
        CALL(obj->call)                                                       \
        return g_lastCallStatus;                                              \
	}

/* Call method on wrapper object, set output value, and return status and error */
#define CATCH(obj, call, ret) {                                               \
        CALL(ret = obj->call);                                                \
        return g_lastCallStatus;                                              \
	}


const char*
nemo_version()
{
	return nemo::version();
}



nemo_status_t
nemo_add_plugin_path(const char* dir)
{
	CALL(nemo::addPluginPath(std::string(dir)));
	return g_lastCallStatus;
}


nemo_status_t
nemo_cuda_device_count(unsigned* count)
{
	*count = 0;
	CALL(*count = nemo::cudaDeviceCount());
	return g_lastCallStatus;
}


nemo_status_t
nemo_cuda_device_description(unsigned device, const char** descr)
{
	CALL(*descr = nemo::cudaDeviceDescription(device));
	return g_lastCallStatus;
}


nemo_network_t
nemo_new_network()
{
	return static_cast<nemo_network_t>(new nemo::Network());
}


void
nemo_delete_network(nemo_network_t net)
{
	delete static_cast<nemo::Network*>(net);
}



nemo_configuration_t
nemo_new_configuration()
{
	try {
		return static_cast<nemo_configuration_t>(new nemo::Configuration());
	} catch(nemo::exception& e) {
		setResult(e.what(), e.errorNumber());
		return NULL;
	} catch(std::exception& e) {
		setResult(e.what(), NEMO_UNKNOWN_ERROR);
		return NULL;
	}
}



void
nemo_delete_configuration(nemo_configuration_t conf)
{
	delete static_cast<nemo::Configuration*>(conf);
}



nemo_simulation_t
nemo_new_simulation(nemo_network_t net_ptr, nemo_configuration_t conf_ptr)
{
	try {
		nemo::Network* net = static_cast<nemo::Network*>(net_ptr);
		nemo::Configuration* conf = static_cast<nemo::Configuration*>(conf_ptr);
		return static_cast<nemo_simulation_t>(nemo::simulationBackend(*net, *conf));
	} catch(nemo::exception& e) {
		setResult(e.what(), e.errorNumber());
		return NULL;
	} catch(std::exception& e) {
		setResult(e.what(), NEMO_UNKNOWN_ERROR);
		return NULL;
	} catch(...) {
		setResult("Unknown error", NEMO_UNKNOWN_ERROR);
		return NULL;

	}
}



void
nemo_delete_simulation(nemo_simulation_t sim)
{
	delete static_cast<nemo::SimulationBackend*>(sim);
}



nemo_status_t
nemo_add_neuron_type(nemo_network_t net,
		const char* name,
		unsigned* type)
{
	CATCH(net, addNeuronType(name), *type);
}



nemo_status_t
nemo_add_neuron_iz(nemo_network_t net,
		unsigned idx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	CATCH_(net, addNeuron(idx, a, b, c, d, u, v, sigma));
}



nemo_status_t
nemo_add_neuron(nemo_network_t net,
		unsigned type, unsigned idx,
		unsigned nargs, float args[])
{
	CATCH_(net, addNeuron(type, idx, nargs, args));
}



nemo_status_t
nemo_add_synapse(nemo_network_t net,
		unsigned source,
		unsigned target,
		unsigned delay,
		float weight,
		unsigned char is_plastic,
		synapse_id* id)
{
	synapse_id sid = 0;
	CALL(sid = net->addSynapse(source, target, delay, weight, is_plastic));
	if(id != NULL) {
		*id = sid;
	}
	return g_lastCallStatus;
}



nemo_status_t
nemo_neuron_count(nemo_network_t net, unsigned* ncount)
{
	CATCH(net, neuronCount(), *ncount);
}



nemo_status_t
nemo_get_membrane_potential(nemo_simulation_t sim, unsigned neuron, float* v)
{
	CATCH(sim, getMembranePotential(neuron), *v);
}


nemo_status_t
nemo_get_neuron_state_n(nemo_network_t net, unsigned neuron, unsigned var, float* val)
{
	CATCH(net, getNeuronState(neuron, var), *val);
}


nemo_status_t
nemo_get_neuron_parameter_n(nemo_network_t net, unsigned neuron, unsigned param, float* val)
{
	CATCH(net, getNeuronParameter(neuron, param), *val);
}


nemo_status_t
nemo_get_neuron_state_s(nemo_simulation_t sim, unsigned neuron, unsigned var, float* val)
{
	CATCH(sim, getNeuronState(neuron, var), *val);
}


nemo_status_t
nemo_get_neuron_parameter_s(nemo_simulation_t sim, unsigned neuron, unsigned param, float* val)
{
	CATCH(sim, getNeuronParameter(neuron, param), *val);
}



nemo_status_t
nemo_set_neuron_n(nemo_network_t net, unsigned idx, unsigned nargs, float args[])
{
	CATCH_(net, setNeuron(idx, nargs, args));
}



nemo_status_t
nemo_set_neuron_s(nemo_simulation_t sim, unsigned idx, unsigned nargs, float args[])
{
	CATCH_(sim, setNeuron(idx, nargs, args));
}



nemo_status_t
nemo_set_neuron_iz_n(nemo_network_t net,
		unsigned idx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	CATCH_(net, setNeuron(idx, a, b, c, d, u, v, sigma));
}


nemo_status_t
nemo_set_neuron_iz_s(nemo_simulation_t sim,
		unsigned idx,
		float a, float b, float c, float d,
		float u, float v, float sigma)
{
	CATCH_(sim, setNeuron(idx, a, b, c, d, u, v, sigma));
}


nemo_status_t
nemo_set_neuron_state_n(nemo_network_t net, unsigned neuron, unsigned var, float val)
{
	CATCH_(net, setNeuronState(neuron, var, val));
}


nemo_status_t
nemo_set_neuron_parameter_n(nemo_network_t net, unsigned neuron, unsigned param, float val)
{
	CATCH_(net, setNeuronParameter(neuron, param, val));
}


nemo_status_t
nemo_set_neuron_state_s(nemo_simulation_t sim, unsigned neuron, unsigned var, float val)
{
	CATCH_(sim, setNeuronState(neuron, var, val));
}


nemo_status_t
nemo_set_neuron_parameter_s(nemo_simulation_t sim, unsigned neuron, unsigned param, float val)
{
	CATCH_(sim, setNeuronParameter(neuron, param, val));
}


void
getSynapsesFrom(
		nemo::ReadableNetwork* net,
		unsigned source,
		synapse_id *synapses[],
		size_t* len)
{
	const std::vector<synapse_id>& ids = net->getSynapsesFrom(source);
	if(g_lastCallStatus == NEMO_OK && !ids.empty()) {
		*synapses = const_cast<synapse_id*>(&ids[0]);
		*len = ids.size();
	} else {
		*synapses = NULL;
		*len = 0;
	}
}


nemo_status_t
nemo_get_synapses_from_n(nemo_network_t net,
		unsigned source,
		synapse_id *synapses[],
		size_t* len)
{
	CALL(getSynapsesFrom(net, source, synapses, len));
	return g_lastCallStatus;
}



nemo_status_t
nemo_get_synapses_from_s(nemo_simulation_t sim,
		unsigned source,
		synapse_id *synapses[],
		size_t* len)
{
	CALL(getSynapsesFrom(sim, source, synapses, len));
	return g_lastCallStatus;
}


nemo_status_t
nemo_get_synapse_source_n(nemo_network_t ptr, synapse_id synapse, unsigned* source)
{
	CATCH(ptr, getSynapseSource(synapse), *source);
}


nemo_status_t
nemo_get_synapse_source_s(nemo_simulation_t ptr, synapse_id synapse, unsigned* source)
{
	CATCH(ptr, getSynapseSource(synapse), *source);
}


nemo_status_t
nemo_get_synapse_target_n(nemo_network_t ptr, synapse_id synapse, unsigned* target)
{
	CATCH(ptr, getSynapseTarget(synapse), *target);
}


nemo_status_t
nemo_get_synapse_target_s(nemo_simulation_t ptr, synapse_id synapse, unsigned* target)
{
	CATCH(ptr, getSynapseTarget(synapse), *target);
}


nemo_status_t
nemo_get_synapse_delay_n(nemo_network_t ptr, synapse_id synapse, unsigned* delay)
{
	CATCH(ptr, getSynapseDelay(synapse), *delay);
}


nemo_status_t
nemo_get_synapse_delay_s(nemo_simulation_t ptr, synapse_id synapse, unsigned* delay)
{
	CATCH(ptr, getSynapseDelay(synapse), *delay);
}


nemo_status_t
nemo_get_synapse_weight_n(nemo_network_t ptr, synapse_id synapse, float* weight)
{
	CATCH(ptr, getSynapseWeight(synapse), *weight);
}


nemo_status_t
nemo_get_synapse_weight_s(nemo_simulation_t ptr, synapse_id synapse, float* weight)
{
	CATCH(ptr, getSynapseWeight(synapse), *weight);
}


nemo_status_t
nemo_get_synapse_plastic_n(nemo_network_t ptr, synapse_id synapse, unsigned char* plastic)
{
	CATCH(ptr, getSynapsePlastic(synapse), *plastic);
}


nemo_status_t
nemo_get_synapse_plastic_s(nemo_simulation_t ptr, synapse_id synapse, unsigned char* plastic)
{
	CATCH(ptr, getSynapsePlastic(synapse), *plastic);
}


void
step(nemo::SimulationBackend* sim,
		const std::vector<unsigned>& fstim,
		unsigned istim_nidx[], float istim_current[], size_t istim_len,
		unsigned *fired[], size_t* fired_len)
{
	sim->setFiringStimulus(fstim);
	sim->initCurrentStimulus(istim_len);
	for(size_t i = 0; i < istim_len; ++i) {
		sim->addCurrentStimulus(istim_nidx[i], istim_current[i]);
	}
	sim->finalizeCurrentStimulus(istim_len);
	sim->prefire();
	sim->fire();
	sim->postfire();
	const std::vector<unsigned>& fired_ = sim->readFiring().neurons;
	if(fired != NULL) {
		*fired = fired_.empty() ? NULL : const_cast<unsigned*>(&fired_[0]);
	}
	if(fired_len != NULL) {
		*fired_len = fired_.size();
	}
}



nemo_status_t
nemo_step(nemo_simulation_t sim,
		unsigned fstim_nidx[], size_t fstim_count,
		unsigned istim_nidx[], float istim_current[], size_t istim_count,
		unsigned* fired[], size_t* fired_count)
{
	CALL(step(sim,
			std::vector<unsigned>(fstim_nidx, fstim_nidx + fstim_count),
			istim_nidx, istim_current, istim_count,
			fired, fired_count));
	return g_lastCallStatus;
}



nemo_status_t
nemo_apply_stdp(nemo_simulation_t sim, float reward)
{
	CATCH_(sim, applyStdp(reward));
}



//-----------------------------------------------------------------------------
// Timing
//-----------------------------------------------------------------------------


nemo_status_t
nemo_log_stdout(nemo_configuration_t conf)
{
	CATCH_(conf, enableLogging());
}



nemo_status_t
nemo_elapsed_wallclock(nemo_simulation_t sim, unsigned long* elapsed)
{
	CATCH(sim, elapsedWallclock(), *elapsed);
}



nemo_status_t
nemo_elapsed_simulation(nemo_simulation_t sim, unsigned long* elapsed)
{
	CATCH(sim, elapsedSimulation(), *elapsed);
}



nemo_status_t
nemo_reset_timer(nemo_simulation_t sim)
{
	CATCH_(sim, resetTimer());
}



//-----------------------------------------------------------------------------
// CONFIGURATION
//-----------------------------------------------------------------------------


nemo_status_t
nemo_set_stdp_function(nemo_configuration_t conf,
		float* pre_fn, size_t pre_len,
		float* post_fn, size_t post_len,
		float we_min, float we_max,
		float wi_min, float wi_max)
{
	CATCH_(conf, setStdpFunction(
				std::vector<float>(pre_fn, pre_fn+pre_len),
				std::vector<float>(post_fn, post_fn+post_len),
				we_min, we_max, wi_min, wi_max));
}


nemo_status_t
nemo_set_cpu_backend(nemo_configuration_t conf)
{
	CATCH_(conf, setCpuBackend());
}



nemo_status_t
nemo_set_cuda_backend(nemo_configuration_t conf, int dev)
{
	CATCH_(conf, setCudaBackend(dev));
}



nemo_status_t
nemo_cuda_device(nemo_configuration_t conf, int* dev)
{
	CATCH(conf, cudaDevice(), *dev);
}



nemo_status_t
nemo_backend(nemo_configuration_t conf, backend_t* backend)
{
	CATCH(conf, backend(), *backend);
}



nemo_status_t
nemo_backend_description(nemo_configuration_t conf, const char** descr)
{
	CATCH(conf, backendDescription(), *descr);
}


nemo_status_t
nemo_set_write_only_synapses(nemo_configuration_t conf)
{
	CATCH_(conf, setWriteOnlySynapses());
}


const char*
nemo_strerror()
{
	return g_lastError.c_str();
}
