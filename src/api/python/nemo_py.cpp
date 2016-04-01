/* Copyright 2010 Imperial College London
 *
 * This file is part of NeMo.
 *
 * This software is licenced for non-commercial academic use under the GNU
 * General Public Licence (GPL). You should have received a copy of this
 * licence along with NeMo. If not, see <http://www.gnu.org/licenses/>.
 */

#include <sstream>
#include <stdexcept>
#include <functional>

#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <nemo.hpp>
#include <nemo/config.h>

#include "docstrings.h" // auto-generated

#ifdef NEMO_BRIAN_ENABLED
const char* SIMULATION_PROPAGATE_DOC =
	"Propagate spikes on GPU given firing\n"
	"\n"
	"This function is intended purely for integration with Brian\n"
	"\n"
	"Inputs:\n"
	"fired  -- device pointer non-compact list of fired neurons (on CUDA)\n"
	"          or host pointer to compact list of fired neurons (on CPU)\n"
	"nfired -- length of fired if on CPU\n"
	"\n"
	"Returns tuple of pointers to per-neuron accumulated weights, the first\n"
	"one for excitatory, the second for inhbitiory weights.\n";
#endif


using namespace boost::python;


/* Py_ssize_t only introduced in Python 2.5 */
#if PY_VERSION_HEX < 0x02050000 && !defined(PY_SSIZE_T_MIN)
typedef int Py_ssize_t;
#	define PY_SSIZE_T_MAX INT_MAX
#	define PY_SSIZE_T_MIN INT_MIN
#endif


/* The simulation is only created via a factory and only accessed throught the
 * returned pointer */
boost::shared_ptr<nemo::Simulation>
makeSimulation(const nemo::Network& net, const nemo::Configuration& conf)
{
	return boost::shared_ptr<nemo::Simulation>(simulation(net, conf));
}



template<typename T>
std::string
std_vector_str(std::vector<T>& self)
{
	std::stringstream out;

	if(self.size() > 0) {
		out << "[";
		if(self.size() > 1) {
			std::copy(self.begin(), self.end() - 1, std::ostream_iterator<T>(out, ", "));
		}
		out << self.back() << "]";
	}

	return out.str();
}



/* We use uchar to stand in for booleans */
std::string
std_bool_vector_str(std::vector<unsigned char>& self)
{
	std::stringstream out;

	if(self.size() > 0) {
		out << "[";
		for(std::vector<unsigned char>::const_iterator i = self.begin(); i != self.end() - 1; ++i) {
			out << (*i ? "True" : "False") << ", ";

		}
		out << (self.back() ? "True" : "False") << ", ";
	}
	return out.str();
}



/* Converter from python pair to std::pair */
template<typename T1, typename T2>
struct from_py_list_of_pairs
{
	typedef std::pair<T1, T2> pair_t;
	typedef std::vector<pair_t> vector_t;

	from_py_list_of_pairs() {
		converter::registry::push_back(
			&convertible,
			&construct,
			boost::python::type_id<vector_t>()
		);
	}

	static void* convertible(PyObject* obj_ptr) {
		if (!PyList_Check(obj_ptr)) {
			return 0;
		}
		/* It's possible for the list to contain data of different type. In
         * that case, fall over later, during the actual conversion. */
		return obj_ptr;
	}

	/* Convert obj_ptr into a std::vector */
	static void construct(
			PyObject* list,
			boost::python::converter::rvalue_from_python_stage1_data* data)
	{
		/* grab pointer to memory into which to construct vector */
		typedef converter::rvalue_from_python_storage<vector_t> storage_t;
		void* storage = reinterpret_cast<storage_t*>(data)->storage.bytes;

		/* in-place construct vector */
		Py_ssize_t len = PyList_Size(list);
		// vector_t* instance = new (storage) vector_t(len, 0);
		vector_t* instance = new (storage) vector_t(len);

		/* stash the memory chunk pointer for later use by boost.python */
		data->convertible = storage;

		/* populate the vector */
		vector_t& vec = *instance;
		for(unsigned i=0, i_end=len; i != i_end; ++i) {
			PyObject* pair = PyList_GetItem(list, i);
			PyObject* first = PyTuple_GetItem(pair, 0);
			PyObject* second = PyTuple_GetItem(pair, 1);
			vec[i] = std::make_pair<T1, T2>(extract<T1>(first), extract<T2>(second));
		}
	}
};



//! \todo make this a more generic sequence converter
/* Python list to std::vector convertor */
template<typename T>
struct from_py_list
{
	typedef std::vector<T> vector_t;

	from_py_list() {
		converter::registry::push_back(
			&convertible,
			&construct,
			boost::python::type_id<vector_t>()
		);
	}

	static void* convertible(PyObject* obj_ptr) {
		if (!PyList_Check(obj_ptr)) {
			return 0;
		}
		/* It's possible for the list to contain data of different type. In
         * that case, fall over later, during the actual conversion. */
		return obj_ptr;
	}

	/* Convert obj_ptr into a std::vector */
	static void construct(
			PyObject* list,
			boost::python::converter::rvalue_from_python_stage1_data* data)
	{
		/* grab pointer to memory into which to construct vector */
		typedef converter::rvalue_from_python_storage<vector_t> storage_t;
		void* storage = reinterpret_cast<storage_t*>(data)->storage.bytes;

		/* in-place construct vector */
		Py_ssize_t len = PyList_Size(list);
		vector_t* instance = new (storage) vector_t(len, 0);

		/* stash the memory chunk pointer for later use by boost.python */
		data->convertible = storage;

		/* populate the vector */
		vector_t& vec = *instance;
		for(unsigned i=0, i_end=len; i != i_end; ++i) {
			vec[i] = extract<T>(PyList_GetItem(list, i));
		}
	}
};



/*! Determine if input is scalar or vector.
 *
 * If it is a vector, verify that the vector length is the same as other
 * vectors (whose length is already set in \a vectorLength.
 *
 * \param obj either a scalar or a vector
 * \param vectorLength length of any other vectors in the same parameter list,
 * 		or '0' if there are no others.
 * \return true if the object is vector (Python list), false if it's a scalar
 */
bool
checkInputVector(PyObject* obj, unsigned &vectorLength)
{
	unsigned length = PySequence_Check(obj) ? PySequence_Size(obj) : 0;
	if(length > 0) {
		if(vectorLength > 0 && length != vectorLength) {
			throw std::invalid_argument("input vectors of different length");
		}
		vectorLength = length;
	}
	return length > 0;
}



/*! Add one or more synapses
 *
 * \return synapse id
 *
 * The arguments (other than net) may be either scalar or vector. All vectors
 * must be of the same length. If any of the inputs are vectors, the scalar
 * arguments are replicated for each synapse.
 */
PyObject*
add_synapse(nemo::Network& net, PyObject* sources, PyObject* targets,
		PyObject* delays, PyObject* weights, PyObject* plastics)
{
	unsigned len = 0;

	bool vectorSources  = checkInputVector(sources, len);
	bool vectorTargets  = checkInputVector(targets, len);
	bool vectorDelays   = checkInputVector(delays, len);
	bool vectorWeights  = checkInputVector(weights, len);
	bool vectorPlastics = checkInputVector(plastics, len);

	to_python_value<synapse_id&> get_id;

	if(len == 0) {
		/* All inputs are scalars */
		return get_id(net.addSynapse(
					extract<unsigned>(sources),
					extract<unsigned>(targets),
					extract<unsigned>(delays),
					extract<float>(weights),
					extract<unsigned char>(plastics))
				);
	} else {
		/* At least some inputs are vectors, so we need to return a list */
		PyObject* list = PyList_New(len);
		for(unsigned i=0; i != len; ++i) {
			unsigned source = extract<unsigned>(vectorSources ? PySequence_GetItem(sources, i) : sources);
			unsigned target = extract<unsigned>(vectorTargets ? PySequence_GetItem(targets, i) : targets);
			unsigned delay = extract<unsigned>(vectorDelays ? PySequence_GetItem(delays, i) : delays);
			float weight = extract<float>(vectorWeights ? PySequence_GetItem(weights, i) : weights);
			unsigned char plastic = extract<unsigned char>(vectorPlastics ? PySequence_GetItem(plastics, i) : plastics);
			PyList_SetItem(list, i, get_id(net.addSynapse(source, target, delay, weight, plastic)));
		}
		return list;
	}
}



/*! Add one ore more neurons of arbitrary type
 *
 * This function expects the following arguments:
 *
 * - neuron type (unsigned scalar)
 * - neuron index (unsigned scalar/vector)
 * - a variable number of parameters (float scalar/vector)
 * - a variable number of state variables (float scalar/vector)
 *
 * The corresponding python prototype would be
 *
 * add_neuron(self, neuron_type, neuron_idx, *args)
 *
 * The neuron type is expected to be a scalar. The remaining arguments
 * may be either scalar or vector. All vectors must be of the same
 * length. If any of the inputs are vectors, the scalar arguments are
 * replicated for each neuron.
 *
 * \param args parameters and state variables
 * \param kwargs unused, but required by boost::python
 * \return None
 *
 * \see nemo::Network::addNeuron nemo::Network::addNeuronType
 */
boost::python::object
add_neuron_va(boost::python::tuple py_args, boost::python::dict /*kwargs*/)
{
	using namespace boost::python;

	unsigned vlen = 0;
	unsigned nargs = boost::python::len(py_args);
	nemo::Network& net = boost::python::extract<nemo::Network&>(py_args[0])();

	/* The neuron type should always be a scalar */
	unsigned neuron_type = extract<unsigned>(py_args[1]);

	/* Get raw pointers and determine the mix of scalar and vector arguments */
	//! \todo skip initial objects if possible
	PyObject* objects[NEMO_MAX_NEURON_ARGS];
	bool vectorized[NEMO_MAX_NEURON_ARGS];

	assert(nargs < NEMO_MAX_NEURON_ARGS-3);

	for(unsigned i=2; i<nargs; ++i) {
		objects[i] = static_cast<boost::python::object>(py_args[i]).ptr();
		vectorized[i] = checkInputVector(objects[i], vlen);
	}

	/* Get the neuron index, if it's a scalar */
	unsigned neuron_index = 0;
	if(!vectorized[2]) {
		neuron_index = extract<unsigned>(py_args[2]);
	}

	/* Get all scalar parameters and state variables */
	float args[NEMO_MAX_NEURON_ARGS];
	for(unsigned i=3; i<nargs; ++i) {
		if(!vectorized[i]) {
			args[i] = extract<float>(objects[i]);
		}
	}

	if(vlen == 0) {
		/* All inputs are scalars, the 'scalars' array has already been
		 * populated. */
		//! \todo deal with empty list
		net.addNeuron(neuron_type, neuron_index, nargs-3, &args[3]);
	} else {
		/* At least some inputs are vectors */
		for(unsigned i=0; i < vlen; ++i) {
			/* Fill in the vector arguments */
			if(vectorized[2]) {
				neuron_index = extract<unsigned>(PySequence_GetItem(objects[2], i));
			}
			for(unsigned j=3; j<nargs; ++j) {
				if(vectorized[j]) {
					args[j] = extract<float>(PySequence_GetItem(objects[j], i));
				}
			}
			net.addNeuron(neuron_type, neuron_index, nargs-3, &args[3]);
		}
	}
	return object();
}



/*! Modify one or more neurons of arbitrary type
 *
 * \param args parameters and state variables, see below
 * \param kwargs unused, but required by boost::python
 * \return None
 *
 * This function expects the following arguments:
 *
 * - 0  : network or simulation (class reference scalar)
 * - 1  : neuron index (unsigned scalar/vector)
 * - >2 : parameters and state variables (float scalar/vector)
 *
 * The corresponding python prototype would be
 *
 * set_neuron(self, neuron_idx, *args)
 *
 * The non-self arguments may be either scalar or vector. All vectors must be
 * of the same length. If any of the inputs are vectors, the scalar arguments
 * are replicated for each neuron.
 *
 * \see nemo::Network::setNeuron nemo::Simulation::setNeuron
 */
template<class T>
boost::python::object
set_neuron_va(boost::python::tuple py_args, boost::python::dict /*kwargs*/)
{
	using namespace boost::python;

	/* Common length of all non-scalar arguments */
	unsigned vlen = 0;
	unsigned nargs = boost::python::len(py_args);
	T& net = boost::python::extract<T&>(py_args[0])();

	PyObject* objects[NEMO_MAX_NEURON_ARGS];
	bool vectorized[NEMO_MAX_NEURON_ARGS];

	assert(nargs < NEMO_MAX_NEURON_ARGS-2);

	//! \todo shift everything down to minimise space usage
	/* Get raw pointers and determine the mix of scalar and vector arguments */

	for(unsigned i=1; i<nargs; ++i) {
		objects[i] = static_cast<boost::python::object>(py_args[i]).ptr();
		vectorized[i] = checkInputVector(objects[i], vlen);
	}

	/* Get the neuron index, if it's a scalar */
	unsigned neuron_index = 0;
	if(!vectorized[1]) {
		neuron_index = extract<unsigned>(py_args[1]);
	}

	/* Get all scalar parameters and state variables */
	//! \todo fold this back into previous loop? Need to deal with index first, then loop
	float args[NEMO_MAX_NEURON_ARGS];
	for(unsigned i=2; i<nargs; ++i) {
		if(!vectorized[i]) {
			args[i] = extract<float>(objects[i]);
		}
	}

	if(vlen == 0) {
		/* All inputs are scalars */
		//! \todo deal with empty list
		net.setNeuron(neuron_index, nargs-2, &args[2]);
	} else {
		/* At least some inputs are vectors */
		for(unsigned i=0; i < vlen; ++i) {
			/* Fill in the vector arguments */
			if(vectorized[1]) {
				neuron_index = extract<unsigned>(PySequence_GetItem(objects[1], i));
			}
			for(unsigned j=2; j<nargs; ++j) {
				if(vectorized[j]) {
					args[j] = extract<float>(PySequence_GetItem(objects[j], i));
				}
			}
			net.setNeuron(neuron_index, nargs-2, &args[2]);
		}
	}
	return object();
}



unsigned
set_neuron_x_length(PyObject* a, PyObject* b)
{
	unsigned len = 0;

	bool vectorA = checkInputVector(a, len);
	bool vectorB = checkInputVector(b, len);

	if(vectorA != vectorB) {
		throw std::invalid_argument("first and third argument must either both be scalar or lists of same length");
	}
	return len;
}



/*! Set neuron parameters for one or more neurons
 *
 * On the Python side the syntax is net.set_neuron_parameter(neurons, param,
 * values). Either these are all scalar, or neurons and values are both lists
 * of the same length.
 */
template<class T>
void
set_neuron_parameter(T& obj, PyObject* neurons, unsigned param, PyObject* values)
{
	const unsigned len = set_neuron_x_length(neurons, values);
	if(len == 0) {
		obj.setNeuronParameter(extract<unsigned>(neurons), param, extract<float>(values));
	} else {
		for(unsigned i=0; i < len; ++i) {
			unsigned neuron = extract<unsigned>(PySequence_GetItem(neurons, i));
			float value = extract<float>(PySequence_GetItem(values, i));
			obj.setNeuronParameter(neuron, param, value);
		}
	}
}



/*! Set neuron state for one or more neurons
 *
 * On the Python side the syntax is net.set_neuron_state(neurons, param,
 * values). Either these are all scalar, or neurons and values are both lists
 * of the same length.
 */
template<class T>
void
set_neuron_state(T& obj, PyObject* neurons, unsigned param, PyObject* values)
{
	const unsigned len = set_neuron_x_length(neurons, values);
	if(len == 0) {
		obj.setNeuronState(extract<unsigned>(neurons), param, extract<float>(values));
	} else {
		for(unsigned i=0; i < len; ++i) {
			unsigned neuron = extract<unsigned>(PySequence_GetItem(neurons, i));
			float value = extract<float>(PySequence_GetItem(values, i));
			obj.setNeuronState(neuron, param, value);
		}
	}
}



template<class T>
PyObject*
get_neuron_parameter(T& obj, PyObject* neurons, unsigned param)
{
	const Py_ssize_t len = PySequence_Check(neurons) ? PySequence_Size(neurons) : 0;
	if(len == 0) {
		return PyFloat_FromDouble(obj.getNeuronParameter(extract<unsigned>(neurons), param));
	} else {
		PyObject* list = PyList_New(len);
		for(Py_ssize_t i=0; i < len; ++i) {
			const unsigned neuron = extract<unsigned>(PySequence_GetItem(neurons, i));
			const float val = obj.getNeuronParameter(neuron, param);
			PyList_SetItem(list, i, PyFloat_FromDouble(val));
		}
		return list;
	}
}



template<class T>
PyObject*
get_neuron_state(T& obj, PyObject* neurons, unsigned param)
{
	const Py_ssize_t len = PySequence_Check(neurons) ? PySequence_Size(neurons) : 0;
	if(len == 0) {
		return PyFloat_FromDouble(obj.getNeuronState(extract<unsigned>(neurons), param));
	} else {
		PyObject* list = PyList_New(len);
		for(Py_ssize_t i=0; i < len; ++i) {
			const unsigned neuron = extract<unsigned>(PySequence_GetItem(neurons, i));
			const float val = obj.getNeuronState(neuron, param);
			PyList_SetItem(list, i, PyFloat_FromDouble(val));
		}
		return list;
	}
}



/*! Return the membrane potential of one or more neurons */
PyObject*
get_membrane_potential(nemo::Simulation& sim, PyObject* neurons)
{
	const Py_ssize_t len = PySequence_Check(neurons) ? PySequence_Size(neurons) : 0;
	if(len == 0) {
		return PyFloat_FromDouble(sim.getMembranePotential(extract<unsigned>(neurons)));
	} else {
		PyObject* list = PyList_New(len);
		for(Py_ssize_t i=0; i < len; ++i) {
			const unsigned neuron = extract<unsigned>(PySequence_GetItem(neurons, i));
			const float val = sim.getMembranePotential(neuron);
			PyList_SetItem(list, i, PyFloat_FromDouble(val));
		}
		return list;
	}
}



/* Convert scalar type to corresponding C++ type. Oddly, boost::python does not
 * seem to have this */
template<typename T>
PyObject*
insert(T)
{
	throw std::logic_error("invalid static type conversion in Python/C++ interface");
}


template<> PyObject* insert<float>(float x) { return PyFloat_FromDouble(x); }
template<> PyObject* insert<unsigned>(unsigned x) { return PyInt_FromLong(x); }
template<> PyObject* insert<unsigned char>(unsigned char x) { return PyBool_FromLong(x); }



/*! Return scalar or vector synapse parameter/state of type R from a
 * ReadableNetwork instance of type Net */
template<typename T>
PyObject*
get_synapse_x(const nemo::ReadableNetwork& net,
		PyObject* ids,
		std::const_mem_fun1_ref_t<T, nemo::ReadableNetwork, const synapse_id&> get_x)
{
	const Py_ssize_t len = PySequence_Check(ids) ? PySequence_Size(ids) : 0;
	if(len == 0) {
		return insert<T>(get_x(net, extract<synapse_id>(ids)));
	} else {
		PyObject* list = PyList_New(len);
		for(Py_ssize_t i=0; i < len; ++i) {
			synapse_id id = extract<synapse_id>(PySequence_GetItem(ids, i));
			const T val = get_x(net, id);
			PyList_SetItem(list, i, insert<T>(val));
		}
		return list;
	}
}



template<class Net>
PyObject*
get_synapse_source(const Net& net, PyObject* ids)
{
	return get_synapse_x<unsigned>(
			static_cast<const nemo::ReadableNetwork&>(net),
			ids, std::mem_fun_ref(&nemo::ReadableNetwork::getSynapseSource));
}


template<class Net>
PyObject*
get_synapse_target(const Net& net, PyObject* ids)
{
	return get_synapse_x<unsigned>(
			static_cast<const nemo::ReadableNetwork&>(net),
			ids, std::mem_fun_ref(&nemo::ReadableNetwork::getSynapseTarget));
}


template<class Net>
PyObject*
get_synapse_delay(const Net& net, PyObject* ids)
{
	return get_synapse_x<unsigned>(
			static_cast<const nemo::ReadableNetwork&>(net),
			ids, std::mem_fun_ref(&nemo::ReadableNetwork::getSynapseDelay));
}


template<class Net>
PyObject*
get_synapse_weight(const Net& net, PyObject* ids)
{
	return get_synapse_x<float>(
			static_cast<const nemo::ReadableNetwork&>(net),
			ids, std::mem_fun_ref(&nemo::ReadableNetwork::getSynapseWeight));
}


template<class Net>
PyObject*
get_synapse_plastic(const Net& net, PyObject* ids)
{
	return get_synapse_x<unsigned char>(
			static_cast<const nemo::ReadableNetwork&>(net),
			ids, std::mem_fun_ref(&nemo::ReadableNetwork::getSynapsePlastic));
}



/* This wrappers for overloads of nemo::Simulation::step */
const std::vector<unsigned>&
step_noinput(nemo::Simulation& sim)
{
	return sim.step();
}


const std::vector<unsigned>&
step_f(nemo::Simulation& sim, const std::vector<unsigned>& fstim)
{
	return sim.step(fstim);
}


const std::vector<unsigned>&
step_i(nemo::Simulation& sim, const std::vector< std::pair<unsigned, float> >& istim)
{
	return sim.step(istim);
}


const std::vector<unsigned>&
step_fi(nemo::Simulation& sim,
		const std::vector<unsigned>& fstim,
		const std::vector< std::pair<unsigned, float> >& istim)
{
	return sim.step(fstim, istim);
}



#ifdef NEMO_BRIAN_ENABLED

/*! \copydoc nemo::Simulation::propagate */
tuple
propagate(nemo::Simulation& sim, size_t fired, int nfired)
{
	std::pair<size_t, size_t> ret = sim.propagate(fired, nfired);
	return make_tuple(ret.first, ret.second);
}

#endif



void
initializeConverters()
{
	// register the from-python converter
	from_py_list<synapse_id>();
	from_py_list<unsigned>();
	from_py_list<unsigned char>();
	from_py_list<float>();
	from_py_list_of_pairs<unsigned, float>();
}


/* The STDP configuration comes in two forms in the C++ API. Use just the
 * original form here, in order to avoid breaking existing code. */
void (nemo::Configuration::*stdp2)(
		const std::vector<float>& prefire,
		const std::vector<float>& postfire,
		float minWeight,
		float maxWeight) = &nemo::Configuration::setStdpFunction;


BOOST_PYTHON_MODULE(_nemo)
{
	def("init", initializeConverters);

	class_<std::vector<unsigned> >("std_vector_unsigned")
		.def(vector_indexing_suite<std::vector<unsigned> >())
		.def("__str__", &std_vector_str<unsigned>)
	;

	class_<std::vector<float> >("std_vector_float")
		.def(vector_indexing_suite<std::vector<float> >())
		.def("__str__", &std_vector_str<float>)
	;

	class_<std::vector<unsigned char> >("std_vector_uchar")
		.def(vector_indexing_suite<std::vector<unsigned char> >())
		.def("__str__", &std_bool_vector_str)
	;

	class_<std::vector<uint64_t> >("std_vector_uint64")
		.def(vector_indexing_suite<std::vector<uint64_t> >())
		.def("__str__", &std_vector_str<uint64_t>)
	;

	class_<nemo::Configuration>("Configuration", CONFIGURATION_DOC)
		//.def("enable_logging", &nemo::Configuration::enableLogging)
		//.def("disable_logging", &nemo::Configuration::disableLogging)
		//.def("logging_enabled", &nemo::Configuration::loggingEnabled)
		.def("set_stdp_function", stdp2, CONFIGURATION_SET_STDP_FUNCTION_DOC)
		.def("set_cuda_backend", &nemo::Configuration::setCudaBackend, CONFIGURATION_SET_CUDA_BACKEND_DOC)
		.def("set_cpu_backend", &nemo::Configuration::setCpuBackend, CONFIGURATION_SET_CPU_BACKEND_DOC)
		.def("backend_description", &nemo::Configuration::backendDescription, CONFIGURATION_BACKEND_DESCRIPTION_DOC)
	;

	class_<nemo::Network, boost::noncopyable>("Network", NETWORK_DOC)
		.def("add_neuron_type", &nemo::Network::addNeuronType, NETWORK_ADD_NEURON_TYPE_DOC)
		.def("add_neuron", raw_function(add_neuron_va, 3), NETWORK_ADD_NEURON_DOC)
		.def("add_synapse", add_synapse, NETWORK_ADD_SYNAPSE_DOC)
		.def("set_neuron", raw_function(set_neuron_va<nemo::Network>, 2), CONSTRUCTABLE_SET_NEURON_DOC)
		.def("get_neuron_state", get_neuron_state<nemo::Network>, CONSTRUCTABLE_GET_NEURON_STATE_DOC)
		.def("get_neuron_parameter", get_neuron_parameter<nemo::Network>, CONSTRUCTABLE_GET_NEURON_PARAMETER_DOC)
		.def("set_neuron_state", set_neuron_state<nemo::Network>, CONSTRUCTABLE_SET_NEURON_STATE_DOC)
		.def("set_neuron_parameter", set_neuron_parameter<nemo::Network>, CONSTRUCTABLE_SET_NEURON_PARAMETER_DOC)
		.def("get_synapse_source", &nemo::Network::getSynapseSource)
		.def("neuron_count", &nemo::Network::neuronCount, NETWORK_NEURON_COUNT_DOC)
		.def("get_synapses_from", &nemo::Network::getSynapsesFrom, return_value_policy<copy_const_reference>(), CONSTRUCTABLE_GET_SYNAPSES_FROM_DOC)
		.def("get_synapse_source", get_synapse_source<nemo::Network>, CONSTRUCTABLE_GET_SYNAPSE_SOURCE_DOC)
		.def("get_synapse_target", get_synapse_target<nemo::Network>, CONSTRUCTABLE_GET_SYNAPSE_TARGET_DOC)
		.def("get_synapse_delay", get_synapse_delay<nemo::Network>, CONSTRUCTABLE_GET_SYNAPSE_DELAY_DOC)
		.def("get_synapse_weight", get_synapse_weight<nemo::Network>, CONSTRUCTABLE_GET_SYNAPSE_WEIGHT_DOC)
		.def("get_synapse_plastic", get_synapse_plastic<nemo::Network>, CONSTRUCTABLE_GET_SYNAPSE_PLASTIC_DOC)
	;

	class_<nemo::Simulation, boost::shared_ptr<nemo::Simulation>, boost::noncopyable>(
			"Simulation", SIMULATION_DOC, no_init)
		.def("__init__", make_constructor(makeSimulation))
		/* For the step function(s) named optional input arguments is handled
		 * in pure python. See __init__.py. */
		/* May want to make a copy here, for some added safety:
		 * return_value_policy<copy_const_reference>()
		 *
		 * In the current form we return a reference to memory handled by the
		 * simulation object, which may be overwritten by subsequent calls to
		 * to this function. */
		.def("step_noinput", step_noinput, return_internal_reference<1>())
		.def("step_f", step_f, return_internal_reference<1>())
		.def("step_i", step_i, return_internal_reference<1>())
		.def("step_fi", step_fi, return_internal_reference<1>())
#ifdef NEMO_BRIAN_ENABLED
		.def("propagate", propagate, SIMULATION_PROPAGATE_DOC)
#endif
		.def("apply_stdp", &nemo::Simulation::applyStdp, SIMULATION_APPLY_STDP_DOC)
		.def("set_neuron", raw_function(set_neuron_va<nemo::Simulation>, 2), CONSTRUCTABLE_SET_NEURON_DOC)
		.def("get_neuron_state", get_neuron_state<nemo::Simulation>, CONSTRUCTABLE_GET_NEURON_STATE_DOC)
		.def("get_neuron_parameter", get_neuron_parameter<nemo::Simulation>, CONSTRUCTABLE_GET_NEURON_PARAMETER_DOC)
		.def("set_neuron_state", set_neuron_state<nemo::Simulation>, CONSTRUCTABLE_SET_NEURON_STATE_DOC)
		.def("set_neuron_parameter", set_neuron_parameter<nemo::Simulation>, CONSTRUCTABLE_SET_NEURON_PARAMETER_DOC)
		.def("get_membrane_potential", get_membrane_potential, SIMULATION_GET_MEMBRANE_POTENTIAL_DOC)
		.def("get_synapses_from", &nemo::Simulation::getSynapsesFrom, return_value_policy<copy_const_reference>(), CONSTRUCTABLE_GET_SYNAPSES_FROM_DOC)
		.def("get_synapse_source", get_synapse_source<nemo::Simulation>, CONSTRUCTABLE_GET_SYNAPSE_SOURCE_DOC)
		.def("get_synapse_target", get_synapse_target<nemo::Simulation>, CONSTRUCTABLE_GET_SYNAPSE_TARGET_DOC)
		.def("get_synapse_delay", get_synapse_delay<nemo::Simulation>, CONSTRUCTABLE_GET_SYNAPSE_DELAY_DOC)
		.def("get_synapse_weight", get_synapse_weight<nemo::Simulation>, CONSTRUCTABLE_GET_SYNAPSE_WEIGHT_DOC)
		.def("get_synapse_plastic", get_synapse_plastic<nemo::Simulation>, CONSTRUCTABLE_GET_SYNAPSE_PLASTIC_DOC)
		.def("elapsed_wallclock", &nemo::Simulation::elapsedWallclock, SIMULATION_ELAPSED_WALLCLOCK_DOC)
		.def("elapsed_simulation", &nemo::Simulation::elapsedSimulation, SIMULATION_ELAPSED_SIMULATION_DOC)
		.def("reset_timer", &nemo::Simulation::resetTimer, SIMULATION_RESET_TIMER_DOC)
	;
}
