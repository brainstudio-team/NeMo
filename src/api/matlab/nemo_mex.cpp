#include <vector>
#include <algorithm>
#include <iterator>
#include <string>

#include <boost/numeric/conversion/cast.hpp>
#include <mex.h>

#include <nemo.h>
#include <nemo/types.h>


/* Unique global objects containing all NeMo state */
static nemo_network_t g_network = NULL;
static nemo_configuration_t g_configuration = NULL;
static nemo_simulation_t g_simulation = NULL;



/* When returning data to Matlab we need to specify the class of the data.
 * There should be a one-to-one mapping between the supported C types and
 * Matlab classes. The full list of supported classes can be found in 
 * /path-to-matlab/extern/include/matrix.h */
template<typename M> mxClassID classId() { 
	mexErrMsgIdAndTxt("nemo:mex", "programming error: class id requested for unknown class");
	return mxUNKNOWN_CLASS;
}
template<> mxClassID classId<char*>() { return mxCHAR_CLASS; }
template<> mxClassID classId<uint8_t>() { return mxUINT8_CLASS; }
template<> mxClassID classId<uint32_t>() { return mxUINT32_CLASS; }
template<> mxClassID classId<int32_t>() { return mxINT32_CLASS; }
template<> mxClassID classId<uint64_t>() { return mxUINT64_CLASS; }
template<> mxClassID classId<double>() { return mxDOUBLE_CLASS; }



inline
void
checkNemoStatus(nemo_status_t s)
{
	if(s != NEMO_OK) {
		mexErrMsgIdAndTxt("nemo:backend", nemo_strerror());
	}
}



template<typename T>
const mxArray*
numeric(const mxArray* arr)
{
	if(!mxIsNumeric(arr)) {
		mexErrMsgIdAndTxt("nemo:api", "argument should be numeric\n");
	}
	if(mxGetClassID(arr) != classId<T>()) {
		/* to be able to report the expected class name we have to create a
		 * temporary array, as there's no simpler way of getting the name. 
		 * Matlab will garbage collect this. */
		mxArray* tmp = mxCreateNumericMatrix(1, 1, classId<T>(), mxREAL);
		mexErrMsgIdAndTxt("nemo:api", "expected input of class %s, but found %s",
				mxGetClassName(tmp), mxGetClassName(arr));
	}
	return arr;
}



/* Return scalar from a numeric 1x1 array provided by Matlab. M and N refers to
 * Matlab and Nemo types, rather than array dimensions. */
template<typename N, typename M>
N
scalar(const mxArray* arr)
{
	if(mxGetN(arr) != 1 || mxGetM(arr) != 1) {
		mexErrMsgIdAndTxt("nemo:api", "argument should be scalar");
	}
	// target, source
	return boost::numeric_cast<N, M>(*static_cast<M*>(mxGetData(numeric<M>(arr))));
}



/* Return scalar from a given position in a Matlab array. M and N refers to
 * Matlab and Nemo types, rather than array dimensions. Vectorised functions
 * can use a mix of scalar and vector inputs. The length of the relevant input
 * is found in len. Scalar inputs ignore the offset. */
template<typename N, typename M>
N
scalarAt(const mxArray* arr, size_t i, unsigned len)
{
	size_t offset = len == 1 ? 0 : i;
	/* No bounds checking here since we have already verified the bonds for all input vectors */
	return boost::numeric_cast<N, M>(*(static_cast<M*>(mxGetData(numeric<M>(arr)))+offset));
}



/* Return vector from a numeric 1 x m array provided by Matlab. M and N refers
 * to Matlab and Nemo types, rather than array dimensions. */
template<typename N, typename M>
std::vector<N> // let's hope the compiler can optimise the return...
vector(const mxArray* arr)
{
	std::vector<N> ret;
	if(mxGetM(arr) == 0 && mxGetN(arr) == 0) {
		return ret;
	}

	if(mxGetM(arr) != 1) {
		mexErrMsgIdAndTxt("nemo:api", 
				"argument should be 1 x m vector. Size is %u x %u",
				mxGetM(arr), mxGetN(arr));
	}

	size_t length = mxGetN(arr);
	M* begin = static_cast<M*>(mxGetData(numeric<M>(arr)));
	std::transform(begin, begin + length,
			std::back_inserter(ret),
			boost::numeric_cast<N, M>);
	return ret;
}



void
checkInputCount(int actualArgs, int expectedArgs)
{
	// The function id is always an extra parameter
	if(actualArgs - 1 != expectedArgs) {
		mexErrMsgIdAndTxt("nemo:api", "found %u input arguments, but expected %u",
			actualArgs - 1, expectedArgs);
	}
}


void
checkOutputCount(int actualArgs, int expectedArgs)
{
	if(actualArgs != expectedArgs) {
		mexErrMsgIdAndTxt("nemo:api", "found %u output arguments, but expected %u",
			actualArgs, expectedArgs);
	}
}



/* Print out all dimensions and lengths of all arguments (either input or output) */
void
reportVectorDimensions(int argc, const mxArray* argv[])
{
	for(int i=0; i < argc; ++i) {
		size_t n = mxGetN(argv[i]);
		size_t m = mxGetM(argv[i]);
		mexPrintf("argument %u: size=(%u,%u) length=%u\n", i, m, n, m*n);
	}
}



/* For the vector form of functions which are scalar in the C++ API (i.e. all
 * inputs and outputs are scalars) we allow using a vector form in Matlab. For
 * these functions we support a mix of scalar and vector input, as long as all
 * vectors have the same length. scalar inputs are simply replicated as
 * appropriate. The precise shape of the matrices do not matter. */
size_t
vectorDimension(int nrhs, const mxArray* prhs[], unsigned arglen[])
{
	if(nrhs < 1) {
		mexErrMsgIdAndTxt("nemo:api", "function should have at least one input argument");
	}

	size_t dim = 1U;
	size_t i = 0U;

	/* Skip initial scalars */
	for(size_t i_max = nrhs; i < i_max && dim == 1U; ++i) {
		dim = arglen[i] = mxGetN(prhs[i]) * mxGetM(prhs[i]);
	}

	/* Verify remaining vectors */
	for(size_t i_max = nrhs; i < i_max; ++i) {
		arglen[i] = mxGetN(prhs[i]) * mxGetM(prhs[i]);
		if(arglen[i] != dim && arglen[i] != 1) {
			reportVectorDimensions(nrhs, prhs);
			mexErrMsgIdAndTxt("nemo:api", "vector arguments do not have the same size");
		}
	}

	return dim;
}



/* Return numeric scalar in output argument 0 after doing the appropriate
 * conversion from the type used by nemo (N) to the type used by matlab (M). */
template<typename N, typename M>
void
returnScalar(mxArray* plhs[], int argno, N val)
{
	plhs[argno] = mxCreateNumericMatrix(1, 1, classId<M>(), mxREAL);
	*(static_cast<M*>(mxGetData(plhs[argno]))) = val;
}


template<>
void
returnScalar<const char*, char*>(mxArray* plhs[], int argno, const char* str)
{
	plhs[argno] = mxCreateString(str);
}



/* Allocate memory for return vector */
template<typename M>
void
allocateOutputVector(mxArray* plhs[], int argno, size_t len)
{
	plhs[argno] = mxCreateNumericMatrix(1, len, classId<M>(), mxREAL);
}



/* Set an element in output array and do the required casting from the type
 * used by nemo (N) to the type used by matlab (M). */
template<typename N, typename M>
void
returnScalarAt(mxArray* plhs[], int argno, size_t offset, N val)
{
	static_cast<M*>(mxGetData(plhs[argno]))[offset] = boost::numeric_cast<N, M>(val);
}



/* Return numeric vector in output argument n after doing the appropriate
 * conversion from the type used by nemo (N) to the type used by matlab (M). */
template<typename N, typename M>
void
returnVector(mxArray* plhs[], int argno, const std::vector<N>& vec)
{
	mxArray* ret = mxCreateNumericMatrix(1, vec.size(), classId<M>(), mxREAL);
	std::transform(vec.begin(), vec.end(), 
			static_cast<M*>(mxGetData(ret)),
			boost::numeric_cast<N, M>);
	plhs[argno] = ret;
}


/* Return numeric vector in output argument n after doing the appropriate
 * conversion from the type used by nemo (N) to the type used by matlab (M). */
template<typename N, typename M>
void
returnVector(mxArray* plhs[], int argno, N* arr, unsigned len)
{
	mxArray* ret = mxCreateNumericMatrix(1, len, classId<M>(), mxREAL);
	std::transform(arr, arr+len,
			static_cast<M*>(mxGetData(ret)),
			boost::numeric_cast<N, M>);
	plhs[argno] = ret;
}



void
deleteGlobals()
{
	if(g_simulation != NULL) {
		nemo_delete_simulation(g_simulation);
		g_simulation = NULL;
	}
	if(g_network != NULL) {
		nemo_delete_network(g_network);
		g_network = NULL;
	}
	if(g_configuration != NULL) {
		nemo_delete_configuration(g_configuration);
		g_configuration = NULL;
	}
}



nemo_network_t
getNetwork()
{
	if(g_network == NULL) {
		g_network = nemo_new_network();
		if(g_network == NULL) {
			mexErrMsgIdAndTxt("nemo:backend", "failed to create network: %s", nemo_strerror());
		}
		mexAtExit(deleteGlobals);
	}
	return g_network;
}



void
clearNetwork(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if(g_network == NULL) {
		/* This should not be considered a usage an error. If the user has not
		 * added any neurons or synapses the network object is NULL. Clearing
		 * this should be perfectly valid. */
		return;
	}
	nemo_delete_network(g_network);
	g_network = NULL;
}



nemo_configuration_t
getConfiguration()
{
	if(g_configuration == NULL) {
		g_configuration = nemo_new_configuration();
		if(g_configuration == NULL) {
			mexErrMsgIdAndTxt("nemo:backend", "failed to create configuration: %s", nemo_strerror());
		}
		mexAtExit(deleteGlobals);
	}
	return g_configuration;
}



/*! Reset the configuration object to the default settings */
void
resetConfiguration(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if(g_configuration != NULL) {
		nemo_delete_configuration(g_configuration);
		g_configuration = NULL;
	}
	g_configuration = nemo_new_configuration();
	if(g_configuration == NULL) {
		mexErrMsgIdAndTxt("nemo:backend", "failed to create configuration: %s", nemo_strerror());
	}
}


/*! The simulation only exists between calls to \a createSimulation and
 * \a destroySimulation. Asking for the simulation object outside this region is
 * an error. */
nemo_simulation_t
getSimulation()
{
	if(g_simulation == NULL) {
		mexErrMsgIdAndTxt("nemo:api", "Non-existing simulation object requested. Make sure nemoCreateSimulation is called before any simulation commands.");
	}
	return g_simulation;
}



void
createSimulation(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if(g_simulation != NULL) {
		mexErrMsgIdAndTxt("nemo:api", "Simulation already exists. Call nemoDestroySimulation if you want to delete the existing simulation.");
		return;
	}
	nemo_network_t net = getNetwork();
	nemo_configuration_t conf = getConfiguration();
	nemo_simulation_t sim = nemo_new_simulation(net, conf);
	if(sim == NULL) {
		mexErrMsgIdAndTxt("nemo:backend", "Failed to create simulation: %s", nemo_strerror());
	}
	g_simulation = sim;
	mexAtExit(deleteGlobals);
}



bool
isSimulating()
{
	return g_simulation != NULL;
}



void
destroySimulation(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if(g_simulation == NULL) {
		mexErrMsgIdAndTxt("nemo:api", "Attempt to stop simulation when simulation is not running");
	}
	nemo_delete_simulation(g_simulation);
	g_simulation = NULL;
}



void
reset(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	if(g_simulation != NULL) {
		nemo_delete_simulation(g_simulation);
		g_simulation = NULL;
	}
	clearNetwork(nlhs, plhs, nrhs, prhs);
	resetConfiguration(nlhs, plhs, nrhs, prhs);
}



void
addNeuron(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	unsigned arglen[NEMO_MAX_NEURON_ARGS];
	float args[NEMO_MAX_NEURON_ARGS];

	/* Number of arguments in variable-length argument list */
	int nargs = nrhs-3;
	if(nargs < 0) {
		mexErrMsgIdAndTxt("nemo:api", "missing arguments");
	}
	if(nargs > NEMO_MAX_NEURON_ARGS) {
		mexErrMsgIdAndTxt("nemo:mex", "too many arguments");
	}
	size_t elems = vectorDimension(nargs+2, prhs + 1, arglen);
	checkOutputCount(nlhs, 0);
	nemo_network_t hdl = getNetwork();
	for(size_t i=0; i<elems; ++i){
		for(int a=0; a<nargs; ++a) {
			args[a] = scalarAt<float,double>(prhs[3+a], i, arglen[2+a]);
		}
		checkNemoStatus(
				nemo_add_neuron(
					hdl,
					scalarAt<unsigned,uint32_t>(prhs[1], i, arglen[0]),
					scalarAt<unsigned,uint32_t>(prhs[2], i, arglen[1]),
					nargs,
					args
				)
		);
	}
}



void
setNeuron(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	unsigned arglen[NEMO_MAX_NEURON_ARGS];
	float args[NEMO_MAX_NEURON_ARGS];

	/* Number of arguments in variable-length argument list */
	int nargs = nrhs-2;
	if(nargs < 0) {
		mexErrMsgIdAndTxt("nemo:api", "missing arguments");
	}
	if(nargs > NEMO_MAX_NEURON_ARGS) {
		mexErrMsgIdAndTxt("nemo:mex", "too many arguments");
	}
	size_t elems = vectorDimension(nargs+1, prhs+1, arglen);
	checkOutputCount(nlhs, 0);
	if(isSimulating()) {
		nemo_simulation_t hdl = getSimulation();
		for(size_t i=0; i<elems; ++i){
			for(int a=0; a<nargs; ++a) {
				args[a] = scalarAt<float,double>(prhs[2+a], i, arglen[1+a]);
			}
			checkNemoStatus(nemo_set_neuron_s(hdl, scalarAt<unsigned,uint32_t>(prhs[1], i, arglen[0]), nargs, args));
		}
	} else {
		nemo_network_t hdl = getNetwork();
		for(size_t i=0; i<elems; ++i){
			for(int a=0; a<nargs; ++a) {
				args[a] = scalarAt<float,double>(prhs[2+a], i, arglen[1+a]);
			}
			checkNemoStatus(nemo_set_neuron_n(hdl, scalarAt<unsigned,uint32_t>(prhs[1], i, arglen[0]), nargs, args));
		}
	}
}



void
setStdpFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 4);
    checkOutputCount(nlhs, 0);
    std::vector<float> prefire = vector<float, double>(prhs[1]);
    std::vector<float> postfire = vector<float, double>(prhs[2]);
    checkNemoStatus(
            nemo_set_stdp_function(
                    getConfiguration(),
                    &prefire[0], prefire.size(),
                    &postfire[0], postfire.size(),
					0.0f, scalar<float,double>(prhs[4]),
                    0.0f, scalar<float,double>(prhs[3])
            )
    );
}



/* AUTO-GENERATED CODE START */

void
addNeuronType(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 1);
    char* name = mxArrayToString(prhs[1]);
    unsigned type;
    checkNemoStatus(nemo_add_neuron_type(getNetwork(), name, &type));
    mxFree(name);
    returnScalar<unsigned, uint32_t>(plhs, 0, type);
}



void
addSynapse(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    static unsigned arglen[5];
    size_t elems = vectorDimension(5, prhs + 1, arglen);
    checkInputCount(nrhs, 5);
    checkOutputCount(nlhs, 1);
    allocateOutputVector<uint64_t>(plhs, 0, elems);
    nemo_network_t hdl = getNetwork();
    for(size_t i=0; i<elems; ++i){
        uint64_t id;
        checkNemoStatus( 
                nemo_add_synapse( 
                        hdl, 
                        scalarAt<unsigned,uint32_t>(prhs[1], i, arglen[0]), 
                        scalarAt<unsigned,uint32_t>(prhs[2], i, arglen[1]), 
                        scalarAt<unsigned,uint32_t>(prhs[3], i, arglen[2]), 
                        scalarAt<float,double>(prhs[4], i, arglen[3]), 
                        scalarAt<unsigned char,uint8_t>(prhs[5], i, arglen[4]), 
                        &id 
                ) 
        );
        returnScalarAt<uint64_t, uint64_t>(plhs, 0, i, id);
    }
}



void
neuronCount(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 0);
    checkOutputCount(nlhs, 1);
    unsigned ncount;
    checkNemoStatus(nemo_neuron_count(getNetwork(), &ncount));
    returnScalar<unsigned, uint32_t>(plhs, 0, ncount);
}



void
setCpuBackend(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 0);
    checkOutputCount(nlhs, 0);
    checkNemoStatus(nemo_set_cpu_backend(getConfiguration()));
}



void
setCudaBackend(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 0);
    checkNemoStatus( 
            nemo_set_cuda_backend(getConfiguration(), scalar<int,int32_t>(prhs[1])) 
    );
}



void
backendDescription(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 0);
    checkOutputCount(nlhs, 1);
    const char* description;
    checkNemoStatus(nemo_backend_description(getConfiguration(), &description));
    returnScalar<const char*, char*>(plhs, 0, description);
}



void
setWriteOnlySynapses(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 0);
    checkOutputCount(nlhs, 0);
    checkNemoStatus(nemo_set_write_only_synapses(getConfiguration()));
}



void
logStdout(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 0);
    checkOutputCount(nlhs, 0);
    checkNemoStatus(nemo_log_stdout(getConfiguration()));
}



void
step(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 3);
    checkOutputCount(nlhs, 1);
    std::vector<unsigned> fstim = vector<unsigned, uint32_t>(prhs[1]);
    std::vector<unsigned> istim_nidx = vector<unsigned, uint32_t>(prhs[2]);
    std::vector<float> istim_current = vector<float, double>(prhs[3]);
    unsigned* fired;
    size_t fired_len;
    checkNemoStatus( 
            nemo_step( 
                    getSimulation(), 
                    &fstim[0], fstim.size(), 
                    &istim_nidx[0], 
                    &istim_current[0], istim_current.size(), 
                    &fired, &fired_len 
            ) 
    );
    returnVector<unsigned, uint32_t>(plhs, 0, fired, fired_len);
}



void
applyStdp(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 0);
    checkNemoStatus( 
            nemo_apply_stdp(getSimulation(), scalar<float,double>(prhs[1])) 
    );
}



void
getMembranePotential(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    static unsigned arglen[1];
    size_t elems = vectorDimension(1, prhs + 1, arglen);
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 1);
    allocateOutputVector<double>(plhs, 0, elems);
    nemo_simulation_t hdl = getSimulation();
    for(size_t i=0; i<elems; ++i){
        float v;
        checkNemoStatus( 
                nemo_get_membrane_potential(hdl, scalarAt<unsigned,uint32_t>(prhs[1], i, arglen[0]), &v) 
        );
        returnScalarAt<float, double>(plhs, 0, i, v);
    }
}



void
elapsedWallclock(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 0);
    checkOutputCount(nlhs, 1);
    unsigned long elapsed;
    checkNemoStatus(nemo_elapsed_wallclock(getSimulation(), &elapsed));
    returnScalar<unsigned long, uint64_t>(plhs, 0, elapsed);
}



void
elapsedSimulation(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 0);
    checkOutputCount(nlhs, 1);
    unsigned long elapsed;
    checkNemoStatus(nemo_elapsed_simulation(getSimulation(), &elapsed));
    returnScalar<unsigned long, uint64_t>(plhs, 0, elapsed);
}



void
resetTimer(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 0);
    checkOutputCount(nlhs, 0);
    checkNemoStatus(nemo_reset_timer(getSimulation()));
}



void
setNeuronState(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    static unsigned arglen[3];
    size_t elems = vectorDimension(3, prhs + 1, arglen);
    checkInputCount(nrhs, 3);
    checkOutputCount(nlhs, 0);
    if(isSimulating()){
        nemo_simulation_t hdl = getSimulation();
        for(size_t i=0; i<elems; ++i){
            checkNemoStatus( 
                    nemo_set_neuron_state_s( 
                            hdl, 
                            scalarAt<unsigned,uint32_t>(prhs[1], i, arglen[0]), 
                            scalarAt<unsigned,uint32_t>(prhs[2], i, arglen[1]), 
                            scalarAt<float,double>(prhs[3], i, arglen[2]) 
                    ) 
            );
        }
    } else {
        nemo_network_t hdl = getNetwork();
        for(size_t i=0; i<elems; ++i){
            checkNemoStatus( 
                    nemo_set_neuron_state_n( 
                            hdl, 
                            scalarAt<unsigned,uint32_t>(prhs[1], i, arglen[0]), 
                            scalarAt<unsigned,uint32_t>(prhs[2], i, arglen[1]), 
                            scalarAt<float,double>(prhs[3], i, arglen[2]) 
                    ) 
            );
        }
    }
}



void
setNeuronParameter(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    static unsigned arglen[3];
    size_t elems = vectorDimension(3, prhs + 1, arglen);
    checkInputCount(nrhs, 3);
    checkOutputCount(nlhs, 0);
    if(isSimulating()){
        nemo_simulation_t hdl = getSimulation();
        for(size_t i=0; i<elems; ++i){
            checkNemoStatus( 
                    nemo_set_neuron_parameter_s( 
                            hdl, 
                            scalarAt<unsigned,uint32_t>(prhs[1], i, arglen[0]), 
                            scalarAt<unsigned,uint32_t>(prhs[2], i, arglen[1]), 
                            scalarAt<float,double>(prhs[3], i, arglen[2]) 
                    ) 
            );
        }
    } else {
        nemo_network_t hdl = getNetwork();
        for(size_t i=0; i<elems; ++i){
            checkNemoStatus( 
                    nemo_set_neuron_parameter_n( 
                            hdl, 
                            scalarAt<unsigned,uint32_t>(prhs[1], i, arglen[0]), 
                            scalarAt<unsigned,uint32_t>(prhs[2], i, arglen[1]), 
                            scalarAt<float,double>(prhs[3], i, arglen[2]) 
                    ) 
            );
        }
    }
}



void
getNeuronState(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    static unsigned arglen[2];
    size_t elems = vectorDimension(2, prhs + 1, arglen);
    checkInputCount(nrhs, 2);
    checkOutputCount(nlhs, 1);
    allocateOutputVector<double>(plhs, 0, elems);
    if(isSimulating()){
        nemo_simulation_t hdl = getSimulation();
        for(size_t i=0; i<elems; ++i){
            float val;
            checkNemoStatus( 
                    nemo_get_neuron_state_s( 
                            hdl, 
                            scalarAt<unsigned,uint32_t>(prhs[1], i, arglen[0]), 
                            scalarAt<unsigned,uint32_t>(prhs[2], i, arglen[1]), 
                            &val 
                    ) 
            );
            returnScalarAt<float, double>(plhs, 0, i, val);
        }
    } else {
        nemo_network_t hdl = getNetwork();
        for(size_t i=0; i<elems; ++i){
            float val;
            checkNemoStatus( 
                    nemo_get_neuron_state_n( 
                            hdl, 
                            scalarAt<unsigned,uint32_t>(prhs[1], i, arglen[0]), 
                            scalarAt<unsigned,uint32_t>(prhs[2], i, arglen[1]), 
                            &val 
                    ) 
            );
            returnScalarAt<float, double>(plhs, 0, i, val);
        }
    }
}



void
getNeuronParameter(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    static unsigned arglen[2];
    size_t elems = vectorDimension(2, prhs + 1, arglen);
    checkInputCount(nrhs, 2);
    checkOutputCount(nlhs, 1);
    allocateOutputVector<double>(plhs, 0, elems);
    if(isSimulating()){
        nemo_simulation_t hdl = getSimulation();
        for(size_t i=0; i<elems; ++i){
            float val;
            checkNemoStatus( 
                    nemo_get_neuron_parameter_s( 
                            hdl, 
                            scalarAt<unsigned,uint32_t>(prhs[1], i, arglen[0]), 
                            scalarAt<unsigned,uint32_t>(prhs[2], i, arglen[1]), 
                            &val 
                    ) 
            );
            returnScalarAt<float, double>(plhs, 0, i, val);
        }
    } else {
        nemo_network_t hdl = getNetwork();
        for(size_t i=0; i<elems; ++i){
            float val;
            checkNemoStatus( 
                    nemo_get_neuron_parameter_n( 
                            hdl, 
                            scalarAt<unsigned,uint32_t>(prhs[1], i, arglen[0]), 
                            scalarAt<unsigned,uint32_t>(prhs[2], i, arglen[1]), 
                            &val 
                    ) 
            );
            returnScalarAt<float, double>(plhs, 0, i, val);
        }
    }
}



void
getSynapsesFrom(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 1);
    uint64_t* synapses;
    size_t synapses_len;
    if(isSimulating()){
        checkNemoStatus( 
                nemo_get_synapses_from_s( 
                        getSimulation(), 
                        scalar<unsigned,uint32_t>(prhs[1]), 
                        &synapses, &synapses_len 
                ) 
        );
    } else {
        checkNemoStatus( 
                nemo_get_synapses_from_n( 
                        getNetwork(), 
                        scalar<unsigned,uint32_t>(prhs[1]), 
                        &synapses, &synapses_len 
                ) 
        );
    }
    returnVector<uint64_t, uint64_t>(plhs, 0, synapses, synapses_len);
}



void
getSynapseSource(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    static unsigned arglen[1];
    size_t elems = vectorDimension(1, prhs + 1, arglen);
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 1);
    allocateOutputVector<uint32_t>(plhs, 0, elems);
    if(isSimulating()){
        nemo_simulation_t hdl = getSimulation();
        for(size_t i=0; i<elems; ++i){
            unsigned source;
            checkNemoStatus( 
                    nemo_get_synapse_source_s( 
                            hdl, 
                            scalarAt<uint64_t,uint64_t>(prhs[1], i, arglen[0]), 
                            &source 
                    ) 
            );
            returnScalarAt<unsigned, uint32_t>(plhs, 0, i, source);
        }
    } else {
        nemo_network_t hdl = getNetwork();
        for(size_t i=0; i<elems; ++i){
            unsigned source;
            checkNemoStatus( 
                    nemo_get_synapse_source_n( 
                            hdl, 
                            scalarAt<uint64_t,uint64_t>(prhs[1], i, arglen[0]), 
                            &source 
                    ) 
            );
            returnScalarAt<unsigned, uint32_t>(plhs, 0, i, source);
        }
    }
}



void
getSynapseTarget(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    static unsigned arglen[1];
    size_t elems = vectorDimension(1, prhs + 1, arglen);
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 1);
    allocateOutputVector<uint32_t>(plhs, 0, elems);
    if(isSimulating()){
        nemo_simulation_t hdl = getSimulation();
        for(size_t i=0; i<elems; ++i){
            unsigned target;
            checkNemoStatus( 
                    nemo_get_synapse_target_s( 
                            hdl, 
                            scalarAt<uint64_t,uint64_t>(prhs[1], i, arglen[0]), 
                            &target 
                    ) 
            );
            returnScalarAt<unsigned, uint32_t>(plhs, 0, i, target);
        }
    } else {
        nemo_network_t hdl = getNetwork();
        for(size_t i=0; i<elems; ++i){
            unsigned target;
            checkNemoStatus( 
                    nemo_get_synapse_target_n( 
                            hdl, 
                            scalarAt<uint64_t,uint64_t>(prhs[1], i, arglen[0]), 
                            &target 
                    ) 
            );
            returnScalarAt<unsigned, uint32_t>(plhs, 0, i, target);
        }
    }
}



void
getSynapseDelay(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    static unsigned arglen[1];
    size_t elems = vectorDimension(1, prhs + 1, arglen);
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 1);
    allocateOutputVector<uint32_t>(plhs, 0, elems);
    if(isSimulating()){
        nemo_simulation_t hdl = getSimulation();
        for(size_t i=0; i<elems; ++i){
            unsigned delay;
            checkNemoStatus( 
                    nemo_get_synapse_delay_s( 
                            hdl, 
                            scalarAt<uint64_t,uint64_t>(prhs[1], i, arglen[0]), 
                            &delay 
                    ) 
            );
            returnScalarAt<unsigned, uint32_t>(plhs, 0, i, delay);
        }
    } else {
        nemo_network_t hdl = getNetwork();
        for(size_t i=0; i<elems; ++i){
            unsigned delay;
            checkNemoStatus( 
                    nemo_get_synapse_delay_n( 
                            hdl, 
                            scalarAt<uint64_t,uint64_t>(prhs[1], i, arglen[0]), 
                            &delay 
                    ) 
            );
            returnScalarAt<unsigned, uint32_t>(plhs, 0, i, delay);
        }
    }
}



void
getSynapseWeight(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    static unsigned arglen[1];
    size_t elems = vectorDimension(1, prhs + 1, arglen);
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 1);
    allocateOutputVector<double>(plhs, 0, elems);
    if(isSimulating()){
        nemo_simulation_t hdl = getSimulation();
        for(size_t i=0; i<elems; ++i){
            float weight;
            checkNemoStatus( 
                    nemo_get_synapse_weight_s( 
                            hdl, 
                            scalarAt<uint64_t,uint64_t>(prhs[1], i, arglen[0]), 
                            &weight 
                    ) 
            );
            returnScalarAt<float, double>(plhs, 0, i, weight);
        }
    } else {
        nemo_network_t hdl = getNetwork();
        for(size_t i=0; i<elems; ++i){
            float weight;
            checkNemoStatus( 
                    nemo_get_synapse_weight_n( 
                            hdl, 
                            scalarAt<uint64_t,uint64_t>(prhs[1], i, arglen[0]), 
                            &weight 
                    ) 
            );
            returnScalarAt<float, double>(plhs, 0, i, weight);
        }
    }
}



void
getSynapsePlastic(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    static unsigned arglen[1];
    size_t elems = vectorDimension(1, prhs + 1, arglen);
    checkInputCount(nrhs, 1);
    checkOutputCount(nlhs, 1);
    allocateOutputVector<uint8_t>(plhs, 0, elems);
    if(isSimulating()){
        nemo_simulation_t hdl = getSimulation();
        for(size_t i=0; i<elems; ++i){
            unsigned char plastic;
            checkNemoStatus( 
                    nemo_get_synapse_plastic_s( 
                            hdl, 
                            scalarAt<uint64_t,uint64_t>(prhs[1], i, arglen[0]), 
                            &plastic 
                    ) 
            );
            returnScalarAt<unsigned char, uint8_t>(plhs, 0, i, plastic);
        }
    } else {
        nemo_network_t hdl = getNetwork();
        for(size_t i=0; i<elems; ++i){
            unsigned char plastic;
            checkNemoStatus( 
                    nemo_get_synapse_plastic_n( 
                            hdl, 
                            scalarAt<uint64_t,uint64_t>(prhs[1], i, arglen[0]), 
                            &plastic 
                    ) 
            );
            returnScalarAt<unsigned char, uint8_t>(plhs, 0, i, plastic);
        }
    }
}



typedef void (*fn_ptr)(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);
#define FN_COUNT 32
fn_ptr fn_arr[FN_COUNT] = {
    addNeuronType,
    addNeuron,
    addSynapse,
    neuronCount,
    clearNetwork,
    setCpuBackend,
    setCudaBackend,
    setStdpFunction,
    backendDescription,
    setWriteOnlySynapses,
    logStdout,
    resetConfiguration,
    step,
    applyStdp,
    getMembranePotential,
    elapsedWallclock,
    elapsedSimulation,
    resetTimer,
    createSimulation,
    destroySimulation,
    reset,
    setNeuron,
    setNeuronState,
    setNeuronParameter,
    getNeuronState,
    getNeuronParameter,
    getSynapsesFrom,
    getSynapseSource,
    getSynapseTarget,
    getSynapseDelay,
    getSynapseWeight,
    getSynapsePlastic
};

/* AUTO-GENERATED CODE END */
void
mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	/* The first argument specifies the function */
	uint32_t fn_idx = scalar<uint32_t, uint32_t>(prhs[0]);
	if(fn_idx >= FN_COUNT || fn_idx < 0) {
		mexErrMsgIdAndTxt("nemo:mex", "Unknown function index %u", fn_idx);
	}
	fn_arr[fn_idx](nlhs, plhs, nrhs, prhs);
}
