#include "neuron_model.h"

extern "C"
NEMO_PLUGIN_DLL_PUBLIC
cudaError_t
cuda_init_neurons( 
		unsigned globalPartitionCount,
		unsigned localPartitionCount,
		unsigned basePartition,
		unsigned* d_partitionSize,
		param_t* d_params,
		float* df_neuronParameters,
		float* df_neuronState,
		nrng_t /* rng */,
		uint32_t* d_valid)
{
	return cudaSuccess;
}

	            
cuda_init_neurons_t* test_init = &cuda_init_neurons;

