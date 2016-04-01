function type = nemoAddNeuronType(name)
% nemoAddNeuronType - register a new neuron type with the network
%  
% Synopsis:
%   type = nemoAddNeuronType(name)
%  
% Inputs:
%   name    - canonical name of the neuron type. The neuron type data
%             is loaded from a plugin configuration file of the same name.
%    
% Outputs:
%   type    - index of the the neuron type, to be used when adding
%             neurons
%    
% This function must be called before neurons of the specified type
% can be added to the network.
    type = nemo_mex(uint32(0), name);
end