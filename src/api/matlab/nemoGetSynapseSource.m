function source = nemoGetSynapseSource(synapse)
% nemoGetSynapseSource - return the source neuron of the specified synapse
%  
% Synopsis:
%   source = nemoGetSynapseSource(synapse)
%  
% Inputs:
%   synapse - synapse id (as returned by addSynapse)
%    
% Outputs:
%   source  - source neuron index
%    
%  
% The input arguments can be a mix of scalars and vectors as long as
% all vectors have the same length. Scalar arguments are replicated
% the appropriate number of times. If all input arguments are scalar,
% the output is scalar. Otherwise the output has the same length as
% the vector input arguments.
    source = nemo_mex(uint32(27), uint64(synapse));
end