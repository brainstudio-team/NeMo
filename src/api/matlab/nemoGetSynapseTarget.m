function target = nemoGetSynapseTarget(synapse)
% nemoGetSynapseTarget - return the target of the specified synapse
%  
% Synopsis:
%   target = nemoGetSynapseTarget(synapse)
%  
% Inputs:
%   synapse - synapse id (as returned by addSynapse)
%    
% Outputs:
%   target  - target neuron index
%    
%  
% The input arguments can be a mix of scalars and vectors as long as
% all vectors have the same length. Scalar arguments are replicated
% the appropriate number of times. If all input arguments are scalar,
% the output is scalar. Otherwise the output has the same length as
% the vector input arguments.
    target = nemo_mex(uint32(28), uint64(synapse));
end