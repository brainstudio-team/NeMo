function weight = nemoGetSynapseWeight(synapse)
% nemoGetSynapseWeight - return the weight for the specified synapse
%  
% Synopsis:
%   weight = nemoGetSynapseWeight(synapse)
%  
% Inputs:
%   synapse - synapse id (as returned by addSynapse)
%    
% Outputs:
%   weight  - weight of the specified synapse
%    
%  
% The input arguments can be a mix of scalars and vectors as long as
% all vectors have the same length. Scalar arguments are replicated
% the appropriate number of times. If all input arguments are scalar,
% the output is scalar. Otherwise the output has the same length as
% the vector input arguments.
    weight = nemo_mex(uint32(30), uint64(synapse));
end