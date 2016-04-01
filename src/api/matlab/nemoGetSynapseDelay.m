function delay = nemoGetSynapseDelay(synapse)
% nemoGetSynapseDelay - return the conduction delay for the specified synapse
%  
% Synopsis:
%   delay = nemoGetSynapseDelay(synapse)
%  
% Inputs:
%   synapse - synapse id (as returned by addSynapse)
%    
% Outputs:
%   delay   - conduction delay of the specified synapse
%    
%  
% The input arguments can be a mix of scalars and vectors as long as
% all vectors have the same length. Scalar arguments are replicated
% the appropriate number of times. If all input arguments are scalar,
% the output is scalar. Otherwise the output has the same length as
% the vector input arguments.
    delay = nemo_mex(uint32(29), uint64(synapse));
end