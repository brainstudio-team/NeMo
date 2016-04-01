function plastic = nemoGetSynapsePlastic(synapse)
% nemoGetSynapsePlastic - return the boolean plasticity status for the specified synapse
%  
% Synopsis:
%   plastic = nemoGetSynapsePlastic(synapse)
%  
% Inputs:
%   synapse - synapse id (as returned by addSynapse)
%    
% Outputs:
%   plastic - plasticity status of the specified synapse
%    
%  
% The input arguments can be a mix of scalars and vectors as long as
% all vectors have the same length. Scalar arguments are replicated
% the appropriate number of times. If all input arguments are scalar,
% the output is scalar. Otherwise the output has the same length as
% the vector input arguments.
    plastic = nemo_mex(uint32(31), uint64(synapse));
end