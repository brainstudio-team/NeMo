function v = nemoGetMembranePotential(idx)
% nemoGetMembranePotential - get neuron membane potential
%  
% Synopsis:
%   v = nemoGetMembranePotential(idx)
%  
% Inputs:
%   idx     - neuron index
%    
% Outputs:
%   v       - membrane potential
%    
%  
% The input arguments can be a mix of scalars and vectors as long as
% all vectors have the same length. Scalar arguments are replicated
% the appropriate number of times. If all input arguments are scalar,
% the output is scalar. Otherwise the output has the same length as
% the vector input arguments.
    v = nemo_mex(uint32(14), uint32(idx));
end