function nemoSetNeuronState(idx, varno, val)
% nemoSetNeuronState - set neuron state variable
%  
% Synopsis:
%   nemoSetNeuronState(idx, varno, val)
%  
% Inputs:
%   idx     - neuron index
%   varno   - variable index
%   val     - value of the relevant variable
%    
% For the Izhikevich model: 0=u, 1=v.
%  
% The input arguments can be a mix of scalars and vectors as long as
% all vectors have the same length. Scalar arguments are replicated
% the appropriate number of times.
    nemo_mex(uint32(22), uint32(idx), uint32(varno), double(val));
end