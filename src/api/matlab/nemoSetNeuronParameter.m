function nemoSetNeuronParameter(idx, varno, val)
% nemoSetNeuronParameter - set neuron parameter
%  
% Synopsis:
%   nemoSetNeuronParameter(idx, varno, val)
%  
% Inputs:
%   idx     - neuron index
%   varno   - variable index
%   val     - value of the neuron parameter
%    
% The neuron parameters do not change during simulation. For the
% Izhikevich model: 0=a, 1=b, 2=c, 3=d.
%  
% The input arguments can be a mix of scalars and vectors as long as
% all vectors have the same length. Scalar arguments are replicated
% the appropriate number of times.
    nemo_mex(uint32(23), uint32(idx), uint32(varno), double(val));
end