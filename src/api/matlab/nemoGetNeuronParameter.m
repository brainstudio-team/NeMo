function val = nemoGetNeuronParameter(idx, varno)
% nemoGetNeuronParameter - get neuron parameter
%  
% Synopsis:
%   val = nemoGetNeuronParameter(idx, varno)
%  
% Inputs:
%   idx     - neuron index
%   varno   - variable index
%    
% Outputs:
%   val     - value of the neuron parameter
%    
% The neuron parameters do not change during simulation. For the
% Izhikevich model: 0=a, 1=b, 2=c, 3=d.
%  
% The input arguments can be a mix of scalars and vectors as long as
% all vectors have the same length. Scalar arguments are replicated
% the appropriate number of times. If all input arguments are scalar,
% the output is scalar. Otherwise the output has the same length as
% the vector input arguments.
    val = nemo_mex(uint32(25), uint32(idx), uint32(varno));
end