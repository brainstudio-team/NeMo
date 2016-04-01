function val = nemoGetNeuronState(idx, varno)
% nemoGetNeuronState - get neuron state variable
%  
% Synopsis:
%   val = nemoGetNeuronState(idx, varno)
%  
% Inputs:
%   idx     - neuron index
%   varno   - variable index
%    
% Outputs:
%   val     - value of the relevant variable
%    
% For the Izhikevich model: 0=u, 1=v.
%  
% The input arguments can be a mix of scalars and vectors as long as
% all vectors have the same length. Scalar arguments are replicated
% the appropriate number of times. If all input arguments are scalar,
% the output is scalar. Otherwise the output has the same length as
% the vector input arguments.
    val = nemo_mex(uint32(24), uint32(idx), uint32(varno));
end