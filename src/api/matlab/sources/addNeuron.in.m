% nemoAddNeuron - add one or more neurons to the network
%  
% Synopsis:
%   nemoAddNeuron(type, idx, param0, param1..., state0, state1...)
%  
% Inputs:
%   type    - neuron type, as returned by nemoAddNeuronType
%   idx     - neuron index
%   paramX  - neuron parameters
%   stateX  - neuron state variables
%
% The number of parameters and state variables must match the neuron type.
%  
% The input arguments can be a mix of scalars and vectors as long as all
% vectors have the same length. Scalar arguments are replicated the appropriate
% number of times.
%
% The neuron index input must be a vector if any of the parameter or state
% arguments are vectors (as the neuron index cannot be replicated).
function addNeuron(type, idx, varargin)

nemo_mex(uint32(FNID), uint32(type), uint32(idx), varargin{:});
