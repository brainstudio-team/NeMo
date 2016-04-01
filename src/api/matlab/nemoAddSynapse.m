function id = nemoAddSynapse(source, target, delay, weight, plastic)
% nemoAddSynapse - add a single synapse to the network
%  
% Synopsis:
%   id = nemoAddSynapse(source, target, delay, weight, plastic)
%  
% Inputs:
%   source  - Index of source neuron
%   target  - Index of target neuron
%   delay   - Synapse conductance delay in milliseconds
%   weight  - Synapse weights
%   plastic - Boolean specifying whether or not this synapse is plastic
%    
% Outputs:
%   id      - Unique synapse ID
%    
%  
% The input arguments can be a mix of scalars and vectors as long as
% all vectors have the same length. Scalar arguments are replicated
% the appropriate number of times. If all input arguments are scalar,
% the output is scalar. Otherwise the output has the same length as
% the vector input arguments.
    id = nemo_mex(...
                 uint32(2),...
                 uint32(source),...
                 uint32(target),...
                 uint32(delay),...
                 double(weight),...
                 uint8(plastic)...
         );
end