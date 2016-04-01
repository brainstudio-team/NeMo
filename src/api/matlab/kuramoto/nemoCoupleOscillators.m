function id = nemoCoupleOscillators(source, target, delay, strength)
% nemoCoupleOscillators - add one or more couplings between oscillators
%  
% Synopsis:
%   id = nemoCoupleOscillator(source, target, delay, strength)
%  
% Inputs:
%   source   - index of source oscillator(s)
%   target   - index of target oscillator(s)
%   delay    - coupling delay (in time steps)
%   strength - coupling strength
%    
% Outputs:
%   id       - unique coupling ID
%  
% The input arguments can be a mix of scalars and vectors as long as
% all vectors have the same length. Scalar arguments are replicated
% the appropriate number of times. If all input arguments are scalar,
% the output is scalar. Otherwise the output has the same length as
% the vector input arguments.
	id = nemoAddSynapse(source, target, delay, strength, false); 
end
