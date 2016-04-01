function val = nemoGetPhase(idx)
% nemoGetPhase - get oscillator phase
%
% Synopsis
%   phase = nemoGetPhase(idx)
%
% Inputs:
%   idx    - oscillator index
%
% Outputs:
%   phase  - phase of oscillator (in range [0, 2pi))
%
% The oscillator indices can be either scalar or a vector. The output has the
% same shape as the input.
    val = nemoGetNeuronState(idx, 0);
end
