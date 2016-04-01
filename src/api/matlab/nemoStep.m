function fired = nemoStep(fstim, istim_nidx, istim_current)
% step - run simulation for a single cycle (1ms)
%
% Synopsis:
%   fired = step()
%   fired = step(fstim)
%   fired = step(fstim, istim_nidx, istim_current)
%
% Inputs:
%   fstim -
%      An optional list of neurons, which will be forced to fire this cycle 
%   istim_nidx -
%      List of neuron indices for which current will be provided externally
%   istim_current -
%      Input current for this cycle
%
% Output:
%	fired - A list of the neurons which fired this cycle
%
    if nargin < 1
        fired = nemo_mex(uint32(12), uint32(zeros(1, 0)), uint32(zeros(1, 0)), zeros(1, 0));
    elseif nargin < 2
        fired = nemo_mex(uint32(12), uint32(fstim), uint32(zeros(1, 0)), zeros(1, 0));
    else
        fired = nemo_mex(uint32(12), uint32(fstim), uint32(istim_nidx), istim_current);
    end
end
