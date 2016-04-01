function nemoApplyStdp(reward)
% nemoApplyStdp - update synapse weights using the accumulated STDP statistics
%  
% Synopsis:
%   nemoApplyStdp(reward)
%  
% Inputs:
%   reward  - Multiplier for the accumulated weight change
%    
    nemo_mex(uint32(13), double(reward));
end