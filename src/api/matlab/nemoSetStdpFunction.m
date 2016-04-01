function nemoSetStdpFunction(prefire, postfire, minWeight, maxWeight)
% nemoSetStdpFunction - enable STDP and set the global STDP function
%  
% Synopsis:
%   nemoSetStdpFunction(prefire, postfire, minWeight, maxWeight)
%  
% Inputs:
%   prefire - STDP function values for spikes arrival times before the
%             postsynaptic firing, starting closest to the postsynaptic firing
%   postfire -
%             STDP function values for spikes arrival times after the
%             postsynaptic firing, starting closest to the postsynaptic firing
%   minWeight -
%             Lowest (negative) weight beyond which inhibitory synapses are not
%             potentiated
%   maxWeight -
%             Highest (positive) weight beyond which excitatory synapses are not
%             potentiated
%    
% The STDP function is specified by providing the values sampled at
% integer cycles within the STDP window.
    nemo_mex(...
            uint32(7),...
            double(prefire),...
            double(postfire),...
            double(minWeight),...
            double(maxWeight)...
    );
end