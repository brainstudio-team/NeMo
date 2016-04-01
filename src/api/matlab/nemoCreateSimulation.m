function nemoCreateSimulation()
% nemoCreateSimulation - Initialise simulation data
%  
% Synopsis:
%   nemoCreateSimulation()
%  
% Initialise simulation data, but do not start running. Call step to
% run simulation. The initialisation step can be time-consuming.
    nemo_mex(uint32(18));
end