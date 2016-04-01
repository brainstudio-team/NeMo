function nemoDestroySimulation()
% nemoDestroySimulation - Stop simulation and free associated data
%  
% Synopsis:
%   nemoDestroySimulation()
%  
% The simulation can have a significant amount of memory associated
% with it. Calling destroySimulation frees up this memory.
    nemo_mex(uint32(19));
end