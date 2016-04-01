function nemoSetWriteOnlySynapses()
% nemoSetWriteOnlySynapses - Specify that synapses will not be read back at run-time
%  
% Synopsis:
%   nemoSetWriteOnlySynapses()
%  
% By default synapse state can be read back at run-time. This may
% require setting up data structures of considerable size before
% starting the simulation. If the synapse state is not required at
% run-time, specify that synapses are write-only in order to save
% memory and setup time. By default synapses are readable
    nemo_mex(uint32(9));
end