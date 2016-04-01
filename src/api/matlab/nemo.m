% To use the NeMo simulator:
%
% 1. Set any global configuration options, if desired
% 2. Create a network piecemeal by adding neurons and synapses
% 3. Create simulation and step through it.
%
% For example:
%
%    Ne = 800;
%    Ni = 200;
%    N = Ne + Ni;
%    
%    iz = nemoAddNeuronType('Izhikevich');
%    
%    re = rand(Ne,1);
%    nemoAddNeuron(iz, 0:Ne-1, 0.02, 0.2, -65+15*re.^2, 8-6*re.^2, 5, -65*0.2, -65);
%    ri = rand(Ni,1);
%    nemoAddNeuron(iz, Ne:Ne+Ni-1, 0.02+0.08*ri, 0.25-0.05*ri, -65, 2, 2, -65*(0.25-0.05*ri), -65);
%    
%    for n = 1:Ne-1
%    	nemoAddSynapse(n, 0:N-1, 1, 0.5*rand(N,1), false);
%    end
%    
%    for n = Ne:N-1
%    	nemoAddSynapse(n, 0:N-1, 1, -rand(N,1), false);
%    end
%    
%    firings = [];
%    nemoCreateSimulation;
%    for t=1:1000
%    	fired = nemoStep;
%    	firings=[firings; t+0*fired',fired'];
%    end
%    nemoDestroySimulation;
%    nemoClearNetwork;
%    plot(firings(:,1),firings(:,2),'.');
%
% The library is modal: it is either in the construction/configuration stage or
% in the simulation stage. nemoCreateSimulation switches from
% construction/configuration to simulation and nemoDestroySimulation switches
% back again. Functions are classified as either configuration, construction,
% or simulation functions, and can only be used in the appropriate stage.
%
% NeMo provides the functions listed below. See the documentation for the
% respective functions for more detail.
%
% Construction:
%  nemoAddNeuronType
%  nemoAddNeuron
%  nemoAddSynapse
%  nemoNeuronCount
%  nemoClearNetwork
%  nemoSetNeuron
%  nemoSetNeuronState
%  nemoSetNeuronParameter
%  nemoGetNeuronState
%  nemoGetNeuronParameter
%  nemoGetSynapsesFrom
%  nemoGetSynapseSource
%  nemoGetSynapseTarget
%  nemoGetSynapseDelay
%  nemoGetSynapseWeight
%  nemoGetSynapsePlastic
%
% Configuration:
%  nemoSetCpuBackend
%  nemoSetCudaBackend
%  nemoSetStdpFunction
%  nemoBackendDescription
%  nemoSetWriteOnlySynapses
%  nemoLogStdout
%  nemoResetConfiguration
%
% Simulation:
%  nemoStep
%  nemoApplyStdp
%  nemoGetMembranePotential
%  nemoElapsedWallclock
%  nemoElapsedSimulation
%  nemoResetTimer
%  nemoCreateSimulation
%  nemoDestroySimulation
%  nemoSetNeuron
%  nemoSetNeuronState
%  nemoSetNeuronParameter
%  nemoGetNeuronState
%  nemoGetNeuronParameter
%  nemoGetSynapsesFrom
%  nemoGetSynapseSource
%  nemoGetSynapseTarget
%  nemoGetSynapseDelay
%  nemoGetSynapseWeight
%  nemoGetSynapsePlastic
%
% Others:
%  nemoReset
