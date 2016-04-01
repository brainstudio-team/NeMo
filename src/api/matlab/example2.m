% Simple example showing usage of Matlab bindings of nemo.
% Create a network with 1k neurons randomly connected and run it for a bit.

% Populate the network

% 800 excitatory neurons
Ne = 800;
re = rand(Ne,1);
nemoAddNeuron(0:Ne-1, 0.02, 0.2, -65+15*re.^2, 8-6*re.^2, 5, -65*0.2, -65);

% 200 inhibitory neurons
Ni = 200;
ri = rand(Ni,1);
nemoAddNeuron(Ne:Ne+Ni-1, 0.02+0.08*ri, 0.25-0.05*ri, -65, 2, 2, -65*(0.25-0.05*ri), -65);

N = Ne + Ni;

% 1000 synapses per excitatory neuron
for n = 1:Ne-1
	nemoAddSynapse(n, 0:N-1, 1, 0.5*rand(N,1), false);
end

% 1000 synapses per inhibitory neuron
for n = Ne:N-1
	nemoAddSynapse(n, 0:N-1, 1, -rand(N,1), false);
end


% Set up STDP
prefire = 0.1 * exp(-(0:20)./20);
postfire = -0.08 * exp(-(0:20)./20);

nemoSetStdpFunction(prefire, postfire, -1.0, 1.0);


nemoCreateSimulation;

% Run for 5s with STDP enabled
for s=0:4
	for ms=1:1000
		fired = nemoStep;
		t = s*1000 + ms;
		disp([ones(size(fired')) * t, fired'])
	end

	% Change neuron parameter during simulation
	nemoSetNeuron(50, 0.05, 0.25, -70, 2, -70*0.05, -70, 2 * rand);

	% Read back membrane potential of a single neuron
	v = nemoGetMembranePotential(100)

	% Read back membrane potential of a couple of neurons
	v = nemoGetMembranePotential([101, 105])

	nemoApplyStdp(1.0);
end
elapsed = nemoElapsedWallclock

% Test that the synapse queries work by reading back the
% synapse data for all outgoing connections from neuron 100.
ids = nemoGetSynapsesFrom(100)

weights = nemoGetWeights(ids)
targets = nemoGetTargets(ids)
delays = nemoGetDelays(ids)
plastic = nemoGetPlastic(ids)

nemoDestroySimulation;
