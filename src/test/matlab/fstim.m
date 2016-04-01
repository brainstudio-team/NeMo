% Make sure firing stimulus work
nemoReset
input = nemoAddNeuronType('Input');
nemoAddNeuron(input, 0);
nemoCreateSimulation;
for t = 1:1000
	fired = nemoStep([0]);
	if length(fired) ~= 1
		error('nemo:test', 'stimulated neuron does not fire');
	end
end
nemoDestroySimulation;
nemoClearNetwork;
