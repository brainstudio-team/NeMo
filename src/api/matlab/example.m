Ne = 800;
Ni = 200;
N = Ne + Ni;

iz = nemoAddNeuronType('Izhikevich');

re = rand(Ne,1);
nemoAddNeuron(iz, 0:Ne-1, 0.02, 0.2, -65+15*re.^2, 8-6*re.^2, 5, -65*0.2, -65);
ri = rand(Ni,1);
nemoAddNeuron(iz, Ne:Ne+Ni-1, 0.02+0.08*ri, 0.25-0.05*ri, -65, 2, 2, -65*(0.25-0.05*ri), -65);

for n = 1:Ne-1
	nemoAddSynapse(n, 0:N-1, 1, 0.5*rand(N,1), false);
end

for n = Ne:N-1
	nemoAddSynapse(n, 0:N-1, 1, -rand(N,1), false);
end

firings = [];
nemoCreateSimulation;
for t=1:1000
	fired = nemoStep;
	firings=[firings; t+0*fired',fired'];
end
nemoDestroySimulation;
nemoClearNetwork;
plot(firings(:,1),firings(:,2),'.');
