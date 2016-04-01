% Create a network of 128x128 oscillators with nearest neighbour connections
% and periodic boundary conditions.
function example

nn = 128;
tmax = 1000;

km = nemoAddNeuronType('Kuramoto');

tt = 10;
w0 = 0.5/tt;
K = 50*w0/4;
tau = 1;

tic
for y = 0:nn-1
	src = idx(0:nn-1,y,nn);

	% Random natural frequency, identical initial phase
	nemoAddNeuron(km, src, w0+randn(1,nn)*w0, 0);
	nemoCoupleOscillators(src, idx(0:nn-1,mod(y-1,nn),nn), tau, K);
	nemoCoupleOscillators(src, idx(0:nn-1,mod(y+1,nn),nn), tau, K);
	nemoCoupleOscillators(src, idx(mod(1:nn,nn),y,nn), tau, K);
	nemoCoupleOscillators(src, idx(mod(-1:nn-2,nn),y,nn), tau, K);
end

% Add a couple of inhomoegenous  regions
for y = 10:20
	nemoSetNeuronState(idx(10:20,y,nn), 0, pi);
end
for y = nn-20:nn-10
	nemoSetNeuronState(idx(nn-20:nn-10,y,nn), 0, -pi);
end

fprintf(1, 'Constructed network in %f seconds (%u oscillators, %u couplings)\n',...
	toc, nn^2, nn^2*4);

all = [0:nn^2-1];

tic;
nemoCreateSimulation;
fprintf(1, 'Created simulation in %f seconds\n', toc); 

figure(1);
colormap(jet(512));
scale = 512/(2*pi);

for t = 1:tmax
	nemoStep;
	phase = reshape(nemoGetPhase(all), nn, nn);
	image(mod(phase,2*pi)*scale);
	axis square;
	colorbar;
	title(sprintf('t=%u', t));
	drawnow;
end

nemoDestroySimulation;
nemoClearNetwork;

end

function i = idx(x, y, nn)
	i = y*nn+x;
end
