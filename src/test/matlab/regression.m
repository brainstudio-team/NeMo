function regression(create, stdp)
% regression - run regression test or create test data
%
% Synopsis:
%	regression(create, stdp)

	if ~create
		fprintf(1, 'Testing for stdp=%u\n', stdp);
	end

	fn = filename(stdp);
	t_max = 1000;
	s = RandStream('mt19937ar','Seed', 5489);
	RandStream.setDefaultStream(s);
	createRandom(stdp);
	firings = run(stdp, t_max);
	if create
		save(fn, 'firings');
	else
		verify(fn, firings, t_max);
	end
end



function createRandom(stdp)
	Ne=800;
	Ni=200;
	N=Ne+Ni;
	M=1000;
	re=rand(Ne,1);
	v=-65;
	iz = nemoAddNeuronType('Izhikevich');
	nemoAddNeuron(iz, 0:Ne-1, 0.02, 0.2, -65+15*re.^2, 8-6*re.^2, 5, v*0.2, v);
	for src = 0:Ne-1
		nemoAddSynapse(src, 0:N-1, floor(20*rand(N,1))+1, 0.5*rand(N,1), stdp);
	end
	ri=rand(Ni,1);
	bi=0.25-0.05*ri;
	nemoAddNeuron(iz, Ne:Ne+Ni-1, 0.02+0.08*ri, bi, -65, 2, 2, bi*v, v);
	for src = Ne:Ne+Ni-1
		nemoAddSynapse(src, 0:N-1, 1, -rand(N,1), false);
	end
end


function firings = run(stdp, t_max)
	firings = {};
	if stdp
		prefire = 0.1 * exp(-(0:20)./20);
		postfire = -0.08 * exp(-(0:20)./20);
		nemoSetStdpFunction(prefire, postfire, -1.0, 1.0);
	end

	nemoCreateSimulation;

	for t = 1:t_max
		% TODO: add tests with input stimulus as well
		fired = nemoStep();
		firings{t} = fired;
		if mod(t, 100) == 0 && stdp
			nemoApplyStdp(1);
		end
	end

	nemoDestroySimulation;
	nemoClearNetwork;
end


function name = filename(stdp)
	[s, hostname] = unix('hostname');
	if s ~= 0
		error('failed to get hostname');
	end
	name = sprintf('rdata_%s_%u.mat', strtrim(hostname), stdp);
end


function verify(fn, firings1, t_max)
	load(fn, 'firings');
	for t = 1:t_max
		f0 = firings{t};
		f1 = firings1{t};
		assert(numel(f0) == numel(f1));
		assert(all(f0 == f1));
	end
end
