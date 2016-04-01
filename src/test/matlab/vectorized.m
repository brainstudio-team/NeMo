function vectorized
% Test that vectorized functions work as expected

nemoReset;

iz = nemoAddNeuronType('Izhikevich');

nmin = 1;

% generate random combinations of parameters, scalar and vector
for test = 1:100
	v = randi([0 1], 1, 8);
	if any(v)
		idx = 0:10;
	else
		idx = 0;
	end
	idx = idx + nmin;
	nmin = nmin + size(idx,2);
	nemoAddNeuron(iz, idx,...
		arg(v, 2), arg(v, 3), arg(v, 4), arg(v, 5), arg(v, 6), ...
		arg(v, 7), arg(v, 8));
	nemoSetNeuron(idx,...
		1+arg(v, 2), 1+arg(v, 3), 1+arg(v, 4), 1+arg(v, 5), 1+arg(v, 6), ...
		1+arg(v, 7), 1+arg(v, 8));
end

nemoReset;
end


function a = arg(vectorized, i)
	if vectorized(i)
		a = 0:10;
	else
		a = 0;
	end
end
