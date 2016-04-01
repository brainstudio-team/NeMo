% Test that neuron getters/setters work 
function testNeuronGetSet

	for param = 0:4
		val_in = rand(1, 1);
		nemoSetNeuronParameter(0, param, val_in);
		val_out = nemoGetNeuronParameter(0, param);
		% Some difference expected due to different number formats
		if val_in - val_out > 0.00001
			error('get/set mismatch')
		end
	end

	% Test failure conditions

	% Invalid neuron
	try
		nemoSetNeuronParameter(5000, 0, 0);
		error('setting invalid neuron did not fail correctly');
	catch e
		% all good	
	end

	try
		nemoSetNeuronState(5000, 0, 0);
		error('setting invalid neuron did not fail correctly');
	catch e
		% all good	
	end

	try
		x = nemoGetNeuronParameter(5000, 0);
		error('getting invalid neuron did not fail correctly');
	catch e
		% all good	
	end

	try
		x = nemoGetNeuronState(5000, 0);
		error('getting invalid neuron did not fail correctly');
	catch e
		% all good	
	end

	% Invalid parameter
	try
		nemoSetNeuronParameter(0, 5, 0);
		error('setting invalid neuron parameter did not fail correctly');
	catch e
		% all good	
	end

	try
		x = nemoGetNeuronParameter(0, 5, 0);
		error('getting invalid neuron parameter did not fail correctly');
	catch e
		% all good	
	end

	% Invalid state variable
	try
		nemoSetNeuronState(0, 2, 0);
		error('setting invalid neuron state variable did not fail correctly');
	catch e
		% all good	
	end

	try
		x = nemoGetNeuronState(0, 2, 0);
		error('getting invalid neuron state variable did not fail correctly');
	catch e
		% all good	
	end

	for varno = 0:1
		val_in = rand(1, 1);
		nemoSetNeuronState(0, varno, val_in);
		val_out = nemoGetNeuronState(0, varno);
		% Some difference expected due to different number formats
		if val_in - val_out > 0.00001
			error('get/set mismatch')
		end
	end
end
