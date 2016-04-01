function weights = nemoGetWeights(synapses)
% nemoGetWeights - return the weights for the specified synapses
%  
% Synopsis:
%   weights = nemoGetWeights(synapses)
%  
% Inputs:
%   synapses -
%             synapse ids (as returned by addSynapse)
%    
% Outputs:
%   weights - weights of the specified synapses
%    
	warning('nemo:api', 'nemoGetWeights is deprecated; Use nemoGetSynapseWeight instead');
	weights = nemoGetSynapseWeight(synapses);
end
