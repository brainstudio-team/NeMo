function targets = nemoGetTargets(synapses)
% nemoGetTargets - return the targets for the specified synapses
%  
% Synopsis:
%   targets = nemoGetTargets(synapses)
%  
% Inputs:
%   synapses -
%             synapse ids (as returned by addSynapse)
%    
% Outputs:
%   targets - indices of target neurons
%    
	warning('nemo:api', 'nemoGetTargets is deprecated; Use nemoGetSynapseTarget instead');
	targets = nemoGetSynapseTarget(synapses);
end
