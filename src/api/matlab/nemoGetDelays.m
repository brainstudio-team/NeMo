function delays = nemoGetDelays(synapses)
% nemoGetDelays - return the conductance delays for the specified synapses
%  
% Synopsis:
%   delays = nemoGetDelays(synapses)
%  
% Inputs:
%   synapses -
%             synapse ids (as returned by addSynapse)
%    
% Outputs:
%   delays  - conductance delays of the specified synpases
%    
	warning('nemo:api', 'nemoGetDelays is deprecated; Use nemoGetSynapseDelay instead');
	delays = nemoGetSynapseWeight(synapses);
end
