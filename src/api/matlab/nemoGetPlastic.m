function plastic = nemoGetPlastic(synapses)
% nemoGetPlastic - return the boolean plasticity status for the specified synapses
%  
% Synopsis:
%   plastic = nemoGetPlastic(synapses)
%  
% Inputs:
%   synapses -
%             synapse ids (as returned by addSynapse)
%    
% Outputs:
%   plastic - plasticity status of the specified synpases
%    
	warning('nemo:api', 'nemoGetPlastic is deprecated; Use nemoGetSynapsePlastic instead');
	plastic = nemoGetSynapsePlastic(synapses);
end
