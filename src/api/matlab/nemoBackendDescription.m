function description = nemoBackendDescription()
% nemoBackendDescription - Description of the currently selected simulation backend
%  
% Synopsis:
%   description = nemoBackendDescription()
%  
% Outputs:
%   description -
%             Textual description of the currently selected backend
%    
% The backend can be changed using setCudaBackend or setCpuBackend
    description = nemo_mex(uint32(8));
end