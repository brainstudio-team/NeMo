function nemoSetCudaBackend(deviceNumber)
% nemoSetCudaBackend - specify that the CUDA backend should be used
%  
% Synopsis:
%   nemoSetCudaBackend(deviceNumber)
%  
% Inputs:
%   deviceNumber -
%    
% Specify that the CUDA backend should be used and optionally specify
% a desired device. If the (default) device value of -1 is used the
% backend will choose the best available device. If the cuda backend
% (and the chosen device) cannot be used for whatever reason, an
% exception is raised. The device numbering is the numbering used
% internally by nemo (see cudaDeviceCount and cudaDeviceDescription).
% This device numbering may differ from the one provided by the CUDA
% driver directly, since NeMo ignores any devices it cannot use.
    nemo_mex(uint32(6), int32(deviceNumber));
end