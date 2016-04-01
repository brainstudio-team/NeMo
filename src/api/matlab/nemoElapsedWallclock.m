function elapsed = nemoElapsedWallclock()
% nemoElapsedWallclock - 
%  
% Synopsis:
%   elapsed = nemoElapsedWallclock()
%  
% Outputs:
%   elapsed - number of milliseconds of wall-clock time elapsed since
%             first simulation step (or last timer reset)
%    
    elapsed = nemo_mex(uint32(15));
end