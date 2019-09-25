% acf_est.m Help file for ACF_EST MEX file
%
% ACF_EST Calculate Bartlett's estimate of the Auto Correlation Function 
%   C = ACF_EST(A) Calculate one estimate of A for every column using
%   r[k] = 1/N * sum{n=0 -> N - |k| - 1}(x[n + k]x[n]).
%   Returns the columnwise result in C, with length length(A). The spectra is one sided, so use flipud(:,2:end) for the double sided spectra.
%   Works along the first nonsingular dimension of A
%
%   MEX File function.