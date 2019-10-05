% ACF_EST   Fast implementation of Bartlett's estimate of the Auto Correlation 
%   Function.
%
%   Bartlett's estimate is defined by
%   r[k] = 1/N * sum{n=0 -> N - |k| - 1}(x[n + k]x[n]).
%
%   C = ACF_EST(X) is the Bartlett estimate of X if X is a vector.
%
%   For matrices, each column in C holds the Bartlett estimate of each column 
%   in X.
%
%   The spectra is one sided with r[0] at index 1, so flipud(r(:,2:end)) gives 
%   the other side of the spectra. For ACF averaging after calculation, use 
%   avg = mean(r, 2).