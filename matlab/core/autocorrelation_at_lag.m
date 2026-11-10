function rho = autocorrelation_at_lag(x, k)
%AUTOCORRELATION_AT_LAG Positive-lag aperiodic autocorrelation.
%
%   rho = autocorrelation_at_lag(x, k)
%
%   Mirrors src/waveform_optimization/core/autocorrelation.py.
%
%       rho(k) = sum_i x(i) * conj(x(i + k))
%
%   IMPORTANT INDEXING NOTE (Python vs MATLAB):
%   Python (0-based): rho(k) = sum(x[0 : N-k] * conj(x[k : N]))
%   MATLAB (1-based) equivalent:
%       first factor  -> x(1 : N-k)      (same N-k elements, starting at 1)
%       second factor -> x(k+1 : N)      (shifted by k, so index k+1 not k)
%   This +1 shift on the second slice is exactly the 0-based -> 1-based
%   translation and is the main place cross-language bugs creep in.

    x = x(:).';
    N = numel(x);

    if k < 1 || k >= N
        error('autocorrelation_at_lag:BadLag', ...
            'k must satisfy 1 <= k < length(x).');
    end

    rho = sum(x(1:N-k) .* conj(x(k+1:N)));
end
