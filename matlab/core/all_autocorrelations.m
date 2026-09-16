function rhos = all_autocorrelations(x)
%ALL_AUTOCORRELATIONS Positive-lag aperiodic autocorrelations for k = 1..N-1.
%
%   rhos = all_autocorrelations(x)
%
%   Mirrors src/waveform_optimization/core/autocorrelation.py.
%   Returns a 1-by-(N-1) complex row vector, rhos(k) = rho(k), for
%   k = 1, ..., N-1 (lag zero is excluded, exactly as in the objective).

    x = x(:).';
    N = numel(x);

    if N < 2
        error('all_autocorrelations:TooShort', ...
            'x must contain at least two chips.');
    end

    rhos = zeros(1, N - 1);
    for k = 1:(N - 1)
        rhos(k) = autocorrelation_at_lag(x, k);
    end
end
