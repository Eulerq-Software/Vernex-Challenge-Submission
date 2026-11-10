function F = islr_from_waveform(x)
%ISLR_FROM_WAVEFORM Integrated sidelobe objective from a complex waveform.
%
%   F = islr_from_waveform(x)
%
%   Mirrors src/waveform_optimization/core/objective.py.
%
%       F(x) = sum_{k=1}^{N-1} |rho(k)|^2

    rhos = all_autocorrelations(x);
    F = sum(abs(rhos) .^ 2);
end
