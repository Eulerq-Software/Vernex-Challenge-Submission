function validate_waveform(x, N, tolerance)
%VALIDATE_WAVEFORM Validate a complex unit-magnitude waveform.
%
%   validate_waveform(x)
%   validate_waveform(x, N)
%   validate_waveform(x, N, tolerance)
%
%   Mirrors the Python function of the same name in validation.py.
%   Pass [] for N to skip the length check.

    if nargin < 2
        N = [];
    end
    if nargin < 3 || isempty(tolerance)
        tolerance = 1e-10;
    end

    x = x(:).';

    if isempty(x)
        error('validate_waveform:EmptyWaveform', 'Waveform must not be empty.');
    end

    if ~isempty(N) && numel(x) ~= N
        error('validate_waveform:WrongLength', ...
            'Expected %d waveform samples, got %d.', N, numel(x));
    end

    magnitudes = abs(x);

    if any(abs(magnitudes - 1.0) > tolerance)
        error('validate_waveform:BadMagnitude', ...
            'Every waveform sample must have unit magnitude.');
    end
end
