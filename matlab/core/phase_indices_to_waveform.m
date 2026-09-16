function x = phase_indices_to_waveform(q, Q)
%PHASE_INDICES_TO_WAVEFORM Convert integer phase indices to a complex waveform.
%
%   x = phase_indices_to_waveform(q, Q)
%
%   Mirrors src/waveform_optimization/core/encoding.py.
%
%   Inputs
%   ------
%   q : vector of phase indices in {0, 1, ..., Q-1}
%   Q : number of available phase values
%
%   Output
%   ------
%   x : complex row vector, one unit-magnitude sample per phase index,
%       using the fixed project convention x_i = exp(-1j * 2*pi*q_i / Q)

    if Q < 2
        error('phase_indices_to_waveform:QTooSmall', 'Q must be at least 2.');
    end

    validate_phase_indices(q, [], Q);

    q = q(:).';
    x = exp(-1i * 2 * pi * q / Q);
end
