function F = islr_from_phase_indices(q, Q)
%ISLR_FROM_PHASE_INDICES Convert phase indices to a waveform and score it.
%
%   F = islr_from_phase_indices(q, Q)
%
%   Mirrors src/waveform_optimization/core/objective.py. This is the
%   ONE function every MATLAB solver must call to score a candidate
%   sequence, exactly as the Python side requires solvers to call
%   islr_from_phase_indices(q=phase_indices, Q=Q).

    x = phase_indices_to_waveform(q, Q);
    F = islr_from_waveform(x);
end
