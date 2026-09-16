function validate_phase_indices(q, N, Q)
%VALIDATE_PHASE_INDICES Validate a discrete phase-index sequence.
%
%   validate_phase_indices(q)
%   validate_phase_indices(q, N)
%   validate_phase_indices(q, N, Q)
%
%   Mirrors the Python function of the same name in validation.py.
%   Raises an error (Octave/MATLAB's equivalent of Python's ValueError)
%   if the sequence is invalid. Pass [] for N or Q to skip that check.
%
%   Inputs
%   ------
%   q : numeric row/column vector of phase indices
%   N : expected length, or [] to skip the length check
%   Q : number of phase levels, or [] to skip the range check

    if nargin < 2
        N = [];
    end
    if nargin < 3
        Q = [];
    end

    q = q(:).';  % force to a row vector, same role as np.asarray(q).ndim check

    if isempty(q)
        error('validate_phase_indices:EmptySequence', ...
            'Phase sequence must not be empty.');
    end

    if ~isempty(N) && numel(q) ~= N
        error('validate_phase_indices:WrongLength', ...
            'Expected %d phase indices, got %d.', N, numel(q));
    end

    if ~isempty(Q) && Q < 2
        error('validate_phase_indices:QTooSmall', 'Q must be at least 2.');
    end

    if any(q ~= floor(q))
        error('validate_phase_indices:NotInteger', ...
            'All phase indices must be integers.');
    end

    if ~isempty(Q) && (any(q < 0) || any(q >= Q))
        error('validate_phase_indices:OutOfRange', ...
            'Phase indices must lie in [0, Q - 1].');
    end
end
