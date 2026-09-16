function result = enumeration_solver(N, Q, time_limit, seed)
%ENUMERATION_SOLVER Exhaustive search over all Q^N phase-index sequences.
%
%   result = enumeration_solver(N, Q)
%   result = enumeration_solver(N, Q, time_limit)
%   result = enumeration_solver(N, Q, time_limit, seed)
%
%   Inputs
%   ------
%   N          : waveform length (positive integer)
%   Q          : number of phase levels (positive integer, >= 2 in
%                practice, though this solver itself only requires > 0
%                to mirror the Python bound checks)
%   time_limit : optional runtime limit in seconds ([] or omitted = none)
%   seed       : optional, recorded but unused, for interface parity
%
%   Output
%   ------
%   result : struct with fields matching the shared SolverResult schema:
%       algorithm, N, Q, best_phase_indices, best_objective,
%       runtime_seconds, objective_evaluations, status, seed,
%       time_limit_seconds
%
%   Ordering note: sequences are enumerated in the same order as
%   Python's itertools.product(range(Q), repeat=N) - i.e. counting in
%   base Q with N digits, most-significant digit first (index 0 varies
%   slowest, index N-1 varies fastest). This matters for tie-breaking:
%   both solvers keep the FIRST sequence that achieves the best
%   objective, so matching the enumeration order matters if you ever
%   compare best_phase_indices directly between Python and MATLAB.

    if nargin < 3
        time_limit = [];
    end
    if nargin < 4
        seed = [];
    end

    if ~isscalar(N) || N ~= floor(N) || N <= 0
        error('enumeration_solver:BadN', 'N must be a positive integer.');
    end
    if ~isscalar(Q) || Q ~= floor(Q) || Q <= 0
        error('enumeration_solver:BadQ', 'Q must be a positive integer.');
    end
    if ~isempty(time_limit) && (time_limit <= 0)
        error('enumeration_solver:BadTimeLimit', ...
            'time_limit must be a positive number or [].');
    end

    tic_id = tic;

    best_phase_indices = [];
    best_objective = Inf;
    objective_evaluations = 0;
    status = 'completed';

    total = Q ^ N;

    for idx = 0:(total - 1)
        if ~isempty(time_limit) && toc(tic_id) >= time_limit
            status = 'time_limit_reached';
            break;
        end

        q = local_decode_mixed_radix(idx, N, Q);

        objective = islr_from_phase_indices(q, Q);
        objective_evaluations = objective_evaluations + 1;

        if objective < best_objective
            best_objective = objective;
            best_phase_indices = q;
        end
    end

    runtime_seconds = toc(tic_id);

    if isempty(best_phase_indices)
        error('enumeration_solver:NoEvaluations', ...
            'Enumeration did not evaluate any solution.');
    end

    % Final validity check before returning, per the shared spec
    % ("every solver must validate its final solution before returning it").
    validate_phase_indices(best_phase_indices, N, Q);

    result = struct( ...
        'algorithm', 'enumeration', ...
        'N', N, ...
        'Q', Q, ...
        'best_phase_indices', best_phase_indices, ...
        'best_objective', best_objective, ...
        'runtime_seconds', runtime_seconds, ...
        'objective_evaluations', objective_evaluations, ...
        'status', status, ...
        'seed', seed, ...
        'time_limit_seconds', time_limit ...
    );
end

function q = local_decode_mixed_radix(idx, N, Q)
    % Decode idx (0-based) into an N-digit base-Q sequence, most
    % significant digit first, matching itertools.product's order.
    q = zeros(1, N);
    for i = N:-1:1
        q(i) = mod(idx, Q);
        idx = floor(idx / Q);
    end
end
