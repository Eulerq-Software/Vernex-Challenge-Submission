function result = greedy_solver(N, Q, time_limit, seed, initial_solution)
%GREEDY_SOLVER Construct a waveform left to right, one coordinate at a time.
%
%   result = greedy_solver(N, Q)
%   result = greedy_solver(N, Q, time_limit)
%   result = greedy_solver(N, Q, time_limit, seed)
%   result = greedy_solver(N, Q, time_limit, seed, initial_solution)
%
%   Completion strategy: unassigned positions are held at phase 0
%   ("zero fill") while every candidate value for the current position is
%   tried in turn.
%
%   Tie-breaking: for a fixed position, candidate phase values are tried
%   in increasing order 0, 1, ..., Q-1 and a STRICT less-than comparison
%   is used, so the first (lowest) phase that attains the best objective
%   for that position is kept - matching the Python solver's comment
%   "the first (lowest) phase is retained".
%
%   Time limit: checked once per candidate evaluation, exactly as in the
%   Python version. If the limit is hit partway through a position's
%   phase sweep, that position is left at whatever value it already had
%   (0 if untouched) and the outer loop stops immediately - the current
%   position is NOT finalized with whatever partial best was seen. This
%   matches solve() in greedy.py line for line.
%
%   `seed` has no effect on this solver (it is deterministic); it is
%   accepted and recorded only for interface parity with every other
%   solver's SolverResult, exactly as in greedy.py.
%
%   Output
%   ------
%   result : struct with fields matching the shared SolverResult schema:
%       algorithm, N, Q, best_phase_indices, best_objective,
%       runtime_seconds, objective_evaluations, status, seed,
%       time_limit_seconds, metadata

    if nargin < 3
        time_limit = [];
    end
    if nargin < 4
        seed = [];
    end
    if nargin < 5
        initial_solution = [];
    end

    if ~isscalar(N) || N ~= floor(N) || N < 1
        error('greedy_solver:BadN', 'N must be an integer >= 1.');
    end
    if ~isscalar(Q) || Q ~= floor(Q) || Q < 2
        error('greedy_solver:BadQ', 'Q must be an integer >= 2.');
    end
    if ~isempty(time_limit) && (time_limit < 0)
        error('greedy_solver:BadTimeLimit', ...
            'time_limit must be nonnegative or [].');
    end

    tic_id = tic;
    objective_evaluations = 0;
    status = 'completed';

    % Zero-fill completion strategy.
    current_solution = zeros(1, N);
    current_objective = islr_from_phase_indices(current_solution, Q);
    objective_evaluations = objective_evaluations + 1;

    best_solution = current_solution;
    best_objective = current_objective;

    % An optional initial solution is treated as another valid candidate.
    if ~isempty(initial_solution)
        validate_phase_indices(initial_solution, N, Q);
        initial_objective = islr_from_phase_indices(initial_solution, Q);
        objective_evaluations = objective_evaluations + 1;

        if initial_objective < best_objective
            best_solution = initial_solution(:)';
            best_objective = initial_objective;
        end
    end

    for position = 1:N
        best_phase = current_solution(position);
        best_candidate_objective = Inf;
        hit_time_limit = false;

        for phase = 0:(Q - 1)
            if ~isempty(time_limit) && toc(tic_id) >= time_limit
                status = 'time_limit_reached';
                hit_time_limit = true;
                break;
            end

            candidate = current_solution;
            candidate(position) = phase;

            candidate_objective = islr_from_phase_indices(candidate, Q);
            objective_evaluations = objective_evaluations + 1;

            % Strict comparison makes ties deterministic: the first
            % (lowest) phase is retained.
            if candidate_objective < best_candidate_objective
                best_phase = phase;
                best_candidate_objective = candidate_objective;
            end

            if candidate_objective < best_objective
                best_solution = candidate;
                best_objective = candidate_objective;
            end
        end

        if hit_time_limit
            break;
        end

        current_solution(position) = best_phase;
        current_objective = best_candidate_objective; %#ok<NASGU>
    end

    validate_phase_indices(best_solution, N, Q);

    result = struct( ...
        'algorithm', 'greedy', ...
        'N', N, ...
        'Q', Q, ...
        'best_phase_indices', best_solution, ...
        'best_objective', best_objective, ...
        'runtime_seconds', toc(tic_id), ...
        'objective_evaluations', objective_evaluations, ...
        'status', status, ...
        'seed', seed, ...
        'time_limit_seconds', time_limit, ...
        'metadata', struct( ...
            'coordinate_order', 'left_to_right', ...
            'completion_strategy', 'zero_fill', ...
            'tie_breaking', 'lowest_phase_index' ...
        ) ...
    );
end