function xlang_run_solver(solver_fn, N, Q, seed, out_path, params_file)
%XLANG_RUN_SOLVER Run one MATLAB solver and write a JSON record to disk.
%
%   xlang_run_solver('enumeration_solver', 8, 2, 'none', '/tmp/ml.json')
%   xlang_run_solver('genetic_algorithm_solver', 8, 2, 'none', ...
%       '/tmp/ml.json', 'experiments/configs/ga_xlang_params.json')
%
%   Called by scripts/xlang_benchmark.sh through `matlab -batch` (or
%   `octave --eval`). Every argument may arrive as a char array, because
%   that is how the shell passes them, so they are converted here.
%
%   The solver is expected to follow the common MATLAB solver signature
%
%       result = solver_fn(N, Q, time_limit, seed)
%
%   and to return a struct with the shared SolverResult fields. Time limits
%   are deliberately not used by this harness: [] is always passed, so the
%   amount of computational work is identical on both sides of the
%   comparison (see section 6.6 of the challenge document).
%
%   params_file (optional): path to a JSON file of extra solver arguments
%   (e.g. GA population_size, mutation_prob, ...). The SAME file is read
%   by xlang_run_python_solver.py, so both languages run with identical
%   hyperparameters. It is only forwarded to the solver function when that
%   function actually accepts a 5th (params struct) argument - this keeps
%   solvers like greedy_solver, whose 5th argument means something else
%   (initial_solution), unaffected. Pass '' or omit when not needed.
%
%   The record is written to a file rather than stdout so that MATLAB's
%   banner, warnings, or any solver printout cannot corrupt the JSON.

    if nargin < 6
        params_file = '';
    end
    if ischar(N) || isstring(N)
        N = str2double(N);
    end
    if ischar(Q) || isstring(Q)
        Q = str2double(Q);
    end
    if ischar(seed) || isstring(seed)
        s = lower(strtrim(char(seed)));
        if isempty(s) || strcmp(s, 'none') || strcmp(s, 'null') || strcmp(s, 'na')
            seed = [];
        else
            seed = str2double(s);
        end
    end

    payload = struct();
    payload.language = 'matlab';
    payload.solver_spec = char(solver_fn);
    payload.N = N;
    payload.Q = Q;
    payload.status = 'error';
    payload.error = '';

    try
        fh = str2func(char(solver_fn));

        params_struct = [];
        if ~isempty(params_file)
            raw_json = fileread(char(params_file));
            params_struct = jsondecode(raw_json);
            payload.params = params_struct;
        end

        wall_start = tic;
        % Only solvers declaring a 5th (params) argument receive it, e.g.
        % genetic_algorithm_solver(N, Q, time_limit, seed, params).
        if ~isempty(params_struct) && abs(nargin(fh)) >= 5
            result = fh(N, Q, [], seed, params_struct);
        else
            result = fh(N, Q, [], seed);
        end
        wall_seconds = toc(wall_start);

        q = double(result.best_phase_indices(:)');

        % Independent recomputation through the shared MATLAB core.
        recomputed = double(islr_from_phase_indices(q, Q));

        valid = (numel(q) == N) && all(q >= 0) && all(q < Q) && all(q == floor(q));

        payload.status = char(result.status);
        payload.algorithm = char(result.algorithm);
        payload.N = double(result.N);
        payload.Q = double(result.Q);
        if isempty(seed)
            payload.seed = [];
        else
            payload.seed = double(seed);
        end
        payload.best_objective = double(result.best_objective);
        payload.recomputed_objective = recomputed;
        payload.objective_evaluations = double(result.objective_evaluations);
        payload.runtime_seconds = double(result.runtime_seconds);
        payload.wall_seconds = double(wall_seconds);
        payload.best_phase_indices = q;
        payload.valid = logical(valid);
        payload.error = '';
    catch err
        payload.status = 'error';
        payload.error = sprintf('%s: %s', err.identifier, err.message);
    end

    text = jsonencode(payload);

    fid = fopen(char(out_path), 'w');
    if fid < 0
        error('xlang_run_solver:CannotWrite', ...
            'Could not open %s for writing.', char(out_path));
    end
    fwrite(fid, text, 'char');
    fclose(fid);
end