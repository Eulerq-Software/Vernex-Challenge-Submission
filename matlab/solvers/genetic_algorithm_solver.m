function result = genetic_algorithm_solver(N, Q, time_limit, seed, params)
%GENETIC_ALGORITHM_SOLVER Evolve a population of phase-index sequences.
%
%   result = genetic_algorithm_solver(N, Q)
%   result = genetic_algorithm_solver(N, Q, time_limit)
%   result = genetic_algorithm_solver(N, Q, time_limit, seed)
%   result = genetic_algorithm_solver(N, Q, time_limit, seed, params)
%
%   Mirrors the shared solver interface (project spec Section 3.2) and
%   the Genetic Algorithm design in Section 5.4.
%
%   ALGORITHM
%   ---------
%     1. Initialize a random population of valid phase-index sequences.
%     2. Evaluate every individual with islr_from_phase_indices.
%     3. Repeat until time_limit or max_generations is reached:
%          a. Copy the `elite_count` best individuals unchanged into the
%             next generation. This is a reuse of an already-known
%             objective value, so per Section 3.4 of the spec it is NOT
%             counted as a new evaluation.
%          b. Fill the rest of the next generation by: tournament-select
%             two parents, uniform-crossover them with probability
%             crossover_prob (otherwise just copy parent 1), then apply
%             per-gene random-reset mutation with probability
%             mutation_prob. The resulting offspring IS evaluated, and
%             that DOES count.
%          c. Replace the population with the next generation.
%     4. Return the best individual seen across the whole run (not just
%        the final generation - elitism already keeps it in the
%        population, but this is kept explicit for clarity and safety).
%
%   FITNESS TRANSFORM (required design decision, Section 5.4.1):
%   selection here is tournament-based, i.e. it only ever compares two
%   or more raw objective values and keeps the smaller one. That never
%   needs a positive "higher is better" fitness score, so the spec's
%   suggested fitness(q) = 1/(1+F(q)) transform is NOT applied - this
%   solver minimizes F directly throughout. That transform would only
%   be needed for proportional/roulette-wheel selection, which this
%   implementation does not use.
%
%   SEEDING (deviation from Section 3.5, documented): MATLAB's
%   per-instance RandStream class is not implemented in GNU Octave, so
%   this solver seeds the GLOBAL random generator via rng(seed) rather
%   than using a local, isolated stream. This is fully reproducible
%   (same seed -> same run) but is not isolated from other code that
%   also touches the global stream in the same session. If you only run
%   this under real MATLAB and want isolation, replace the rng(...)
%   calls with a RandStream object and thread it through the local
%   helper functions below.
%
%   Inputs
%   ------
%   N          : waveform length (positive integer)
%   Q          : number of phase levels (integer >= 2)
%   time_limit : optional runtime limit in seconds ([] or omitted = none)
%   seed       : optional random seed ([] or omitted = nondeterministic)
%   params     : optional struct overriding any of these defaults:
%       population_size   (default 40)
%       max_generations   (default 300)
%       crossover_prob    (default 0.9)
%       mutation_prob     (default 1/N, i.e. ~1 mutated gene/individual)
%       tournament_size   (default 3)
%       elite_count       (default 2)
%
%   Output
%   ------
%   result : struct matching the shared SolverResult schema:
%       algorithm, N, Q, best_phase_indices, best_objective,
%       runtime_seconds, objective_evaluations, status, seed,
%       time_limit_seconds, metadata
%       (metadata records every GA parameter actually used, so a run is
%       fully reproducible from the result struct alone together with
%       the seed field)

    if nargin < 3
        time_limit = [];
    end
    if nargin < 4
        seed = [];
    end
    if nargin < 5 || isempty(params)
        params = struct();
    end

    if ~isscalar(N) || N ~= floor(N) || N < 1
        error('genetic_algorithm_solver:BadN', 'N must be an integer >= 1.');
    end
    if ~isscalar(Q) || Q ~= floor(Q) || Q < 2
        error('genetic_algorithm_solver:BadQ', 'Q must be an integer >= 2.');
    end
    if ~isempty(time_limit) && (time_limit <= 0)
        error('genetic_algorithm_solver:BadTimeLimit', ...
            'time_limit must be a positive number or [].');
    end

    % ---- parameters, with defaults filled in ----
    population_size = get_param(params, 'population_size', 40);
    max_generations = get_param(params, 'max_generations', 300);
    crossover_prob  = get_param(params, 'crossover_prob', 0.9);
    mutation_prob   = get_param(params, 'mutation_prob', 1 / N);
    tournament_size = get_param(params, 'tournament_size', 3);
    elite_count     = get_param(params, 'elite_count', 2);

    if ~isscalar(population_size) || population_size < 2
        error('genetic_algorithm_solver:BadPopulationSize', ...
            'population_size must be >= 2.');
    end
    if elite_count < 0 || elite_count >= population_size
        error('genetic_algorithm_solver:BadEliteCount', ...
            'elite_count must satisfy 0 <= elite_count < population_size.');
    end
    if tournament_size < 1 || tournament_size > population_size
        error('genetic_algorithm_solver:BadTournamentSize', ...
            'tournament_size must be between 1 and population_size.');
    end

    % ---- seeding (see note above re: RandStream vs global rng) ----
    if isempty(seed)
        rng('shuffle');
    else
        rng(seed);
    end

    tic_id = tic;
    objective_evaluations = 0;
    status = 'completed';

    % ---- 1. initial population ----
    population = randi([0, Q - 1], population_size, N);
    objectives = zeros(population_size, 1);
    for i = 1:population_size
        objectives(i) = islr_from_phase_indices(population(i, :), Q);
        objective_evaluations = objective_evaluations + 1;
    end

    [best_objective, best_idx] = min(objectives);
    best_solution = population(best_idx, :);

    generation = 0;
    hit_time_limit = false;

    while generation < max_generations
        if ~isempty(time_limit) && toc(tic_id) >= time_limit
            status = 'time_limit_reached';
            hit_time_limit = true;
            break;
        end

        % ---- elitism: carry the best `elite_count` individuals forward
        %      without re-evaluating them (their objective is already
        %      known) ----
        [~, order] = sort(objectives, 'ascend');
        elite_idx = order(1:elite_count);

        next_population = zeros(population_size, N);
        next_objectives = zeros(population_size, 1);
        next_population(1:elite_count, :) = population(elite_idx, :);
        next_objectives(1:elite_count) = objectives(elite_idx);
        filled = elite_count;

        % ---- fill the rest of the next generation ----
        while filled < population_size
            if ~isempty(time_limit) && toc(tic_id) >= time_limit
                status = 'time_limit_reached';
                hit_time_limit = true;
                break;
            end

            parent1 = population(tournament_select(objectives, tournament_size), :);
            parent2 = population(tournament_select(objectives, tournament_size), :);

            if rand() < crossover_prob
                child = uniform_crossover(parent1, parent2);
            else
                child = parent1;
            end

            child = mutate(child, Q, mutation_prob);

            child_objective = islr_from_phase_indices(child, Q);
            objective_evaluations = objective_evaluations + 1;

            filled = filled + 1;
            next_population(filled, :) = child;
            next_objectives(filled) = child_objective;

            if child_objective < best_objective
                best_objective = child_objective;
                best_solution = child;
            end
        end

        if hit_time_limit
            % Whatever slots got filled before the cutoff are discarded;
            % the CURRENT population (already fully evaluated) is kept
            % as-is, and best_solution/best_objective already reflect
            % every individual evaluated so far, elite or not.
            break;
        end

        population = next_population;
        objectives = next_objectives;
        generation = generation + 1;
    end

    runtime_seconds = toc(tic_id);

    validate_phase_indices(best_solution, N, Q);

    result = struct( ...
        'algorithm', 'genetic_algorithm', ...
        'N', N, ...
        'Q', Q, ...
        'best_phase_indices', best_solution, ...
        'best_objective', best_objective, ...
        'runtime_seconds', runtime_seconds, ...
        'objective_evaluations', objective_evaluations, ...
        'status', status, ...
        'seed', seed, ...
        'time_limit_seconds', time_limit, ...
        'metadata', struct( ...
            'population_size', population_size, ...
            'generations_completed', generation, ...
            'max_generations', max_generations, ...
            'crossover_prob', crossover_prob, ...
            'mutation_prob', mutation_prob, ...
            'tournament_size', tournament_size, ...
            'elite_count', elite_count, ...
            'selection', 'tournament', ...
            'crossover', 'uniform', ...
            'mutation', 'per_gene_random_reset', ...
            'fitness_transform', 'none_direct_minimization' ...
        ) ...
    );
end

% ------------------------------------------------------------------------
% local helpers (only callable from within this file, same rule as
% enumeration_solver.m's local_decode_mixed_radix)
% ------------------------------------------------------------------------

function v = get_param(params, name, default)
    if isfield(params, name) && ~isempty(params.(name))
        v = params.(name);
    else
        v = default;
    end
end

function idx = tournament_select(objectives, tournament_size)
    n = numel(objectives);
    candidates = randi(n, 1, tournament_size);
    [~, local_best] = min(objectives(candidates));
    idx = candidates(local_best);
end

function child = uniform_crossover(parent1, parent2)
    mask = rand(1, numel(parent1)) < 0.5;
    child = parent1;
    child(mask) = parent2(mask);
end

function child = mutate(child, Q, mutation_prob)
    mask = rand(1, numel(child)) < mutation_prob;
    n_mutate = sum(mask);
    if n_mutate > 0
        child(mask) = randi([0, Q - 1], 1, n_mutate);
    end
end