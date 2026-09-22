function test_genetic_algorithm()
%TEST_GENETIC_ALGORITHM Tests for solvers/genetic_algorithm_solver.m
%
%   Run with:  test_genetic_algorithm
%   (matlab/core and matlab/solvers must be on the path)
%
%   Covers every bullet in the project spec's testing requirements
%   (Section 3.8): basic execution, result type, N/Q, validity of the
%   returned sequence, nonnegative objective, agreement between the
%   reported and independently recomputed objective, reproducibility
%   with a seed, time-limit behavior, and invalid input handling.
%
%   Small-instance GA runs are intentionally given a generous budget so
%   they reliably match the known enumeration optimum - this is a
%   correctness check, not a speed benchmark (see
%   experiments/scripts/cross_lang_bench.sh for timing comparisons).

    tests = {
        @test_basic_execution, ...
        @test_result_has_expected_fields, ...
        @test_result_N_and_Q, ...
        @test_best_phase_indices_is_valid, ...
        @test_objective_is_nonnegative, ...
        @test_reported_objective_matches_recomputed, ...
        @test_reproducible_with_same_seed, ...
        @test_different_seeds_can_differ, ...
        @test_matches_enumeration_optimum_on_small_instance, ...
        @test_time_limit_status, ...
        @test_invalid_N_is_rejected, ...
        @test_invalid_Q_is_rejected, ...
        @test_invalid_elite_count_is_rejected, ...
        @test_invalid_tournament_size_is_rejected ...
    };

    n_pass = 0; n_fail = 0;
    for i = 1:numel(tests)
        fn = tests{i};
        name = func2str(fn);
        try
            fn();
            printf('[PASS] %s\n', name);
            n_pass = n_pass + 1;
        catch err
            printf('[FAIL] %s : %s\n', name, err.message);
            n_fail = n_fail + 1;
        end
    end
    printf('\n%d passed, %d failed (of %d)\n', n_pass, n_fail, numel(tests));
    if n_fail > 0
        error('test_genetic_algorithm:Failures', '%d test(s) failed.', n_fail);
    end
end

% ---- shared small-instance settings for the correctness checks --------

function p = small_params()
    p = struct('population_size', 20, 'max_generations', 60);
end

% ---- individual tests --------------------------------------------------

function test_basic_execution()
    r = genetic_algorithm_solver(5, 2, [], 1, small_params());
    assert(isstruct(r), 'result should be a struct');
end

function test_result_has_expected_fields()
    r = genetic_algorithm_solver(5, 2, [], 1, small_params());
    required = {'algorithm','N','Q','best_phase_indices','best_objective', ...
        'runtime_seconds','objective_evaluations','status','seed', ...
        'time_limit_seconds','metadata'};
    for i = 1:numel(required)
        assert(isfield(r, required{i}), sprintf('missing field %s', required{i}));
    end
end

function test_result_N_and_Q()
    r = genetic_algorithm_solver(6, 3, [], 1, small_params());
    assert(r.N == 6, 'N mismatch');
    assert(r.Q == 3, 'Q mismatch');
    assert(numel(r.best_phase_indices) == 6, 'best_phase_indices wrong length');
end

function test_best_phase_indices_is_valid()
    r = genetic_algorithm_solver(6, 3, [], 1, small_params());
    validate_phase_indices(r.best_phase_indices, r.N, r.Q);  % should not throw
end

function test_objective_is_nonnegative()
    r = genetic_algorithm_solver(6, 3, [], 1, small_params());
    assert(r.best_objective >= 0, 'objective should be nonnegative');
end

function test_reported_objective_matches_recomputed()
    r = genetic_algorithm_solver(6, 3, [], 1, small_params());
    recomputed = islr_from_phase_indices(r.best_phase_indices, r.Q);
    assert(abs(recomputed - r.best_objective) < 1e-9, ...
        'reported objective does not match independent recomputation');
end

function test_reproducible_with_same_seed()
    p = small_params();
    r1 = genetic_algorithm_solver(8, 3, [], 7, p);
    r2 = genetic_algorithm_solver(8, 3, [], 7, p);
    assert(isequal(r1.best_phase_indices, r2.best_phase_indices), ...
        'same seed should give the same best sequence');
    assert(r1.objective_evaluations == r2.objective_evaluations, ...
        'same seed should give the same evaluation count');
end

function test_different_seeds_can_differ()
    p = small_params();
    r1 = genetic_algorithm_solver(10, 3, [], 1, p);
    r2 = genetic_algorithm_solver(10, 3, [], 2, p);
    % Not a strict requirement that they MUST differ, but for a
    % nontrivial instance and a modest budget they very likely will;
    % this guards against an accidental hardcoded/ignored seed.
    if isequal(r1.best_phase_indices, r2.best_phase_indices)
        warning('test_genetic_algorithm:SameResultDifferentSeeds', ...
            'Different seeds happened to converge to the same solution; not a failure by itself.');
    end
end

function test_matches_enumeration_optimum_on_small_instance()
    N = 4; Q = 3;
    ga_r = genetic_algorithm_solver(N, Q, [], 1, small_params());
    en_r = enumeration_solver(N, Q);
    assert(abs(ga_r.best_objective - en_r.best_objective) < 1e-9, ...
        sprintf('GA (%.6f) did not reach the known enumeration optimum (%.6f)', ...
        ga_r.best_objective, en_r.best_objective));
end

function test_time_limit_status()
    % A larger instance with a very tight time limit should not finish
    % max_generations in time.
    r = genetic_algorithm_solver(20, 4, 0.05, 1, struct('population_size', 40, 'max_generations', 100000));
    assert(strcmp(r.status, 'time_limit_reached'), ...
        'expected time_limit_reached for a tight limit on a large instance');
    assert(r.runtime_seconds < 2.0, ...
        'runtime should stay close to the requested time limit');
end

function test_invalid_N_is_rejected()
    threw = false;
    try
        genetic_algorithm_solver(0, 2);
    catch
        threw = true;
    end
    assert(threw, 'expected N=0 to be rejected');
end

function test_invalid_Q_is_rejected()
    threw = false;
    try
        genetic_algorithm_solver(5, 1);
    catch
        threw = true;
    end
    assert(threw, 'expected Q=1 to be rejected');
end

function test_invalid_elite_count_is_rejected()
    threw = false;
    try
        genetic_algorithm_solver(5, 2, [], [], struct('elite_count', 40, 'population_size', 10));
    catch
        threw = true;
    end
    assert(threw, 'expected elite_count >= population_size to be rejected');
end

function test_invalid_tournament_size_is_rejected()
    threw = false;
    try
        genetic_algorithm_solver(5, 2, [], [], struct('tournament_size', 100, 'population_size', 10));
    catch
        threw = true;
    end
    assert(threw, 'expected tournament_size > population_size to be rejected');
end