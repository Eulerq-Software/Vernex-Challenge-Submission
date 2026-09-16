function test_enumeration_solver()
%TEST_ENUMERATION_SOLVER Tests for solvers/enumeration_solver.m
%
%   Run with:  test_enumeration_solver
%   (matlab/core and matlab/solvers must be on the path)

    tests = {
        @test_basic_execution, ...
        @test_result_has_expected_fields, ...
        @test_result_N_and_Q, ...
        @test_best_phase_indices_is_valid, ...
        @test_objective_is_nonnegative, ...
        @test_reported_objective_matches_recomputed, ...
        @test_evaluation_count_equals_Q_pow_N, ...
        @test_time_limit_status, ...
        @test_invalid_N_is_rejected, ...
        @test_invalid_Q_is_rejected ...
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
        error('test_enumeration_solver:Failures', '%d test(s) failed.', n_fail);
    end
end

function test_basic_execution()
    r = enumeration_solver(3, 2);
    assert(isstruct(r), 'result should be a struct');
end

function test_result_has_expected_fields()
    r = enumeration_solver(3, 2);
    required = {'algorithm','N','Q','best_phase_indices','best_objective', ...
        'runtime_seconds','objective_evaluations','status','seed', ...
        'time_limit_seconds'};
    for i = 1:numel(required)
        assert(isfield(r, required{i}), sprintf('missing field %s', required{i}));
    end
end

function test_result_N_and_Q()
    r = enumeration_solver(5, 2);
    assert(r.N == 5, 'N mismatch');
    assert(r.Q == 2, 'Q mismatch');
    assert(numel(r.best_phase_indices) == 5, 'best_phase_indices wrong length');
end

function test_best_phase_indices_is_valid()
    r = enumeration_solver(4, 3);
    % Should not throw:
    validate_phase_indices(r.best_phase_indices, r.N, r.Q);
end

function test_objective_is_nonnegative()
    r = enumeration_solver(4, 3);
    assert(r.best_objective >= 0, 'objective should be nonnegative');
end

function test_reported_objective_matches_recomputed()
    r = enumeration_solver(5, 2);
    recomputed = islr_from_phase_indices(r.best_phase_indices, r.Q);
    assert(abs(recomputed - r.best_objective) < 1e-9, ...
        'reported objective does not match independent recomputation');
end

function test_evaluation_count_equals_Q_pow_N()
    r = enumeration_solver(3, 2);
    assert(r.objective_evaluations == 2^3, 'evaluation count mismatch');
end

function test_time_limit_status()
    r = enumeration_solver(8, 4, 0.05);
    assert(strcmp(r.status, 'time_limit_reached') || strcmp(r.status, 'completed'), ...
        'unexpected status value');
    % With a tight limit on a 4^8 search space this should not complete:
    assert(strcmp(r.status, 'time_limit_reached'), ...
        'expected time_limit_reached for a tight limit on a large space');
end

function test_invalid_N_is_rejected()
    threw = false;
    try
        enumeration_solver(0, 2);
    catch
        threw = true;
    end
    assert(threw, 'expected N=0 to be rejected');
end

function test_invalid_Q_is_rejected()
    threw = false;
    try
        enumeration_solver(3, -1);
    catch
        threw = true;
    end
    assert(threw, 'expected negative Q to be rejected');
end
