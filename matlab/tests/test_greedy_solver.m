function test_greedy_solver()
%TEST_GREEDY_SOLVER Validation checks for greedy_solver, mirroring
%   tests/solvers/weak_baselines/test_greedy.py test-for-test.

    tests = { ...
        @check_returns_valid_result, ...
        @check_objective_matches_shared_objective, ...
        @check_is_reproducible, ...
        @check_respects_immediate_time_limit, ...
        @check_rejects_invalid_inputs ...
    };

    n_pass = 0;
    n_fail = 0;

    for i = 1:numel(tests)
        test_fn = tests{i};
        name = func2str(test_fn);
        try
            test_fn();
            fprintf('PASS: %s\n', name);
            n_pass = n_pass + 1;
        catch err
            fprintf('FAIL: %s\n    %s\n', name, err.message);
            n_fail = n_fail + 1;
        end
    end

    fprintf('\n%d passed, %d failed\n', n_pass, n_fail);

    if n_fail > 0
        error('test_greedy_solver:Failures', '%d test(s) failed.', n_fail);
    end
end

function check_returns_valid_result()
    result = greedy_solver(7, 2, [], []);

    assert(strcmp(result.algorithm, 'greedy'), 'algorithm mismatch');
    assert(result.N == 7, 'N mismatch');
    assert(result.Q == 2, 'Q mismatch');
    assert(numel(result.best_phase_indices) == 7, 'length mismatch');
    assert(all(result.best_phase_indices >= 0) && all(result.best_phase_indices < 2), ...
        'phase indices out of range');
    assert(result.best_objective >= 0, 'objective must be nonnegative');
    assert(strcmp(result.status, 'completed'), 'status mismatch');
    assert(result.objective_evaluations > 0, 'no evaluations counted');
end

function check_objective_matches_shared_objective()
    result = greedy_solver(7, 2, [], []);

    expected_objective = islr_from_phase_indices(result.best_phase_indices, 2);

    assert(abs(result.best_objective - expected_objective) < 1e-9, ...
        'reported objective disagrees with recomputed objective');
end

function check_is_reproducible()
    % Greedy has no randomness, so this holds for any seed value,
    % matching the Python test which fixes seed=42.
    first = greedy_solver(8, 4, [], 42);
    second = greedy_solver(8, 4, [], 42);

    assert(isequal(first.best_phase_indices, second.best_phase_indices), ...
        'best_phase_indices differ between runs with the same seed');
    assert(first.best_objective == second.best_objective, ...
        'best_objective differs between runs with the same seed');
end

function check_respects_immediate_time_limit()
    result = greedy_solver(20, 4, 0.0, []);

    assert(strcmp(result.status, 'time_limit_reached'), ...
        'status should be time_limit_reached');
    assert(numel(result.best_phase_indices) == 20, 'length mismatch');
    assert(all(result.best_phase_indices >= 0) && all(result.best_phase_indices < 4), ...
        'phase indices out of range');
end

function check_rejects_invalid_inputs()
    threw = false;
    try
        greedy_solver(0, 2, [], []);
    catch
        threw = true;
    end
    assert(threw, 'N=0 should raise an error');

    threw = false;
    try
        greedy_solver(5, 1, [], []);
    catch
        threw = true;
    end
    assert(threw, 'Q=1 should raise an error');
end