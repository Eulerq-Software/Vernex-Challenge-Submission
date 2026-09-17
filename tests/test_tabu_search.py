import pytest

from waveform_optimization.benchmarking.schemas import SolverResult
from waveform_optimization.core.objective import islr_from_phase_indices
from waveform_optimization.solvers.strong_baselines.tabu_search import TabuSearchSolver


def test_basic_execution():
    solver = TabuSearchSolver()
    result = solver.solve(N=11, Q=4, seed=1, max_iterations=20)

    assert result.status in ("completed", "time_limit_reached")


def test_result_type():
    solver = TabuSearchSolver()
    result = solver.solve(N=11, Q=4, seed=1, max_iterations=20)

    assert isinstance(result, SolverResult)


def test_result_N_and_Q():
    solver = TabuSearchSolver()
    result = solver.solve(N=11, Q=4, seed=1, max_iterations=20)

    assert result.N == 11
    assert result.Q == 4
    assert len(result.best_phase_indices) == 11


def test_solution_validity():
    solver = TabuSearchSolver()
    result = solver.solve(N=15, Q=5, seed=2, max_iterations=20)

    assert all(0 <= q < result.Q for q in result.best_phase_indices)


def test_nonnegative_objective():
    solver = TabuSearchSolver()
    result = solver.solve(N=15, Q=5, seed=2, max_iterations=20)

    assert result.best_objective >= 0


def test_objective_agrees_with_recomputation():
    solver = TabuSearchSolver()
    result = solver.solve(N=13, Q=3, seed=3, max_iterations=20)

    recomputed = islr_from_phase_indices(q=result.best_phase_indices, Q=result.Q)

    assert result.best_objective == pytest.approx(recomputed)


def test_reproducibility_with_seed():
    solver = TabuSearchSolver()
    result_a = solver.solve(N=13, Q=4, seed=42, max_iterations=25)
    result_b = solver.solve(N=13, Q=4, seed=42, max_iterations=25)

    assert result_a.best_phase_indices == result_b.best_phase_indices
    assert result_a.best_objective == result_b.best_objective


def test_different_seeds_can_diverge():
    solver = TabuSearchSolver()
    result_a = solver.solve(N=25, Q=4, seed=1, max_iterations=25)
    result_b = solver.solve(N=25, Q=4, seed=2, max_iterations=25)

    assert result_a.best_phase_indices != result_b.best_phase_indices


def test_time_limit_behavior():
    solver = TabuSearchSolver()
    result = solver.solve(
        N=40,
        Q=6,
        seed=4,
        time_limit=0.05,
        max_iterations=1000000,
    )

    assert result.status == "time_limit_reached"
    assert result.runtime_seconds <= 1.0


def test_iterations_do_not_exceed_max_iterations():
    solver = TabuSearchSolver()
    result = solver.solve(N=10, Q=3, seed=5, max_iterations=7)

    assert result.metadata["iterations_completed"] <= 7


def test_objective_evaluation_counting_full_neighborhood():
    solver = TabuSearchSolver()
    N, Q, max_iterations = 8, 3, 5
    result = solver.solve(
        N=N,
        Q=Q,
        seed=6,
        max_iterations=max_iterations,
        no_improvement_limit=None,
    )

    expected = 1 + result.metadata["iterations_completed"] * N * (Q - 1)

    assert result.objective_evaluations == expected


def test_neighborhood_size_limits_moves_per_iteration():
    solver = TabuSearchSolver()
    N, Q, max_iterations, neighborhood_size = 10, 4, 5, 6
    result = solver.solve(
        N=N,
        Q=Q,
        seed=6,
        max_iterations=max_iterations,
        neighborhood_size=neighborhood_size,
    )

    expected = 1 + result.metadata["iterations_completed"] * neighborhood_size

    assert result.objective_evaluations == expected


def test_tabu_tenure_prevents_immediate_reversal():
    solver = TabuSearchSolver()
    result = solver.solve(
        N=6,
        Q=3,
        seed=9,
        tabu_tenure=1000,
        max_iterations=1,
        use_aspiration=False,
    )

    assert result.metadata["iterations_completed"] == 1


def test_accepts_initial_solution():
    solver = TabuSearchSolver()
    initial = [0] * 10
    result = solver.solve(
        N=10,
        Q=3,
        seed=7,
        initial_solution=initial,
        max_iterations=15,
    )

    assert result.best_objective <= islr_from_phase_indices(q=initial, Q=3)


def test_invalid_N():
    solver = TabuSearchSolver()

    with pytest.raises(ValueError):
        solver.solve(N=0, Q=4)


def test_invalid_Q():
    solver = TabuSearchSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=1)


def test_invalid_time_limit():
    solver = TabuSearchSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, time_limit=-1)


def test_invalid_tabu_tenure():
    solver = TabuSearchSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, tabu_tenure=0)


def test_invalid_max_iterations():
    solver = TabuSearchSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, max_iterations=0)


def test_invalid_neighborhood_size():
    solver = TabuSearchSolver()

    with pytest.raises(ValueError):
        solver.solve(N=5, Q=2, neighborhood_size=100)


def test_invalid_no_improvement_limit():
    solver = TabuSearchSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, no_improvement_limit=0)


def test_invalid_initial_solution_length():
    solver = TabuSearchSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, initial_solution=[0, 1, 2])