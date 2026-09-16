import pytest

from waveform_optimization.core.objective import islr_from_phase_indices
from waveform_optimization.solvers.weak_baselines.random_search import (
    RandomSearchSolver,
)


def test_random_search_returns_valid_result():
    result = RandomSearchSolver().solve(
        N=7,
        Q=2,
        seed=42,
        max_iterations=25,
    )

    assert result.algorithm == "random_search"
    assert result.N == 7
    assert result.Q == 2
    assert len(result.best_phase_indices) == 7
    assert all(0 <= phase < 2 for phase in result.best_phase_indices)
    assert result.best_objective >= 0
    assert result.status == "completed"
    assert result.objective_evaluations == 26
    assert result.metadata["iterations"] == 25


def test_random_search_objective_matches_shared_objective():
    result = RandomSearchSolver().solve(
        N=8,
        Q=4,
        seed=42,
        max_iterations=30,
    )

    expected_objective = islr_from_phase_indices(
        result.best_phase_indices,
        Q=4,
    )

    assert result.best_objective == pytest.approx(expected_objective)


def test_random_search_is_reproducible_with_seed():
    solver = RandomSearchSolver()

    first = solver.solve(N=10, Q=4, seed=99, max_iterations=50)
    second = solver.solve(N=10, Q=4, seed=99, max_iterations=50)

    assert first.best_phase_indices == second.best_phase_indices
    assert first.best_objective == second.best_objective


def test_random_search_respects_evaluation_limit():
    result = RandomSearchSolver().solve(
        N=8,
        Q=4,
        seed=42,
        max_evaluations=11,
    )

    assert result.objective_evaluations == 11
    assert result.metadata["iterations"] == 10


def test_random_search_records_history():
    result = RandomSearchSolver().solve(
        N=7,
        Q=2,
        seed=42,
        max_iterations=5,
        record_history=True,
    )

    assert len(result.history) == 5
    assert "elapsed_seconds" in result.history[0]
    assert "objective" in result.history[0]


def test_random_search_respects_immediate_time_limit():
    result = RandomSearchSolver().solve(
        N=20,
        Q=4,
        seed=42,
        time_limit=0.0,
    )

    assert result.status == "time_limit_reached"
    assert len(result.best_phase_indices) == 20
    assert all(0 <= phase < 4 for phase in result.best_phase_indices)


def test_random_search_rejects_invalid_inputs():
    solver = RandomSearchSolver()

    with pytest.raises(ValueError):
        solver.solve(N=0, Q=2)
        
    with pytest.raises(ValueError):
        solver.solve(N=1, Q=2)

    with pytest.raises(ValueError):
        solver.solve(N=5, Q=1)