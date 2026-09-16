import pytest

from waveform_optimization.core.objective import islr_from_phase_indices
from waveform_optimization.solvers.weak_baselines.coordinate_descent import (
    CoordinateDescentSolver,
)


def test_coordinate_descent_returns_valid_result():
    result = CoordinateDescentSolver().solve(
        N=7,
        Q=2,
        seed=42,
        max_iterations=20,
    )

    assert result.algorithm == "coordinate_descent"
    assert result.N == 7
    assert result.Q == 2
    assert len(result.best_phase_indices) == 7
    assert all(0 <= phase < 2 for phase in result.best_phase_indices)
    assert result.best_objective >= 0
    assert result.status == "completed"
    assert result.objective_evaluations > 0
    assert result.metadata["improvement_strategy"] == "best_improvement"


def test_coordinate_descent_objective_matches_shared_objective():
    result = CoordinateDescentSolver().solve(
        N=8,
        Q=4,
        seed=42,
        max_iterations=20,
    )

    expected_objective = islr_from_phase_indices(
        result.best_phase_indices,
        Q=4,
    )

    assert result.best_objective == pytest.approx(expected_objective)


def test_coordinate_descent_is_reproducible_with_seed():
    solver = CoordinateDescentSolver()

    first = solver.solve(N=10, Q=4, seed=99, max_iterations=20)
    second = solver.solve(N=10, Q=4, seed=99, max_iterations=20)

    assert first.best_phase_indices == second.best_phase_indices
    assert first.best_objective == second.best_objective


def test_coordinate_descent_accepts_initial_solution():
    result = CoordinateDescentSolver().solve(
        N=2,
        Q=4,
        initial_solution=[2,0],
        max_iterations=10,
    )

    assert result.metadata["initialization"] == "provided"
    assert result.best_phase_indices == [2,0]


def test_coordinate_descent_respects_evaluation_limit():
    result = CoordinateDescentSolver().solve(
        N=10,
        Q=4,
        seed=42,
        max_evaluations=10,
    )

    assert result.objective_evaluations == 10


def test_coordinate_descent_records_history():
    result = CoordinateDescentSolver().solve(
        N=7,
        Q=2,
        seed=42,
        max_iterations=20,
        record_history=True,
    )

    assert len(result.history) >= 1
    assert "elapsed_seconds" in result.history[0]
    assert "objective" in result.history[0]


def test_coordinate_descent_respects_immediate_time_limit():
    result = CoordinateDescentSolver().solve(
        N=20,
        Q=4,
        seed=42,
        time_limit=0.0,
    )

    assert result.status == "time_limit_reached"
    assert len(result.best_phase_indices) == 20
    assert all(0 <= phase < 4 for phase in result.best_phase_indices)


def test_coordinate_descent_rejects_invalid_inputs():
    solver = CoordinateDescentSolver()

    with pytest.raises(ValueError):
        solver.solve(N=0, Q=2)

    with pytest.raises(ValueError):
        solver.solve(N=1, Q=2)

    with pytest.raises(ValueError):
        solver.solve(N=5, Q=1)