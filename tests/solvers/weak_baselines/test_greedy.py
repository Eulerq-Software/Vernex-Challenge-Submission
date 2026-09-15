import pytest

from waveform_optimization.core.objective import islr_from_phase_indices
from waveform_optimization.solvers.weak_baselines.greedy import GreedySolver


def test_greedy_returns_valid_result():
    result = GreedySolver().solve(N=7, Q=2)

    assert result.algorithm == "greedy"
    assert result.N == 7
    assert result.Q == 2
    assert len(result.best_phase_indices) == 7
    assert all(0 <= phase < 2 for phase in result.best_phase_indices)
    assert result.best_objective >= 0
    assert result.status == "completed"
    assert result.objective_evaluations > 0


def test_greedy_objective_matches_shared_objective():
    result = GreedySolver().solve(N=7, Q=2)

    expected_objective = islr_from_phase_indices(
        result.best_phase_indices,
        Q=2,
    )

    assert result.best_objective == pytest.approx(expected_objective)


def test_greedy_is_reproducible():
    solver = GreedySolver()

    first = solver.solve(N=8, Q=4, seed=42)
    second = solver.solve(N=8, Q=4, seed=42)

    assert first.best_phase_indices == second.best_phase_indices
    assert first.best_objective == second.best_objective


def test_greedy_respects_immediate_time_limit():
    result = GreedySolver().solve(N=20, Q=4, time_limit=0.0)

    assert result.status == "time_limit_reached"
    assert len(result.best_phase_indices) == 20
    assert all(0 <= phase < 4 for phase in result.best_phase_indices)


def test_greedy_rejects_invalid_inputs():
    solver = GreedySolver()

    with pytest.raises(ValueError):
        solver.solve(N=0, Q=2)

    with pytest.raises(ValueError):
        solver.solve(N=5, Q=1)