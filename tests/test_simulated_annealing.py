import pytest

from waveform_optimization.benchmarking.schemas import SolverResult
from waveform_optimization.core.objective import islr_from_phase_indices
from waveform_optimization.solvers.strong_baselines.simulated_annealing import (
    SimulatedAnnealingSolver,
)


def test_basic_execution():
    solver = SimulatedAnnealingSolver()
    result = solver.solve(N=11, Q=4, seed=1, iterations_per_temperature=20)

    assert result.status in ("completed", "time_limit_reached")


def test_result_type():
    solver = SimulatedAnnealingSolver()
    result = solver.solve(N=11, Q=4, seed=1, iterations_per_temperature=20)

    assert isinstance(result, SolverResult)


def test_result_N_and_Q():
    solver = SimulatedAnnealingSolver()
    result = solver.solve(N=11, Q=4, seed=1, iterations_per_temperature=20)

    assert result.N == 11
    assert result.Q == 4
    assert len(result.best_phase_indices) == 11


def test_solution_validity():
    solver = SimulatedAnnealingSolver()
    result = solver.solve(N=15, Q=5, seed=2, iterations_per_temperature=20)

    assert all(0 <= q < result.Q for q in result.best_phase_indices)


def test_nonnegative_objective():
    solver = SimulatedAnnealingSolver()
    result = solver.solve(N=15, Q=5, seed=2, iterations_per_temperature=20)

    assert result.best_objective >= 0


def test_objective_agrees_with_recomputation():
    solver = SimulatedAnnealingSolver()
    result = solver.solve(N=13, Q=3, seed=3, iterations_per_temperature=20)

    recomputed = islr_from_phase_indices(q=result.best_phase_indices, Q=result.Q)

    assert result.best_objective == pytest.approx(recomputed)


def test_reproducibility_with_seed():
    solver = SimulatedAnnealingSolver()
    result_a = solver.solve(N=13, Q=4, seed=42, iterations_per_temperature=20)
    result_b = solver.solve(N=13, Q=4, seed=42, iterations_per_temperature=20)

    assert result_a.best_phase_indices == result_b.best_phase_indices
    assert result_a.best_objective == result_b.best_objective


def test_different_seeds_can_diverge():
    solver = SimulatedAnnealingSolver()
    result_a = solver.solve(N=25, Q=4, seed=1, iterations_per_temperature=20)
    result_b = solver.solve(N=25, Q=4, seed=2, iterations_per_temperature=20)

    assert result_a.best_phase_indices != result_b.best_phase_indices


def test_time_limit_behavior():
    solver = SimulatedAnnealingSolver()
    result = solver.solve(
        N=40,
        Q=6,
        seed=4,
        time_limit=0.05,
        initial_temperature=50.0,
        final_temperature=1e-6,
        cooling_rate=0.999,
        iterations_per_temperature=100000,
    )

    assert result.status == "time_limit_reached"
    assert result.runtime_seconds <= 1.0


def test_max_iterations_limits_evaluations():
    solver = SimulatedAnnealingSolver()
    result = solver.solve(
        N=10,
        Q=3,
        seed=5,
        iterations_per_temperature=10,
        max_iterations=25,
    )

    assert result.metadata["total_iterations"] <= 25


def test_objective_evaluation_counting():
    solver = SimulatedAnnealingSolver()
    result = solver.solve(N=10, Q=3, seed=6, iterations_per_temperature=15, max_iterations=40)

    assert result.objective_evaluations == 1 + result.metadata["total_iterations"]


def test_accepts_initial_solution():
    solver = SimulatedAnnealingSolver()
    initial = [0] * 10
    result = solver.solve(
        N=10,
        Q=3,
        seed=7,
        initial_solution=initial,
        iterations_per_temperature=15,
        max_iterations=10,
    )

    assert result.best_objective <= islr_from_phase_indices(q=initial, Q=3)


def test_invalid_N():
    solver = SimulatedAnnealingSolver()

    with pytest.raises(ValueError):
        solver.solve(N=0, Q=4)


def test_invalid_Q():
    solver = SimulatedAnnealingSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=1)


def test_invalid_time_limit():
    solver = SimulatedAnnealingSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, time_limit=-1)


def test_invalid_cooling_rate():
    solver = SimulatedAnnealingSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, cooling_rate=1.5)


def test_invalid_temperature_ordering():
    solver = SimulatedAnnealingSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, initial_temperature=1.0, final_temperature=1.0)


def test_invalid_initial_solution_length():
    solver = SimulatedAnnealingSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, initial_solution=[0, 1, 2])