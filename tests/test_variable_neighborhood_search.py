import pytest

from waveform_optimization.benchmarking.schemas import SolverResult
from waveform_optimization.core.objective import islr_from_phase_indices
from waveform_optimization.solvers.strong_baselines.variable_neighborhood_search import (
    VariableNeighborhoodSearchSolver,
)


def test_basic_execution():
    solver = VariableNeighborhoodSearchSolver()
    result = solver.solve(N=11, Q=4, seed=1, max_iterations=10)

    assert result.status in ("completed", "time_limit_reached")


def test_result_type():
    solver = VariableNeighborhoodSearchSolver()
    result = solver.solve(N=11, Q=4, seed=1, max_iterations=10)

    assert isinstance(result, SolverResult)


def test_result_N_and_Q():
    solver = VariableNeighborhoodSearchSolver()
    result = solver.solve(N=11, Q=4, seed=1, max_iterations=10)

    assert result.N == 11
    assert result.Q == 4
    assert len(result.best_phase_indices) == 11


def test_solution_validity():
    solver = VariableNeighborhoodSearchSolver()
    result = solver.solve(N=15, Q=5, seed=2, max_iterations=10)

    assert all(0 <= q < result.Q for q in result.best_phase_indices)


def test_nonnegative_objective():
    solver = VariableNeighborhoodSearchSolver()
    result = solver.solve(N=15, Q=5, seed=2, max_iterations=10)

    assert result.best_objective >= 0


def test_objective_agrees_with_recomputation():
    solver = VariableNeighborhoodSearchSolver()
    result = solver.solve(N=13, Q=3, seed=3, max_iterations=10)

    recomputed = islr_from_phase_indices(q=result.best_phase_indices, Q=result.Q)

    assert result.best_objective == pytest.approx(recomputed)


def test_reproducibility_with_seed():
    solver = VariableNeighborhoodSearchSolver()
    result_a = solver.solve(N=13, Q=4, seed=42, max_iterations=15)
    result_b = solver.solve(N=13, Q=4, seed=42, max_iterations=15)

    assert result_a.best_phase_indices == result_b.best_phase_indices
    assert result_a.best_objective == result_b.best_objective


def test_different_seeds_can_diverge():
    solver = VariableNeighborhoodSearchSolver()
    result_a = solver.solve(N=25, Q=4, seed=1, max_iterations=15)
    result_b = solver.solve(N=25, Q=4, seed=2, max_iterations=15)

    assert result_a.best_phase_indices != result_b.best_phase_indices


def test_time_limit_behavior():
    solver = VariableNeighborhoodSearchSolver()
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
    solver = VariableNeighborhoodSearchSolver()
    result = solver.solve(N=10, Q=3, seed=5, max_iterations=7)

    assert result.metadata["iterations_completed"] <= 7


def test_local_search_never_worsens_best_solution():
    solver = VariableNeighborhoodSearchSolver()
    initial = [0] * 12
    result = solver.solve(
        N=12,
        Q=3,
        seed=6,
        initial_solution=initial,
        max_iterations=15,
    )

    assert result.best_objective <= islr_from_phase_indices(q=initial, Q=3)


def test_neighborhood_level_resets_on_improvement():
    solver = VariableNeighborhoodSearchSolver()
    result = solver.solve(N=20, Q=5, seed=6, max_iterations=30)

    assert 0 <= result.metadata["final_neighborhood_level"] < 3
    assert result.metadata["accepted_shakes"] <= result.metadata["iterations_completed"]


def test_block_size_is_clamped_to_N():
    solver = VariableNeighborhoodSearchSolver()
    result = solver.solve(N=4, Q=3, seed=7, max_iterations=5, block_size=100)

    assert result.metadata["block_size"] == 4


def test_objective_evaluations_are_positive():
    solver = VariableNeighborhoodSearchSolver()
    result = solver.solve(N=10, Q=3, seed=8, max_iterations=10)

    assert result.objective_evaluations > 0


def test_accepts_initial_solution():
    solver = VariableNeighborhoodSearchSolver()
    initial = [0] * 10
    result = solver.solve(
        N=10,
        Q=3,
        seed=9,
        initial_solution=initial,
        max_iterations=15,
    )

    assert result.best_objective <= islr_from_phase_indices(q=initial, Q=3)


def test_invalid_N():
    solver = VariableNeighborhoodSearchSolver()

    with pytest.raises(ValueError):
        solver.solve(N=0, Q=4)


def test_invalid_Q():
    solver = VariableNeighborhoodSearchSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=1)


def test_invalid_time_limit():
    solver = VariableNeighborhoodSearchSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, time_limit=-1)


def test_invalid_max_iterations():
    solver = VariableNeighborhoodSearchSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, max_iterations=0)


def test_invalid_block_size():
    solver = VariableNeighborhoodSearchSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, block_size=0)


def test_invalid_local_search_max_sweeps():
    solver = VariableNeighborhoodSearchSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, local_search_max_sweeps=0)


def test_invalid_initial_solution_length():
    solver = VariableNeighborhoodSearchSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, initial_solution=[0, 1, 2])