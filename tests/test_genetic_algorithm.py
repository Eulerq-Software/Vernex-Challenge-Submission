import pytest

from waveform_optimization.benchmarking.schemas import SolverResult
from waveform_optimization.core.objective import islr_from_phase_indices
from waveform_optimization.solvers.strong_baselines.genetic_algorithm import (
    GeneticAlgorithmSolver,
)


def test_basic_execution():
    solver = GeneticAlgorithmSolver()
    result = solver.solve(N=11, Q=4, seed=1, population_size=20, max_generations=10)

    assert result.status in ("completed", "time_limit_reached")


def test_result_type():
    solver = GeneticAlgorithmSolver()
    result = solver.solve(N=11, Q=4, seed=1, population_size=20, max_generations=10)

    assert isinstance(result, SolverResult)


def test_result_N_and_Q():
    solver = GeneticAlgorithmSolver()
    result = solver.solve(N=11, Q=4, seed=1, population_size=20, max_generations=10)

    assert result.N == 11
    assert result.Q == 4
    assert len(result.best_phase_indices) == 11


def test_solution_validity():
    solver = GeneticAlgorithmSolver()
    result = solver.solve(N=15, Q=5, seed=2, population_size=20, max_generations=10)

    assert all(0 <= q < result.Q for q in result.best_phase_indices)


def test_nonnegative_objective():
    solver = GeneticAlgorithmSolver()
    result = solver.solve(N=15, Q=5, seed=2, population_size=20, max_generations=10)

    assert result.best_objective >= 0


def test_objective_agrees_with_recomputation():
    solver = GeneticAlgorithmSolver()
    result = solver.solve(N=13, Q=3, seed=3, population_size=20, max_generations=10)

    recomputed = islr_from_phase_indices(q=result.best_phase_indices, Q=result.Q)

    assert result.best_objective == pytest.approx(recomputed)


def test_reproducibility_with_seed():
    solver = GeneticAlgorithmSolver()
    result_a = solver.solve(N=13, Q=4, seed=42, population_size=20, max_generations=15)
    result_b = solver.solve(N=13, Q=4, seed=42, population_size=20, max_generations=15)

    assert result_a.best_phase_indices == result_b.best_phase_indices
    assert result_a.best_objective == result_b.best_objective


def test_different_seeds_can_diverge():
    solver = GeneticAlgorithmSolver()
    result_a = solver.solve(N=25, Q=4, seed=1, population_size=20, max_generations=15)
    result_b = solver.solve(N=25, Q=4, seed=2, population_size=20, max_generations=15)

    assert result_a.best_phase_indices != result_b.best_phase_indices


def test_time_limit_behavior():
    solver = GeneticAlgorithmSolver()
    result = solver.solve(
        N=40,
        Q=6,
        seed=4,
        time_limit=0.05,
        population_size=200,
        max_generations=100000,
    )

    assert result.status == "time_limit_reached"
    assert result.runtime_seconds <= 1.0


def test_generations_do_not_exceed_max_generations():
    solver = GeneticAlgorithmSolver()
    result = solver.solve(N=10, Q=3, seed=5, population_size=20, max_generations=7)

    assert result.metadata["generations_completed"] <= 7


def test_objective_evaluation_counting():
    solver = GeneticAlgorithmSolver()
    population_size = 20
    elite_count = 2
    result = solver.solve(
        N=10,
        Q=3,
        seed=6,
        population_size=population_size,
        elite_count=elite_count,
        max_generations=5,
    )

    generations = result.metadata["generations_completed"]
    expected = population_size + generations * (population_size - elite_count)

    assert result.objective_evaluations == expected


def test_elitism_never_loses_best_solution():
    solver = GeneticAlgorithmSolver()
    result = solver.solve(
        N=12,
        Q=3,
        seed=8,
        population_size=15,
        elite_count=3,
        max_generations=25,
        mutation_rate=0.3,
    )

    recomputed = islr_from_phase_indices(q=result.best_phase_indices, Q=result.Q)

    assert result.best_objective <= recomputed + 1e-9


def test_accepts_initial_solution():
    solver = GeneticAlgorithmSolver()
    initial = [0] * 10
    result = solver.solve(
        N=10,
        Q=3,
        seed=7,
        initial_solution=initial,
        population_size=15,
        max_generations=15,
    )

    assert result.best_objective <= islr_from_phase_indices(q=initial, Q=3)


def test_invalid_N():
    solver = GeneticAlgorithmSolver()

    with pytest.raises(ValueError):
        solver.solve(N=0, Q=4)


def test_invalid_Q():
    solver = GeneticAlgorithmSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=1)


def test_invalid_time_limit():
    solver = GeneticAlgorithmSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, time_limit=-1)


def test_invalid_population_size():
    solver = GeneticAlgorithmSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, population_size=1)


def test_invalid_crossover_rate():
    solver = GeneticAlgorithmSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, crossover_rate=1.5)


def test_invalid_mutation_rate():
    solver = GeneticAlgorithmSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, mutation_rate=-0.1)


def test_invalid_tournament_size():
    solver = GeneticAlgorithmSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, population_size=10, tournament_size=11)


def test_invalid_elite_count():
    solver = GeneticAlgorithmSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, population_size=10, elite_count=10)


def test_invalid_initial_solution_length():
    solver = GeneticAlgorithmSolver()

    with pytest.raises(ValueError):
        solver.solve(N=10, Q=4, initial_solution=[0, 1, 2])