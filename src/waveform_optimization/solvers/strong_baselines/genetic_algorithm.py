from __future__ import annotations

import random
import time
from typing import Any

from waveform_optimization.benchmarking.schemas import SolverResult
from waveform_optimization.core.objective import islr_from_phase_indices
from waveform_optimization.core.validation import validate_phase_indices
from waveform_optimization.solvers.base import Solver


class GeneticAlgorithmSolver(Solver):
    def solve(
        self,
        N: int,
        Q: int,
        time_limit: float | None = None,
        seed: int | None = None,
        initial_solution: list[int] | None = None,
        population_size: int = 50,
        max_generations: int = 200,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.05,
        tournament_size: int = 3,
        elite_count: int = 2,
        **kwargs: Any,
    ) -> SolverResult:
        if not isinstance(N, int) or isinstance(N, bool) or N <= 0:
            raise ValueError("N must be a positive integer.")

        if not isinstance(Q, int) or isinstance(Q, bool) or Q < 2:
            raise ValueError("Q must be an integer >= 2.")

        if time_limit is not None:
            if (
                not isinstance(time_limit, (int, float))
                or isinstance(time_limit, bool)
                or time_limit <= 0
            ):
                raise ValueError("time_limit must be a positive number or None.")

        if not isinstance(population_size, int) or population_size < 2:
            raise ValueError("population_size must be an integer >= 2.")

        if not isinstance(max_generations, int) or max_generations <= 0:
            raise ValueError("max_generations must be a positive integer.")

        if not (0 <= crossover_rate <= 1):
            raise ValueError("crossover_rate must be in [0, 1].")

        if not (0 <= mutation_rate <= 1):
            raise ValueError("mutation_rate must be in [0, 1].")

        if not isinstance(tournament_size, int) or not (1 <= tournament_size <= population_size):
            raise ValueError("tournament_size must be an integer in [1, population_size].")

        if not isinstance(elite_count, int) or not (0 <= elite_count < population_size):
            raise ValueError("elite_count must be an integer in [0, population_size).")

        if initial_solution is not None:
            validate_phase_indices(initial_solution, N=N, Q=Q)

        rng = random.Random(seed)
        start_time = time.perf_counter()

        def time_exceeded() -> bool:
            return time_limit is not None and time.perf_counter() - start_time >= time_limit

        def random_individual() -> list[int]:
            return [rng.randrange(Q) for _ in range(N)]

        def mutate(individual: list[int]) -> None:
            for i in range(N):
                if rng.random() < mutation_rate:
                    offset = rng.randrange(1, Q)
                    individual[i] = (individual[i] + offset) % Q

        def crossover(parent_a: list[int], parent_b: list[int]) -> tuple[list[int], list[int]]:
            if N > 1 and rng.random() < crossover_rate:
                point = rng.randrange(1, N)
                child_a = parent_a[:point] + parent_b[point:]
                child_b = parent_b[:point] + parent_a[point:]
            else:
                child_a = list(parent_a)
                child_b = list(parent_b)
            return child_a, child_b

        population = []
        if initial_solution is not None:
            population.append(list(initial_solution))
        while len(population) < population_size:
            population.append(random_individual())

        objectives = [islr_from_phase_indices(q=ind, Q=Q) for ind in population]
        objective_evaluations = len(population)

        best_index = min(range(population_size), key=lambda i: objectives[i])
        best_phase_indices = list(population[best_index])
        best_objective = objectives[best_index]

        history: list[dict[str, float]] = []
        status = "completed"
        generation = 0

        while generation < max_generations:
            if time_exceeded():
                status = "time_limit_reached"
                break

            ranked = sorted(range(population_size), key=lambda i: objectives[i])
            elite_indices = ranked[:elite_count]

            new_population = [list(population[i]) for i in elite_indices]
            new_objectives = [objectives[i] for i in elite_indices]

            def tournament_select() -> list[int]:
                contestants = [rng.randrange(population_size) for _ in range(tournament_size)]
                winner = min(contestants, key=lambda i: objectives[i])
                return population[winner]

            while len(new_population) < population_size:
                if time_exceeded():
                    status = "time_limit_reached"
                    break

                parent_a = tournament_select()
                parent_b = tournament_select()
                child_a, child_b = crossover(parent_a, parent_b)
                mutate(child_a)
                mutate(child_b)

                for child in (child_a, child_b):
                    if len(new_population) >= population_size:
                        break
                    objective = islr_from_phase_indices(q=child, Q=Q)
                    objective_evaluations += 1
                    new_population.append(child)
                    new_objectives.append(objective)
                    if objective < best_objective:
                        best_objective = objective
                        best_phase_indices = list(child)

            population = new_population
            objectives = new_objectives
            generation += 1

            history.append(
                {
                    "elapsed_seconds": time.perf_counter() - start_time,
                    "generation": generation,
                    "objective": best_objective,
                }
            )

            if status == "time_limit_reached":
                break

        runtime_seconds = time.perf_counter() - start_time

        validate_phase_indices(best_phase_indices, N=N, Q=Q)

        return SolverResult(
            algorithm="genetic_algorithm",
            N=N,
            Q=Q,
            best_phase_indices=best_phase_indices,
            best_objective=best_objective,
            runtime_seconds=runtime_seconds,
            objective_evaluations=objective_evaluations,
            status=status,
            seed=seed,
            time_limit_seconds=time_limit,
            history=history,
            metadata={
                "population_size": population_size,
                "max_generations": max_generations,
                "generations_completed": generation,
                "crossover_rate": crossover_rate,
                "mutation_rate": mutation_rate,
                "tournament_size": tournament_size,
                "elite_count": elite_count,
            },
        )