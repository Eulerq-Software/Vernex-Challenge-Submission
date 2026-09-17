from __future__ import annotations

import random
import time
from typing import Any

from waveform_optimization.benchmarking.schemas import SolverResult
from waveform_optimization.core.objective import islr_from_phase_indices
from waveform_optimization.core.validation import validate_phase_indices
from waveform_optimization.solvers.base import Solver

NUM_NEIGHBORHOOD_LEVELS = 3


class VariableNeighborhoodSearchSolver(Solver):
    def solve(
        self,
        N: int,
        Q: int,
        time_limit: float | None = None,
        seed: int | None = None,
        initial_solution: list[int] | None = None,
        max_iterations: int = 100,
        block_size: int = 3,
        local_search_max_sweeps: int = 10,
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

        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")

        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("block_size must be a positive integer.")

        if not isinstance(local_search_max_sweeps, int) or local_search_max_sweeps <= 0:
            raise ValueError("local_search_max_sweeps must be a positive integer.")

        if initial_solution is not None:
            validate_phase_indices(initial_solution, N=N, Q=Q)

        rng = random.Random(seed)
        start_time = time.perf_counter()
        objective_evaluations = 0

        def time_exceeded() -> bool:
            return time_limit is not None and time.perf_counter() - start_time >= time_limit

        def evaluate(solution: list[int]) -> float:
            nonlocal objective_evaluations
            objective_evaluations += 1
            return islr_from_phase_indices(q=solution, Q=Q)

        def shake(level: int, solution: list[int]) -> list[int]:
            candidate = list(solution)

            if level == 0 or N == 1:
                indices = [rng.randrange(N)]
            elif level == 1:
                indices = rng.sample(range(N), 2) if N >= 2 else [0]
            else:
                size = min(block_size, N)
                start = rng.randrange(N)
                indices = [(start + step) % N for step in range(size)]

            for index in indices:
                offset = rng.randrange(1, Q)
                candidate[index] = (candidate[index] + offset) % Q

            return candidate

        def local_search(solution: list[int], solution_objective: float) -> tuple[list[int], float, bool]:
            current = list(solution)
            current_objective = solution_objective
            hit_time_limit = False
            sweeps = 0
            improved = True

            while improved and sweeps < local_search_max_sweeps:
                improved = False
                sweeps += 1

                for index in range(N):
                    if time_exceeded():
                        hit_time_limit = True
                        break

                    best_value = current[index]
                    best_objective_here = current_objective

                    for value in range(Q):
                        if value == current[index]:
                            continue
                        candidate = list(current)
                        candidate[index] = value
                        candidate_objective = evaluate(candidate)
                        if candidate_objective < best_objective_here:
                            best_objective_here = candidate_objective
                            best_value = value

                    if best_value != current[index]:
                        current[index] = best_value
                        current_objective = best_objective_here
                        improved = True

                if hit_time_limit:
                    break

            return current, current_objective, hit_time_limit

        if initial_solution is not None:
            current = list(initial_solution)
        else:
            current = [rng.randrange(Q) for _ in range(N)]

        current_objective = evaluate(current)

        best_phase_indices = list(current)
        best_objective = current_objective

        history: list[dict[str, float]] = []
        status = "completed"
        level = 0
        iteration = 0
        accepted_shakes = 0

        while iteration < max_iterations:
            if time_exceeded():
                status = "time_limit_reached"
                break

            shaken = shake(level, current)
            shaken_objective = evaluate(shaken)

            candidate, candidate_objective, hit_time_limit = local_search(
                shaken, shaken_objective
            )

            if candidate_objective < current_objective:
                current = candidate
                current_objective = candidate_objective
                level = 0
                accepted_shakes += 1
            else:
                level = (level + 1) % NUM_NEIGHBORHOOD_LEVELS

            if candidate_objective < best_objective:
                best_objective = candidate_objective
                best_phase_indices = list(candidate)

            iteration += 1

            history.append(
                {
                    "elapsed_seconds": time.perf_counter() - start_time,
                    "iteration": iteration,
                    "objective": best_objective,
                }
            )

            if hit_time_limit or time_exceeded():
                status = "time_limit_reached"
                break

        runtime_seconds = time.perf_counter() - start_time

        validate_phase_indices(best_phase_indices, N=N, Q=Q)

        return SolverResult(
            algorithm="variable_neighborhood_search",
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
                "max_iterations": max_iterations,
                "iterations_completed": iteration,
                "num_neighborhood_levels": NUM_NEIGHBORHOOD_LEVELS,
                "block_size": min(block_size, N),
                "local_search_max_sweeps": local_search_max_sweeps,
                "accepted_shakes": accepted_shakes,
                "final_neighborhood_level": level,
            },
        )