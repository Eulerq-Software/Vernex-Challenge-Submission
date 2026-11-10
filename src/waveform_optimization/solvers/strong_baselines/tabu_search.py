from __future__ import annotations

import random
import time
from typing import Any

from waveform_optimization.benchmarking.schemas import SolverResult
from waveform_optimization.core.objective import islr_from_phase_indices
from waveform_optimization.core.validation import validate_phase_indices
from waveform_optimization.solvers.base import Solver


class TabuSearchSolver(Solver):
    def solve(
        self,
        N: int,
        Q: int,
        time_limit: float | None = None,
        seed: int | None = None,
        initial_solution: list[int] | None = None,
        tabu_tenure: int = 10,
        neighborhood_size: int | None = None,
        max_iterations: int = 200,
        no_improvement_limit: int | None = None,
        use_aspiration: bool = True,
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

        if not isinstance(tabu_tenure, int) or tabu_tenure <= 0:
            raise ValueError("tabu_tenure must be a positive integer.")

        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")

        max_neighborhood_size = N * (Q - 1)

        if neighborhood_size is not None:
            if (
                not isinstance(neighborhood_size, int)
                or neighborhood_size <= 0
                or neighborhood_size > max_neighborhood_size
            ):
                raise ValueError(
                    f"neighborhood_size must be an integer in [1, {max_neighborhood_size}]."
                )

        if no_improvement_limit is not None:
            if not isinstance(no_improvement_limit, int) or no_improvement_limit <= 0:
                raise ValueError("no_improvement_limit must be a positive integer or None.")

        if initial_solution is not None:
            validate_phase_indices(initial_solution, N=N, Q=Q)

        rng = random.Random(seed)
        start_time = time.perf_counter()

        def time_exceeded() -> bool:
            return time_limit is not None and time.perf_counter() - start_time >= time_limit

        if initial_solution is not None:
            current = list(initial_solution)
        else:
            current = [rng.randrange(Q) for _ in range(N)]

        current_objective = islr_from_phase_indices(q=current, Q=Q)
        objective_evaluations = 1

        best_phase_indices = list(current)
        best_objective = current_objective

        tabu_list: dict[tuple[int, int], int] = {}
        history: list[dict[str, float]] = []
        status = "completed"
        iterations_since_improvement = 0
        iteration = 0

        def all_moves() -> list[tuple[int, int]]:
            return [(i, v) for i in range(N) for v in range(Q) if v != current[i]]

        while iteration < max_iterations:
            if time_exceeded():
                status = "time_limit_reached"
                break

            moves = all_moves()
            if neighborhood_size is not None and neighborhood_size < len(moves):
                moves = rng.sample(moves, neighborhood_size)

            candidates = []
            for index, value in moves:
                candidate = list(current)
                old_value = candidate[index]
                candidate[index] = value
                candidate_objective = islr_from_phase_indices(q=candidate, Q=Q)
                objective_evaluations += 1

                is_tabu = tabu_list.get((index, value), -1) > iteration
                candidates.append(
                    (candidate_objective, index, old_value, value, candidate, is_tabu)
                )

            admissible = [
                c for c in candidates if not c[5] or (use_aspiration and c[0] < best_objective)
            ]
            if not admissible:
                admissible = candidates

            chosen_objective, index, old_value, new_value, chosen, _ = min(
                admissible, key=lambda c: c[0]
            )

            current = chosen
            current_objective = chosen_objective
            tabu_list[(index, old_value)] = iteration + tabu_tenure

            if current_objective < best_objective:
                best_objective = current_objective
                best_phase_indices = list(current)
                iterations_since_improvement = 0
            else:
                iterations_since_improvement += 1

            if (
                no_improvement_limit is not None
                and iterations_since_improvement >= no_improvement_limit
            ):
                current = [rng.randrange(Q) for _ in range(N)]
                current_objective = islr_from_phase_indices(q=current, Q=Q)
                objective_evaluations += 1
                iterations_since_improvement = 0
                tabu_list.clear()

            iteration += 1

            history.append(
                {
                    "elapsed_seconds": time.perf_counter() - start_time,
                    "iteration": iteration,
                    "objective": best_objective,
                }
            )

            if time_exceeded():
                status = "time_limit_reached"
                break

        runtime_seconds = time.perf_counter() - start_time

        validate_phase_indices(best_phase_indices, N=N, Q=Q)

        return SolverResult(
            algorithm="tabu_search",
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
                "tabu_tenure": tabu_tenure,
                "neighborhood_size": neighborhood_size or max_neighborhood_size,
                "max_iterations": max_iterations,
                "iterations_completed": iteration,
                "no_improvement_limit": no_improvement_limit,
                "use_aspiration": use_aspiration,
            },
        )