from __future__ import annotations

import random
import time
from typing import Any

from waveform_optimization.benchmarking.schemas import SolverResult
from waveform_optimization.core.objective import islr_from_phase_indices
from waveform_optimization.core.validation import validate_phase_indices
from waveform_optimization.solvers.base import Solver


class CoordinateDescentSolver(Solver):
    """
    Improve a waveform one phase index at a time.

    Uses best improvement: for each coordinate, every valid phase is evaluated
    while all other coordinates remain fixed.
    """

    def solve(
        self,
        N: int,
        Q: int,
        time_limit: float | None = None,
        seed: int | None = None,
        initial_solution: list[int] | None = None,
        **kwargs: Any,
    ) -> SolverResult:
        if N < 2:
            raise ValueError("N must be at least 2.")
        if Q < 2:
            raise ValueError("Q must be at least 2.")
        if time_limit is not None and time_limit < 0:
            raise ValueError("time_limit must be nonnegative.")

        max_iterations = kwargs.get("max_iterations")
        max_evaluations = kwargs.get("max_evaluations")
        record_history = kwargs.get("record_history", False)

        if max_iterations is not None and max_iterations < 0:
            raise ValueError("max_iterations must be nonnegative.")
        if max_evaluations is not None and max_evaluations < 1:
            raise ValueError("max_evaluations must be at least 1.")

        rng = random.Random(seed)
        start_time = time.perf_counter()
        objective_evaluations = 0
        iterations = 0
        history: list[dict[str, float]] = []
        status = "completed"

        if initial_solution is None:
            current_solution = [rng.randrange(Q) for _ in range(N)]
            initialization = "random"
        else:
            validate_phase_indices(initial_solution, N=N, Q=Q)
            current_solution = list(initial_solution)
            initialization = "provided"

        current_objective = islr_from_phase_indices(current_solution, Q)
        objective_evaluations += 1

        best_solution = current_solution.copy()
        best_objective = current_objective

        while True:
            if (
                time_limit is not None
                and time.perf_counter() - start_time >= time_limit
            ):
                status = "time_limit_reached"
                break

            if max_iterations is not None and iterations >= max_iterations:
                break

            if (
                max_evaluations is not None
                and objective_evaluations >= max_evaluations
            ):
                break

            improved_in_pass = False

            for position in range(N):
                best_phase = current_solution[position]
                best_coordinate_objective = current_objective

                for phase in range(Q):
                    if (
                        time_limit is not None
                        and time.perf_counter() - start_time >= time_limit
                    ):
                        status = "time_limit_reached"
                        break

                    if (
                        max_evaluations is not None
                        and objective_evaluations >= max_evaluations
                    ):
                        break

                    candidate = current_solution.copy()
                    candidate[position] = phase
                    candidate_objective = islr_from_phase_indices(candidate, Q)
                    objective_evaluations += 1

                    if candidate_objective < best_objective:
                        best_solution = candidate.copy()
                        best_objective = candidate_objective

                    if candidate_objective < best_coordinate_objective:
                        best_phase = phase
                        best_coordinate_objective = candidate_objective

                if status == "time_limit_reached":
                    break

                if max_evaluations is not None and (
                    objective_evaluations >= max_evaluations
                ):
                    break

                if best_coordinate_objective < current_objective:
                    current_solution[position] = best_phase
                    current_objective = best_coordinate_objective
                    improved_in_pass = True

            if status == "time_limit_reached":
                break

            iterations += 1

            if record_history:
                history.append(
                    {
                        "elapsed_seconds": time.perf_counter() - start_time,
                        "objective": best_objective,
                    }
                )

            if not improved_in_pass:
                break

            if max_evaluations is not None and (
                objective_evaluations >= max_evaluations
            ):
                break

        validate_phase_indices(best_solution, N=N, Q=Q)

        return SolverResult(
            algorithm="coordinate_descent",
            N=N,
            Q=Q,
            best_phase_indices=best_solution,
            best_objective=best_objective,
            runtime_seconds=time.perf_counter() - start_time,
            objective_evaluations=objective_evaluations,
            status=status,
            seed=seed,
            time_limit_seconds=time_limit,
            history=history,
            metadata={
                "initialization": initialization,
                "improvement_strategy": "best_improvement",
                "iterations": iterations,
                "max_iterations": max_iterations,
                "max_evaluations": max_evaluations,
            },
        )