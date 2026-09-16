from __future__ import annotations

import random
import time
from typing import Any

from waveform_optimization.benchmarking.schemas import SolverResult
from waveform_optimization.core.objective import islr_from_phase_indices
from waveform_optimization.core.validation import validate_phase_indices
from waveform_optimization.solvers.base import Solver


class RandomSearchSolver(Solver):
    """
    Randomly sample valid phase-index sequences and retain the best one found.
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
        if N < 1:
            raise ValueError("N must be at least 1.")
        if Q < 2:
            raise ValueError("Q must be at least 2.")
        if time_limit is not None and time_limit < 0:
            raise ValueError("time_limit must be nonnegative.")

        max_iterations = kwargs.get("max_iterations")
        max_evaluations = kwargs.get("max_evaluations")
        record_history = kwargs.get("record_history", False)

        if max_iterations is not None and max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        if max_evaluations is not None and max_evaluations < 1:
            raise ValueError("max_evaluations must be at least 1.")

        rng = random.Random(seed)
        start_time = time.perf_counter()
        objective_evaluations = 0
        iterations = 0
        history: list[dict[str, float]] = []
        status = "completed"

        # A valid incumbent ensures a valid result even with time_limit=0.
        best_solution = [0] * N
        best_objective = islr_from_phase_indices(best_solution, Q)
        objective_evaluations += 1

        if initial_solution is not None:
            validate_phase_indices(initial_solution, N=N, Q=Q)
            initial_objective = islr_from_phase_indices(initial_solution, Q)
            objective_evaluations += 1

            if initial_objective < best_objective:
                best_solution = list(initial_solution)
                best_objective = initial_objective

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

            candidate = [rng.randrange(Q) for _ in range(N)]
            candidate_objective = islr_from_phase_indices(candidate, Q)
            objective_evaluations += 1
            iterations += 1

            if candidate_objective < best_objective:
                best_solution = candidate
                best_objective = candidate_objective

            if record_history:
                history.append(
                    {
                        "elapsed_seconds": time.perf_counter() - start_time,
                        "objective": best_objective,
                    }
                )

        validate_phase_indices(best_solution, N=N, Q=Q)

        return SolverResult(
            algorithm="random_search",
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
                "iterations": iterations,
                "max_iterations": max_iterations,
                "max_evaluations": max_evaluations,
            },
        )