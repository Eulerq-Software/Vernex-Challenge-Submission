from __future__ import annotations

import time
from typing import Any

from waveform_optimization.benchmarking.schemas import SolverResult
from waveform_optimization.core.objective import islr_from_phase_indices
from waveform_optimization.core.validation import validate_phase_indices
from waveform_optimization.solvers.base import Solver


class GreedySolver(Solver):
    """
    Construct a waveform left to right.

    For each coordinate, try every phase value. Unassigned positions are
    temporarily set to zero and the shared ISLR objective is evaluated.
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

        start_time = time.perf_counter()
        objective_evaluations = 0
        status = "completed"

        # Zero-fill completion strategy.
        current_solution = [0] * N
        current_objective = islr_from_phase_indices(current_solution, Q)
        objective_evaluations += 1

        best_solution = current_solution.copy()
        best_objective = current_objective

        # Optional initial solution is treated as another valid candidate.
        if initial_solution is not None:
            validate_phase_indices(initial_solution, N=N, Q=Q)
            initial_objective = islr_from_phase_indices(initial_solution, Q)
            objective_evaluations += 1

            if initial_objective < best_objective:
                best_solution = list(initial_solution)
                best_objective = initial_objective

        for position in range(N):
            best_phase = current_solution[position]
            best_candidate_objective = float("inf")

            for phase in range(Q):
                if (
                    time_limit is not None
                    and time.perf_counter() - start_time >= time_limit
                ):
                    status = "time_limit_reached"
                    break

                candidate = current_solution.copy()
                candidate[position] = phase

                candidate_objective = islr_from_phase_indices(candidate, Q)
                objective_evaluations += 1

                # Strict comparison makes ties deterministic:
                # the first (lowest) phase is retained.
                if candidate_objective < best_candidate_objective:
                    best_phase = phase
                    best_candidate_objective = candidate_objective

                if candidate_objective < best_objective:
                    best_solution = candidate.copy()
                    best_objective = candidate_objective

            if status == "time_limit_reached":
                break

            current_solution[position] = best_phase
            current_objective = best_candidate_objective

        validate_phase_indices(best_solution, N=N, Q=Q)

        return SolverResult(
            algorithm="greedy",
            N=N,
            Q=Q,
            best_phase_indices=best_solution,
            best_objective=best_objective,
            runtime_seconds=time.perf_counter() - start_time,
            objective_evaluations=objective_evaluations,
            status=status,
            seed=seed,
            time_limit_seconds=time_limit,
            metadata={
                "coordinate_order": "left_to_right",
                "completion_strategy": "zero_fill",
                "tie_breaking": "lowest_phase_index",
            },
        )