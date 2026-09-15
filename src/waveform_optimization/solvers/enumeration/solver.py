from __future__ import annotations

import itertools
import time
from typing import Any

from waveform_optimization.benchmarking.schemas import SolverResult
from waveform_optimization.core.objective import islr_from_phase_indices
from waveform_optimization.solvers.base import Solver


class EnumerationSolver(Solver):
    """Exhaustive enumeration solver for waveform optimization."""

    def solve(
        self,
        N: int,
        Q: int,
        time_limit: float | None = None,
        seed: int | None = None,
        initial_solution: list[int] | None = None,
        **kwargs: Any,
    ) -> SolverResult:
        """
        Search every possible phase-index sequence.

        Args:
            N: Waveform length.
            Q: Number of available phase values.
            time_limit: Optional runtime limit in seconds.
            seed: Unused by exhaustive enumeration.
            initial_solution: Unused by exhaustive enumeration.
            **kwargs: Additional solver options.

        Returns:
            The best solution found as a SolverResult.
        """
        if not isinstance(N, int) or isinstance(N, bool) or N <= 0:
            raise ValueError("N must be a positive integer.")

        if not isinstance(Q, int) or isinstance(Q, bool) or Q <= 0:
            raise ValueError("Q must be a positive integer.")

        if time_limit is not None:
            if (
                not isinstance(time_limit, (int, float))
                or isinstance(time_limit, bool)
                or time_limit <= 0
            ):
                raise ValueError("time_limit must be a positive number or None.")

        start_time = time.perf_counter()

        best_phase_indices: list[int] | None = None
        best_objective = float("inf")
        objective_evaluations = 0
        status = "completed"

        for phase_indices in itertools.product(range(Q), repeat=N):
            if (
                time_limit is not None
                and time.perf_counter() - start_time >= time_limit
            ):
                status = "time_limit_reached"
                break

            objective = islr_from_phase_indices(
                q=phase_indices,
                Q=Q,
            )

            objective_evaluations += 1

            if objective < best_objective:
                best_objective = float(objective)
                best_phase_indices = list(phase_indices)

        runtime_seconds = time.perf_counter() - start_time

        if best_phase_indices is None:
            raise RuntimeError("Enumeration did not evaluate any solution.")

        result = SolverResult(
            algorithm="enumeration",
            N=N,
            Q=Q,
            best_phase_indices=best_phase_indices,
            best_objective=best_objective,
            runtime_seconds=runtime_seconds,
            objective_evaluations=objective_evaluations,
            status=status,
            seed=seed,
            time_limit_seconds=time_limit,
        )

        return result