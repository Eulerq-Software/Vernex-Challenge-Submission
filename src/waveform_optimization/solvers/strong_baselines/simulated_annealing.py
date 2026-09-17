from __future__ import annotations

import math
import random
import time
from typing import Any

from waveform_optimization.benchmarking.schemas import SolverResult
from waveform_optimization.core.objective import islr_from_phase_indices
from waveform_optimization.core.validation import validate_phase_indices
from waveform_optimization.solvers.base import Solver


class SimulatedAnnealingSolver(Solver):
    def solve(
        self,
        N: int,
        Q: int,
        time_limit: float | None = None,
        seed: int | None = None,
        initial_solution: list[int] | None = None,
        initial_temperature: float = 10.0,
        final_temperature: float = 1e-3,
        cooling_rate: float = 0.95,
        iterations_per_temperature: int = 50,
        max_iterations: int | None = None,
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

        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive.")

        if final_temperature <= 0:
            raise ValueError("final_temperature must be positive.")

        if final_temperature >= initial_temperature:
            raise ValueError("final_temperature must be less than initial_temperature.")

        if not (0 < cooling_rate < 1):
            raise ValueError("cooling_rate must be in (0, 1).")

        if not isinstance(iterations_per_temperature, int) or iterations_per_temperature <= 0:
            raise ValueError("iterations_per_temperature must be a positive integer.")

        if max_iterations is not None:
            if not isinstance(max_iterations, int) or max_iterations <= 0:
                raise ValueError("max_iterations must be a positive integer or None.")

        rng = random.Random(seed)
        start_time = time.perf_counter()

        if initial_solution is not None:
            validate_phase_indices(initial_solution, N=N, Q=Q)
            current = list(initial_solution)
        else:
            current = [rng.randrange(Q) for _ in range(N)]

        current_objective = islr_from_phase_indices(q=current, Q=Q)
        objective_evaluations = 1

        best_phase_indices = list(current)
        best_objective = current_objective

        accepted_moves = 0
        improving_moves = 0
        history: list[dict[str, float]] = []

        temperature = initial_temperature
        total_iterations = 0
        status = "completed"

        def time_exceeded() -> bool:
            return time_limit is not None and time.perf_counter() - start_time >= time_limit

        while temperature > final_temperature:
            if time_exceeded():
                status = "time_limit_reached"
                break

            for _ in range(iterations_per_temperature):
                if time_exceeded():
                    status = "time_limit_reached"
                    break

                if max_iterations is not None and total_iterations >= max_iterations:
                    break

                index = rng.randrange(N)
                offset = rng.randrange(1, Q)
                candidate = list(current)
                candidate[index] = (candidate[index] + offset) % Q

                candidate_objective = islr_from_phase_indices(q=candidate, Q=Q)
                objective_evaluations += 1
                total_iterations += 1

                delta = candidate_objective - current_objective

                if delta <= 0:
                    accept = True
                else:
                    accept = rng.random() < math.exp(-delta / temperature)

                if accept:
                    current = candidate
                    current_objective = candidate_objective
                    accepted_moves += 1
                    if delta < 0:
                        improving_moves += 1

                if current_objective < best_objective:
                    best_phase_indices = list(current)
                    best_objective = current_objective

            history.append(
                {
                    "elapsed_seconds": time.perf_counter() - start_time,
                    "temperature": temperature,
                    "objective": best_objective,
                }
            )

            if status == "time_limit_reached":
                break

            if max_iterations is not None and total_iterations >= max_iterations:
                break

            temperature *= cooling_rate

        runtime_seconds = time.perf_counter() - start_time

        validate_phase_indices(best_phase_indices, N=N, Q=Q)

        return SolverResult(
            algorithm="simulated_annealing",
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
                "initial_temperature": initial_temperature,
                "final_temperature_configured": final_temperature,
                "final_temperature_reached": temperature,
                "cooling_rate": cooling_rate,
                "iterations_per_temperature": iterations_per_temperature,
                "total_iterations": total_iterations,
                "accepted_moves": accepted_moves,
                "improving_moves": improving_moves,
            },
        )