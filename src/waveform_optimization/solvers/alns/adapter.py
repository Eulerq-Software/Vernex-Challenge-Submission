from __future__ import annotations

from typing import Any

import numpy as np

from waveform_optimization.benchmarking.schemas import SolverResult
from waveform_optimization.core.binary_objective import BinaryObjectiveState
from waveform_optimization.core.objective import islr_from_phase_indices
from waveform_optimization.solvers.base import Solver
from waveform_optimization.solvers.alns.solver import solve_alns


class ALNSWaveformSolver(Solver):
    """
    Standard benchmarking adapter for the binary ALNS solver.

    Supports Q=2 only.

    The underlying ALNS implementation remains available
    independently through solve_alns().
    """

    def __init__(self, **config_kwargs: Any) -> None:
        """
        Store default ALNS configuration.

        Example
        -------
        solver = ALNSWaveformSolver(
            destroy_fraction=0.10,
            exact_max_neighborhood_size=12,
        )
        """

        self.config_kwargs = dict(config_kwargs)

    def solve(
        self,
        N: int,
        Q: int,
        time_limit: float | None = None,
        seed: int | None = None,
        initial_solution: list[int] | None = None,
        **kwargs: Any,
    ) -> SolverResult:

        # --------------------------------------------------
        # Validate problem
        # --------------------------------------------------

        if not isinstance(N, int) or isinstance(N, bool):
            raise TypeError("N must be an integer.")

        if N < 2:
            raise ValueError("N must be at least 2.")

        if Q != 2:
            raise ValueError(
                "ALNSWaveformSolver currently supports Q=2 only."
            )

        # --------------------------------------------------
        # Prepare optional initial solution
        # --------------------------------------------------

        initial_state = None

        if initial_solution is not None:

            if len(initial_solution) != N:
                raise ValueError(
                    "initial_solution length must equal N."
                )

            initial_state = BinaryObjectiveState(
                initial_solution
            )

        # --------------------------------------------------
        # Combine default and per-call configuration
        # --------------------------------------------------

        config = {
            **self.config_kwargs,
            **kwargs,
        }

        # These arguments belong to the standard
        # solver interface, not the extra configuration.

        config.pop("time_limit", None)
        config.pop("seed", None)

        # --------------------------------------------------
        # Run ALNS
        # --------------------------------------------------

        alns_result = solve_alns(
            N=N,
            initial_state=initial_state,
            time_limit=time_limit,
            seed=seed,
            **config,
        )

        # --------------------------------------------------
        # Extract solution
        # --------------------------------------------------

        best_phase_indices = (
            alns_result.best_sequence
            .astype(int)
            .tolist()
        )

        # Use the challenge's reference objective
        # for the standard benchmark result.

        reference_objective = float(
            islr_from_phase_indices(
                best_phase_indices,
                Q=2,
            )
        )

        # Verify agreement with incremental objective.

        if not np.isclose(
            reference_objective,
            alns_result.best_energy,
            rtol=1e-10,
            atol=1e-8,
        ):
            raise RuntimeError(
                "ALNS incremental objective does not "
                "match the reference objective. "
                f"Incremental={alns_result.best_energy}, "
                f"Reference={reference_objective}"
            )

        # --------------------------------------------------
        # Convert history
        # --------------------------------------------------

        history = []

        for entry in alns_result.history:

            history.append(
                {
                    "iteration": float(
                        entry["iteration"]
                    ),
                    "elapsed_time": float(
                        entry["elapsed_time"]
                    ),
                    "current_energy": float(
                        entry["current_energy"]
                    ),
                    "best_energy": float(
                        entry["best_energy"]
                    ),
                }
            )

        # --------------------------------------------------
        # Determine termination status
        # --------------------------------------------------

        if (
            time_limit is not None
            and alns_result.elapsed_time >= time_limit
        ):

            status = "time_limit_reached"

        elif (
            config.get("max_iterations") is not None
            and alns_result.iterations
            >= config["max_iterations"]
        ):

            status = "iteration_limit_reached"

        else:

            status = "completed"

        # --------------------------------------------------
        # Build standard result
        # --------------------------------------------------

        return SolverResult(
            algorithm="alns",
            N=N,
            Q=2,
            best_phase_indices=best_phase_indices,
            best_objective=reference_objective,
            runtime_seconds=float(
                alns_result.elapsed_time
            ),

            # Objective evaluations are not yet
            # instrumented in the ALNS objective engine.
            #
            # Do not confuse ALNS iterations with
            # objective evaluations.

            objective_evaluations=0,

            status=status,
            seed=seed,
            time_limit_seconds=time_limit,
            history=history,
            metadata={
                "initial_energy": (
                    alns_result.initial_energy
                ),
                "iterations": (
                    alns_result.iterations
                ),
                "accepted_moves": (
                    alns_result.accepted_moves
                ),
                "rejected_moves": (
                    alns_result.rejected_moves
                ),
                "improving_moves": (
                    alns_result.improving_moves
                ),
                "new_best_moves": (
                    alns_result.new_best_moves
                ),
                "destroy_statistics": (
                    alns_result.destroy_statistics
                ),
                "repair_statistics": (
                    alns_result.repair_statistics
                ),
                "objective_evaluations_instrumented": False,
                "internal_best_energy": (
                    alns_result.best_energy
                ),
                "barker_reference_ratio": (
                    reference_objective
                    / ((N - 1) / 2)
                ),
            },
        )