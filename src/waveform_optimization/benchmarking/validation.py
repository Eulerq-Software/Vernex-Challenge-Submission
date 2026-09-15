from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

from waveform_optimization.core.validation import validate_phase_indices


REQUIRED_RESULT_FIELDS = {
    "algorithm",
    "N",
    "Q",
    "best_phase_indices",
    "best_objective",
    "runtime_seconds",
    "objective_evaluations",
    "status",
}

OPTIONAL_RESULT_FIELDS = {
    "seed",
    "time_limit_seconds",
    "history",
    "metadata",
}


def validate_solver_result(result: Mapping[str, Any]) -> None:
    """
    Validate the structure and values of a solver result.

    Raises:
        TypeError: If the result has an invalid type.
        ValueError: If a required field is missing or invalid.
    """
    if not isinstance(result, Mapping):
        raise TypeError("Solver result must be a mapping.")

    missing_fields = REQUIRED_RESULT_FIELDS - result.keys()

    if missing_fields:
        raise ValueError(
            f"Solver result is missing required fields: "
            f"{sorted(missing_fields)}"
        )

    unexpected_fields = set(result.keys()) - (
        REQUIRED_RESULT_FIELDS | OPTIONAL_RESULT_FIELDS
    )

    if unexpected_fields:
        raise ValueError(
            f"Solver result contains unexpected fields: "
            f"{sorted(unexpected_fields)}"
        )

    algorithm = result["algorithm"]
    N = result["N"]
    Q = result["Q"]
    best_phase_indices = result["best_phase_indices"]
    best_objective = result["best_objective"]
    runtime_seconds = result["runtime_seconds"]
    objective_evaluations = result["objective_evaluations"]
    status = result["status"]

    if not isinstance(algorithm, str) or not algorithm.strip():
        raise ValueError("'algorithm' must be a non-empty string.")

    if not isinstance(N, int) or isinstance(N, bool) or N <= 0:
        raise ValueError("'N' must be a positive integer.")

    if not isinstance(Q, int) or isinstance(Q, bool) or Q <= 0:
        raise ValueError("'Q' must be a positive integer.")

    if not isinstance(best_phase_indices, list):
        raise TypeError("'best_phase_indices' must be a list.")

    validate_phase_indices(
        q=best_phase_indices,
        N=N,
        Q=Q,
    )

    if not isinstance(best_objective, Real) or isinstance(best_objective, bool):
        raise ValueError("'best_objective' must be a real number.")

    if best_objective < 0:
        raise ValueError("'best_objective' cannot be negative.")

    if (
        not isinstance(runtime_seconds, Real)
        or isinstance(runtime_seconds, bool)
        or runtime_seconds < 0
    ):
        raise ValueError("'runtime_seconds' must be a non-negative number.")

    if (
        not isinstance(objective_evaluations, int)
        or isinstance(objective_evaluations, bool)
        or objective_evaluations < 0
    ):
        raise ValueError(
            "'objective_evaluations' must be a non-negative integer."
        )

    if not isinstance(status, str) or not status.strip():
        raise ValueError("'status' must be a non-empty string.")

    if "seed" in result:
        seed = result["seed"]

        if seed is not None and (
            not isinstance(seed, int) or isinstance(seed, bool)
        ):
            raise ValueError("'seed' must be an integer or None.")

    if "time_limit_seconds" in result:
        time_limit_seconds = result["time_limit_seconds"]

        if time_limit_seconds is not None and (
            not isinstance(time_limit_seconds, Real)
            or isinstance(time_limit_seconds, bool)
            or time_limit_seconds <= 0
        ):
            raise ValueError(
                "'time_limit_seconds' must be a positive number or None."
            )

    if "history" in result:
        history = result["history"]

        if not isinstance(history, list):
            raise TypeError("'history' must be a list.")

        for entry in history:
            if not isinstance(entry, dict):
                raise TypeError("Each history entry must be a dictionary.")

    if "metadata" in result:
        metadata = result["metadata"]

        if not isinstance(metadata, dict):
            raise TypeError("'metadata' must be a dictionary.")