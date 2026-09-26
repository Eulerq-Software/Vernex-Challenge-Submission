from __future__ import annotations

import time

import numpy as np

from waveform_optimization.core.binary_objective import (
    BinaryObjectiveState,
)


def steepest_descent(
    state: BinaryObjectiveState,
    *,
    allowed_indices: np.ndarray | list[int] | None = None,
    max_iterations: int | None = None,
    deadline: float | None = None,
    rng: np.random.Generator | None = None,
    inplace: bool = False,
) -> BinaryObjectiveState:
    """
    Single-bit steepest-descent local search.

    At every iteration:

        1. Evaluate all permitted single-bit flips.
        2. Select the flip producing the largest improvement.
        3. Apply the flip.
        4. Repeat until no improving flip exists.

    Parameters
    ----------
    state:
        Initial BinaryObjectiveState.

    allowed_indices:
        Positions permitted to change.

        None:
            All positions are permitted.

        list or ndarray:
            Only the specified positions are permitted.

        This is particularly useful for ALNS repair,
        where positions outside the destroyed set
        must remain fixed.

    max_iterations:
        Maximum number of accepted improving flips.

        None means no iteration limit.

    deadline:
        Absolute deadline obtained using
        time.perf_counter().

        None means no time limit.

    rng:
        NumPy random generator for tie-breaking.

    inplace:
        If True, modify the supplied state.

        If False, operate on an independent copy.

    Returns
    -------
    BinaryObjectiveState:
        Improved solution.

    Notes
    -----
    If the search terminates due to an iteration
    or time limit, the returned solution is not
    necessarily a local optimum.
    """

    # --------------------------------------------------
    # Random generator
    # --------------------------------------------------

    if rng is None:
        rng = np.random.default_rng()

    # --------------------------------------------------
    # Validate iteration limit
    # --------------------------------------------------

    if max_iterations is not None:

        if max_iterations < 0:
            raise ValueError(
                "max_iterations must be non-negative."
            )

    # --------------------------------------------------
    # Copy state if required
    # --------------------------------------------------

    if not inplace:
        state = state.copy()

    # --------------------------------------------------
    # Prepare allowed indices
    # --------------------------------------------------

    if allowed_indices is None:

        indices = np.arange(
            state.N,
            dtype=np.int64,
        )

    else:

        raw_indices = np.asarray(
            allowed_indices
        )

        if raw_indices.ndim != 1:
            raise ValueError(
                "allowed_indices must be one-dimensional."
            )

        if not np.issubdtype(
            raw_indices.dtype,
            np.integer,
        ):

            if raw_indices.size != 0:
                raise ValueError(
                    "allowed_indices must contain integers."
                )

        indices = raw_indices.astype(
            np.int64,
            copy=True,
        )

        if np.any(indices < 0):

            raise ValueError(
                "allowed_indices contains "
                "negative positions."
            )

        if np.any(indices >= state.N):

            raise ValueError(
                "allowed_indices contains "
                "positions outside the sequence."
            )

        # Remove duplicate positions.

        indices = np.unique(indices)

    # --------------------------------------------------
    # Empty neighborhood
    # --------------------------------------------------

    if len(indices) == 0:
        return state

    # --------------------------------------------------
    # Main steepest-descent loop
    # --------------------------------------------------

    iterations = 0

    while True:

        # ----------------------------------------------
        # Check deadline
        # ----------------------------------------------

        if deadline is not None:

            if time.perf_counter() >= deadline:
                break

        # ----------------------------------------------
        # Check iteration limit
        # ----------------------------------------------

        if max_iterations is not None:

            if iterations >= max_iterations:
                break

        # ----------------------------------------------
        # Evaluate permitted flips
        # ----------------------------------------------

        deltas = np.empty(
            len(indices),
            dtype=np.int64,
        )

        for j, i in enumerate(indices):

            deltas[j] = state.flip_delta(
                int(i)
            )

        # ----------------------------------------------
        # Identify best improvement
        # ----------------------------------------------

        best_delta = int(
            np.min(deltas)
        )

        # ----------------------------------------------
        # Stop at local optimum
        # ----------------------------------------------

        if best_delta >= 0:
            break

        # ----------------------------------------------
        # Random tie-breaking
        # ----------------------------------------------

        best_positions = np.flatnonzero(
            deltas == best_delta
        )

        selected_position = int(
            rng.choice(best_positions)
        )

        selected_index = int(
            indices[selected_position]
        )

        # ----------------------------------------------
        # Apply improving flip
        # ----------------------------------------------

        state.apply_flip(
            selected_index
        )

        iterations += 1

    return state