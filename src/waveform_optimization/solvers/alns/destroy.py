from __future__ import annotations

from typing import Callable

import numpy as np

from waveform_optimization.core.binary_objective import (
    BinaryObjectiveState,
)


# ============================================================
# Type alias
# ============================================================

DestroyOperator = Callable[
    [BinaryObjectiveState, int, np.random.Generator],
    np.ndarray,
]


# ============================================================
# Common validation
# ============================================================

def validate_destroy_size(
    state: BinaryObjectiveState,
    destroy_size: int,
) -> None:
    """
    Validate the number of positions selected
    for destruction.
    """

    if not isinstance(
        destroy_size,
        (int, np.integer),
    ):
        raise TypeError(
            "destroy_size must be an integer."
        )

    if not 1 <= destroy_size <= state.N:

        raise ValueError(
            "destroy_size must be between "
            "1 and N."
        )


# ============================================================
# D1: Random destroy
# ============================================================

def random_destroy(
    state: BinaryObjectiveState,
    destroy_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Select destroy_size distinct positions
    uniformly at random.

    Complexity:
        O(N) for NumPy's general sampling
        implementation.
    """

    validate_destroy_size(
        state,
        destroy_size,
    )

    indices = rng.choice(
        state.N,
        size=destroy_size,
        replace=False,
    )

    return np.sort(
        indices.astype(np.int64)
    )


# ============================================================
# D2: Contiguous block destroy
# ============================================================

def block_destroy(
    state: BinaryObjectiveState,
    destroy_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Select a contiguous block of positions.

    The block may wrap around the end
    of the sequence.

    For example:

        N = 10
        start = 8
        destroy_size = 4

        selected = [8, 9, 0, 1]

    Returns sorted indices.
    """

    validate_destroy_size(
        state,
        destroy_size,
    )

    start = int(
        rng.integers(
            0,
            state.N,
        )
    )

    indices = (
        start
        + np.arange(
            destroy_size,
            dtype=np.int64,
        )
    ) % state.N

    return np.sort(indices)


# ============================================================
# D3: Best-flip destroy
# ============================================================

def best_flip_destroy(
    state: BinaryObjectiveState,
    destroy_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Select positions with the smallest
    single-bit objective changes.

    For each position i:

        delta_i = E(flip_i(x)) - E(x)

    Positions with the smallest delta_i
    are selected first.

    Random tie-breaking is used.

    Complexity:
        O(N^2) for evaluating all flip deltas.
    """

    validate_destroy_size(
        state,
        destroy_size,
    )

    # --------------------------------------------------
    # Evaluate all single-bit flips
    # --------------------------------------------------

    deltas = state.all_flip_deltas()

    # --------------------------------------------------
    # Random tie-breaking
    # --------------------------------------------------

    random_keys = rng.random(
        state.N
    )

    # Primary key: objective delta.
    # Secondary key: random number.

    ranking = np.lexsort(
        (
            random_keys,
            deltas,
        )
    )

    selected = ranking[
        :destroy_size
    ]

    return np.sort(
        selected.astype(np.int64)
    )


# ============================================================
# Correlation-guided position scores
# ============================================================

def correlation_position_scores(
    state: BinaryObjectiveState,
) -> np.ndarray:
    """
    Calculate correlation-guided scores
    for every position.

    Score:

        s_i = sum_k C_k * a_{i,k}

    where:

        a_{i,k} =
            x_i * (
                x_{i+k}, if valid
                +
                x_{i-k}, if valid
            )

    Large positive scores indicate
    promising positions for reducing
    correlation contributions.

    Complexity:
        O(N^2)
    """

    N = state.N

    x = state.waveform.astype(
        np.int64,
        copy=False,
    )

    C = state.correlations

    scores = np.zeros(
        N,
        dtype=np.int64,
    )

    # --------------------------------------------------
    # Iterate over positive lags
    # --------------------------------------------------

    for k in range(1, N):

        ck = int(
            C[k - 1]
        )

        # ----------------------------------------------
        # Pair contribution
        # ----------------------------------------------

        # Each pair (i, i+k) contributes
        # to the scores of both endpoints.

        pair_contribution = (
            ck
            * x[:N-k]
            * x[k:]
        )

        scores[:N-k] += (
            pair_contribution
        )

        scores[k:] += (
            pair_contribution
        )

    return scores


# ============================================================
# D4: Correlation-guided destroy
# ============================================================

def correlation_guided_destroy(
    state: BinaryObjectiveState,
    destroy_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Select positions with the largest
    correlation-guided scores.

    Random tie-breaking is used.

    Complexity:
        O(N^2)
    """

    validate_destroy_size(
        state,
        destroy_size,
    )

    scores = correlation_position_scores(
        state
    )

    random_keys = rng.random(
        state.N
    )

    # Negative scores produce
    # descending score order.

    ranking = np.lexsort(
        (
            random_keys,
            -scores,
        )
    )

    selected = ranking[
        :destroy_size
    ]

    return np.sort(
        selected.astype(np.int64)
    )


# ============================================================
# D5: Mixed destroy
# ============================================================

def mixed_destroy(
    state: BinaryObjectiveState,
    destroy_size: int,
    rng: np.random.Generator,
    *,
    guided_fraction: float = 0.5,
) -> np.ndarray:
    """
    Combine best-flip-guided selection
    with random selection.

    Example:

        destroy_size = 20
        guided_fraction = 0.5

        10 positions from best-flip ranking
        10 positions selected randomly

    Random positions are selected from
    the remaining unselected positions.

    Complexity:
        O(N^2)
    """

    validate_destroy_size(
        state,
        destroy_size,
    )

    if not 0 <= guided_fraction <= 1:

        raise ValueError(
            "guided_fraction must be "
            "between 0 and 1."
        )

    # --------------------------------------------------
    # Determine guided/random counts
    # --------------------------------------------------

    guided_count = int(
        round(
            guided_fraction
            * destroy_size
        )
    )

    random_count = (
        destroy_size
        - guided_count
    )

    # --------------------------------------------------
    # Guided selection
    # --------------------------------------------------

    if guided_count > 0:

        guided_indices = (
            best_flip_destroy(
                state,
                guided_count,
                rng,
            )
        )

    else:

        guided_indices = np.empty(
            0,
            dtype=np.int64,
        )

    # --------------------------------------------------
    # Random selection
    # --------------------------------------------------

    if random_count > 0:

        available = np.setdiff1d(
            np.arange(
                state.N,
                dtype=np.int64,
            ),
            guided_indices,
        )

        random_indices = rng.choice(
            available,
            size=random_count,
            replace=False,
        )

    else:

        random_indices = np.empty(
            0,
            dtype=np.int64,
        )

    # --------------------------------------------------
    # Combine selections
    # --------------------------------------------------

    selected = np.concatenate(
        (
            guided_indices,
            random_indices,
        )
    )

    return np.sort(
        selected.astype(np.int64)
    )


# ============================================================
# Destroy operator registry
# ============================================================

DESTROY_OPERATORS = {
    "random": random_destroy,
    "block": block_destroy,
    "best_flip": best_flip_destroy,
    "correlation_guided": correlation_guided_destroy,
    "mixed": mixed_destroy,
}


# ============================================================
# Common destroy interface
# ============================================================

def apply_destroy(
    state: BinaryObjectiveState,
    *,
    method: str,
    destroy_size: int,
    rng: np.random.Generator,
    **kwargs,
) -> np.ndarray:
    """
    Apply a destroy operator by name.

    Returns:
        Sorted array of distinct positions.

    Does not modify the supplied state.
    """

    if method not in DESTROY_OPERATORS:

        raise ValueError(
            f"Unknown destroy method: {method}. "
            f"Available methods: "
            f"{list(DESTROY_OPERATORS)}"
        )

    operator = DESTROY_OPERATORS[
        method
    ]

    indices = operator(
        state,
        destroy_size,
        rng,
        **kwargs,
    )

    return indices