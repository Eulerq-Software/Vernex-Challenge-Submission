from __future__ import annotations

import math
import time

import numpy as np

from waveform_optimization.core.binary_objective import (
    BinaryObjectiveState,
)

from waveform_optimization.solvers.alns.local_search import (
    steepest_descent,
)


# ============================================================
# Common utilities
# ============================================================

def validate_allowed_indices(
    state: BinaryObjectiveState,
    allowed_indices: np.ndarray | list[int],
) -> np.ndarray:
    """
    Validate and normalize the selected-bit neighborhood.

    Returns
    -------
    np.ndarray:
        Sorted, unique, int64 indices.

    Notes
    -----
    An empty neighborhood is permitted. Repair operators
    return an unchanged copy in that case.
    """

    raw = np.asarray(allowed_indices)

    if raw.ndim != 1:
        raise ValueError(
            "allowed_indices must be one-dimensional."
        )

    if raw.size == 0:
        return np.empty(0, dtype=np.int64)

    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(
            "allowed_indices must contain integers."
        )

    indices = raw.astype(
        np.int64,
        copy=True,
    )

    if np.any(indices < 0):
        raise ValueError(
            "allowed_indices contains negative positions."
        )

    if np.any(indices >= state.N):
        raise ValueError(
            "allowed_indices contains positions outside "
            "the sequence."
        )

    return np.unique(indices)


def deadline_reached(
    deadline: float | None,
) -> bool:
    """
    Check an absolute perf_counter deadline.
    """

    return (
        deadline is not None
        and time.perf_counter() >= deadline
    )


# ============================================================
# R1: Greedy repair
# ============================================================

def greedy_repair(
    state: BinaryObjectiveState,
    allowed_indices: np.ndarray | list[int],
    rng: np.random.Generator,
    *,
    deadline: float | None = None,
    max_iterations: int | None = None,
) -> BinaryObjectiveState:
    """
    Restricted steepest-descent repair.

    Repeatedly select the allowed single-bit flip
    with the greatest objective improvement.

    Stops when:
        - No improving allowed flip exists.
        - The iteration limit is reached.
        - The deadline is reached.

    Returns an independent improved state.
    """

    indices = validate_allowed_indices(
        state,
        allowed_indices,
    )

    return steepest_descent(
        state,
        allowed_indices=indices,
        max_iterations=max_iterations,
        deadline=deadline,
        rng=rng,
        inplace=False,
    )


# ============================================================
# R2: Randomized greedy repair
# ============================================================

def randomized_greedy_repair(
    state: BinaryObjectiveState,
    allowed_indices: np.ndarray | list[int],
    rng: np.random.Generator,
    *,
    deadline: float | None = None,
    max_iterations: int | None = None,
    top_k: int = 3,
    reference_state: BinaryObjectiveState | None = None,
) -> BinaryObjectiveState:
    """
    Randomized restricted greedy repair.

    At each iteration:

        1. Evaluate all allowed single-bit flips.
        2. Retain only improving flips.
        3. Identify the top_k best improving flips.
        4. Select one uniformly at random.
        5. Apply the selected flip.

    This introduces diversification while ensuring
    that every accepted flip improves the objective.

    Parameters
    ----------
    top_k:
        Number of promising improving moves from
        which to choose randomly.

        top_k=1 is equivalent to steepest descent,
        apart from tie-breaking details.
    """

    indices = validate_allowed_indices(
        state,
        allowed_indices,
    )

    if not isinstance(top_k, (int, np.integer)):
        raise TypeError(
            "top_k must be an integer."
        )

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    if max_iterations is not None:
        if max_iterations < 0:
            raise ValueError(
                "max_iterations must be non-negative."
            )

    current = state.copy()

    if indices.size == 0:
        return current

    iterations = 0

    while True:

        if deadline_reached(deadline):
            break

        if (
            max_iterations is not None
            and iterations >= max_iterations
        ):
            break

        # ----------------------------------------------
        # Evaluate all permitted single-bit flips
        # ----------------------------------------------

        deltas = np.array(
            [
                current.flip_delta(int(i))
                for i in indices
            ],
            dtype=np.int64,
        )

        # ----------------------------------------------
        # Keep only improving moves
        # ----------------------------------------------

        improving = np.flatnonzero(
            deltas < 0
        )

        if reference_state is not None and improving.size:
            retained = []
            for candidate_position in improving:
                trial = current.copy()
                trial.apply_flip(int(indices[candidate_position]))
                if not np.array_equal(trial.sequence, reference_state.sequence):
                    retained.append(candidate_position)
            improving = np.asarray(retained, dtype=np.int64)

        if improving.size == 0:
            break

        # ----------------------------------------------
        # Randomize ties, then sort by objective delta
        # ----------------------------------------------

        shuffled = rng.permutation(
            improving
        )

        ranking = shuffled[
            np.argsort(
                deltas[shuffled],
                kind="stable",
            )
        ]

        # ----------------------------------------------
        # Randomly select among top-k moves
        # ----------------------------------------------

        candidates = ranking[
            :min(top_k, ranking.size)
        ]

        chosen_position = int(
            rng.choice(candidates)
        )

        chosen_index = int(
            indices[chosen_position]
        )

        # ----------------------------------------------
        # Apply improving flip
        # ----------------------------------------------

        current.apply_flip(
            chosen_index
        )

        iterations += 1

    return current


# ============================================================
# R3: Exact repair
# ============================================================

def exact_repair(
    state: BinaryObjectiveState,
    allowed_indices: np.ndarray | list[int],
    rng: np.random.Generator,
    *,
    deadline: float | None = None,
    max_neighborhood_size: int = 16,
) -> BinaryObjectiveState:
    """
    Exhaustively optimize the selected-bit neighborhood.

    For m selected bits, there are 2^m possible
    assignments.

    All positions outside allowed_indices remain fixed.

    Uses Gray-code enumeration so that consecutive
    assignments differ in exactly one bit.

    Consequently, the objective is updated using
    BinaryObjectiveState.apply_flip() rather than
    recomputed from scratch for every assignment.

    Parameters
    ----------
    max_neighborhood_size:
        Maximum permitted number of selected bits.

        Default: 16

        This limits the worst-case enumeration to
        65,536 assignments.

    Notes
    -----
    If the deadline is reached, the best assignment
    examined so far is returned.

    Therefore, exact optimality within the neighborhood
    is guaranteed only when enumeration completes.
    """

    # rng is retained for compatibility with the
    # common repair-operator interface.
    _ = rng

    indices = validate_allowed_indices(
        state,
        allowed_indices,
    )

    if not isinstance(
        max_neighborhood_size,
        (int, np.integer),
    ):
        raise TypeError(
            "max_neighborhood_size must be an integer."
        )

    if max_neighborhood_size < 1:
        raise ValueError(
            "max_neighborhood_size must be at least 1."
        )

    m = len(indices)

    if m > max_neighborhood_size:
        raise ValueError(
            f"Exact repair received {m} selected bits, "
            f"but max_neighborhood_size="
            f"{max_neighborhood_size}."
        )

    # --------------------------------------------------
    # Initial assignment is already one valid member
    # of the neighborhood.
    # --------------------------------------------------

    current = state.copy()

    best = current.copy()

    if m == 0:
        return best

    total_assignments = 1 << m

    previous_gray = 0

    # --------------------------------------------------
    # Gray-code enumeration
    # --------------------------------------------------

    for counter in range(
        1,
        total_assignments,
    ):

        if deadline_reached(deadline):
            break

        gray = (
            counter ^ (counter >> 1)
        )

        # Exactly one bit changes between consecutive
        # Gray-code values.

        changed = (
            gray ^ previous_gray
        )

        # Identify which of the m selected positions
        # must be flipped.

        local_position = (
            changed.bit_length() - 1
        )

        sequence_position = int(
            indices[local_position]
        )

        current.apply_flip(
            sequence_position
        )

        # ----------------------------------------------
        # Track best assignment
        # ----------------------------------------------

        if current.energy < best.energy:

            best = current.copy()

        previous_gray = gray

    return best


# ============================================================
# R4: Simulated annealing repair
# ============================================================

def simulated_annealing_repair(
    state: BinaryObjectiveState,
    allowed_indices: np.ndarray | list[int],
    rng: np.random.Generator,
    *,
    deadline: float | None = None,
    max_iterations: int = 1000,
    initial_temperature: float | None = None,
    final_temperature: float | None = None,
    cooling_rate: float = 0.995,
    return_best: bool = True,
) -> BinaryObjectiveState:
    """
    Simulated annealing over the selected-bit neighborhood.

    Each proposed move flips one randomly selected allowed bit.

    Improving and equal-energy moves are always accepted.

    Worsening moves with delta > 0 are accepted with probability:

        exp(-delta / temperature)

    Parameters
    ----------
    state:
        Original solution. Never modified.

    allowed_indices:
        Positions permitted to change. All other positions
        remain fixed.

    rng:
        NumPy random generator.

    deadline:
        Optional absolute deadline obtained using
        time.perf_counter().

    max_iterations:
        Maximum number of proposed single-bit moves.

    initial_temperature:
        Initial annealing temperature. If None, estimate
        it from the positive single-bit objective deltas.

    final_temperature:
        Minimum annealing temperature. If None, use
        a small fraction of the initial temperature.

    cooling_rate:
        Multiplicative cooling factor.

    return_best:
        True:
            Return the best solution encountered,
            including the original solution.

        False:
            Return the final state of the annealing
            trajectory, even if it is worse than
            the original solution.

            This mode allows the outer ALNS acceptance
            mechanism to evaluate worsening candidates.

    Returns
    -------
    BinaryObjectiveState:
        An independent repaired state.

    Notes
    -----
    The original state is never modified.

    When return_best=False, the returned state is
    not guaranteed to improve the objective.

    When return_best=True, the returned state is
    guaranteed to be no worse than the original.
    """

    # --------------------------------------------------
    # Validate selected positions
    # --------------------------------------------------

    indices = validate_allowed_indices(
        state,
        allowed_indices,
    )

    # --------------------------------------------------
    # Validate parameters
    # --------------------------------------------------

    if not isinstance(
        max_iterations,
        (int, np.integer),
    ):
        raise TypeError(
            "max_iterations must be an integer."
        )

    if max_iterations < 0:
        raise ValueError(
            "max_iterations must be non-negative."
        )

    if not 0 < cooling_rate < 1:
        raise ValueError(
            "cooling_rate must satisfy "
            "0 < cooling_rate < 1."
        )

    if not isinstance(
        return_best,
        (bool, np.bool_),
    ):
        raise TypeError(
            "return_best must be a boolean."
        )

    # --------------------------------------------------
    # Copy original state
    # --------------------------------------------------

    current = state.copy()

    best = current.copy()

    # --------------------------------------------------
    # Early termination
    # --------------------------------------------------

    if indices.size == 0:
        return best if return_best else current

    if max_iterations == 0:
        return best if return_best else current

    if deadline_reached(deadline):
        return best if return_best else current

    # --------------------------------------------------
    # Estimate initial temperature
    # --------------------------------------------------

    if initial_temperature is None:

        initial_deltas = np.array(
            [
                current.flip_delta(int(i))
                for i in indices
            ],
            dtype=np.int64,
        )

        positive_deltas = initial_deltas[
            initial_deltas > 0
        ]

        if positive_deltas.size > 0:

            characteristic_delta = float(
                np.median(
                    positive_deltas
                )
            )

            # A worsening move with this delta has
            # approximately 80% acceptance probability
            # at the initial temperature.

            initial_temperature = (
                -characteristic_delta
                / math.log(0.8)
            )

        else:

            initial_temperature = 1.0

    if not math.isfinite(initial_temperature):
        raise ValueError(
            "initial_temperature must be finite."
        )

    if initial_temperature <= 0:
        raise ValueError(
            "initial_temperature must be positive."
        )

    # --------------------------------------------------
    # Determine final temperature
    # --------------------------------------------------

    if final_temperature is None:

        final_temperature = max(
            initial_temperature * 1e-4,
            1e-12,
        )

    if not math.isfinite(final_temperature):
        raise ValueError(
            "final_temperature must be finite."
        )

    if final_temperature <= 0:
        raise ValueError(
            "final_temperature must be positive."
        )

    if final_temperature > initial_temperature:
        raise ValueError(
            "final_temperature cannot exceed "
            "initial_temperature."
        )

    temperature = float(
        initial_temperature
    )

    # --------------------------------------------------
    # Main simulated-annealing loop
    # --------------------------------------------------

    for _ in range(
        max_iterations
    ):

        # ----------------------------------------------
        # Check deadline
        # ----------------------------------------------

        if deadline_reached(deadline):
            break

        # ----------------------------------------------
        # Select one allowed position
        # ----------------------------------------------

        index = int(
            rng.choice(indices)
        )

        # ----------------------------------------------
        # Calculate objective change
        # ----------------------------------------------

        delta = int(
            current.flip_delta(index)
        )

        # ----------------------------------------------
        # Metropolis acceptance
        # ----------------------------------------------

        if delta <= 0:

            accept = True

        else:

            probability = math.exp(
                -delta / temperature
            )

            accept = bool(
                rng.random() < probability
            )

        # ----------------------------------------------
        # Apply accepted move
        # ----------------------------------------------

        if accept:

            current.apply_flip(
                index
            )

            # Track best state independently from
            # the current annealing trajectory.

            if current.energy < best.energy:

                best = current.copy()

        # ----------------------------------------------
        # Update temperature
        # ----------------------------------------------

        temperature = max(
            final_temperature,
            temperature * cooling_rate,
        )

    # --------------------------------------------------
    # Select returned state
    # --------------------------------------------------

    if return_best:

        return best

    return current


# ============================================================
# Repair operator registry
# ============================================================

REPAIR_OPERATORS = {
    "greedy": greedy_repair,
    "randomized_greedy": randomized_greedy_repair,
    "exact": exact_repair,
    "simulated_annealing": simulated_annealing_repair,
}


# ============================================================
# Common repair interface
# ============================================================

def apply_repair(
    state: BinaryObjectiveState,
    *,
    method: str,
    allowed_indices: np.ndarray | list[int],
    rng: np.random.Generator,
    deadline: float | None = None,
    **kwargs,
) -> BinaryObjectiveState:
    """
    Apply a repair operator by name.

    Parameters
    ----------
    state:
        Original solution.

    method:
        One of:

            greedy
            randomized_greedy
            exact
            simulated_annealing

    allowed_indices:
        Positions permitted to change.

    rng:
        NumPy random generator.

    deadline:
        Optional absolute perf_counter deadline.

    **kwargs:
        Operator-specific parameters.

    Returns
    -------
    BinaryObjectiveState:
        Best repaired solution found.

    Notes
    -----
    The original state is never modified.
    """

    if method not in REPAIR_OPERATORS:

        raise ValueError(
            f"Unknown repair method: {method}. "
            f"Available methods: "
            f"{list(REPAIR_OPERATORS)}"
        )

    operator = REPAIR_OPERATORS[
        method
    ]

    return operator(
        state,
        allowed_indices,
        rng,
        deadline=deadline,
        **kwargs,
    )