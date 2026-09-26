from __future__ import annotations

import time

import numpy as np

from waveform_optimization.core.binary_objective import (
    BinaryObjectiveState,
)

from waveform_optimization.solvers.alns.local_search import (
    steepest_descent,
)


# ============================================================
# Random sequence generation
# ============================================================

def random_binary_state(
    N: int,
    rng: np.random.Generator,
) -> BinaryObjectiveState:
    """
    Generate a uniformly random binary sequence.

    Each phase index is independently sampled
    from {0, 1}.
    """

    if N < 2:
        raise ValueError(
            "N must be at least 2."
        )

    sequence = rng.integers(
        low=0,
        high=2,
        size=N,
        dtype=np.int8,
    )

    return BinaryObjectiveState(
        sequence
    )


# ============================================================
# I1: Random initialization
# ============================================================

def random_initialization(
    N: int,
    rng: np.random.Generator,
) -> BinaryObjectiveState:
    """
    Generate one random binary sequence.
    """

    return random_binary_state(
        N=N,
        rng=rng,
    )


# ============================================================
# I2: Random multistart
# ============================================================

def random_multistart_initialization(
    N: int,
    rng: np.random.Generator,
    *,
    deadline: float | None = None,
    max_starts: int | None = None,
) -> BinaryObjectiveState:
    """
    Generate multiple random binary sequences
    and retain the best.

    At least one sequence is generated.
    """

    if max_starts is not None:

        if max_starts < 1:
            raise ValueError(
                "max_starts must be at least 1."
            )

    # --------------------------------------------------
    # Generate first solution
    # --------------------------------------------------

    best = random_binary_state(
        N=N,
        rng=rng,
    )

    starts = 1

    # --------------------------------------------------
    # Multistart loop
    # --------------------------------------------------

    while True:

        if max_starts is not None:

            if starts >= max_starts:
                break

        if deadline is not None:

            if time.perf_counter() >= deadline:
                break

        candidate = random_binary_state(
            N=N,
            rng=rng,
        )

        starts += 1

        if candidate.energy < best.energy:

            best = candidate

    return best


# ============================================================
# I3: Steepest-descent initialization
# ============================================================

def steepest_descent_initialization(
    N: int,
    rng: np.random.Generator,
    *,
    deadline: float | None = None,
    max_iterations: int | None = None,
) -> BinaryObjectiveState:
    """
    Generate one random sequence and improve
    it using unrestricted steepest descent.
    """

    state = random_binary_state(
        N=N,
        rng=rng,
    )

    state = steepest_descent(
        state,
        max_iterations=max_iterations,
        deadline=deadline,
        rng=rng,
        inplace=True,
    )

    return state


# ============================================================
# Perturbation
# ============================================================

def perturb_state(
    state: BinaryObjectiveState,
    rng: np.random.Generator,
    *,
    perturbation_size: int,
) -> BinaryObjectiveState:
    """
    Perturb a solution by flipping a specified
    number of distinct randomly selected bits.

    The original state is not modified.

    Parameters
    ----------
    state:
        Original solution.

    rng:
        NumPy random generator.

    perturbation_size:
        Number of distinct positions to flip.

    Returns
    -------
    BinaryObjectiveState:
        Perturbed solution.
    """

    if not 1 <= perturbation_size <= state.N:

        raise ValueError(
            "perturbation_size must be "
            "between 1 and N."
        )

    # --------------------------------------------------
    # Copy original state
    # --------------------------------------------------

    candidate = state.copy()

    # --------------------------------------------------
    # Select distinct positions
    # --------------------------------------------------

    indices = rng.choice(
        state.N,
        size=perturbation_size,
        replace=False,
    )

    # --------------------------------------------------
    # Apply perturbation
    # --------------------------------------------------

    for i in indices:

        candidate.apply_flip(
            int(i)
        )

    return candidate


# ============================================================
# I4: Iterated local search
# ============================================================

def iterated_local_search_initialization(
    N: int,
    rng: np.random.Generator,
    *,
    deadline: float | None = None,
    max_iterations: int | None = None,
    perturbation_size: int | None = None,
    local_search_iterations: int | None = None,
) -> BinaryObjectiveState:
    """
    Iterated local search initialization.

    Procedure
    ---------
    1. Generate a random sequence.
    2. Apply steepest descent.
    3. Perturb the current solution.
    4. Apply steepest descent again.
    5. Accept improving solutions.
    6. Repeat until termination.

    Returns the best solution found.

    This implementation uses improvement-only
    acceptance.
    """

    # --------------------------------------------------
    # Default perturbation size
    # --------------------------------------------------

    if perturbation_size is None:

        perturbation_size = max(
            1,
            int(round(0.05 * N)),
        )

    if not 1 <= perturbation_size <= N:

        raise ValueError(
            "perturbation_size must be "
            "between 1 and N."
        )

    # --------------------------------------------------
    # Validate iteration limit
    # --------------------------------------------------

    if max_iterations is not None:

        if max_iterations < 0:

            raise ValueError(
                "max_iterations must be non-negative."
            )

    # --------------------------------------------------
    # Initial random solution
    # --------------------------------------------------

    current = random_binary_state(
        N=N,
        rng=rng,
    )

    # --------------------------------------------------
    # Initial local search
    # --------------------------------------------------

    current = steepest_descent(
        current,
        max_iterations=local_search_iterations,
        deadline=deadline,
        rng=rng,
        inplace=True,
    )

    best = current.copy()

    iterations = 0

    # --------------------------------------------------
    # Main ILS loop
    # --------------------------------------------------

    while True:

        # ----------------------------------------------
        # Check iteration limit
        # ----------------------------------------------

        if max_iterations is not None:

            if iterations >= max_iterations:
                break

        # ----------------------------------------------
        # Check deadline
        # ----------------------------------------------

        if deadline is not None:

            if time.perf_counter() >= deadline:
                break

        # ----------------------------------------------
        # Perturb current solution
        # ----------------------------------------------

        candidate = perturb_state(
            current,
            rng,
            perturbation_size=perturbation_size,
        )

        # ----------------------------------------------
        # Improve candidate
        # ----------------------------------------------

        candidate = steepest_descent(
            candidate,
            max_iterations=local_search_iterations,
            deadline=deadline,
            rng=rng,
            inplace=True,
        )

        # ----------------------------------------------
        # Improvement-only acceptance
        # ----------------------------------------------

        if candidate.energy < current.energy:

            current = candidate

        # ----------------------------------------------
        # Update best solution
        # ----------------------------------------------

        if candidate.energy < best.energy:

            best = candidate.copy()

        iterations += 1

    return best


# ============================================================
# Common initialization interface
# ============================================================

def generate_initial_solution(
    N: int,
    *,
    method: str = "ils",
    time_limit: float | None = None,
    seed: int | None = None,
    max_starts: int | None = None,
    max_iterations: int | None = None,
    perturbation_size: int | None = None,
    local_search_iterations: int | None = None,
) -> BinaryObjectiveState:
    """
    Generate an initial solution using one
    of the supported initialization methods.

    Supported methods
    -----------------
    random
    multistart
    steepest
    ils

    Returns
    -------
    BinaryObjectiveState
    """

    # --------------------------------------------------
    # Validate N
    # --------------------------------------------------

    if N < 2:

        raise ValueError(
            "N must be at least 2."
        )

    # --------------------------------------------------
    # Validate time limit
    # --------------------------------------------------

    if time_limit is not None:

        if time_limit <= 0:

            raise ValueError(
                "time_limit must be positive."
            )

    # --------------------------------------------------
    # Random generator
    # --------------------------------------------------

    rng = np.random.default_rng(
        seed
    )

    # --------------------------------------------------
    # Deadline
    # --------------------------------------------------

    deadline = None

    if time_limit is not None:

        deadline = (
            time.perf_counter()
            + time_limit
        )

    # --------------------------------------------------
    # Select initialization method
    # --------------------------------------------------

    if method == "random":

        return random_initialization(
            N=N,
            rng=rng,
        )

    elif method == "multistart":

        if deadline is None and max_starts is None:

            raise ValueError(
                "Multistart requires time_limit "
                "or max_starts."
            )

        return random_multistart_initialization(
            N=N,
            rng=rng,
            deadline=deadline,
            max_starts=max_starts,
        )

    elif method == "steepest":

        return steepest_descent_initialization(
            N=N,
            rng=rng,
            deadline=deadline,
            max_iterations=max_iterations,
        )

    elif method == "ils":

        if deadline is None and max_iterations is None:

            raise ValueError(
                "ILS requires time_limit "
                "or max_iterations."
            )

        return iterated_local_search_initialization(
            N=N,
            rng=rng,
            deadline=deadline,
            max_iterations=max_iterations,
            perturbation_size=perturbation_size,
            local_search_iterations=local_search_iterations,
        )

    else:

        raise ValueError(
            f"Unknown initialization method: "
            f"{method}"
        )