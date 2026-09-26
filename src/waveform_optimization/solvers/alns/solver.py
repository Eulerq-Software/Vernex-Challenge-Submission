from __future__ import annotations

import math
import time

from dataclasses import dataclass, field

import numpy as np

from waveform_optimization.core.binary_objective import (
    BinaryObjectiveState,
)

from waveform_optimization.solvers.alns.initialization import (
    generate_initial_solution,
)

from waveform_optimization.solvers.alns.destroy import (
    DESTROY_OPERATORS,
    apply_destroy,
)

from waveform_optimization.solvers.alns.repair import (
    REPAIR_OPERATORS,
    apply_repair,
)

from waveform_optimization.solvers.alns.acceptance import (
    create_acceptance,
)

from waveform_optimization.solvers.alns.adaptive_selection import (
    AdaptiveOperatorSelector,
    RewardScheme,
    calculate_reward,
)


# ============================================================
# Solver configuration
# ============================================================

@dataclass
class ALNSConfig:
    """
    Configuration for binary radar waveform ALNS.

    At least one of time_limit or max_iterations
    must be specified.
    """

    # --------------------------------------------------
    # Termination
    # --------------------------------------------------

    time_limit: float | None = 30.0

    max_iterations: int | None = None

    seed: int | None = None

    # --------------------------------------------------
    # Initialization
    # --------------------------------------------------

    initialization_method: str = "ils"

    initialization_time_fraction: float = 0.10

    initialization_max_time: float = 2.0

    initialization_max_starts: int | None = None

    # --------------------------------------------------
    # Neighborhood size
    # --------------------------------------------------

    destroy_fraction: float = 0.10

    min_destroy_size: int = 2

    max_destroy_size: int | None = None

    # --------------------------------------------------
    # Exact repair
    # --------------------------------------------------

    exact_max_neighborhood_size: int = 16

    # --------------------------------------------------
    # Per-repair runtime budget
    # --------------------------------------------------

    max_repair_time: float | None = 0.25

    # --------------------------------------------------
    # Greedy repair
    # --------------------------------------------------

    greedy_max_iterations: int | None = None

    randomized_greedy_top_k: int = 3

    # --------------------------------------------------
    # Simulated annealing repair
    # --------------------------------------------------

    repair_sa_iterations: int = 200

    repair_sa_cooling_rate: float = 0.995

    repair_sa_return_best: bool = False

    # --------------------------------------------------
    # Outer acceptance
    # --------------------------------------------------

    acceptance_method: str = "simulated_annealing"

    acceptance_initial_temperature: float | None = None

    acceptance_final_temperature: float | None = None

    acceptance_cooling_rate: float = 0.995

    record_initial_threshold: float | None = None

    record_final_threshold: float = 0.0

    # --------------------------------------------------
    # Adaptive operator selection
    # --------------------------------------------------

    destroy_methods: tuple[str, ...] = (
        "random",
        "block",
        "best_flip",
        "correlation_guided",
        "mixed",
    )

    repair_methods: tuple[str, ...] = (
        "greedy",
        "randomized_greedy",
        "exact",
        "simulated_annealing",
    )

    reaction_factor: float = 0.2

    segment_length: int = 50

    # --------------------------------------------------
    # Operator rewards
    # --------------------------------------------------

    reward_new_best: float = 10.0

    reward_improvement: float = 5.0

    reward_accepted: float = 2.0

    reward_rejected: float = 0.0

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------

    history_interval: int = 10

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def validate(self) -> None:

        if (
            self.time_limit is None
            and self.max_iterations is None
        ):
            raise ValueError(
                "Specify time_limit or max_iterations."
            )

        if self.time_limit is not None:

            if (
                not math.isfinite(self.time_limit)
                or self.time_limit <= 0
            ):
                raise ValueError(
                    "time_limit must be positive and finite."
                )

        if self.max_iterations is not None:

            if self.max_iterations < 0:
                raise ValueError(
                    "max_iterations must be non-negative."
                )

        if not (
            0 < self.initialization_time_fraction <= 1
        ):
            raise ValueError(
                "initialization_time_fraction must "
                "be in (0, 1]."
            )

        if self.initialization_max_time <= 0:
            raise ValueError(
                "initialization_max_time must be positive."
            )

        if not (
            0 < self.destroy_fraction <= 1
        ):
            raise ValueError(
                "destroy_fraction must be in (0, 1]."
            )

        if self.min_destroy_size < 1:
            raise ValueError(
                "min_destroy_size must be at least 1."
            )

        if (
            self.max_destroy_size is not None
            and self.max_destroy_size
            < self.min_destroy_size
        ):
            raise ValueError(
                "max_destroy_size must be at least "
                "min_destroy_size."
            )

        if self.exact_max_neighborhood_size < 1:
            raise ValueError(
                "exact_max_neighborhood_size must "
                "be at least 1."
            )

        if self.repair_sa_iterations < 0:
            raise ValueError(
                "repair_sa_iterations must be non-negative."
            )

        if self.randomized_greedy_top_k < 1:
            raise ValueError(
                "randomized_greedy_top_k must "
                "be at least 1."
            )

        if self.history_interval < 1:
            raise ValueError(
                "history_interval must be at least 1."
            )

        if not self.destroy_methods:
            raise ValueError(
                "At least one destroy method is required."
            )

        if not self.repair_methods:
            raise ValueError(
                "At least one repair method is required."
            )

        unknown_destroy = (
            set(self.destroy_methods)
            - set(DESTROY_OPERATORS)
        )

        if unknown_destroy:
            raise ValueError(
                f"Unknown destroy methods: "
                f"{sorted(unknown_destroy)}"
            )

        unknown_repair = (
            set(self.repair_methods)
            - set(REPAIR_OPERATORS)
        )

        if unknown_repair:
            raise ValueError(
                f"Unknown repair methods: "
                f"{sorted(unknown_repair)}"
            )
        
        if self.max_repair_time is not None:

            if (
                not math.isfinite(self.max_repair_time)
                or self.max_repair_time <= 0
            ):

                raise ValueError(
                    "max_repair_time must be positive "
                    "and finite, or None."
                )


# ============================================================
# Solver result
# ============================================================

@dataclass
class ALNSResult:

    best_state: BinaryObjectiveState

    initial_energy: int

    iterations: int

    elapsed_time: float

    accepted_moves: int

    rejected_moves: int

    improving_moves: int

    new_best_moves: int

    destroy_statistics: dict

    repair_statistics: dict

    history: list[dict] = field(
        default_factory=list
    )

    @property
    def best_energy(self) -> int:

        return int(
            self.best_state.energy
        )

    @property
    def best_sequence(self) -> np.ndarray:
        """
        Return binary phase indices, not +/-1 waveform.
        """

        return self.best_state.sequence.copy()

    @property
    def best_waveform(self) -> np.ndarray:

        return self.best_state.waveform.copy()


# ============================================================
# Main ALNS solver
# ============================================================

class ALNSSolver:
    """
    Adaptive large-neighborhood search for binary
    radar waveform optimization.

    The solver maintains:

        current_state
        best_state

    Destroy operators select positions.

    Repair operators optimize or explore the selected
    neighborhood.

    The outer acceptance criterion decides whether
    the candidate becomes the next current state.

    Adaptive selectors update operator probabilities
    based on observed rewards.
    """

    def __init__(
        self,
        config: ALNSConfig | None = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else ALNSConfig()
        )

        self.config.validate()

    # ==================================================
    # Neighborhood size
    # ==================================================

    def _destroy_size(
        self,
        N: int,
        repair_method: str,
    ) -> int:
        """
        Calculate the neighborhood size.

        Exact repair is automatically restricted
        to its configured maximum neighborhood size.
        """

        config = self.config

        size = max(
            config.min_destroy_size,
            math.ceil(
                config.destroy_fraction * N
            ),
        )

        if config.max_destroy_size is not None:

            size = min(
                size,
                config.max_destroy_size,
            )

        size = min(
            size,
            N,
        )

        if repair_method == "exact":

            size = min(
                size,
                config.exact_max_neighborhood_size,
            )

        return max(
            1,
            size,
        )

    # ==================================================
    # Acceptance construction
    # ==================================================

    def _create_acceptance(
        self,
        initial_energy: int,
    ):
        """
        Construct the configured outer acceptance
        criterion.

        Default temperatures and thresholds are
        scaled using the initial objective value.
        """

        config = self.config

        if config.acceptance_method == "greedy":

            return create_acceptance(
                "greedy"
            )

        if (
            config.acceptance_method
            == "simulated_annealing"
        ):

            initial_temperature = (
                config.acceptance_initial_temperature
            )

            if initial_temperature is None:

                initial_temperature = max(
                    1.0,
                    0.05 * initial_energy,
                )

            final_temperature = (
                config.acceptance_final_temperature
            )

            if final_temperature is None:

                final_temperature = max(
                    1e-6,
                    0.001 * initial_temperature,
                )

            return create_acceptance(
                "simulated_annealing",
                initial_temperature=initial_temperature,
                final_temperature=final_temperature,
                cooling_rate=(
                    config.acceptance_cooling_rate
                ),
            )

        if (
            config.acceptance_method
            == "record_to_record"
        ):

            threshold = (
                config.record_initial_threshold
            )

            if threshold is None:

                threshold = (
                    0.05 * initial_energy
                )

            return create_acceptance(
                "record_to_record",
                initial_threshold=threshold,
                final_threshold=(
                    config.record_final_threshold
                ),
                cooling_rate=(
                    config.acceptance_cooling_rate
                ),
            )

        raise ValueError(
            "Unknown acceptance method: "
            f"{config.acceptance_method}"
        )

    # ==================================================
    # Repair-specific parameters
    # ==================================================

    def _repair_kwargs(
        self,
        method: str,
    ) -> dict:
        """
        Construct parameters for the selected
        repair operator.
        """

        config = self.config

        if method == "greedy":

            return {
                "max_iterations": (
                    config.greedy_max_iterations
                ),
            }

        if method == "randomized_greedy":

            return {
                "max_iterations": (
                    config.greedy_max_iterations
                ),
                "top_k": (
                    config.randomized_greedy_top_k
                ),
            }

        if method == "exact":

            return {
                "max_neighborhood_size": (
                    config.exact_max_neighborhood_size
                ),
            }

        if method == "simulated_annealing":

            return {
                "max_iterations": (
                    config.repair_sa_iterations
                ),
                "cooling_rate": (
                    config.repair_sa_cooling_rate
                ),
                "return_best": (
                    config.repair_sa_return_best
                ),
            }

        raise ValueError(
            f"Unknown repair method: {method}"
        )

    # ==================================================
    # Solve
    # ==================================================

    def solve(
        self,
        N: int,
        *,
        initial_state: BinaryObjectiveState | None = None,
    ) -> ALNSResult:
        """
        Run classical ALNS for a binary waveform
        of length N.

        Parameters
        ----------
        N:
            Waveform length.

        initial_state:
            Optional externally generated initial
            solution.

            If supplied, initialization is skipped.

        Returns
        -------
        ALNSResult:
            Best solution and solver statistics.
        """

        if not isinstance(
            N,
            (int, np.integer),
        ):
            raise TypeError(
                "N must be an integer."
            )

        if N < 2:
            raise ValueError(
                "N must be at least 2."
            )

        config = self.config

        config.validate()

        rng = np.random.default_rng(
            config.seed
        )

        start_time = (
            time.perf_counter()
        )

        # --------------------------------------------------
        # Global deadline
        # --------------------------------------------------

        deadline = None

        if config.time_limit is not None:

            deadline = (
                start_time
                + config.time_limit
            )

        # ==================================================
        # 1. Initialization
        # ==================================================

        if initial_state is not None:

            if initial_state.N != N:

                raise ValueError(
                    "initial_state length does "
                    "not match N."
                )

            current_state = (
                initial_state.copy()
            )

        else:

            # Allocate a limited portion of the
            # total runtime to initialization.

            if config.time_limit is not None:

                initialization_time = min(
                    config.initialization_max_time,
                    config.time_limit
                    * config.initialization_time_fraction,
                )

            else:

                initialization_time = (
                    config.initialization_max_time
                )

            initialization_seed = int(
                rng.integers(
                    0,
                    2**32,
                )
            )

            current_state = (
                generate_initial_solution(
                    N=N,
                    method=(
                        config.initialization_method
                    ),
                    time_limit=(
                        initialization_time
                    ),
                    seed=initialization_seed,
                    max_starts=(
                        config.initialization_max_starts
                    ),
                )
            )

        best_state = (
            current_state.copy()
        )

        initial_energy = int(
            current_state.energy
        )

        # ==================================================
        # 2. Construct acceptance mechanism
        # ==================================================

        acceptance = (
            self._create_acceptance(
                initial_energy
            )
        )

        # ==================================================
        # 3. Construct adaptive selectors
        # ==================================================

        destroy_selector = (
            AdaptiveOperatorSelector(
                operators=(
                    config.destroy_methods
                ),
                reaction_factor=(
                    config.reaction_factor
                ),
                segment_length=(
                    config.segment_length
                ),
            )
        )

        repair_selector = (
            AdaptiveOperatorSelector(
                operators=(
                    config.repair_methods
                ),
                reaction_factor=(
                    config.reaction_factor
                ),
                segment_length=(
                    config.segment_length
                ),
            )
        )

        # ==================================================
        # 4. Reward scheme
        # ==================================================

        reward_scheme = RewardScheme(
            new_best=(
                config.reward_new_best
            ),
            improvement=(
                config.reward_improvement
            ),
            accepted=(
                config.reward_accepted
            ),
            rejected=(
                config.reward_rejected
            ),
        )

        # ==================================================
        # 5. Statistics
        # ==================================================

        iterations = 0

        accepted_moves = 0

        rejected_moves = 0

        improving_moves = 0

        new_best_moves = 0

        history = []

        history.append(
            {
                "iteration": 0,
                "elapsed_time": (
                    time.perf_counter()
                    - start_time
                ),
                "current_energy": (
                    current_state.energy
                ),
                "best_energy": (
                    best_state.energy
                ),
            }
        )

        # ==================================================
        # 6. Main ALNS loop
        # ==================================================

        while True:

            # ----------------------------------------------
            # Termination: runtime
            # ----------------------------------------------

            if (
                deadline is not None
                and time.perf_counter()
                >= deadline
            ):
                break

            # ----------------------------------------------
            # Termination: iterations
            # ----------------------------------------------

            if (
                config.max_iterations is not None
                and iterations
                >= config.max_iterations
            ):
                break

            # ----------------------------------------------
            # Select destroy and repair operators
            # ----------------------------------------------

            destroy_method = (
                destroy_selector.select(
                    rng
                )
            )

            repair_method = (
                repair_selector.select(
                    rng
                )
            )

            # ----------------------------------------------
            # Determine neighborhood size
            # ----------------------------------------------

            destroy_size = (
                self._destroy_size(
                    N,
                    repair_method,
                )
            )

            # ----------------------------------------------
            # Destroy: select positions
            # ----------------------------------------------

            indices = apply_destroy(
                current_state,
                method=destroy_method,
                destroy_size=destroy_size,
                rng=rng,
            )

            # ----------------------------------------------
            # Repair selected positions
            # ----------------------------------------------

            candidate = apply_repair(
                current_state,
                method=repair_method,
                allowed_indices=indices,
                rng=rng,
                deadline=deadline,
                **self._repair_kwargs(
                    repair_method
                ),
            )

            # ----------------------------------------------
            # Preserve pre-acceptance energies
            # ----------------------------------------------

            previous_current_energy = int(
                current_state.energy
            )

            previous_best_energy = int(
                best_state.energy
            )

            candidate_energy = int(
                candidate.energy
            )

            # ----------------------------------------------
            # Outer acceptance decision
            # ----------------------------------------------

            accepted = acceptance.accept(
                current_energy=(
                    previous_current_energy
                ),
                candidate_energy=(
                    candidate_energy
                ),
                best_energy=(
                    previous_best_energy
                ),
                rng=rng,
            )

            # ----------------------------------------------
            # Calculate operator reward
            # ----------------------------------------------

            reward = calculate_reward(
                candidate_energy=(
                    candidate_energy
                ),
                current_energy=(
                    previous_current_energy
                ),
                best_energy=(
                    previous_best_energy
                ),
                accepted=accepted,
                scheme=reward_scheme,
            )

            # ----------------------------------------------
            # Update current state
            # ----------------------------------------------

            if accepted:

                current_state = (
                    candidate
                )

                accepted_moves += 1

                if (
                    candidate_energy
                    < previous_current_energy
                ):

                    improving_moves += 1

            else:

                rejected_moves += 1

            # ----------------------------------------------
            # Update global best
            # ----------------------------------------------

            if (
                candidate_energy
                < previous_best_energy
            ):

                best_state = (
                    candidate.copy()
                )

                new_best_moves += 1

            # ----------------------------------------------
            # Record operator performance
            # ----------------------------------------------

            destroy_selector.record(
                destroy_method,
                reward,
            )

            repair_selector.record(
                repair_method,
                reward,
            )

            # ----------------------------------------------
            # Advance acceptance schedule
            # ----------------------------------------------

            acceptance.advance()

            # ----------------------------------------------
            # Increment iteration counter
            # ----------------------------------------------

            iterations += 1

            # ----------------------------------------------
            # Record history
            # ----------------------------------------------

            if (
                iterations
                % config.history_interval
                == 0
            ):

                history.append(
                    {
                        "iteration": (
                            iterations
                        ),
                        "elapsed_time": (
                            time.perf_counter()
                            - start_time
                        ),
                        "current_energy": (
                            current_state.energy
                        ),
                        "best_energy": (
                            best_state.energy
                        ),
                        "destroy_method": (
                            destroy_method
                        ),
                        "repair_method": (
                            repair_method
                        ),
                    }
                )

        # ==================================================
        # 7. Finalize incomplete adaptive segments
        # ==================================================

        destroy_selector.update_weights()

        repair_selector.update_weights()

        # ==================================================
        # 8. Final statistics
        # ==================================================

        elapsed_time = (
            time.perf_counter()
            - start_time
        )

        if (
            history[-1]["iteration"]
            != iterations
        ):

            history.append(
                {
                    "iteration": (
                        iterations
                    ),
                    "elapsed_time": (
                        elapsed_time
                    ),
                    "current_energy": (
                        current_state.energy
                    ),
                    "best_energy": (
                        best_state.energy
                    ),
                }
            )

        # ==================================================
        # 9. Return best solution
        # ==================================================

        return ALNSResult(
            best_state=best_state,
            initial_energy=(
                initial_energy
            ),
            iterations=iterations,
            elapsed_time=elapsed_time,
            accepted_moves=(
                accepted_moves
            ),
            rejected_moves=(
                rejected_moves
            ),
            improving_moves=(
                improving_moves
            ),
            new_best_moves=(
                new_best_moves
            ),
            destroy_statistics=(
                destroy_selector.get_statistics()
            ),
            repair_statistics=(
                repair_selector.get_statistics()
            ),
            history=history,
        )


# ============================================================
# Convenience function
# ============================================================

def solve_alns(
    N: int,
    *,
    initial_state: BinaryObjectiveState | None = None,
    **config_kwargs,
) -> ALNSResult:
    """
    Convenient functional interface.

    Example
    -------
    result = solve_alns(
        N=100,
        time_limit=30.0,
        seed=42,
    )
    """

    config = ALNSConfig(
        **config_kwargs
    )

    solver = ALNSSolver(
        config
    )

    return solver.solve(
        N,
        initial_state=initial_state,
    )