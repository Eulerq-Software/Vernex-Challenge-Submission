from __future__ import annotations

import math

from collections.abc import Sequence

import numpy as np


# ============================================================
# Reward configuration
# ============================================================

class RewardScheme:
    """
    Rewards assigned to operator outcomes.

    Default values:

        New global best:       10
        Improvement:            5
        Accepted non-improving: 2
        Rejected:               0
    """

    def __init__(
        self,
        *,
        new_best: float = 10.0,
        improvement: float = 5.0,
        accepted: float = 2.0,
        rejected: float = 0.0,
    ) -> None:

        values = {
            "new_best": new_best,
            "improvement": improvement,
            "accepted": accepted,
            "rejected": rejected,
        }

        for name, value in values.items():

            if not math.isfinite(value):
                raise ValueError(
                    f"{name} reward must be finite."
                )

            if value < 0:
                raise ValueError(
                    f"{name} reward must be "
                    "non-negative."
                )

        self.new_best = float(
            new_best
        )

        self.improvement = float(
            improvement
        )

        self.accepted = float(
            accepted
        )

        self.rejected = float(
            rejected
        )


# ============================================================
# Calculate reward
# ============================================================

def calculate_reward(
    *,
    candidate_energy: float,
    current_energy: float,
    best_energy: float,
    accepted: bool,
    scheme: RewardScheme | None = None,
) -> float:
    """
    Calculate reward for one ALNS iteration.

    Priority:

        1. New global best.
        2. Improvement over current.
        3. Accepted non-improving candidate.
        4. Rejected candidate.

    IMPORTANT:
        current_energy and best_energy must refer
        to their values BEFORE the candidate is
        processed by the ALNS main loop.
    """

    if scheme is None:

        scheme = RewardScheme()

    # --------------------------------------------------
    # New global best
    # --------------------------------------------------

    if candidate_energy < best_energy:

        return scheme.new_best

    # --------------------------------------------------
    # Improvement over current solution
    # --------------------------------------------------

    if (
        accepted
        and candidate_energy < current_energy
    ):

        return scheme.improvement

    # --------------------------------------------------
    # Accepted non-improving candidate
    # --------------------------------------------------

    if accepted:

        return scheme.accepted

    # --------------------------------------------------
    # Rejected candidate
    # --------------------------------------------------

    return scheme.rejected


# ============================================================
# Adaptive operator selector
# ============================================================

class AdaptiveOperatorSelector:
    """
    Adaptive roulette-wheel operator selection.

    Each operator has:

        - Current weight
        - Accumulated segment reward
        - Number of uses in the segment
        - Total number of uses

    Selection probability:

        p_i = w_i / sum_j w_j

    Segment-end weight update:

        w_i_new =
            (1 - reaction_factor) * w_i
            +
            reaction_factor
            * (segment_reward_i / segment_uses_i)

    Operators not used during a segment retain
    their previous weights.

    A minimum weight prevents operators from
    becoming permanently unselectable.
    """

    def __init__(
        self,
        operators: Sequence[str],
        *,
        reaction_factor: float = 0.2,
        segment_length: int = 50,
        initial_weight: float = 1.0,
        min_weight: float = 0.05,
    ) -> None:

        # ----------------------------------------------
        # Validate operators
        # ----------------------------------------------

        if len(operators) == 0:
            raise ValueError(
                "At least one operator is required."
            )

        if len(set(operators)) != len(operators):
            raise ValueError(
                "Operator names must be unique."
            )

        if not all(
            isinstance(name, str)
            for name in operators
        ):
            raise TypeError(
                "Operator names must be strings."
            )

        # ----------------------------------------------
        # Validate parameters
        # ----------------------------------------------

        if not math.isfinite(reaction_factor):
            raise ValueError(
                "reaction_factor must be finite."
            )

        if not 0 < reaction_factor <= 1:
            raise ValueError(
                "reaction_factor must satisfy "
                "0 < reaction_factor <= 1."
            )

        if not isinstance(
            segment_length,
            (int, np.integer),
        ):
            raise TypeError(
                "segment_length must be an integer."
            )

        if segment_length < 1:
            raise ValueError(
                "segment_length must be at least 1."
            )

        if not math.isfinite(initial_weight):
            raise ValueError(
                "initial_weight must be finite."
            )

        if not math.isfinite(min_weight):
            raise ValueError(
                "min_weight must be finite."
            )

        if initial_weight <= 0:
            raise ValueError(
                "initial_weight must be positive."
            )

        if min_weight <= 0:
            raise ValueError(
                "min_weight must be positive."
            )

        if min_weight > initial_weight:
            raise ValueError(
                "min_weight cannot exceed "
                "initial_weight."
            )

        # ----------------------------------------------
        # Store configuration
        # ----------------------------------------------

        self.operators = tuple(
            operators
        )

        self.reaction_factor = float(
            reaction_factor
        )

        self.segment_length = int(
            segment_length
        )

        self.initial_weight = float(
            initial_weight
        )

        self.min_weight = float(
            min_weight
        )

        # ----------------------------------------------
        # Operator state
        # ----------------------------------------------

        self.weights = {
            name: self.initial_weight
            for name in self.operators
        }

        self.segment_rewards = {
            name: 0.0
            for name in self.operators
        }

        self.segment_uses = {
            name: 0
            for name in self.operators
        }

        self.total_uses = {
            name: 0
            for name in self.operators
        }

        self.iterations_in_segment = 0

        self.completed_segments = 0

    # ==================================================
    # Selection probabilities
    # ==================================================

    def probabilities(self) -> dict[str, float]:
        """
        Return current operator selection probabilities.
        """

        total_weight = sum(
            self.weights.values()
        )

        return {
            name: (
                self.weights[name]
                / total_weight
            )
            for name in self.operators
        }

    # ==================================================
    # Select operator
    # ==================================================

    def select(
        self,
        rng: np.random.Generator,
    ) -> str:
        """
        Select one operator using roulette-wheel
        selection.
        """

        probabilities = self.probabilities()

        names = list(
            self.operators
        )

        p = np.array(
            [
                probabilities[name]
                for name in names
            ],
            dtype=np.float64,
        )

        selected = rng.choice(
            names,
            p=p,
        )

        return str(
            selected
        )

    # ==================================================
    # Record outcome
    # ==================================================

    def record(
        self,
        operator: str,
        reward: float,
    ) -> None:
        """
        Record the reward received by an operator.

        Call once for the selected operator after
        every ALNS iteration.
        """

        if operator not in self.weights:

            raise ValueError(
                f"Unknown operator: {operator}"
            )

        if not math.isfinite(reward):
            raise ValueError(
                "reward must be finite."
            )

        if reward < 0:
            raise ValueError(
                "reward must be non-negative."
            )

        self.segment_rewards[operator] += (
            float(reward)
        )

        self.segment_uses[operator] += 1

        self.total_uses[operator] += 1

        self.iterations_in_segment += 1

        # ----------------------------------------------
        # Automatically update at segment boundary
        # ----------------------------------------------

        if (
            self.iterations_in_segment
            >= self.segment_length
        ):

            self.update_weights()

    # ==================================================
    # Update weights
    # ==================================================

    def update_weights(self) -> None:
        """
        Update weights using accumulated rewards.

        Operators not used during the segment
        retain their previous weights.
        """

        if self.iterations_in_segment == 0:
            return

        rho = self.reaction_factor

        for name in self.operators:

            uses = self.segment_uses[
                name
            ]

            if uses == 0:
                continue

            average_reward = (
                self.segment_rewards[name]
                / uses
            )

            old_weight = (
                self.weights[name]
            )

            new_weight = (
                (1.0 - rho)
                * old_weight
                +
                rho
                * average_reward
            )

            self.weights[name] = max(
                self.min_weight,
                new_weight,
            )

        # ----------------------------------------------
        # Reset segment statistics
        # ----------------------------------------------

        self.segment_rewards = {
            name: 0.0
            for name in self.operators
        }

        self.segment_uses = {
            name: 0
            for name in self.operators
        }

        self.iterations_in_segment = 0

        self.completed_segments += 1

    # ==================================================
    # Reset selector
    # ==================================================

    def reset(self) -> None:
        """
        Restore initial weights and clear statistics.
        """

        self.weights = {
            name: self.initial_weight
            for name in self.operators
        }

        self.segment_rewards = {
            name: 0.0
            for name in self.operators
        }

        self.segment_uses = {
            name: 0
            for name in self.operators
        }

        self.total_uses = {
            name: 0
            for name in self.operators
        }

        self.iterations_in_segment = 0

        self.completed_segments = 0

    # ==================================================
    # Diagnostics
    # ==================================================

    def get_statistics(self) -> dict:
        """
        Return a snapshot of current selector state.

        Useful for logging and benchmarking.
        """

        return {
            "weights": self.weights.copy(),
            "probabilities": self.probabilities(),
            "segment_rewards": (
                self.segment_rewards.copy()
            ),
            "segment_uses": (
                self.segment_uses.copy()
            ),
            "total_uses": (
                self.total_uses.copy()
            ),
            "iterations_in_segment": (
                self.iterations_in_segment
            ),
            "completed_segments": (
                self.completed_segments
            ),
        }