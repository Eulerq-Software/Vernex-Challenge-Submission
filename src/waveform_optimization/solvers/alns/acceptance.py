from __future__ import annotations

import math

from typing import Protocol

import numpy as np


# ============================================================
# Common acceptance interface
# ============================================================

class AcceptanceCriterion(Protocol):
    """
    Interface shared by all acceptance mechanisms.
    """

    def accept(
        self,
        *,
        current_energy: float,
        candidate_energy: float,
        best_energy: float,
        rng: np.random.Generator,
    ) -> bool:
        ...

    def advance(self) -> None:
        ...


# ============================================================
# A1: Greedy acceptance
# ============================================================

class GreedyAcceptance:
    """
    Accept a candidate if it is no worse than
    the current solution.

    E_candidate <= E_current
    """

    def accept(
        self,
        *,
        current_energy: float,
        candidate_energy: float,
        best_energy: float,
        rng: np.random.Generator,
    ) -> bool:

        return (
            candidate_energy
            <= current_energy
        )

    def advance(self) -> None:
        """
        Greedy acceptance has no schedule.
        """

        pass


# ============================================================
# A2: Simulated annealing acceptance
# ============================================================

class SimulatedAnnealingAcceptance:
    """
    Simulated annealing acceptance criterion.

    Improving or equal candidates are always accepted.

    Worsening candidates are accepted with probability:

        exp(-(E_candidate - E_current) / T)

    Temperature follows a geometric cooling schedule:

        T_new = max(T_final, T_current * cooling_rate)

    The acceptance mechanism does not modify
    any BinaryObjectiveState.
    """

    def __init__(
        self,
        *,
        initial_temperature: float,
        final_temperature: float = 1e-6,
        cooling_rate: float = 0.995,
    ) -> None:

        # ----------------------------------------------
        # Validate parameters
        # ----------------------------------------------

        if not math.isfinite(initial_temperature):
            raise ValueError(
                "initial_temperature must be finite."
            )

        if not math.isfinite(final_temperature):
            raise ValueError(
                "final_temperature must be finite."
            )

        if not math.isfinite(cooling_rate):
            raise ValueError(
                "cooling_rate must be finite."
            )

        if initial_temperature <= 0:
            raise ValueError(
                "initial_temperature must be positive."
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

        if not 0 < cooling_rate < 1:
            raise ValueError(
                "cooling_rate must satisfy "
                "0 < cooling_rate < 1."
            )

        # ----------------------------------------------
        # Store configuration
        # ----------------------------------------------

        self.initial_temperature = float(
            initial_temperature
        )

        self.final_temperature = float(
            final_temperature
        )

        self.cooling_rate = float(
            cooling_rate
        )

        self.temperature = float(
            initial_temperature
        )

        self.iteration = 0

    # --------------------------------------------------
    # Acceptance decision
    # --------------------------------------------------

    def accept(
        self,
        *,
        current_energy: float,
        candidate_energy: float,
        best_energy: float,
        rng: np.random.Generator,
    ) -> bool:

        delta = (
            candidate_energy
            - current_energy
        )

        # Always accept improving or equal solutions.

        if delta <= 0:
            return True

        probability = math.exp(
            -delta / self.temperature
        )

        return bool(
            rng.random() < probability
        )

    # --------------------------------------------------
    # Advance cooling schedule
    # --------------------------------------------------

    def advance(self) -> None:

        self.temperature = max(
            self.final_temperature,
            self.temperature
            * self.cooling_rate,
        )

        self.iteration += 1

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------

    def reset(self) -> None:

        self.temperature = (
            self.initial_temperature
        )

        self.iteration = 0


# ============================================================
# A3: Record-to-record acceptance
# ============================================================

class RecordToRecordAcceptance:
    """
    Accept candidates sufficiently close to
    the best solution discovered so far.

    Acceptance condition:

        E_candidate <= E_best + threshold

    The threshold decreases geometrically:

        threshold_new =
            max(
                final_threshold,
                threshold * cooling_rate
            )

    This is an absolute-energy threshold,
    not a percentage.
    """

    def __init__(
        self,
        *,
        initial_threshold: float,
        final_threshold: float = 0.0,
        cooling_rate: float = 0.995,
    ) -> None:

        # ----------------------------------------------
        # Validate parameters
        # ----------------------------------------------

        if not math.isfinite(initial_threshold):
            raise ValueError(
                "initial_threshold must be finite."
            )

        if not math.isfinite(final_threshold):
            raise ValueError(
                "final_threshold must be finite."
            )

        if not math.isfinite(cooling_rate):
            raise ValueError(
                "cooling_rate must be finite."
            )

        if initial_threshold < 0:
            raise ValueError(
                "initial_threshold must be "
                "non-negative."
            )

        if final_threshold < 0:
            raise ValueError(
                "final_threshold must be "
                "non-negative."
            )

        if final_threshold > initial_threshold:
            raise ValueError(
                "final_threshold cannot exceed "
                "initial_threshold."
            )

        if not 0 < cooling_rate < 1:
            raise ValueError(
                "cooling_rate must satisfy "
                "0 < cooling_rate < 1."
            )

        self.initial_threshold = float(
            initial_threshold
        )

        self.final_threshold = float(
            final_threshold
        )

        self.cooling_rate = float(
            cooling_rate
        )

        self.threshold = float(
            initial_threshold
        )

        self.iteration = 0

    # --------------------------------------------------
    # Acceptance decision
    # --------------------------------------------------

    def accept(
        self,
        *,
        current_energy: float,
        candidate_energy: float,
        best_energy: float,
        rng: np.random.Generator,
    ) -> bool:

        return bool(
            candidate_energy
            <= best_energy + self.threshold
        )

    # --------------------------------------------------
    # Advance threshold schedule
    # --------------------------------------------------

    def advance(self) -> None:

        self.threshold = max(
            self.final_threshold,
            self.threshold
            * self.cooling_rate,
        )

        self.iteration += 1

    # --------------------------------------------------
    # Reset
    # --------------------------------------------------

    def reset(self) -> None:

        self.threshold = (
            self.initial_threshold
        )

        self.iteration = 0


# ============================================================
# Acceptance factory
# ============================================================

ACCEPTANCE_METHODS = {
    "greedy": GreedyAcceptance,
    "simulated_annealing": SimulatedAnnealingAcceptance,
    "record_to_record": RecordToRecordAcceptance,
}


def create_acceptance(
    method: str,
    **kwargs,
) -> AcceptanceCriterion:
    """
    Construct an acceptance mechanism by name.

    Examples
    --------
    create_acceptance("greedy")

    create_acceptance(
        "simulated_annealing",
        initial_temperature=100.0,
        cooling_rate=0.995,
    )

    create_acceptance(
        "record_to_record",
        initial_threshold=50.0,
    )
    """

    if method not in ACCEPTANCE_METHODS:

        raise ValueError(
            f"Unknown acceptance method: {method}. "
            f"Available methods: "
            f"{list(ACCEPTANCE_METHODS)}"
        )

    return ACCEPTANCE_METHODS[
        method
    ](
        **kwargs
    )