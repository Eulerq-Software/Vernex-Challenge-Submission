from __future__ import annotations

import numpy as np


class BinaryObjectiveState:
    """
    Incremental objective evaluator for binary-phase
    radar waveform optimization (Q = 2).

    Phase-index representation:
        q[i] = 0 -> x[i] = +1
        q[i] = 1 -> x[i] = -1

    Objective:
        E(x) = sum_{k=1}^{N-1} C_k(x)^2

    where:
        C_k(x) = sum_{j=0}^{N-k-1} x[j] * x[j+k]

    Maintains:
        - Binary phase indices
        - Real-valued waveform (+1/-1)
        - Aperiodic autocorrelations
        - Objective value

    Supports:
        - O(N) flip-delta evaluation
        - O(N) incremental flip application
        - Full objective recomputation
        - State copying
        - Consistency validation
    """

    def __init__(
        self,
        sequence: np.ndarray | list[int],
    ) -> None:

        q = np.asarray(sequence)

        # -----------------------------
        # Input validation
        # -----------------------------

        if q.ndim != 1:
            raise ValueError(
                "Sequence must be one-dimensional."
            )

        if len(q) < 2:
            raise ValueError(
                "Sequence length must be at least 2."
            )

        if not np.all((q == 0) | (q == 1)):
            raise ValueError(
                "Sequence must contain only 0 and 1."
            )

        # -----------------------------
        # Store phase indices
        # -----------------------------

        self.sequence = q.astype(
            np.int8,
            copy=True,
        )

        self.N = len(self.sequence)

        # -----------------------------
        # Convert to waveform
        # -----------------------------

        self.waveform = (
            1 - 2 * self.sequence
        ).astype(np.int8)

        # -----------------------------
        # Compute autocorrelations
        # -----------------------------

        self.correlations = (
            self._compute_correlations()
        )

        # -----------------------------
        # Compute objective
        # -----------------------------

        self.energy = int(
            np.dot(
                self.correlations,
                self.correlations,
            )
        )

    # ==================================================
    # Full autocorrelation computation
    # ==================================================

    def _compute_correlations(
        self,
    ) -> np.ndarray:
        """
        Compute all positive-lag aperiodic
        autocorrelations.

        Returns:
            correlations[k-1] = C_k

        Complexity:
            O(N^2)
        """

        N = self.N

        x = self.waveform.astype(
            np.int64,
            copy=False,
        )

        correlations = np.empty(
            N - 1,
            dtype=np.int64,
        )

        for k in range(1, N):

            correlations[k - 1] = (
                np.dot(
                    x[:N-k],
                    x[k:],
                )
            )

        return correlations

    # ==================================================
    # Full objective computation
    # ==================================================

    def recompute_energy(
        self,
    ) -> int:
        """
        Recompute the objective from scratch.

        Complexity:
            O(N^2)

        Does not modify the stored state.
        """

        correlations = (
            self._compute_correlations()
        )

        return int(
            np.dot(
                correlations,
                correlations,
            )
        )

    # ==================================================
    # Incremental correlation changes
    # ==================================================

    def _flip_correlation_delta(
        self,
        i: int,
    ) -> np.ndarray:
        """
        Calculate changes in all autocorrelations
        caused by flipping position i.

        Does not modify the state.

        Complexity:
            O(N)
        """

        if not 0 <= i < self.N:
            raise IndexError(
                f"Bit index {i} out of range."
            )

        N = self.N
        x = self.waveform

        delta = np.zeros(
            N - 1,
            dtype=np.int64,
        )

        xi = int(x[i])

        # ------------------------------------------
        # Contributions where i is the left index
        # ------------------------------------------

        # C_k contains x[i] * x[i+k]
        # when i+k < N.

        right_count = N - i - 1

        if right_count > 0:

            delta[:right_count] += (
                -2
                * xi
                * x[i+1:].astype(np.int64)
            )

        # ------------------------------------------
        # Contributions where i is the right index
        # ------------------------------------------

        # C_k contains x[i-k] * x[i]
        # when i-k >= 0.

        if i > 0:

            delta[:i] += (
                -2
                * xi
                * x[:i][::-1].astype(np.int64)
            )

        return delta

    # ==================================================
    # Flip-delta evaluation
    # ==================================================

    def flip_delta(
        self,
        i: int,
    ) -> int:
        """
        Calculate the change in objective
        caused by flipping bit i.

        Returns:
            E_after - E_before

        Does not modify the state.

        Complexity:
            O(N)
        """

        delta_c = (
            self._flip_correlation_delta(i)
        )

        # E_new - E_old
        #
        # = sum(
        #     2*C_k*delta_C_k
        #     + delta_C_k^2
        # )

        delta_energy = (
            2
            * np.dot(
                self.correlations,
                delta_c,
            )
            +
            np.dot(
                delta_c,
                delta_c,
            )
        )

        return int(delta_energy)

    # ==================================================
    # Apply bit flip
    # ==================================================

    def apply_flip(
        self,
        i: int,
    ) -> int:
        """
        Flip bit i and update the stored state.

        Updates:
            - sequence
            - waveform
            - correlations
            - energy

        Returns:
            Change in objective.

        Complexity:
            O(N)
        """

        delta_c = (
            self._flip_correlation_delta(i)
        )

        delta_energy = int(
            2
            * np.dot(
                self.correlations,
                delta_c,
            )
            +
            np.dot(
                delta_c,
                delta_c,
            )
        )

        # -----------------------------
        # Update correlations
        # -----------------------------

        self.correlations += delta_c

        # -----------------------------
        # Update energy
        # -----------------------------

        self.energy += delta_energy

        # -----------------------------
        # Flip phase index
        # -----------------------------

        self.sequence[i] ^= 1

        # -----------------------------
        # Flip waveform
        # -----------------------------

        self.waveform[i] *= -1

        return delta_energy

    # ==================================================
    # Evaluate all single-bit flips
    # ==================================================

    def all_flip_deltas(
        self,
    ) -> np.ndarray:
        """
        Calculate objective changes for flipping
        every bit independently.

        Returns:
            Array of length N.

        Complexity:
            O(N^2)
        """

        deltas = np.empty(
            self.N,
            dtype=np.int64,
        )

        for i in range(self.N):

            deltas[i] = (
                self.flip_delta(i)
            )

        return deltas

    # ==================================================
    # Copy state
    # ==================================================

    def copy(
        self,
    ) -> BinaryObjectiveState:
        """
        Create an independent copy of the state.

        Avoids recomputing autocorrelations.
        """

        new_state = object.__new__(
            BinaryObjectiveState
        )

        new_state.N = self.N

        new_state.sequence = (
            self.sequence.copy()
        )

        new_state.waveform = (
            self.waveform.copy()
        )

        new_state.correlations = (
            self.correlations.copy()
        )

        new_state.energy = self.energy

        return new_state

    # ==================================================
    # Consistency validation
    # ==================================================

    def validate(
        self,
    ) -> bool:
        """
        Verify that the stored state is consistent.

        Recomputes:
            - waveform
            - autocorrelations
            - energy

        Raises AssertionError if inconsistent.
        """

        expected_waveform = (
            1 - 2 * self.sequence
        ).astype(np.int8)

        assert np.array_equal(
            self.waveform,
            expected_waveform,
        ), "Waveform mismatch."

        expected_correlations = (
            self._compute_correlations()
        )

        assert np.array_equal(
            self.correlations,
            expected_correlations,
        ), "Autocorrelation mismatch."

        expected_energy = int(
            np.dot(
                expected_correlations,
                expected_correlations,
            )
        )

        assert (
            self.energy == expected_energy
        ), "Objective mismatch."

        return True

    # ==================================================
    # Representation
    # ==================================================

    def __repr__(
        self,
    ) -> str:

        return (
            f"BinaryObjectiveState("
            f"N={self.N}, "
            f"energy={self.energy})"
        )