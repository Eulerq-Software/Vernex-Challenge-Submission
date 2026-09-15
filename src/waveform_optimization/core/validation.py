
from __future__ import annotations

import numpy as np


def validate_phase_indices(
    q: np.ndarray | list[int],
    N: int | None = None,
    Q: int | None = None,
) -> None:
    """
    Validate a discrete phase-index sequence.

    Raises
    ------
    ValueError
        If the sequence is invalid.
    """
    q = np.asarray(q)

    if q.ndim != 1:
        raise ValueError("Phase indices must be one-dimensional.")

    if q.size == 0:
        raise ValueError("Phase sequence must not be empty.")

    if N is not None and len(q) != N:
        raise ValueError(f"Expected {N} phase indices, got {len(q)}.")

    if Q is not None and Q < 2:
        raise ValueError("Q must be at least 2.")

    if not np.issubdtype(q.dtype, np.integer):
        if not np.all(np.equal(q, np.floor(q))):
            raise ValueError("All phase indices must be integers.")

    q = q.astype(int)

    if Q is not None and (np.any(q < 0) or np.any(q >= Q)):
        raise ValueError("Phase indices must lie in [0, Q - 1].")


def validate_waveform(
    x: np.ndarray | list[complex],
    N: int | None = None,
    tolerance: float = 1e-10,
) -> None:
    """
    Validate a complex unit-magnitude waveform.
    """
    x = np.asarray(x, dtype=complex)

    if x.ndim != 1:
        raise ValueError("Waveform must be one-dimensional.")

    if x.size == 0:
        raise ValueError("Waveform must not be empty.")

    if N is not None and len(x) != N:
        raise ValueError(f"Expected {N} waveform samples, got {len(x)}.")

    magnitudes = np.abs(x)

    if not np.allclose(magnitudes, 1.0, atol=tolerance):
        raise ValueError("Every waveform sample must have unit magnitude.")