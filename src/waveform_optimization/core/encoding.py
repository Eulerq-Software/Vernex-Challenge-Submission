
from __future__ import annotations

import numpy as np


def phase_indices_to_waveform(
    q: np.ndarray | list[int],
    Q: int,
) -> np.ndarray:
    """
    Convert integer phase indices into a complex unit-magnitude waveform.

    Parameters
    ----------
    q:
        Phase indices in {0, 1, ..., Q - 1}.
    Q:
        Number of available phase values.

    Returns
    -------
    np.ndarray
        Complex waveform with one value per phase index.
    """
    q = np.asarray(q)

    if Q < 2:
        raise ValueError("Q must be at least 2.")

    if q.ndim != 1:
        raise ValueError("q must be a one-dimensional sequence.")

    if not np.issubdtype(q.dtype, np.integer):
        if not np.all(np.equal(q, np.floor(q))):
            raise ValueError("All phase indices must be integers.")

    q = q.astype(int)

    if np.any(q < 0) or np.any(q >= Q):
        raise ValueError("Phase indices must lie in [0, Q - 1].")

    return np.exp(-1j * 2 * np.pi * q / Q)