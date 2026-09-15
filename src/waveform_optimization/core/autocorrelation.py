
from __future__ import annotations

import numpy as np


def autocorrelation_at_lag(
    x: np.ndarray | list[complex],
    k: int,
) -> complex:
    """
    Calculate the positive-lag aperiodic autocorrelation.

    rho(k) = sum_i x[i] * conjugate(x[i + k])
    """
    x = np.asarray(x, dtype=complex)

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional.")

    N = len(x)

    if not 1 <= k < N:
        raise ValueError("k must satisfy 1 <= k < len(x).")

    return np.sum(x[: N - k] * np.conj(x[k:]))


def all_autocorrelations(
    x: np.ndarray | list[complex],
) -> np.ndarray:
    """
    Return positive-lag aperiodic autocorrelations for k = 1, ..., N - 1.
    """
    x = np.asarray(x, dtype=complex)

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional.")

    N = len(x)

    if N < 2:
        raise ValueError("x must contain at least two chips.")

    return np.array(
        [autocorrelation_at_lag(x, k) for k in range(1, N)],
        dtype=complex,
    )