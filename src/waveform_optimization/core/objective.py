
from __future__ import annotations

import numpy as np

from .autocorrelation import all_autocorrelations
from .encoding import phase_indices_to_waveform


def islr_from_waveform(
    x: np.ndarray | list[complex],
) -> float:
    """
    Calculate the integrated sidelobe objective:

        F(x) = sum_{k=1}^{N-1} |rho(k)|^2
    """
    correlations = all_autocorrelations(x)
    return float(np.sum(np.abs(correlations) ** 2))


def islr_from_phase_indices(
    q: np.ndarray | list[int],
    Q: int,
) -> float:
    """
    Convert phase indices to a waveform and calculate the objective.
    """
    x = phase_indices_to_waveform(q, Q)
    return islr_from_waveform(x)