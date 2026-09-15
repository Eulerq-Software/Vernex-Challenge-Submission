
import numpy as np

from waveform_optimization.core import phase_indices_to_waveform


def test_binary_phase_encoding():
    waveform = phase_indices_to_waveform([0, 1], Q=2)

    np.testing.assert_allclose(
        waveform,
        np.array([1, -1], dtype=complex),
    )


def test_waveform_has_unit_magnitude():
    waveform = phase_indices_to_waveform([0, 1, 2, 3], Q=4)

    np.testing.assert_allclose(
        np.abs(waveform),
        np.ones(4),
    )