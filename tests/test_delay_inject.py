"""Tests for src/delay_inject.py — 5 tests."""

from __future__ import annotations

import numpy as np
import pytest
from src.delay_inject import inject_delays_hrf, inject_delays_bw, apply_global_coupling
from src.bw_model import BWParams

DT = 0.001
T  = 30.0
N_REGIONS = 3
params = BWParams()


def make_neural_2d(n_regions=N_REGIONS):
    n = int(T / DT)
    signals = np.zeros((n_regions, n))
    for i in range(n_regions):
        signals[i, int(5/DT):int(6/DT)] = 1.0
    return signals


def test_inject_delays_hrf_output_shape():
    """Output should be shape (n_regions, n_timepoints)."""
    neural = make_neural_2d()
    delays = np.array([0.0, 1.0, 2.0])
    result = inject_delays_hrf(neural, delays, dt=DT)
    assert result.shape == (N_REGIONS, int(T / DT))


def test_zero_delay_identity_hrf():
    """Zero delay should give the same BOLD as calling convolve_hrf directly."""
    from src.hrf import double_gamma_hrf, convolve_hrf
    neural = make_neural_2d(1)
    delays = np.array([0.0])
    result = inject_delays_hrf(neural, delays, dt=DT)
    t_hrf = np.arange(0, 32.0, DT)
    hrf = double_gamma_hrf(t_hrf)
    expected = convolve_hrf(neural[0], hrf, DT)
    np.testing.assert_allclose(result[0], expected, rtol=1e-4)


def test_delay_2s_shifts_peak_correctly():
    """A 2-second delay should shift the BOLD peak by approximately 2s/DT samples."""
    n = int(T / DT)
    neural = np.zeros((2, n))
    neural[:, int(5/DT):int(6/DT)] = 1.0
    delays = np.array([0.0, 2.0])
    result = inject_delays_bw(neural, delays, params, DT, T)
    peak_0 = np.argmax(result[0])
    peak_1 = np.argmax(result[1])
    expected_shift = 2.0 / DT
    actual_shift = peak_1 - peak_0
    assert abs(actual_shift - expected_shift) < 0.1 * expected_shift


def test_G_zero_uncoupled():
    """apply_global_coupling with G=0 should give zero output."""
    neural = make_neural_2d()
    C = np.ones((N_REGIONS, N_REGIONS))
    np.fill_diagonal(C, 0)
    result = apply_global_coupling(neural, G=0.0, coupling_matrix=C)
    np.testing.assert_array_equal(result, np.zeros_like(neural))


def test_bw_and_hrf_injection_differ():
    """BW ODE injection and HRF convolution injection should produce different arrays."""
    neural = make_neural_2d()
    delays = np.array([0.0, 1.0, 2.0])
    result_hrf = inject_delays_hrf(neural, delays, dt=DT)
    result_bw  = inject_delays_bw(neural, delays, params, DT, T)
    assert not np.allclose(result_hrf, result_bw, atol=1e-4)
