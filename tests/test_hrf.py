"""Tests for src/hrf.py — 5 tests."""

from __future__ import annotations

import numpy as np
import pytest
from src.hrf import (
    double_gamma_hrf, canonical_spm_hrf, shifted_hrf,
    hrf_matrix, convolve_hrf
)

DT = 0.001
T_HRF = 32.0
t = np.arange(0, T_HRF, DT)


def test_double_gamma_integrates_to_approximately_one():
    """The integral of the double-gamma HRF should be approximately 1 (normalised)."""
    hrf = double_gamma_hrf(t)
    integral = np.trapz(hrf, t)
    assert abs(integral - 1.0) < 0.05, f"HRF integral={integral:.4f}, expected ≈1.0"


def test_shifted_hrf_peak_delayed():
    """Shifted HRF with tau_i=2s should have its peak ~2s later."""
    tau_i = 2.0
    hrf_base = double_gamma_hrf(t)
    hrf_shift = shifted_hrf(t, tau_i)
    peak_base = t[np.argmax(hrf_base)]
    peak_shift = t[np.argmax(hrf_shift)]
    assert abs((peak_shift - peak_base) - tau_i) < 0.5, (
        f"Shifted peak at {peak_shift:.2f}s, expected ≈{peak_base + tau_i:.2f}s"
    )


def test_hrf_matrix_shape():
    """hrf_matrix should return shape (n_regions, len_t)."""
    n_regions = 5
    delays = np.linspace(0, 2, n_regions)
    mat = hrf_matrix(t, delays)
    assert mat.shape == (n_regions, len(t))


def test_convolve_hrf_output_length():
    """convolve_hrf output should have the same length as the input signal."""
    signal = np.random.randn(int(30 / DT))
    hrf = double_gamma_hrf(t)
    result = convolve_hrf(signal, hrf, DT)
    assert len(result) == len(signal)


def test_double_gamma_known_values():
    """Double-gamma HRF should have peak near t=5-6s."""
    t_test = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20], dtype=float)
    hrf = double_gamma_hrf(t_test)
    peak_idx = np.argmax(hrf)
    assert t_test[peak_idx] in [5, 6], f"Expected peak near t=5-6s, got t={t_test[peak_idx]}"
    assert hrf[10] < hrf[5], "HRF should be falling at t=10s"
    assert abs(hrf[-1]) < 0.1, f"HRF at t=20s should be near zero, got {hrf[-1]:.4f}"
