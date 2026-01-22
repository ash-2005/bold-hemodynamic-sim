"""Tests for src/fc_from_bold.py — 5 tests."""

from __future__ import annotations

import numpy as np
import pytest
from src.fc_from_bold import (
    compute_fc, fc_legacy, fc_delay_corrected, fc_difference, pearson_r_fc
)

TR = 2.0
N  = 10
T_BOLD = 200


def make_bold(n_regions=N, n_timepoints=T_BOLD, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_regions, n_timepoints))


def test_fc_matrix_symmetric():
    bold = make_bold()
    fc = fc_legacy(bold, TR)
    np.testing.assert_allclose(fc, fc.T, atol=1e-10)


def test_fc_diagonal_is_one():
    bold = make_bold()
    fc = fc_legacy(bold, TR)
    np.testing.assert_allclose(np.diag(fc), np.ones(N), atol=1e-10)


def test_delay_correction_changes_fc():
    bold = make_bold()
    delays = np.linspace(0, 4.0, N)
    fc_leg  = fc_legacy(bold, TR)
    fc_corr = fc_delay_corrected(bold, delays, TR)
    assert not np.allclose(fc_leg, fc_corr, atol=1e-4)


def test_zero_delay_gives_identical_fc():
    bold = make_bold()
    delays = np.zeros(N)
    fc_leg  = fc_legacy(bold, TR)
    fc_corr = fc_delay_corrected(bold, delays, TR)
    np.testing.assert_allclose(fc_leg, fc_corr, atol=1e-10)


def test_pearson_r_identical_matrices():
    bold = make_bold()
    fc = fc_legacy(bold, TR)
    r = pearson_r_fc(fc, fc)
    assert abs(r - 1.0) < 1e-10
