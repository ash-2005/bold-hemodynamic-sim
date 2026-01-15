"""Tests for src/bw_model.py — 8 tests."""

from __future__ import annotations

import numpy as np
import pytest
from src.bw_model import (
    BWParams, BWState, BWResult, simulate, bold_signal,
    check_physiological_bounds, downsample
)

DT = 0.001
T  = 30.0


def make_neural(amplitude=1.0):
    """Simple event-related neural signal for testing."""
    n = int(T / DT)
    u = np.zeros(n)
    u[int(5/DT):int(6/DT)] = amplitude  # 1-second pulse at t=5s
    return u


def test_bold_range():
    """BOLD signal should remain within physiological range [-0.05, 0.05]."""
    neural = make_neural()
    result = simulate(neural, DT, T)
    assert np.all(result.bold >= -0.05), "BOLD dropped below -5%"
    assert np.all(result.bold <= 0.05), "BOLD exceeded +5%"


def test_zero_neural_input_resting_state():
    """Zero neural input should yield approximately zero BOLD (resting state)."""
    neural = np.zeros(int(T / DT))
    result = simulate(neural, DT, T)
    assert np.max(np.abs(result.bold)) < 1e-4, (
        f"Expected near-zero BOLD for zero input, got max={np.max(np.abs(result.bold))}"
    )


def test_state_trajectory_shape():
    """State trajectory should be shape (4, n_timepoints)."""
    neural = make_neural()
    result = simulate(neural, DT, T)
    expected_len = int(T / DT)
    assert result.state_trajectory.shape[0] == 4
    assert result.state_trajectory.shape[1] == expected_len


def test_downsample_length():
    """Downsampled BOLD should have correct length for given TR."""
    neural = make_neural()
    result = simulate(neural, DT, T)
    target_tr = 2.0
    ds = downsample(result.bold, DT, target_tr)
    expected_len = int(T / target_tr)
    assert abs(len(ds) - expected_len) <= 1


def test_ode_conservation_positive_v_q():
    """Blood volume v and deoxyhaemoglobin q must remain positive throughout."""
    neural = make_neural()
    result = simulate(neural, DT, T)
    v = result.state_trajectory[2]
    q = result.state_trajectory[3]
    assert np.all(v > 0), f"v went negative: min={v.min()}"
    assert np.all(q > 0), f"q went negative: min={q.min()}"


def test_kappa_effect_monotonic():
    """Higher kappa should give lower peak BOLD amplitude (monotonically)."""
    neural = make_neural()
    kappa_values = [0.3, 0.65, 1.0, 1.5]
    peaks = []
    for kappa in kappa_values:
        result = simulate(neural, DT, T, BWParams(kappa=kappa))
        peaks.append(np.max(result.bold))
    for i in range(len(peaks) - 1):
        assert peaks[i] > peaks[i+1], (
            f"Peak BOLD did not decrease monotonically with kappa: {peaks}"
        )


def test_simulate_vs_euler_within_1_percent():
    """RK45 output should match Euler integration to within 1%."""
    dt_test = 0.0001
    T_test  = 5.0
    n       = int(T_test / dt_test)
    neural  = np.zeros(n)
    neural[int(1/dt_test):int(2/dt_test)] = 0.5
    params  = BWParams()
    result_rk45 = simulate(neural, dt_test, T_test, params)
    from src.bw_model import bw_ode
    from scipy.interpolate import interp1d
    t_arr = np.arange(0, T_test, dt_test)
    u_func = interp1d(t_arr, neural, bounds_error=False, fill_value=0.0)
    y = np.array([0.0, 1.0, 1.0, 1.0])
    bold_euler = []
    for t_i in t_arr:
        v, q = y[2], y[3]
        bold_euler.append(bold_signal(v, q, params))
        dy = bw_ode(t_i, y, u_func, params)
        y = y + dt_test * dy
    bold_euler = np.array(bold_euler)
    max_diff = np.max(np.abs(result_rk45.bold - bold_euler))
    assert max_diff < 0.01 * np.max(np.abs(result_rk45.bold)) + 1e-8


def test_check_physiological_bounds_catches_violation():
    """check_physiological_bounds should detect injected out-of-range state."""
    neural = make_neural()
    result = simulate(neural, DT, T)
    bad_result = BWResult(
        time=result.time,
        bold=result.bold.copy(),
        state_trajectory=result.state_trajectory.copy()
    )
    bad_result.state_trajectory[2, 100] = -0.5  # v = -0.5 is nonphysical
    report = check_physiological_bounds(bad_result)
    assert report['v']['n_violations'] > 0
