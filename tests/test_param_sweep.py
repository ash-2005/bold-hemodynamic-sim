"""Tests for src/param_sweep.py — 4 tests."""

from __future__ import annotations

import os
import numpy as np
import pytest
import tempfile
from src.param_sweep import (
    sweep_single_param, extract_bold_features, plot_sweep_grid, sweep_all_params
)
from src.bw_model import BWParams
from src.neural_generator import generate_event_related

DT = 0.001
T  = 30.0

neural = generate_event_related(2, 8.0, 1.0, T, DT, noise_std=0.0)
base_params = BWParams()


def test_sweep_output_has_correct_number_of_entries():
    values = np.linspace(0.3, 1.2, 10)
    results = sweep_single_param('kappa', values, base_params, neural, DT, T)
    assert len(results) == 10


def test_higher_kappa_shorter_time_to_peak():
    kappa_vals = np.linspace(0.3, 1.5, 6)
    results = sweep_single_param('kappa', kappa_vals, base_params, neural, DT, T)
    ttps = [extract_bold_features(results[k])['time_to_peak'] for k in kappa_vals]
    for i in range(len(ttps) - 1):
        assert ttps[i] >= ttps[i+1] - 0.5


def test_feature_extraction_returns_all_keys():
    from src.bw_model import simulate
    result = simulate(neural, DT, T, base_params)
    features = extract_bold_features(result)
    expected_keys = {'peak_amplitude', 'time_to_peak', 'fwhm', 'undershoot_depth'}
    assert set(features.keys()) == expected_keys


def test_plot_sweep_grid_creates_file():
    values = np.linspace(0.4, 1.0, 5)
    results_kappa = sweep_single_param('kappa', values, base_params, neural, DT, T)
    results_tau   = sweep_single_param('tau', np.linspace(0.5, 1.5, 5), base_params, neural, DT, T)
    sweep_results = {'kappa': results_kappa, 'tau': results_tau}
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, 'sweep_grid.png')
        plot_sweep_grid(sweep_results, out)
        assert os.path.exists(out)
