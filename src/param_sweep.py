"""BW parameter sensitivity analysis: sweeps, feature extraction, plotting."""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.bw_model import BWParams, BWResult, simulate


def sweep_single_param(
    param_name: str,
    values: np.ndarray,
    base_params: BWParams,
    neural_signal: np.ndarray,
    dt: float,
    T: float,
) -> dict:
    """Sweep a single BW parameter across a range of values.

    Args:
        param_name: Name of the parameter to sweep.
        values: Array of values to test.
        base_params: Base parameter set.
        neural_signal: Neural input timeseries.
        dt: Timestep [s].
        T: Duration [s].

    Returns:
        dict: Keys are float values, values are BWResult objects.
    """
    results = {}
    for v in tqdm(values, desc=f"Sweep {param_name}", leave=False):
        params = dataclasses.replace(base_params, **{param_name: float(v)})
        results[float(v)] = simulate(neural_signal, dt, T, params)
    return results


def sweep_all_params(
    base_params: BWParams,
    n_points: int,
    neural_signal: np.ndarray,
    dt: float,
    T: float,
) -> dict:
    """Sweep all six BW parameters independently across n_points values.

    Each parameter is swept over ±50% of its default value on a linear grid.

    Returns:
        dict: Outer keys are parameter names; inner dicts are {value: BWResult}.
    """
    param_ranges = {
        'kappa': (base_params.kappa * 0.5, base_params.kappa * 1.5),
        'gamma': (base_params.gamma * 0.5, base_params.gamma * 1.5),
        'tau':   (base_params.tau   * 0.5, base_params.tau   * 1.5),
        'alpha': (base_params.alpha * 0.5, base_params.alpha * 1.5),
        'E0':    (base_params.E0    * 0.5, min(base_params.E0 * 1.5, 0.9)),
        'V0':    (base_params.V0    * 0.5, base_params.V0    * 1.5),
    }
    all_results = {}
    for name, (lo, hi) in param_ranges.items():
        values = np.linspace(lo, hi, n_points)
        all_results[name] = sweep_single_param(name, values, base_params, neural_signal, dt, T)
    return all_results


def extract_bold_features(bw_result: BWResult) -> dict:
    """Extract scalar features from a BW simulation BOLD timeseries.

    Returns:
        dict with keys: 'peak_amplitude', 'time_to_peak', 'fwhm', 'undershoot_depth'.
    """
    bold = bw_result.bold
    time = bw_result.time

    peak_idx = int(np.argmax(bold))
    peak_amplitude = float(bold[peak_idx])
    time_to_peak = float(time[peak_idx])

    # FWHM: time interval where bold > 0.5 * peak
    half = 0.5 * peak_amplitude
    above = bold > half
    if above.any():
        first = int(np.argmax(above))
        last = int(len(above) - 1 - np.argmax(above[::-1]))
        fwhm = float(time[last] - time[first])
    else:
        fwhm = float('nan')

    # Undershoot: min after peak
    if peak_idx < len(bold) - 1:
        undershoot_depth = float(np.min(bold[peak_idx:]))
    else:
        undershoot_depth = float('nan')

    return {
        'peak_amplitude':  peak_amplitude,
        'time_to_peak':   time_to_peak,
        'fwhm':            fwhm,
        'undershoot_depth': undershoot_depth,
    }


def plot_sweep_grid(sweep_results: dict, output_path: str) -> None:
    """Produce a figure grid with one subplot per parameter.

    Args:
        sweep_results: Output of sweep_all_params() or a subset dict.
        output_path: Path to save the figure (PNG or PDF).
    """
    params_list = list(sweep_results.keys())
    n_params = len(params_list)
    ncols = min(3, n_params)
    nrows = (n_params + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, name in enumerate(params_list):
        ax = axes[idx // ncols][idx % ncols]
        results = sweep_results[name]
        vals = sorted(results.keys())
        features = [extract_bold_features(results[v]) for v in vals]
        peaks = [f['peak_amplitude'] for f in features]
        ttps  = [f['time_to_peak'] for f in features]
        ax.plot(vals, peaks, 'o-', color='tomato', label='peak')
        ax2 = ax.twinx()
        ax2.plot(vals, ttps, 's--', color='royalblue', label='TTP')
        ax.set_xlabel(name)
        ax.set_ylabel('Peak amplitude', color='tomato')
        ax2.set_ylabel('Time to peak (s)', color='royalblue')
        ax.set_title(f'BOLD vs {name}')

    # Hide unused subplots
    for idx in range(n_params, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle('BW Parameter Sensitivity Sweep', fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
