"""G-coupling bifurcation analysis for the BW model."""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.bw_model import BWParams, downsample
from src.neural_generator import generate_coupled_oscillators
from src.delay_inject import inject_delays_bw, apply_global_coupling
from src.fc_from_bold import compute_fc, pearson_r_fc


def g_sweep(
    G_values: np.ndarray,
    coupling_matrix: np.ndarray,
    n_regions: int,
    T: float,
    dt: float,
    params: BWParams,
    tr: float,
) -> dict:
    """Sweep global coupling G and compute FC at each value.

    Returns:
        dict: Keys are G values; values are FC matrices.
    """
    freq_vector = np.full(n_regions, 0.05)
    neural_base = generate_coupled_oscillators(
        n_regions, T, dt, coupling_matrix, freq_vector, noise_std=0.01
    )
    delays = np.zeros(n_regions)
    fc_dict = {}
    for G in tqdm(G_values, desc="G sweep"):
        neural_coupled = apply_global_coupling(neural_base, G, coupling_matrix)
        bold_full = inject_delays_bw(neural_coupled, delays, params, dt, T)
        bold_ds = np.stack([downsample(bold_full[i], dt, tr) for i in range(n_regions)])
        fc_dict[float(G)] = compute_fc(bold_ds, tr=tr)
    return fc_dict


def compute_model_fit(
    G_values: np.ndarray,
    fc_simulated_dict: dict,
    fc_empirical: np.ndarray,
) -> dict:
    """Compute Pearson r model fit at each G value.

    Returns:
        dict: Keys are G values; values are Pearson r (float).
    """
    return {float(G): pearson_r_fc(fc_simulated_dict[float(G)], fc_empirical)
            for G in G_values}


def find_g_optimal(fit_dict: dict) -> float:
    """Return the G value that maximises model fit Pearson r."""
    return max(fit_dict, key=fit_dict.get)


def find_g_max_delta(fit_dict_legacy: dict, fit_dict_delayed: dict) -> float:
    """Find G where the difference r_delayed - r_legacy is maximised."""
    G_keys = sorted(set(fit_dict_legacy.keys()) & set(fit_dict_delayed.keys()))
    deltas = {G: fit_dict_delayed[G] - fit_dict_legacy[G] for G in G_keys}
    return max(deltas, key=deltas.get)


def plot_g_sweep(
    fit_dict: dict,
    output_path: str,
    fit_dict_delayed: dict = None,
) -> None:
    """Plot model fit Pearson r vs G on a log-scale x-axis.

    Args:
        fit_dict: Legacy FC fit results.
        output_path: Path to save figure.
        fit_dict_delayed: Delayed FC fit results (optional).
    """
    G_vals = sorted(fit_dict.keys())
    r_vals = [fit_dict[G] for G in G_vals]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.semilogx(G_vals, r_vals, 'o-', color='royalblue', label='FC legacy')

    if fit_dict_delayed is not None:
        r_delayed = [fit_dict_delayed.get(G, float('nan')) for G in G_vals]
        ax.semilogx(G_vals, r_delayed, 's--', color='tomato', label='FC delay-corrected')
        ax2 = ax.twinx()
        delta_r = [d - l for d, l in zip(r_delayed, r_vals)]
        ax2.plot(G_vals, delta_r, '^:', color='seagreen', alpha=0.7, label='ΔR')
        ax2.set_ylabel('ΔR (delayed − legacy)', color='seagreen')

    ax.set_xlabel('G (global coupling)')
    ax.set_ylabel('Pearson r (simulated vs empirical FC)')
    ax.set_title('G-coupling bifurcation sweep')
    ax.legend(loc='upper left')
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def delay_sensitivity_analysis(
    G_values: np.ndarray,
    delay_ranges: np.ndarray,
    coupling_matrix: np.ndarray,
    n_regions: int,
    params: BWParams,
    tr: float,
) -> np.ndarray:
    """Compute delta_r over a 2D grid of G values and delay magnitudes.

    Returns:
        np.ndarray: Shape (n_G, n_d). Entry [i,j] is delta_r.
    """
    from src.fc_from_bold import fc_legacy as _fc_legacy, fc_delay_corrected
    dt = 0.001
    T = 60.0
    freq_vector = np.full(n_regions, 0.05)
    neural_base = generate_coupled_oscillators(n_regions, T, dt, coupling_matrix, freq_vector)

    fc_empirical = np.eye(n_regions) + 0.05 * np.random.randn(n_regions, n_regions)
    fc_empirical = (fc_empirical + fc_empirical.T) / 2
    np.fill_diagonal(fc_empirical, 1.0)

    delta_grid = np.zeros((len(G_values), len(delay_ranges)))
    for i, G in enumerate(tqdm(G_values, desc="G axis")):
        for j, max_delay in enumerate(delay_ranges):
            neural_coupled = apply_global_coupling(neural_base, G, coupling_matrix)
            delays_zero = np.zeros(n_regions)
            bold_z = inject_delays_bw(neural_coupled, delays_zero, params, dt, T)
            bold_z_ds = np.stack([downsample(bold_z[k], dt, tr) for k in range(n_regions)])
            fc_z = compute_fc(bold_z_ds, tr=tr)
            r_legacy = pearson_r_fc(fc_z, fc_empirical)

            delays_d = np.linspace(0, max_delay, n_regions)
            bold_d = inject_delays_bw(neural_coupled, delays_d, params, dt, T)
            bold_d_ds = np.stack([downsample(bold_d[k], dt, tr) for k in range(n_regions)])
            fc_d = fc_delay_corrected(bold_d_ds, delays_d, tr)
            r_delayed = pearson_r_fc(fc_d, fc_empirical)

            delta_grid[i, j] = r_delayed - r_legacy
    return delta_grid
