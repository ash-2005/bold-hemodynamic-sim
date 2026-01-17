"""Per-region haemodynamic delay injection via HRF convolution or BW ODE."""

from __future__ import annotations

import numpy as np

from src.bw_model import BWParams, simulate
from src.hrf import shifted_hrf, convolve_hrf


def inject_delays_hrf(
    neural_signals: np.ndarray,
    delay_vector: np.ndarray,
    hrf_type: str = 'double_gamma',
    dt: float = 0.001,
) -> np.ndarray:
    """Produce multiregion BOLD by convolving each region with a delay-shifted HRF.

    Args:
        neural_signals: Shape (n_regions, n_timepoints).
        delay_vector: Per-region HRF onset delays [s], shape (n_regions,).
        hrf_type: HRF type. Default 'double_gamma'.
        dt: Timestep [s]. Default 0.001.

    Returns:
        np.ndarray: BOLD matrix, shape (n_regions, n_timepoints).
    """
    n_regions, n_t = neural_signals.shape
    bold = np.zeros((n_regions, n_t))
    t_hrf = np.arange(0, 32.0, dt)
    for i in range(n_regions):
        hrf = shifted_hrf(t_hrf, delay_vector[i], hrf_type=hrf_type)
        bold[i] = convolve_hrf(neural_signals[i], hrf, dt)
    return bold


def inject_delays_bw(
    neural_signals: np.ndarray,
    delay_vector: np.ndarray,
    params: BWParams,
    dt: float,
    T: float,
) -> np.ndarray:
    """Produce multiregion BOLD via full BW ODE integration per region.

    Per-region delays are applied as integer sample offsets to the neural input.

    Args:
        neural_signals: Shape (n_regions, n_timepoints).
        delay_vector: Per-region delays [s], shape (n_regions,).
        params: BW model parameters.
        dt: Integration timestep [s].
        T: Total duration [s].

    Returns:
        np.ndarray: BOLD matrix, shape (n_regions, n_timepoints).
    """
    n_regions, n_t = neural_signals.shape
    bold = np.zeros((n_regions, n_t))
    for i in range(n_regions):
        shift = int(round(delay_vector[i] / dt))
        shifted = np.zeros(n_t)
        if shift < n_t:
            shifted[shift:] = neural_signals[i, :n_t - shift]
        result = simulate(shifted, dt, T, params)
        bold[i] = result.bold[:n_t]
    return bold


def apply_global_coupling(
    neural_base: np.ndarray,
    G: float,
    coupling_matrix: np.ndarray,
) -> np.ndarray:
    """Scale a multiregion neural signal by global coupling G and coupling matrix.

    Args:
        neural_base: Shape (n_regions, n_timepoints).
        G: Global coupling scalar.
        coupling_matrix: Shape (n_regions, n_regions).

    Returns:
        np.ndarray: Shape (n_regions, n_timepoints). G * C @ neural_base.
    """
    return G * (coupling_matrix @ neural_base)


def delay_vector_from_rapidtide(npy_path: str) -> np.ndarray:
    """Load a per-region delay vector from a .npy file.

    Args:
        npy_path: Path to .npy file with a 1D delay array.

    Returns:
        np.ndarray: Delay vector in seconds, shape (n_regions,).

    Raises:
        ValueError: If the file does not contain a valid 1D finite delay vector.
    """
    arr = np.load(npy_path)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D delay array, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Delay vector contains non-finite values")
    return arr
