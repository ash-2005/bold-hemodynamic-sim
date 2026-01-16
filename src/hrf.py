"""HRF library: double-gamma (Glover 1999), SPM canonical, delay-shifted HRFs."""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as gamma_func


def double_gamma_hrf(
    t: np.ndarray,
    peak1: float = 6.0,
    peak2: float = 16.0,
    ratio: float = 6.0,
) -> np.ndarray:
    """Compute the double-gamma HRF (Glover 1999).

    h(t) = gamma_pdf(t, peak1) - (1/ratio) * gamma_pdf(t, peak2)
    Normalised so that max(h) = 1.

    Args:
        t: Time vector [s].
        peak1: Time-to-peak of positive response [s]. Default 6.0.
        peak2: Time-to-peak of negative undershoot [s]. Default 16.0.
        ratio: Ratio of positive to negative amplitude. Default 6.0.

    Returns:
        np.ndarray: HRF values, normalised to peak=1.
    """
    t = np.asarray(t, dtype=float)
    safe_t = np.maximum(t, 1e-10)

    def gamma_pdf(x, a):
        """Gamma PDF with shape a, scale b=1."""
        return (x ** (a - 1)) * np.exp(-x) / gamma_func(a)

    h = gamma_pdf(safe_t, peak1) - (1.0 / ratio) * gamma_pdf(safe_t, peak2)
    h[t < 0] = 0.0
    peak_val = np.max(np.abs(h))
    if peak_val > 0:
        h = h / peak_val
    return h


def canonical_spm_hrf(t: np.ndarray) -> np.ndarray:
    """Compute the canonical SPM haemodynamic response function.

    SPM parameters: peak1=6s, peak2=16s, ratio=6, normalised to unit peak.

    Args:
        t: Time vector [s].

    Returns:
        np.ndarray: SPM canonical HRF, normalised to peak=1.
    """
    return double_gamma_hrf(t, peak1=6.0, peak2=16.0, ratio=6.0)


def shifted_hrf(
    t: np.ndarray,
    tau_i: float,
    hrf_type: str = 'double_gamma',
) -> np.ndarray:
    """Return an HRF evaluated at t - tau_i (onset delay).

    Args:
        t: Time vector [s].
        tau_i: Onset delay [s]. Positive = later response.
        hrf_type: One of 'double_gamma', 'spm'. Default 'double_gamma'.

    Returns:
        np.ndarray: HRF shifted by tau_i seconds; zero-padded before onset.
    """
    t_shifted = t - tau_i
    if hrf_type == 'spm':
        h = canonical_spm_hrf(t_shifted)
    else:
        h = double_gamma_hrf(t_shifted)
    h[t_shifted < 0] = 0.0
    return h


def hrf_matrix(
    t: np.ndarray,
    delay_vector: np.ndarray,
    hrf_type: str = 'double_gamma',
) -> np.ndarray:
    """Build a matrix of per-region delay-shifted HRFs.

    Args:
        t: Time vector [s], length L.
        delay_vector: Per-region delays [s], shape (n_regions,).
        hrf_type: HRF type. Default 'double_gamma'.

    Returns:
        np.ndarray: Shape (n_regions, L).
    """
    return np.array([shifted_hrf(t, d, hrf_type) for d in delay_vector])


def convolve_hrf(
    neural_signal: np.ndarray,
    hrf: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Convolve a neural signal with an HRF to produce a BOLD estimate.

    Args:
        neural_signal: Neural activation timeseries.
        hrf: HRF kernel at resolution dt.
        dt: Timestep [s].

    Returns:
        np.ndarray: BOLD estimate, same length as neural_signal.
    """
    full = np.convolve(neural_signal, hrf, mode='full')
    return full[:len(neural_signal)] * dt


def compare_hrfs(t: np.ndarray) -> dict:
    """Compute multiple HRF types on the same time axis for comparison.

    Args:
        t: Time vector [s].

    Returns:
        dict: Keys are HRF names, values are np.ndarray.
    """
    return {
        'double_gamma': double_gamma_hrf(t),
        'spm':          canonical_spm_hrf(t),
    }
