"""Functional connectivity computation from BOLD data."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def _bandpass_filter(
    bold_matrix: np.ndarray,
    bandpass_low: float,
    bandpass_high: float,
    tr: float,
) -> np.ndarray:
    """Apply Butterworth bandpass filter to each region."""
    fs = 1.0 / tr
    nyq = 0.5 * fs
    low = bandpass_low / nyq
    high = bandpass_high / nyq
    low = max(low, 1e-4)
    high = min(high, 0.9999)
    if low >= high:
        return bold_matrix
    b, a = butter(3, [low, high], btype='band')
    filtered = np.zeros_like(bold_matrix)
    for i in range(bold_matrix.shape[0]):
        filtered[i] = filtfilt(b, a, bold_matrix[i])
    return filtered


def compute_fc(
    bold_matrix: np.ndarray,
    bandpass_low: float = 0.01,
    bandpass_high: float = 0.1,
    tr: float = 2.0,
) -> np.ndarray:
    """Compute functional connectivity as Pearson correlation from bandpass-filtered BOLD.

    Args:
        bold_matrix: Shape (n_regions, n_timepoints).
        bandpass_low: Low-frequency cutoff [Hz]. Default 0.01.
        bandpass_high: High-frequency cutoff [Hz]. Default 0.1.
        tr: Repetition time [s]. Default 2.0.

    Returns:
        np.ndarray: Symmetric FC matrix, shape (n_regions, n_regions).
    """
    if np.any(~np.isfinite(bold_matrix)):
        raise ValueError("bold_matrix contains NaN or Inf values")
    filtered = _bandpass_filter(bold_matrix, bandpass_low, bandpass_high, tr)
    return np.corrcoef(filtered)


def fc_legacy(bold_matrix: np.ndarray, tr: float = 2.0) -> np.ndarray:
    """Compute FC without any delay correction (legacy/standard method).

    Args:
        bold_matrix: Shape (n_regions, n_timepoints).
        tr: TR [s].

    Returns:
        np.ndarray: FC matrix, shape (n_regions, n_regions).
    """
    return compute_fc(bold_matrix, tr=tr)


def fc_delay_corrected(
    bold_matrix: np.ndarray,
    delay_vector: np.ndarray,
    tr: float = 2.0,
) -> np.ndarray:
    """Compute FC after correcting for known per-region haemodynamic delays.

    For each region i, shifts its BOLD timeseries backward by
    round(delay_vector[i] / tr) samples before computing FC.

    Args:
        bold_matrix: Shape (n_regions, n_timepoints).
        delay_vector: Per-region delays [s], shape (n_regions,).
        tr: TR [s].

    Returns:
        np.ndarray: Delay-corrected FC matrix, shape (n_regions, n_regions).
    """
    corrected = bold_matrix.copy()
    for i, d in enumerate(delay_vector):
        shift = int(round(d / tr))
        if shift != 0:
            corrected[i] = np.roll(corrected[i], -shift)
    return compute_fc(corrected, tr=tr)


def fc_difference(fc1: np.ndarray, fc2: np.ndarray) -> np.ndarray:
    """Compute the elementwise difference between two FC matrices.

    Args:
        fc1: First FC matrix, shape (n, n).
        fc2: Second FC matrix, shape (n, n).

    Returns:
        np.ndarray: Delta FC matrix = fc1 - fc2.
    """
    return fc1 - fc2


def pearson_r_fc(fc_simulated: np.ndarray, fc_empirical: np.ndarray) -> float:
    """Compute Pearson r between upper triangles of two FC matrices.

    Args:
        fc_simulated: Simulated FC, shape (n, n).
        fc_empirical: Empirical/target FC, shape (n, n).

    Returns:
        float: Pearson r scalar in [-1, 1].
    """
    if fc_simulated.shape != fc_empirical.shape:
        raise ValueError("FC matrix shapes must match")
    n = fc_simulated.shape[0]
    idx = np.triu_indices(n, k=1)
    x = fc_simulated[idx]
    y = fc_empirical[idx]
    r = np.corrcoef(x, y)[0, 1]
    return float(r)


def fc_summary(fc_matrix: np.ndarray, network_labels: list) -> dict:
    """Compute mean FC for each pair of functional networks.

    Args:
        fc_matrix: FC matrix, shape (n_regions, n_regions).
        network_labels: List of network label strings, length n_regions.

    Returns:
        dict: Keys 'NetworkA-NetworkB' (sorted), values mean FC float.
    """
    labels = list(set(network_labels))
    result = {}
    for i, la in enumerate(labels):
        for lb in labels[i:]:
            idx_a = [j for j, l in enumerate(network_labels) if l == la]
            idx_b = [j for j, l in enumerate(network_labels) if l == lb]
            vals = []
            for ia in idx_a:
                for ib in idx_b:
                    if ia != ib:
                        vals.append(fc_matrix[ia, ib])
            key = '-'.join(sorted([la, lb]))
            result[key] = float(np.mean(vals)) if vals else 0.0
    return result
