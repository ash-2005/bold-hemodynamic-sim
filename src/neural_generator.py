"""Synthetic neural activation signal generators for BW model input."""

from __future__ import annotations

import numpy as np


def generate_event_related(
    n_events: int,
    isi: float,
    event_duration: float,
    T: float,
    dt: float,
    noise_std: float = 0.0,
) -> np.ndarray:
    """Generate an event-related neural activation timeseries.

    Args:
        n_events: Number of stimulus events.
        isi: Inter-stimulus interval [s] (onset to onset).
        event_duration: Duration of each event [s].
        T: Total timeseries duration [s].
        dt: Timestep [s].
        noise_std: Std of additive Gaussian noise. Default 0.0.

    Returns:
        np.ndarray: Activation timeseries, shape (int(T/dt),).
    """
    n = int(T / dt)
    u = np.zeros(n)
    for k in range(1, n_events + 1):
        onset = int(k * isi / dt)
        offset = int((k * isi + event_duration) / dt)
        onset = min(onset, n)
        offset = min(offset, n)
        u[onset:offset] = 1.0
    if noise_std > 0:
        u = u + np.random.randn(n) * noise_std
    return u


def generate_block_design(
    on_duration: float,
    off_duration: float,
    n_cycles: int,
    T: float,
    dt: float,
    noise_std: float = 0.0,
) -> np.ndarray:
    """Generate a block-design neural activation timeseries.

    Args:
        on_duration: Duration of active block [s].
        off_duration: Duration of rest block [s].
        n_cycles: Number of on/off cycles.
        T: Total duration [s].
        dt: Timestep [s].
        noise_std: Additive noise std. Default 0.0.

    Returns:
        np.ndarray: Block-design activation, shape (int(T/dt),).
    """
    n = int(T / dt)
    u = np.zeros(n)
    cycle_dur = on_duration + off_duration
    for k in range(n_cycles):
        onset = int(k * cycle_dur / dt)
        offset = int((k * cycle_dur + on_duration) / dt)
        onset = min(onset, n)
        offset = min(offset, n)
        u[onset:offset] = 1.0
    if noise_std > 0:
        u = u + np.random.randn(n) * noise_std
    return u


def generate_coupled_oscillators(
    n_regions: int,
    T: float,
    dt: float,
    coupling_matrix: np.ndarray,
    freq_vector: np.ndarray,
    noise_std: float = 0.01,
) -> np.ndarray:
    """Generate a multiregion neural timeseries from coupled oscillators.

    Uses Euler integration of:
        dx_i/dt = -freq_i * x_i + sum_j C_ij * x_j + noise

    Returns:
        np.ndarray: Shape (n_regions, int(T/dt)).
    """
    n = int(T / dt)
    x = np.zeros((n_regions, n))
    x[:, 0] = np.random.randn(n_regions) * 0.1
    for t_idx in range(n - 1):
        coupling = coupling_matrix @ x[:, t_idx]
        dx = -freq_vector * x[:, t_idx] + coupling + np.random.randn(n_regions) * noise_std
        x[:, t_idx + 1] = x[:, t_idx] + dt * dx
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (x - mean) / std


def generate_from_coupling_matrix(
    coupling_strength: float,
    n_regions: int,
    T: float,
    dt: float,
) -> np.ndarray:
    """Generate multiregion neural signal from a uniform random coupling matrix.

    Returns:
        np.ndarray: Shape (n_regions, int(T/dt)).
    """
    C = np.random.randn(n_regions, n_regions)
    C = C / (np.abs(C).max() + 1e-8) * coupling_strength
    np.fill_diagonal(C, 0)
    freq_vector = np.full(n_regions, 0.05)
    return generate_coupled_oscillators(n_regions, T, dt, C, freq_vector)


def add_neural_noise(
    signal: np.ndarray,
    noise_type: str = 'white',
    noise_std: float = 0.01,
) -> np.ndarray:
    """Add noise to a neural signal.

    Args:
        signal: Input signal (1D or 2D, regions × time).
        noise_type: One of 'white', 'pink', 'ar1'. Default 'white'.
        noise_std: Noise amplitude. Default 0.01.

    Returns:
        np.ndarray: Signal with added noise, same shape as input.
    """
    shape = signal.shape
    flat = signal.reshape(-1, shape[-1]) if signal.ndim > 1 else signal[np.newaxis]
    n_r, n_t = flat.shape
    noisy = flat.copy()
    if noise_type == 'white':
        noisy += np.random.randn(n_r, n_t) * noise_std
    elif noise_type == 'pink':
        for i in range(n_r):
            freqs = np.fft.rfftfreq(n_t)
            freqs[0] = 1.0
            power = 1.0 / np.sqrt(freqs)
            power[0] = 0.0
            white = np.fft.rfft(np.random.randn(n_t))
            pink = np.fft.irfft(white * power, n=n_t)
            pink = pink / (pink.std() + 1e-9) * noise_std
            noisy[i] += pink
    elif noise_type == 'ar1':
        for i in range(n_r):
            ar = np.zeros(n_t)
            for t in range(1, n_t):
                ar[t] = 0.9 * ar[t - 1] + np.random.randn() * noise_std
            noisy[i] += ar
    return noisy.reshape(shape)
