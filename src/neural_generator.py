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
