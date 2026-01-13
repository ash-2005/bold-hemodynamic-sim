"""Balloon-Windkessel haemodynamic model — Friston et al. (2003).

Implements the four coupled ODEs (s, f, v, q), the BOLD signal equation,
RK45 integration via SciPy solve_ivp, physiological bounds checking,
and TR-downsampling of BOLD output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


@dataclass
class BWParams:
    """Physiological parameters for the Balloon-Windkessel model.

    All defaults from Friston et al. (2003), Table 1.

    Attributes:
        kappa: Neural efficacy signal decay rate [s⁻¹]. Default 0.65.
        gamma: Autoregulatory feedback gain [s⁻¹]. Default 0.41.
        tau: Haemodynamic transit time [s]. Default 0.98.
        alpha: Grubb's exponent (vessel stiffness). Default 0.32.
        E0: Resting oxygen extraction fraction [0–1]. Default 0.34.
        V0: Resting venous blood volume fraction [0–1]. Default 0.02.
    """
    kappa: float = 0.65
    gamma: float = 0.41
    tau: float = 0.98
    alpha: float = 0.32
    E0: float = 0.34
    V0: float = 0.02


@dataclass
class BWState:
    """Instantaneous state of the BW system at a single time point.

    Attributes:
        s: Neural efficacy signal (resting=0).
        f: Normalised cerebral blood flow (resting=1).
        v: Normalised blood volume (resting=1).
        q: Normalised deoxyhaemoglobin content (resting=1).
        t: Time in seconds.
    """
    s: float = 0.0
    f: float = 1.0
    v: float = 1.0
    q: float = 1.0
    t: float = 0.0


@dataclass
class BWResult:
    """Full output of a BW simulation.

    Attributes:
        time: Time vector [s], shape (n_timepoints,).
        bold: BOLD signal [% signal change], shape (n_timepoints,).
        state_trajectory: Shape (4, n_timepoints) — rows are s, f, v, q.
    """
    time: np.ndarray
    bold: np.ndarray
    state_trajectory: np.ndarray
