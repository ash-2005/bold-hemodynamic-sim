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


def bold_signal(v: float, q: float, params: BWParams) -> float:
    """Compute the BOLD signal from blood volume and deoxyhaemoglobin.

    Friston et al. (2003) Equation 8:
        BOLD = V0 * (k1*(1-q) + k2*(1-q/v) + k3*(1-v))
    where k1=7*E0, k2=2, k3=2*E0-0.2.
    """
    k1 = 7.0 * params.E0
    k2 = 2.0
    k3 = 2.0 * params.E0 - 0.2
    return params.V0 * (k1 * (1.0 - q) + k2 * (1.0 - q / v) + k3 * (1.0 - v))


def bw_ode(
    t: float,
    y: np.ndarray,
    neural_input_func: Callable,
    params: BWParams,
) -> np.ndarray:
    """Compute the four BW ODE derivatives at time t.

    State vector y = [s, f, v, q].
    ODEs (Friston 2003):
        ds/dt = u(t) - kappa*s - gamma*(f-1)
        df/dt = s
        dv/dt = (1/tau) * (f - v^(1/alpha))
        dq/dt = (1/tau) * (f*E(f)/E0 - v^(1/alpha-1)*q)
    where E(f) = 1 - (1-E0)^(1/f).
    """
    s, f, v, q = y
    u = neural_input_func(t)
    f = max(f, 1e-6)
    v = max(v, 1e-6)
    q = max(q, 1e-6)
    E_f = 1.0 - (1.0 - params.E0) ** (1.0 / f)
    ds_dt = u - params.kappa * s - params.gamma * (f - 1.0)
    df_dt = s
    dv_dt = (1.0 / params.tau) * (f - v ** (1.0 / params.alpha))
    dq_dt = (1.0 / params.tau) * (
        f * E_f / params.E0 - (v ** (1.0 / params.alpha - 1.0)) * q
    )
    return np.array([ds_dt, df_dt, dv_dt, dq_dt])


def simulate(
    neural_input: np.ndarray,
    dt: float,
    T: float,
    params: Optional[BWParams] = None,
    initial_state: Optional[BWState] = None,
) -> BWResult:
    """Integrate the BW ODE over time T given a neural input timeseries.

    Uses scipy.integrate.solve_ivp with RK45 adaptive stepping.

    Args:
        neural_input: Neural activation timeseries, shape (n_timepoints,).
        dt: Timestep [s] of the neural input array.
        T: Total simulation duration [s].
        params: BW parameters. Defaults to BWParams() (Friston 2003).
        initial_state: Initial conditions. Defaults to resting state.

    Returns:
        BWResult with time, bold, and state_trajectory fields.
    """
    if params is None:
        params = BWParams()
    if initial_state is None:
        initial_state = BWState()

    t_arr = np.arange(0, T, dt)
    n = len(t_arr)
    if len(neural_input) != n:
        # Truncate or zero-pad to match
        tmp = np.zeros(n)
        tmp[:min(n, len(neural_input))] = neural_input[:min(n, len(neural_input))]
        neural_input = tmp

    u_func = interp1d(t_arr, neural_input, bounds_error=False, fill_value=0.0)
    y0 = [initial_state.s, initial_state.f, initial_state.v, initial_state.q]

    sol = solve_ivp(
        bw_ode,
        t_span=(0.0, T),
        y0=y0,
        method='RK45',
        t_eval=t_arr,
        args=(u_func, params),
        rtol=1e-6,
        atol=1e-8,
    )

    bold_arr = np.array([bold_signal(v, q, params) for v, q in zip(sol.y[2], sol.y[3])])
    return BWResult(time=sol.t, bold=bold_arr, state_trajectory=sol.y)


def check_physiological_bounds(result: BWResult) -> dict:
    """Check whether any state variable leaves its physiological valid range.

    Returns:
        dict with keys 's', 'f', 'v', 'q', 'bold'; each value is a dict
        with 'n_violations', 'min', 'max'.
    """
    bounds = {
        's':    (-5.0, 5.0),
        'f':    (0.0, 5.0),
        'v':    (0.0, 2.0),
        'q':    (0.0, 2.0),
        'bold': (-0.05, 0.05),
    }
    state_labels = ['s', 'f', 'v', 'q']
    report = {}
    for i, label in enumerate(state_labels):
        arr = result.state_trajectory[i]
        lo, hi = bounds[label]
        violations = np.sum((arr < lo) | (arr > hi))
        report[label] = {'n_violations': int(violations), 'min': float(arr.min()), 'max': float(arr.max())}
    bold = result.bold
    lo, hi = bounds['bold']
    report['bold'] = {
        'n_violations': int(np.sum((bold < lo) | (bold > hi))),
        'min': float(bold.min()),
        'max': float(bold.max()),
    }
    return report


def downsample(bold_signal_arr: np.ndarray, original_dt: float, target_tr: float) -> np.ndarray:
    """Downsample a BOLD signal from integration timestep to scanner TR.

    Args:
        bold_signal_arr: BOLD timeseries at original_dt resolution.
        original_dt: Original timestep [s], typically 0.001.
        target_tr: Target TR [s], typically 2.0.

    Returns:
        np.ndarray: Downsampled BOLD signal.
    """
    factor = round(target_tr / original_dt)
    return bold_signal_arr[::factor]
