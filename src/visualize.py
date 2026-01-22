"""Shared visualisation utilities for bold-hemodynamic-sim."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless use


def set_style() -> None:
    """Set consistent Matplotlib rcParams."""
    plt.rcParams.update({
        'figure.dpi': 120,
        'font.size': 11,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_bold(time, bold, ax=None, title='BOLD signal'):
    """Plot single-region BOLD timeseries."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    else:
        fig = ax.get_figure()
    ax.plot(time, bold, color='tomato')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('BOLD')
    ax.set_title(title)
    return fig, ax


def plot_state_variables(result, ax_array=None):
    """4-panel plot of BW state variables s, f, v, q."""
    if ax_array is None:
        fig, ax_array = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    else:
        fig = ax_array[0].get_figure()
    labels = ['s: neural efficacy', 'f: CBF (norm.)', 'v: blood volume', 'q: dHb (norm.)']
    colors = ['royalblue', 'tomato', 'seagreen', 'darkorange']
    for i, (ax, label, color) in enumerate(zip(ax_array, labels, colors)):
        ax.plot(result.time, result.state_trajectory[i], color=color)
        ax.set_ylabel(label, fontsize=9)
    ax_array[-1].set_xlabel('Time (s)')
    return fig, ax_array


def plot_fc_heatmap(fc_matrix, labels=None, ax=None, title='FC matrix'):
    """Seaborn heatmap of FC matrix."""
    try:
        import seaborn as sns
        use_sns = True
    except ImportError:
        use_sns = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()
    if use_sns:
        import seaborn as sns
        sns.heatmap(fc_matrix, ax=ax, cmap='RdBu_r', vmin=-1, vmax=1,
                    xticklabels=labels, yticklabels=labels, square=True)
    else:
        im = ax.imshow(fc_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
    ax.set_title(title)
    return fig, ax


def plot_delta_fc(fc1, fc2, labels=None, ax=None):
    """Heatmap of ΔFC = fc1 - fc2."""
    delta = fc1 - fc2
    return plot_fc_heatmap(delta, labels=labels, ax=ax, title='ΔFC')
