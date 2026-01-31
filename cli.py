"""CLI for bold-hemodynamic-sim.

Subcommands:
  simulate    — Run a single BW simulation and save BOLD output.
  sweep       — Run parameter sensitivity sweep.
  bifurcation — Run G-coupling bifurcation sweep.
  fc          — Compute FC from an existing BOLD .npy file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _simulate(args):
    from src.bw_model import BWParams, simulate, downsample
    from src.neural_generator import generate_event_related, generate_block_design, generate_coupled_oscillators

    params = BWParams(
        kappa=args.kappa, gamma=args.gamma, tau=args.tau,
        alpha=args.alpha, E0=args.E0, V0=args.V0,
    )
    n = int(args.T / args.dt)
    if args.neural_type == 'event_related':
        neural = generate_event_related(3, 10.0, 2.0, args.T, args.dt)
    elif args.neural_type == 'block':
        neural = generate_block_design(15.0, 15.0, 3, args.T, args.dt)
    else:
        C = np.eye(1)
        neural = generate_coupled_oscillators(1, args.T, args.dt, C, np.array([0.05]))[0]

    result = simulate(neural, args.dt, args.T, params)
    ds_bold = downsample(result.bold, args.dt, args.tr)
    np.save(args.output, ds_bold)
    print(f"Saved BOLD ({len(ds_bold)} TRs) to {args.output}")


def _sweep(args):
    from src.bw_model import BWParams
    from src.neural_generator import generate_event_related
    from src.param_sweep import sweep_single_param, plot_sweep_grid

    params = BWParams()
    neural = generate_event_related(3, 10.0, 2.0, args.T, args.dt)
    base = getattr(params, args.param)
    values = np.linspace(base * 0.5, base * 1.5, args.n_points)
    results = sweep_single_param(args.param, values, params, neural, args.dt, args.T)

    os.makedirs(args.output, exist_ok=True)
    plot_sweep_grid({args.param: results}, os.path.join(args.output, 'sweep_grid.png'))
    print(f"Sweep complete. Figures saved to {args.output}/")


def _bifurcation(args):
    from src.bw_model import BWParams
    from src.bifurcation import g_sweep, compute_model_fit, find_g_optimal, plot_g_sweep

    params = BWParams()
    n = args.n_regions
    dist = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(float)
    C = np.exp(-dist / 3.0)
    np.fill_diagonal(C, 0)

    fc_emp = np.eye(n) + 0.1 * np.random.randn(n, n)
    fc_emp = (fc_emp + fc_emp.T) / 2
    np.fill_diagonal(fc_emp, 1.0)

    G_values = np.logspace(np.log10(args.G_min), np.log10(args.G_max), args.n_G)
    fc_dict = g_sweep(G_values, C, n, args.T, args.dt, params, args.tr)
    fit = compute_model_fit(G_values, fc_dict, fc_emp)
    G_opt = find_g_optimal(fit)

    os.makedirs(args.output, exist_ok=True)
    plot_g_sweep(fit, os.path.join(args.output, 'g_sweep.png'))
    print(f"Optimal G: {G_opt:.4f}. Figures saved to {args.output}/")


def _fc(args):
    from src.fc_from_bold import compute_fc, fc_delay_corrected

    bold = np.load(args.bold_file)
    if bold.ndim == 1:
        bold = bold[np.newaxis]
    if args.delay_file and os.path.exists(args.delay_file):
        delays = np.load(args.delay_file)
        fc_mat = fc_delay_corrected(bold, delays, tr=args.tr)
    else:
        fc_mat = compute_fc(bold, tr=args.tr)
    np.save(args.output, fc_mat)
    print(f"FC matrix shape {fc_mat.shape} saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description='bold-hemodynamic-sim CLI')
    sub = parser.add_subparsers(dest='command')

    # simulate
    p_sim = sub.add_parser('simulate', help='Run BW simulation')
    p_sim.add_argument('--T', type=float, default=60.0)
    p_sim.add_argument('--dt', type=float, default=0.001)
    p_sim.add_argument('--tr', type=float, default=2.0)
    p_sim.add_argument('--kappa', type=float, default=0.65)
    p_sim.add_argument('--gamma', type=float, default=0.41)
    p_sim.add_argument('--tau', type=float, default=0.98)
    p_sim.add_argument('--alpha', type=float, default=0.32)
    p_sim.add_argument('--E0', type=float, default=0.34)
    p_sim.add_argument('--V0', type=float, default=0.02)
    p_sim.add_argument('--neural_type', default='event_related')
    p_sim.add_argument('--output', default='bold_out.npy')

    # sweep
    p_sw = sub.add_parser('sweep', help='Parameter sensitivity sweep')
    p_sw.add_argument('--param', default='kappa')
    p_sw.add_argument('--n_points', type=int, default=20)
    p_sw.add_argument('--T', type=float, default=60.0)
    p_sw.add_argument('--dt', type=float, default=0.001)
    p_sw.add_argument('--output', default='sweep_results')

    # bifurcation
    p_bif = sub.add_parser('bifurcation', help='G-coupling bifurcation sweep')
    p_bif.add_argument('--G_min', type=float, default=0.01)
    p_bif.add_argument('--G_max', type=float, default=10.0)
    p_bif.add_argument('--n_G', type=int, default=25)
    p_bif.add_argument('--n_regions', type=int, default=10)
    p_bif.add_argument('--T', type=float, default=120.0)
    p_bif.add_argument('--dt', type=float, default=0.001)
    p_bif.add_argument('--tr', type=float, default=2.0)
    p_bif.add_argument('--output', default='bifurcation_out')

    # fc
    p_fc = sub.add_parser('fc', help='Compute FC from BOLD .npy file')
    p_fc.add_argument('--bold_file', required=True)
    p_fc.add_argument('--tr', type=float, default=2.0)
    p_fc.add_argument('--delay_file', default=None)
    p_fc.add_argument('--output', default='fc_matrix.npy')

    args = parser.parse_args()
    if args.command == 'simulate':
        _simulate(args)
    elif args.command == 'sweep':
        _sweep(args)
    elif args.command == 'bifurcation':
        _bifurcation(args)
    elif args.command == 'fc':
        _fc(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
