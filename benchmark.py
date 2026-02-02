"""
benchmark.py — Full simulation benchmark for bold-hemodynamic-sim.

Runs BW simulation across 100 regions × 25 G values.
Outputs per-step timing as JSON and prints a summary table.

Usage:
    python benchmark.py [--output benchmark_results.json]
"""

import time
import json
import argparse
import numpy as np

from src.bw_model import BWParams, downsample
from src.neural_generator import generate_coupled_oscillators
from src.delay_inject import inject_delays_bw, apply_global_coupling
from src.fc_from_bold import compute_fc, pearson_r_fc

N_REGIONS   = 100
T_SIM       = 300.0
DT          = 0.001
TR          = 2.0
N_G         = 25
G_MIN       = 0.01
G_MAX       = 10.0
LAMBDA_MM   = 30.0


def build_coupling_matrix(n_regions: int, lambda_mm: float) -> np.ndarray:
    positions = np.arange(n_regions, dtype=float)
    dist = np.abs(np.subtract.outer(positions, positions))
    C = np.exp(-dist / lambda_mm)
    np.fill_diagonal(C, 0.0)
    row_sums = C.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return C / row_sums


def build_synthetic_empirical_fc(n_regions: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = n_regions // 2
    fc = rng.uniform(0.0, 0.3, (n_regions, n_regions))
    fc[:half, :half] += rng.uniform(0.3, 0.6, (half, half))
    fc[half:, half:] += rng.uniform(0.3, 0.6, (n_regions - half, n_regions - half))
    fc = (fc + fc.T) / 2.0
    np.fill_diagonal(fc, 1.0)
    return np.clip(fc, -1.0, 1.0)


def run_benchmark(output_path: str) -> dict:
    print("=" * 60)
    print("bold-hemodynamic-sim Benchmark")
    print(f"  Regions: {N_REGIONS}, Duration: {T_SIM}s, G values: {N_G}")
    print("=" * 60)

    params = BWParams()
    t0 = time.perf_counter()
    C = build_coupling_matrix(N_REGIONS, LAMBDA_MM)
    print(f"\n[1/4] Coupling matrix built in {(time.perf_counter()-t0)*1000:.1f} ms")

    t0 = time.perf_counter()
    freq_vector = np.full(N_REGIONS, 0.05)
    neural_base = generate_coupled_oscillators(N_REGIONS, T_SIM, DT, C, freq_vector, noise_std=0.01)
    print(f"[2/4] Neural signal generated in {time.perf_counter()-t0:.2f}s")

    fc_empirical = build_synthetic_empirical_fc(N_REGIONS)
    print(f"[3/4] Synthetic empirical FC built")

    G_values = np.logspace(np.log10(G_MIN), np.log10(G_MAX), N_G)
    delays = np.zeros(N_REGIONS)
    r_values, timing_per_G, bold_time_per_G, fc_time_per_G = [], [], [], []

    print(f"\n[4/4] Running G sweep ({N_G} iterations)...\n")
    total_start = time.perf_counter()

    for G in G_values:
        iter_start = time.perf_counter()
        neural_coupled = apply_global_coupling(neural_base, G, C)
        t_bw_start = time.perf_counter()
        bold_full = inject_delays_bw(neural_coupled, delays, params, DT, T_SIM)
        t_bw = time.perf_counter() - t_bw_start
        bold_ds = np.stack([downsample(bold_full[i], DT, TR) for i in range(N_REGIONS)])
        t_fc_start = time.perf_counter()
        fc_sim = compute_fc(bold_ds, bandpass_low=0.01, bandpass_high=0.1, tr=TR)
        t_fc = time.perf_counter() - t_fc_start
        r = pearson_r_fc(fc_sim, fc_empirical)
        iter_time = time.perf_counter() - iter_start
        r_values.append(float(r))
        timing_per_G.append(float(iter_time))
        bold_time_per_G.append(float(t_bw))
        fc_time_per_G.append(float(t_fc))
        print(f"  G={G:7.4f}  r={r:7.4f}  BW={t_bw:7.2f}s  FC={t_fc*1000:6.1f}ms")

    total_time = time.perf_counter() - total_start
    optimal_idx = int(np.argmax(r_values))
    print(f"\nTotal: {total_time:.1f}s | Optimal G: {G_values[optimal_idx]:.4f} | Max r: {max(r_values):.4f}")

    results = {
        "config": {"n_regions": N_REGIONS, "T_sim": T_SIM, "dt": DT, "tr": TR,
                   "n_G": N_G, "G_min": G_MIN, "G_max": G_MAX},
        "G_values": [float(g) for g in G_values],
        "r_values": r_values,
        "timing_per_G": timing_per_G,
        "total_time": float(total_time),
        "optimal_G": float(G_values[optimal_idx]),
        "max_r": float(max(r_values)),
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bold-hemodynamic-sim performance benchmark")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()
    run_benchmark(args.output)
