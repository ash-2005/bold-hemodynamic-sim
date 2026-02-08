# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.4.0] — 2026-02-15

### Added
- `src/bifurcation.py`: G-coupling bifurcation sweep, `find_g_optimal`, `find_g_max_delta`, dual-axis plot, delay sensitivity analysis.
- `cli.py`: Full command-line interface with `simulate`, `sweep`, `bifurcation`, and `fc` subcommands.
- `benchmark.py`: 100-region × 25 G-value benchmark with distance-decay coupling matrix, JSON timing report.
- `notebooks/bw_walkthrough.ipynb`: 9-cell interactive walkthrough of BW model.
- `notebooks/delay_fc_demo.ipynb`: 7-cell delay-FC demonstration notebook.

### Changed
- `src/visualize.py`: Added shared style configuration and FC heatmap utilities.
- `README.md`: Final polish with verified CLI examples and project structure tree.

---

## [0.3.0] — 2026-02-05

### Added
- `src/fc_from_bold.py`: Pearson FC, bandpass filtering (Butterworth order 3), `fc_legacy`, `fc_delay_corrected`, `fc_difference`, `pearson_r_fc`, `fc_summary`.
- `src/param_sweep.py`: `sweep_single_param`, `sweep_all_params`, `extract_bold_features`, `plot_sweep_grid` with tqdm progress bars.
- `src/visualize.py`: Shared Matplotlib style configuration and plotting utilities.
- `tests/test_fc.py`: 5 tests for FC symmetry, diagonal, delay correction, zero-delay identity, pearson_r.
- `tests/test_param_sweep.py`: 4 tests for sweep count, kappa monotonicity, feature key coverage, plot output.

---

## [0.2.0] — 2026-01-22

### Added
- `src/hrf.py`: `double_gamma_hrf` (Glover 1999), `canonical_spm_hrf`, `shifted_hrf`, `hrf_matrix`, `convolve_hrf`.
- `src/delay_inject.py`: `inject_delays_hrf`, `inject_delays_bw`, `apply_global_coupling`, `delay_vector_from_rapidtide`.
- `tests/test_hrf.py`: 5 tests for normalisation, shifted peak, matrix shape, convolution length, known values.
- `tests/test_delay_inject.py`: 5 tests for output shape, zero-delay identity, 2-second shift, G=0, BW vs HRF.

### Changed
- `src/bw_model.py`: Added `downsample` function.

---

## [0.1.0] — 2026-01-10

### Added
- `src/bw_model.py`: Full Friston 2003 Balloon-Windkessel ODE implementation. `BWParams`, `BWState`, `BWResult` dataclasses. `bold_signal`, `bw_ode`, `simulate` (SciPy RK45), `check_physiological_bounds`.
- `src/neural_generator.py`: `generate_event_related`, `generate_block_design`, `generate_coupled_oscillators`, `generate_from_coupling_matrix`, `add_neural_noise`.
- `tests/test_bw_model.py`: 8 tests.
- `requirements.txt`, `setup.py`, `README.md`, `.gitignore`.

### Scientific foundation
- Friston et al. (2003) — primary BW model reference for all parameters and equations.
- Buxton et al. (1997) — original balloon model, v and q dynamics.
- Glover (1999) — double-gamma HRF parameterisation.
