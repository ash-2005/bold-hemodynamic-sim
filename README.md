# bold-hemodynamic-sim

**A standalone Balloon-Windkessel haemodynamic simulator and parameter analysis toolkit — no TVB dependency, pure Python, built from first principles using Friston et al. (2003).**

---

## Motivation

Before extending The Virtual Brain (TVB) to model haemodynamic delays in fMRI signals, I needed to understand what the BOLD signal actually *is* at the level of the physics. TVB abstracts this away behind a monitor class — it hands you BOLD without showing you the ODEs. That wasn't good enough for me.

So I built this. I took Friston et al. (2003), sat with the four differential equations, implemented them from scratch using SciPy's `solve_ivp`, and then built everything around them: parameter sweeps, delay injection, FC computation, and a global coupling bifurcation analysis. No framework, no shortcuts.

---

## Features

- **Full Friston 2003 BW ODE integration** using SciPy `solve_ivp` with RK45 adaptive stepping
- **Configurable parameters**: κ, γ, τ, α, E₀, V₀
- **Neural signal generators**: event-related, block design, coupled oscillators, noise-modulated
- **HRF library**: double-gamma (Glover 1999), canonical SPM HRF, per-region delay-shifted HRFs
- **Delay injection**: per-region haemodynamic onset offsets via BW integration or HRF convolution
- **FC computation**: Pearson correlation, bandpass filtering, delay-corrected FC, ΔFC visualisation
- **Parameter sensitivity sweep**: sweeps each BW parameter, extracts BOLD features
- **Bifurcation analysis**: global coupling G sweep, finds optimal G and maximum ΔFC point
- **CLI**: `simulate`, `sweep`, `bifurcation`, `fc` commands
- **Benchmark**: 100-region × 25 G-values timing report
- **2 Jupyter notebooks**: BW walkthrough and delay-FC demonstration
- **27 pytest tests**

---

## Installation

```bash
git clone https://github.com/ash-2005/bold-hemodynamic-sim.git
cd bold-hemodynamic-sim
pip install -e .
```

## Quick Start

```python
import numpy as np
from src.neural_generator import generate_event_related
from src.bw_model import simulate, BWParams

dt, T = 0.001, 30.0
neural = generate_event_related(3, 5.0, 1.0, T, dt)
params = BWParams()
result = simulate(neural, dt, T, params)
print(result.bold)
```

## CLI Usage

```bash
python cli.py simulate --T 60 --dt 0.001 --output bold_out.npy
python cli.py sweep --param kappa --n_points 20 --output sweep_results/
python cli.py bifurcation --G_min 0.01 --G_max 10 --n_G 25 --output bifurcation_out/
python cli.py fc --bold_file bold_matrix.npy --tr 2.0 --output fc_matrix.npy
```

## Scientific References

1. **Friston et al. (2003).** Dynamic causal modelling. *NeuroImage, 19*(4), 1273–1302.
2. **Glover (1999).** Deconvolution of impulse response in event-related BOLD fMRI. *NeuroImage, 9*(4), 416–429.
3. **Buxton et al. (1997).** Dynamics of blood flow and oxygenation. *MRM, 39*(6), 855–864.

## License

MIT License.
