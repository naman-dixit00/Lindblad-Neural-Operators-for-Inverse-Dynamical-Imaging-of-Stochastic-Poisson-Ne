# Lindblad Neural Operators for Inverse Dynamical Imaging of Stochastic Poisson–Nernst–Planck Ion Transport in Neuronal Systems

**Project:** *An advanced physics-informed operator-learning framework mapping non-equilibrium electrodiffusive nanoscale transport processes.*

<div align="center">

[![Colab Notebook](https://img.shields.io/badge/Run%20in-Google%20Colab-orange?logo=googlecolab)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Domain](https://img.shields.io/badge/Domain-Scientific%20ML%20%7C%20Computational%20Physics-black)]()

</div>



## Abstract

Ion transport in neuronal nanosystems is governed by highly stochastic, thermally coupled, and dissipative dynamics. At these scales, transport is perturbed by thermal noise and governed by irreversible, open-system thermodynamic environments. Classical solvers struggle with this multiscale stiffness, and standard operators (like FNO) fail under noise, suffering from catastrophic spectral explosions.

The **Lindblad Neural Operator (LNO)** provides a new paradigm. By embedding open-system Lindblad master equations into a continuous function space architecture, the LNO uses a dissipative manifold $D(\rho)$ and $\gamma$ coupling coefficients to enforce thermodynamic boundedness. This enables stable forward simulation and precise inverse dynamical imaging.


## Core Novelty

Standard operators learn unconstrained mappings $u_{t+1} = \mathcal{G}_\theta(u_t)$. We introduce **Physics-Constrained Operator Evolution**:

$$u_{t+1} = \mathcal{G}_\theta^{\mathrm{diss}}(u_t, \mathcal{L})$$

The dissipative generator $\mathcal{L}$ restricts network transformations to physically valid contractive semigroups:

* **Spectral Damping:** Actively suppresses high-frequency modes, dampening them back into the bounded complex unit circle.
* **Strict Energy Boundedness:** Solves the exploding gradient problem endemic to autoregressive unrolling.
* **Entropy-Aware Dynamics:** Ensures local entropy production remains consistent with the Second Law of Thermodynamics.


## Inverse Dynamical Imaging

This framework acts as a **computational dynamical imaging instrument**. By extracting latent macroscopic phenomena from noisy, stochastic nanoscale data, the LNO allows researchers to infer:

* **Dissipation Hotspots:** Spatial coordinates where the system faces maximal energy loss.
* **Instability Emergence:** Detection of critical transport bottlenecks.
* **Metastable Transitions:** Identification of transient structural shifts during ionic channel gating.



## Telemetry & Benchmarks

The LNO architecture outperforms classical baselines (FNO/Koopman) in stability and accuracy.

### Structural Metrics

| Architecture | RMSE ($\downarrow$) | MAE ($\downarrow$) | Mass Error ($\downarrow$) | Spectral Energy |
| --- | --- | --- | --- | --- |
| **FNO (Baseline)** | 0.573130 | 0.520822 | 66.665235 | 0.220555 |
| **Koopman** | 0.604441 | 0.554383 | 70.960995 | 0.194133 |
| **LNO (Ours)** | **0.001233** | **0.000539** | **0.049419** | **86.951497** |

### Ablation Matrix

| Configuration | Validation RMSE ($\downarrow$) | Entropy Error ($\downarrow$) |
| --- | --- | --- |
| **Full LNO Framework** | **0.001233** | **0.031984** |
| Ablated Dissipation | 0.063966 | 0.769123 |
| Ablated Coupling ($\gamma$) | 0.002259 | 0.075911 |

### Long-Horizon Stability ($N=40$)

| Model | Energy Norm Drift ($\Delta E$) | Stability Status |
| --- | --- | --- |
| **LNO (Ours)** | **$< 10^{-6}$** | **Stable Equilibrium** |
| FNO (Standard) | $> 0.299900$ | Catastrophic Collapse |

> **Robustness:** LNO maintains monotonic, bounded scaling even under extreme OOD thermal noise ($\sigma = 1.500$).



## Pipeline Execution

### Setup

```bash
git clone https://github.com/Yuanli-AI-Research/lindblad-neural-operators.git
cd lindblad-neural-operators
pip install -r requirements.txt

```

### Experimental Workflow

1. **`01-data-gen.ipynb`**: Generate synthetic PNP trajectories.
2. **`02-baselines.ipynb`**: Train reference FNO and Koopman models.
3. **`03-train-lno-ipynb.ipynb`**: Execute LNO training and kernel convergence.
4. **`04-analysis-and-experiments.ipynb`**: Run ablation and OOD stress tests.
5. **`05-results-visualization.ipynb`**: Render high-DPI publication plots.

*To execute the full pipeline:* `python run_pipeline.py`



## Contributing

We welcome contributions for adaptive data-driven jump operators or multi-scale grid alignments. Please open a Pull Request or create an Issue to discuss potential research extensions.

## Citation

If you utilize this framework in your research, please cite:

```bibtex
@article{yuanli2026lno,
  title={Lindblad Neural Operators for Inverse Dynamical Imaging of Stochastic Poisson–Nernst–Planck Ion Transport in Neuronal Systems},
  author={Naman Dixit},
  journal={GitHub repository},
  year={2026},
  publisher={GitHub}
}

```
