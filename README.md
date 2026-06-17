# Lindblad Neural Operators for Inverse Dynamical Imaging of Stochastic Poisson–Nernst–Planck Ion Transport in Neuronal Systems
> [!NOTE]
> **Project:** *An advanced physics-informed operator-learning framework mapping non-equilibrium electrodiffusive nanoscale transport processes.*

<div align="center">

[![Colab Notebook](https://img.shields.io/badge/Run%20in-Google%20Colab-orange?logo=googlecolab)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Domain](https://img.shields.io/badge/Domain-Scientific%20ML%20%7C%20Computational%20Physics-black)]()

</div>

<p align="center"> . . . </p>

### 01. ABSTRACT

Ion transport in neuronal nanosystems is governed by highly stochastic, thermally coupled, and dissipative dynamics. At these scales, transport is perturbed by thermal noise and governed by irreversible, open-system thermodynamic environments. Classical solvers struggle with this multiscale stiffness, and standard operators (like FNO) fail under noise, suffering from catastrophic spectral explosions.

The **Lindblad Neural Operator (LNO)** provides a new paradigm. By embedding open-system Lindblad master equations into a continuous function space architecture, the LNO uses a dissipative manifold $D(\rho)$ and $\gamma$ coupling coefficients to enforce thermodynamic boundedness. This enables stable forward simulation and precise inverse dynamical imaging.

<p align="center"> . . . </p>

### 02. CORE NOVELTY

Standard operators learn unconstrained mappings $u_{t+1} = \mathcal{G}_\theta(u_t)$. We introduce **Physics-Constrained Operator Evolution**:

$$u_{t+1} = \mathcal{G}_\theta^{\mathrm{diss}}(u_t, \mathcal{L})$$

The dissipative generator $\mathcal{L}$ restricts network transformations to physically valid contractive semigroups:

* **Spectral Damping:** Actively suppresses high-frequency modes, dampening them back into the bounded complex unit circle.
* **Strict Energy Boundedness:** Solves the exploding gradient problem endemic to autoregressive unrolling.
* **Entropy-Aware Dynamics:** Ensures local entropy production remains consistent with the Second Law of Thermodynamics.

<p align="center"> . . . </p>

### 03. INVERSE DYNAMICAL IMAGING

This framework acts as a **computational dynamical imaging instrument**. By extracting latent macroscopic phenomena from noisy, stochastic nanoscale data, the LNO allows researchers to infer:

* **Dissipation Hotspots:** Spatial coordinates where the system faces maximal energy loss.
* **Instability Emergence:** Detection of critical transport bottlenecks.
* **Metastable Transitions:** Identification of transient structural shifts during ionic channel gating.

<p align="center"> . . . </p>

### 04. TELEMETRY & BENCHMARKS

The LNO architecture outperforms classical baselines (FNO/Koopman) in stability and accuracy.

<p align="center"> . . . </p>

### 05. STRUCTURAL METRICS

| Architecture | RMSE ($\downarrow$) | MAE ($\downarrow$) | Mass Error ($\downarrow$) | Spectral Energy |
| --- | --- | --- | --- | --- |
| **FNO (Baseline)** | 0.573130 | 0.520822 | 66.665235 | 0.220555 |
| **Koopman** | 0.604441 | 0.554383 | 70.960995 | 0.194133 |
| **LNO (Ours)** | **0.001233** | **0.000539** | **0.049419** | **86.951497** |

#### Ablation Matrix

| Configuration | Validation RMSE ($\downarrow$) | Entropy Error ($\downarrow$) |
| --- | --- | --- |
| **Full LNO Framework** | **0.001233** | **0.031984** |
| Ablated Dissipation | 0.063966 | 0.769123 |
| Ablated Coupling ($\gamma$) | 0.002259 | 0.075911 |

#### Long-Horizon Stability ($N=40$)

| Model | Energy Norm Drift ($\Delta E$) | Stability Status |
| --- | --- | --- |
| **LNO (Ours)** | **$< 10^{-6}$** | **Stable Equilibrium** |
| FNO (Standard) | $> 0.299900$ | Catastrophic Collapse |

> **Robustness:** LNO maintains monotonic, bounded scaling even under extreme OOD thermal noise ($\sigma = 1.500$).

<p align="center"> . . . </p>

### 06. PIPELINE EXECUTION

#### Setup

```bash
git clone https://github.com/Yuanli-AI-Research/lindblad-neural-operators.git
cd lindblad-neural-operators
pip install -r requirements.txt

```
<p align="center"> . . . </p>

#### Experimental Workflow

1. **`01-data-gen.ipynb`**: Generate synthetic PNP trajectories.
2. **`02-baselines.ipynb`**: Train reference FNO and Koopman models.
3. **`03-train-lno-ipynb.ipynb`**: Execute LNO training and kernel convergence.
4. **`04-analysis-and-experiments.ipynb`**: Run ablation and OOD stress tests.
5. **`05-results-visualization.ipynb`**: Render high-DPI publication plots.

*To execute the full pipeline:* `python run_pipeline.py`

<p align="center"> . . . </p>

### 07. CONTRIBUTING

> [!WARNING]
> **Open for Contributions**
> We welcome contributions for adaptive data-driven jump operators or multi-scale grid alignments. Please open a Pull Request or create an Issue to discuss potential research extensions.

### 08. CITATION

> [!CAUTION]
> **Cite this research**
> If you utilize this framework in your research, please cite:
> 
> ```bibtex
> @article{lno2026transport,
>   title={Lindblad Neural Operators for Inverse Dynamical Imaging of Stochastic Poisson–Nernst–Planck Ion Transport in Neuronal Systems},
>   author={Naman Dixit},
>   journal={GitHub repository},
>   year={2026},
>   publisher={GitHub}
> }
> ```

### 09. REFERENCES & FOUNDATIONAL LITERATURE

> [!TIP]
>
> ### 1. Neural Operators for Partial Differential Equations
>
> **Reference Source:**
> *Neural Operator for PDE.pdf*
>
> ```bibtex
> @article{neural_operators_pde,
>   title={Neural Operator for Partial Differential Equations},
>   note={As referenced in foundational architecture; file: Neural Operator for PDE.pdf},
>   year={2026}
> }
> ```
>
> **Contribution to this Framework:**
> Establishes the theoretical foundation for learning mappings between infinite-dimensional function spaces through neural operators. These concepts serve as the architectural backbone for operator-learning components and PDE surrogate modeling within the framework.


<p align="center"> . . . </p>

> [!TIP]
>
> ### 2. Poisson–Nernst–Planck Electrodiffusion Theory
>
> **Reference Source:**
> *Possion Nerst Plank Equation for Biomolecular Diffusion.pdf*
>
> ```bibtex
> @article{pnp_biomolecular,
>   title={Poisson-Nernst-Planck Equation for Biomolecular Diffusion},
>   note={Governing electrodiffusive transport equations; file: Possion Nerst Plank Equation for Biomolecular Diffusion.pdf},
>   year={2026}
> }
> ```
>
> **Contribution to this Framework:**
> Provides the governing equations for ionic transport, charge conservation, and electrodiffusive dynamics in biological systems. The Poisson–Nernst–Planck formulation forms the basis for modeling spatial ion concentration evolution and membrane-associated transport processes.

<p align="center"> . . . </p>

> [!TIP]
>
> ### 3. Neuronal Membrane Ion Transport Dynamics
>
> **Reference Source:**
> *Screenshot (497).jpg*
> *Screenshot (498).png*
>
> ```bibtex
> @article{xiang2017model,
>   title={A model of ion transport processes along and across the neuronal membrane},
>   author={Zuoxian, Xiang and Liu, G. Z. and Tang, C. X. and Yan, L. X.},
>   journal={Journal of Integrative Neuroscience},
>   volume={16},
>   number={1},
>   pages={33--55},
>   year={2017},
>   doi={10.3233/JIN-160002},
>   note={Visual context preserved in Screenshot (497).jpg and Screenshot (498).png}
> }
> ```
>
> **Contribution to this Framework:**
> Provides empirical and mathematical descriptions of transmembrane ion transport, intracellular and extracellular diffusion pathways, and stochastic ionic dynamics. These results inform the biological constraints incorporated into the electrodiffusive neural operator formulation.

<p align="center"> . . . </p>

> [!TIP]
>
> ### 4. Open Quantum Systems & Lindblad Dynamics
>
> **Reference Source:**
> *Lindblad master equation approach to superconductivity in open quantum systems.pdf*
>
> ```bibtex
> @article{lindblad_open_systems,
>   title={Lindblad master equation approach to superconductivity in open quantum systems},
>   note={Mathematical framework for dissipative generators; file: Lindblad master equation approach to superconductivity in open quantum systems.pdf},
>   year={2026}
> }
> ```
>
> **Contribution to this Framework:**
> Introduces the mathematical framework of open quantum systems, density matrix evolution, dissipative coupling mechanisms, and contractive semigroup dynamics through Lindblad master equations. These principles motivate the framework's treatment of dissipation, stability, and physically constrained evolution operators.

### 10. LICENSE

> [!IMPORTANT]
> **MIT License**
>
> This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
