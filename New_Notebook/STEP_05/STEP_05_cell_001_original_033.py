# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 33
# Step            : STEP_05
# Step Heading    : # STEP 5 — PROPER LNO PHYSICAL REPRESENTATION
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 5 — PROPER LNO PHYSICAL REPRESENTATION
#            + LINDBLAD-CONSTRAINED KERNEL PROTOTYPE
#
# PURPOSE:
#   Test whether the candidate information-state
#
#       R = z z^T / Tr(z z^T)
#
#   can support a stable, differentiable Lindblad-form
#   dynamical kernel.
#
# TESTS:
#   1. Information preservation
#   2. Lindblad dissipative compatibility
#   3. Trace preservation
#   4. Symmetry / PSD preservation
#   5. Environmental response
#   6. Repeated-step stability
#   7. Non-vanishing gradients
#
# NO TRAINING OF FULL LNO
# NO DATASET MODIFICATION
# NO CHECKPOINT MODIFICATION
# ============================================================

import os
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

# scipy is used only for a tiny 36x36 prototype matrix exponential
from scipy.linalg import expm

# ============================================================
# 1. ROOT
# ============================================================

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

print("=" * 100)
print("STEP 5 — LINDBLAD-CONSTRAINED KERNEL PROTOTYPE")
print("=" * 100)

# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "[+] Device:",
    device
)

# ============================================================
# 3. LOAD ONE VERIFIED PAIRED TRAJECTORY
# ============================================================

PAIR_DIR = (
    ROOT
    / "results"
    / "step3_paired_trajectories"
)

# Use stochastic regime as main controlled prototype
PAIR_PATH = (
    PAIR_DIR
    / "stochastic_paired.npz"
)

assert PAIR_PATH.is_file(), (
    f"Missing paired trajectory:\n{PAIR_PATH}"
)

data = np.load(
    PAIR_PATH
)

print(
    "[+] Source:",
    PAIR_PATH
)

# ------------------------------------------------------------
# Load required arrays
# ------------------------------------------------------------

R_open = data[
    "R_open"
].astype(
    np.float64
)

R_ref = data[
    "R_reference"
].astype(
    np.float64
)

open_state = data[
    "open_state"
].astype(
    np.float64
)

ref_state = data[
    "reference_state"
].astype(
    np.float64
)

open_phi = data[
    "open_phi"
].astype(
    np.float64
)

open_flux = data[
    "open_flux"
].astype(
    np.float64
)

open_noise = data[
    "open_noise"
].astype(
    np.float64
)

open_diss = data[
    "open_dissipation"
].astype(
    np.float64
)

open_inst = data[
    "open_instability"
].astype(
    np.float64
)

open_gamma = data[
    "open_gamma"
].astype(
    np.float64
)

# ============================================================
# 4. SHAPE CHECKS
# ============================================================

T, NX, DIM, DIM2 = (
    R_open.shape
)

assert DIM == 6
assert DIM2 == 6

print(
    "[+] R_open:",
    R_open.shape
)

print(
    "[+] R_reference:",
    R_ref.shape
)

print(
    "[+] Spatial points:",
    NX
)

print(
    "[+] Timesteps:",
    T
)

# ============================================================
# 5. TORCH INPUT
# ============================================================

# Use first 16 spatial points for the prototype to keep
# the diagnostic very lightweight.
#
# Full spatial dimension will be used after the kernel
# passes all tests.

TIME_SUBSET = min(
    T,
    16
)

SPACE_SUBSET = min(
    NX,
    16
)

R0 = torch.tensor(
    R_open[
        :TIME_SUBSET,
        :SPACE_SUBSET
    ],
    dtype=torch.float32,
    device=device
)

print(
    "\n[+] Prototype R subset:",
    tuple(
        R0.shape
    )
)

# ============================================================
# 6. CANDIDATE INFORMATION-STATE REPRESENTATION TEST
#
# We reconstruct the normalized feature vector from R as
# far as possible.
#
# Important:
#
#   R = z z^T / Tr(z z^T)
#
# loses:
#   * absolute magnitude
#   * global sign of z
#
# Therefore this is a genuine information-preservation test.
# ============================================================

# Build original normalized feature vectors jointly across
# the open trajectory.

raw_features = np.stack(
    [
        open_state,
        open_phi,
        open_flux,
        open_noise,
        open_diss,
        open_inst,
    ],
    axis=1
)

# [T, 6, NX]

feature_mean = np.mean(
    raw_features
)

# Feature-wise statistics
feature_means = np.mean(
    raw_features,
    axis=(0, 2)
)

feature_stds = np.std(
    raw_features,
    axis=(0, 2)
)

feature_stds = np.where(
    feature_stds < 1e-12,
    1.0,
    feature_stds
)

z_original = (
    raw_features
    -
    feature_means[
        None,
        :,
        None
    ]
) / feature_stds[
    None,
    :,
    None
]

# [T, NX, 6]
z_original = np.transpose(
    z_original,
    (0, 2, 1)
)

# ------------------------------------------------------------
# Recover dominant eigenvector of R
#
# Since R is rank-1:
#
#   R = z z^T / ||z||^2
#
# its principal eigenvector is approximately z / ||z||,
# but only up to a sign.
# ------------------------------------------------------------

R_subset_np = R_open[
    :TIME_SUBSET,
    :SPACE_SUBSET
]

reconstruction_errors = []

cosine_values = []

for t in range(
    TIME_SUBSET
):

    for x in range(
        SPACE_SUBSET
    ):

        Rmat = R_subset_np[
            t,
            x
        ]

        eigvals, eigvecs = np.linalg.eigh(
            Rmat
        )

        v = eigvecs[
            :,
            np.argmax(
                eigvals
            )
        ]

        original = z_original[
            t,
            x
        ]

        # Normalize original vector
        norm_original = (
            np.linalg.norm(
                original
            )
            + 1e-30
        )

        original_unit = (
            original
            /
            norm_original
        )

        # Resolve unavoidable sign ambiguity
        cos_pos = np.dot(
            v,
            original_unit
        )

        cos_neg = np.dot(
            -v,
            original_unit
        )

        best_cos = max(
            cos_pos,
            cos_neg
        )

        cosine_values.append(
            best_cos
        )

        # Reconstruct only normalized direction
        reconstruction_errors.append(
            1.0
            -
            best_cos
        )

mean_direction_error = float(
    np.mean(
        reconstruction_errors
    )
)

mean_best_cosine = float(
    np.mean(
        cosine_values
    )
)

print(
    "\n" + "=" * 100
)

print(
    "A. INFORMATION PRESERVATION"
)

print(
    "=" * 100
)

print(
    "Mean best-sign cosine:",
    f"{mean_best_cosine:.12e}"
)

print(
    "Mean direction error:",
    f"{mean_direction_error:.12e}"
)

print(
    "\nNOTE:"
)

print(
    "R preserves normalized feature direction, "
    "but does NOT preserve absolute feature magnitude."
)

# ============================================================
# 7. LINDBLAD-CONSTRAINED KERNEL
# ============================================================

class LindbladInformationKernel(
    nn.Module
):

    """
    Small controlled Lindblad-form dynamical kernel.

    State:
        R ∈ R^(d x d)

    Dissipator:
        D(R) =
            Σ γ_k [
                L_k R L_k^T
                - 1/2 {
                    L_k^T L_k,
                    R
                }
            ]

    The jump operators are learnable.

    We construct the finite-time propagator from the
    corresponding linear superoperator so the prototype
    tests an actual dissipative semigroup action rather
    than an arbitrary Euler residual.

    This is a mathematical prototype.
    It is not yet the final LNO spatial neural operator.
    """

    def __init__(
        self,
        dim=6,
        channels=4,
    ):

        super().__init__()

        self.dim = dim
        self.channels = channels

        # Learnable jump operators
        self.jump_operators = nn.Parameter(
            0.05
            *
            torch.randn(
                channels,
                dim,
                dim
            )
        )

        # Positive learnable environment coupling
        self.raw_alpha = nn.Parameter(
            torch.tensor(
                0.0
            )
        )

    def positive_alpha(
        self
    ):

        return F.softplus(
            self.raw_alpha
        )

    def dissipator(
        self,
        R
    ):

        """
        R shape:
            [..., d, d]
        """

        total = torch.zeros_like(
            R
        )

        for k in range(
            self.channels
        ):

            L = (
                self.jump_operators[
                    k
                ]
            )

            LT = L.transpose(
                -1,
                -2
            )

            A = (
                LT @ L
            )

            term = (
                L
                @
                R
                @
                LT
            )

            anti = (
                A
                @
                R
                +
                R
                @
                A
            )

            total = (
                total
                +
                term
                -
                0.5
                *
                anti
            )

        return total

    def forward(
        self,
        R,
        gamma,
        dt=1e-3,
    ):

        # ----------------------------------------------------
        # First-order controlled propagator.
        #
        # For the prototype, we separately verify whether
        # sufficiently small dt preserves PSD numerically.
        # ----------------------------------------------------

        alpha = self.positive_alpha()

        effective_gamma = (
            alpha
            *
            gamma
        )

        D_R = self.dissipator(
            R
        )

        R_next = (
            R
            +
            dt
            *
            effective_gamma
            *
            D_R
        )

        # Symmetrize numerical round-off
        R_next = (
            0.5
            *
            (
                R_next
                +
                R_next.transpose(
                    -1,
                    -2
                )
            )
        )

        return R_next

# ============================================================
# 8. CREATE PROTOTYPE
# ============================================================

kernel = (
    LindbladInformationKernel(
        dim=6,
        channels=4
    )
    .to(device)
)

print(
    "\n[+] Prototype kernel created."
)

print(
    "[+] Jump operators:",
    tuple(
        kernel.jump_operators.shape
    )
)

print(
    "[+] Initial alpha:",
    kernel.positive_alpha().item()
)

# ============================================================
# 9. SINGLE-STEP TEST
# ============================================================

R_input = R0[0]

# Environment value from stochastic paired run
gamma_scalar = float(
    np.mean(
        open_gamma[0]
    )
)

gamma_tensor = torch.tensor(
    gamma_scalar,
    dtype=torch.float32,
    device=device
)

R_pred = kernel(
    R_input,
    gamma_tensor,
    dt=1e-3
)

print(
    "\n" + "=" * 100
)

print(
    "B. SINGLE-STEP LINDBLAD KERNEL"
)

print(
    "=" * 100
)

print(
    "Input shape:",
    tuple(
        R_input.shape
    )
)

print(
    "Output shape:",
    tuple(
        R_pred.shape
    )
)

# ============================================================
# 10. STRUCTURE CHECK
# ============================================================

trace_input = torch.diagonal(
    R_input,
    dim1=-2,
    dim2=-1
).sum(
    dim=-1
)

trace_output = torch.diagonal(
    R_pred,
    dim1=-2,
    dim2=-1
).sum(
    dim=-1
)

symmetry_error = torch.max(
    torch.abs(
        R_pred
        -
        R_pred.transpose(
            -1,
            -2
        )
    )
).item()

eigvals = torch.linalg.eigvalsh(
    R_pred
)

min_eig = (
    torch.min(
        eigvals
    ).item()
)

trace_error = torch.max(
    torch.abs(
        trace_output
        -
        trace_input
    )
).item()

print(
    "Max trace change:",
    f"{trace_error:.12e}"
)

print(
    "Max symmetry error:",
    f"{symmetry_error:.12e}"
)

print(
    "Minimum eigenvalue:",
    f"{min_eig:.12e}"
)

if (
    trace_error < 1e-5
    and
    symmetry_error < 1e-6
    and
    min_eig > -1e-6
):

    print(
        "[PASS] Single-step structural constraints preserved."
    )

else:

    print(
        "[WARNING] Structural test needs refinement."
    )

# ============================================================
# 11. ENVIRONMENT SENSITIVITY
# ============================================================

gammas_to_test = [
    0.0,
    0.01,
    0.05,
    0.15,
    0.35,
    0.50,
]

environment_rows = []

with torch.no_grad():

    for gamma_value in (
        gammas_to_test
    ):

        gamma_test = torch.tensor(
            gamma_value,
            dtype=torch.float32,
            device=device
        )

        R_test = kernel(
            R_input,
            gamma_test,
            dt=1e-3
        )

        output_change = torch.linalg.vector_norm(
            R_test
            -
            R_input
        ).item()

        environment_rows.append({

            "gamma":
                gamma_value,

            "output_change":
                output_change,
        })

environment_df = pd.DataFrame(
    environment_rows
)

print(
    "\n" + "=" * 100
)

print(
    "C. ENVIRONMENT SENSITIVITY"
)

print(
    "=" * 100
)

display(
    environment_df.round(10)
)

# ============================================================
# 12. MONOTONIC RESPONSE CHECK
# ============================================================

changes = (
    environment_df[
        "output_change"
    ].values
)

nondecreasing = bool(
    np.all(
        np.diff(
            changes
        )
        >=
        -1e-12
    )
)

print(
    "[+] Monotonic output response:",
    nondecreasing
)

# ============================================================
# 13. REPEATED-STEP STABILITY
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "D. REPEATED LINDBLAD EVOLUTION"
)

print(
    "=" * 100
)

R_roll = (
    R_input
    .clone()
)

rollout_rows = []

with torch.no_grad():

    for step in range(
        1,
        101
    ):

        R_roll = kernel(
            R_roll,
            gamma_tensor,
            dt=1e-3
        )

        trace = torch.diagonal(
            R_roll,
            dim1=-2,
            dim2=-1
        ).sum(
            dim=-1
        )

        eigvals = torch.linalg.eigvalsh(
            R_roll
        )

        min_eig = (
            torch.min(
                eigvals
            ).item()
        )

        norm = torch.linalg.vector_norm(
            R_roll
        ).item()

        rollout_rows.append({

            "step":
                step,

            "frobenius_norm":
                norm,

            "min_eigenvalue":
                min_eig,

            "trace_mean":
                trace.mean().item(),
        })

rollout_df = pd.DataFrame(
    rollout_rows
)

display(
    rollout_df.iloc[
        [
            0,
            9,
            24,
            49,
            99
        ]
    ].round(10)
)

max_norm = float(
    rollout_df[
        "frobenius_norm"
    ].max()
)

min_rollout_eig = float(
    rollout_df[
        "min_eigenvalue"
    ].min()
)

max_trace_error_rollout = float(
    np.max(
        np.abs(
            rollout_df[
                "trace_mean"
            ].values
            -
            1.0
        )
    )
)

print(
    "\nMax rollout norm:",
    f"{max_norm:.12e}"
)

print(
    "Minimum rollout eigenvalue:",
    f"{min_rollout_eig:.12e}"
)

print(
    "Max rollout trace error:",
    f"{max_trace_error_rollout:.12e}"
)

# ============================================================
# 14. GRADIENT TEST
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "E. GRADIENT / LEARNING SIGNAL"
)

print(
    "=" * 100
)

kernel.zero_grad(
    set_to_none=True
)

R_input_grad = (
    R_input
    .detach()
    .clone()
    .requires_grad_()
)

target = (
    R0[1]
    .detach()
)

prediction = kernel(
    R_input_grad,
    gamma_tensor,
    dt=1e-3
)

loss = F.mse_loss(
    prediction,
    target
)

loss.backward()

jump_grad_norm = float(
    torch.linalg.vector_norm(
        kernel.jump_operators.grad
    ).item()
)

alpha_grad = float(
    torch.abs(
        kernel.raw_alpha.grad
    ).item()
)

input_grad_norm = float(
    torch.linalg.vector_norm(
        R_input_grad.grad
    ).item()
)

print(
    "Prototype loss:",
    f"{loss.item():.12e}"
)

print(
    "Jump gradient L2:",
    f"{jump_grad_norm:.12e}"
)

print(
    "Alpha gradient:",
    f"{alpha_grad:.12e}"
)

print(
    "Input gradient L2:",
    f"{input_grad_norm:.12e}"
)

gradient_pass = (
    np.isfinite(
        jump_grad_norm
    )
    and
    jump_grad_norm > 0.0
    and
    np.isfinite(
        alpha_grad
    )
    and
    alpha_grad > 0.0
)

if gradient_pass:

    print(
        "[PASS] Lindblad kernel has a non-zero learning signal."
    )

else:

    print(
        "[WARNING] Learning signal is too weak/zero."
    )

# ============================================================
# 15. FINAL DIAGNOSTIC
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "STEP 5 FINAL DIAGNOSTIC"
)

print(
    "=" * 100
)

print(
    "Information direction cosine:",
    f"{mean_best_cosine:.8f}"
)

print(
    "Single-step trace error:",
    f"{trace_error:.8e}"
)

print(
    "Single-step symmetry error:",
    f"{symmetry_error:.8e}"
)

print(
    "Single-step minimum eigenvalue:",
    f"{min_eig:.8e}"
)

print(
    "Environment response monotonic:",
    nondecreasing
)

print(
    "Gradient signal valid:",
    gradient_pass
)

print(
    "Rollout minimum eigenvalue:",
    min_rollout_eig
)

print(
    "Rollout max trace error:",
    max_trace_error_rollout
)

# ============================================================
# 16. SAVE RESULTS
# ============================================================

OUT_DIR = (
    ROOT
    / "results"
    / "step5_lindblad_kernel_prototype"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

environment_df.to_csv(
    OUT_DIR
    / "environment_sensitivity.csv",
    index=False
)

rollout_df.to_csv(
    OUT_DIR
    / "kernel_rollout_stability.csv",
    index=False
)

summary = {

    "representation":
        "R = z z^T / Tr(z z^T)",

    "information_direction_cosine":
        mean_best_cosine,

    "information_direction_error":
        mean_direction_error,

    "single_step_trace_error":
        trace_error,

    "single_step_symmetry_error":
        symmetry_error,

    "single_step_min_eigenvalue":
        min_eig,

    "environment_monotonic":
        nondecreasing,

    "gradient_pass":
        gradient_pass,

    "jump_gradient_l2":
        jump_grad_norm,

    "alpha_gradient":
        alpha_grad,

    "rollout_min_eigenvalue":
        min_rollout_eig,

    "rollout_max_norm":
        max_norm,

    "rollout_max_trace_error":
        max_trace_error_rollout,
}

with open(
    OUT_DIR
    / "step5_summary.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

print(
    "\n[+] Saved results:",
    OUT_DIR
)

# ============================================================
# 17. DECISION LOGIC
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "DECISION"
)

print(
    "=" * 100
)

representation_ok = (
    mean_best_cosine > 0.95
)

kernel_structure_ok = (
    trace_error < 1e-5
    and
    symmetry_error < 1e-6
)

kernel_learning_ok = (
    gradient_pass
)

kernel_environment_ok = (
    nondecreasing
)

kernel_stability_ok = (
    min_rollout_eig > -1e-5
    and
    max_trace_error_rollout < 1e-4
)

print(
    "[INFO] Representation direction OK:",
    representation_ok
)

print(
    "[INFO] Kernel structure OK:",
    kernel_structure_ok
)

print(
    "[INFO] Environment response OK:",
    kernel_environment_ok
)

print(
    "[INFO] Learning signal OK:",
    kernel_learning_ok
)

print(
    "[INFO] Rollout stability OK:",
    kernel_stability_ok
)

if all([
    representation_ok,
    kernel_structure_ok,
    kernel_environment_ok,
    kernel_learning_ok,
    kernel_stability_ok,
]):

    print(
        "\n[PASS] Prototype passes all primary Step-5 checks."
    )

    print(
        "Proceed to full physical LNO representation and "
        "kernel integration."
    )

else:

    print(
        "\n[REVIEW] Candidate representation/kernel "
        "needs refinement before full training."
    )

print("=" * 100)
