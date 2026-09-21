# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 35
# Step            : STEP_06
# Step Heading    : "[+] Ready for STEP 6:"
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 5B — COMPLETE INTERNAL INFORMATION STATE
#             + LINDBLAD-CONSTRAINED KERNEL PROTOTYPE
#
# Representation:
#
#   z(x,t) = [state, phi, flux, noise, dissipation, instability]
#
#   R(x,t) = z z^T / Tr(z z^T)
#   a(x,t) = ||z||
#
# Complete information representation:
#
#   S(x,t) = {R(x,t), a(x,t)}
#
# PURPOSE:
#   1. Verify information preservation
#   2. Demonstrate scale information is retained by a
#   3. Verify environmental sensitivity
#   4. Test Lindblad-form kernel compatibility
#   5. Test repeated-step stability
#   6. Verify non-zero gradients
#   7. Verify finite numerical behavior
#
# IMPORTANT:
#   R is a mathematical normalized internal-information
#   structure representation.
#
#   It is NOT being identified with a microscopic density
#   matrix or any quantum state.
#
# NO FULL MODEL TRAINING
# NO DATASET MODIFICATION
# NO CHECKPOINT MODIFICATION
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

# ============================================================
# 1. ROOT
# ============================================================

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

print("=" * 100)
print("STEP 5B — COMPLETE INTERNAL INFORMATION STATE")
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
# 3. SOURCE TRAJECTORY
# ============================================================

PAIR_PATH = (
    ROOT
    / "results"
    / "step3_paired_trajectories"
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

# ============================================================
# 4. LOAD PHYSICAL OBSERVABLES
# ============================================================

def load_fields(prefix):

    return {

        "state":
            data[
                f"{prefix}_state"
            ].astype(
                np.float64
            ),

        "phi":
            data[
                f"{prefix}_phi"
            ].astype(
                np.float64
            ),

        "flux":
            data[
                f"{prefix}_flux"
            ].astype(
                np.float64
            ),

        "noise":
            data[
                f"{prefix}_noise"
            ].astype(
                np.float64
            ),

        "dissipation":
            data[
                f"{prefix}_dissipation"
            ].astype(
                np.float64
            ),

        "instability":
            data[
                f"{prefix}_instability"
            ].astype(
                np.float64
            ),
    }


open_fields = load_fields(
    "open"
)

reference_fields = load_fields(
    "reference"
)

T, NX = open_fields[
    "state"
].shape

print(
    "[+] Timesteps:",
    T
)

print(
    "[+] Spatial points:",
    NX
)

# ============================================================
# 5. RAW INTERNAL INFORMATION VECTORS
# ============================================================

FEATURE_NAMES = [
    "state",
    "phi",
    "flux",
    "noise",
    "dissipation",
    "instability",
]

open_z = np.stack(
    [
        open_fields[name]
        for name in FEATURE_NAMES
    ],
    axis=-1
)

reference_z = np.stack(
    [
        reference_fields[name]
        for name in FEATURE_NAMES
    ],
    axis=-1
)

assert open_z.shape == (
    T,
    NX,
    6
)

assert reference_z.shape == (
    T,
    NX,
    6
)

print(
    "\n[+] Open information vector:",
    open_z.shape
)

print(
    "[+] Reference information vector:",
    reference_z.shape
)

# ============================================================
# 6. JOINT FEATURE NORMALIZATION
#
# Important:
# Open + reference use the SAME normalization statistics.
# ============================================================

combined_z = np.concatenate(
    [
        open_z,
        reference_z
    ],
    axis=0
)

feature_mean = np.mean(
    combined_z,
    axis=(0, 1)
)

feature_std = np.std(
    combined_z,
    axis=(0, 1)
)

feature_std = np.where(
    feature_std < 1e-12,
    1.0,
    feature_std
)

open_zn = (
    open_z
    -
    feature_mean[
        None,
        None,
        :
    ]
) / feature_std[
    None,
    None,
    :
]

reference_zn = (
    reference_z
    -
    feature_mean[
        None,
        None,
        :
    ]
) / feature_std[
    None,
    None,
    :
]

# ============================================================
# 7. CONSTRUCT R + AMPLITUDE
# ============================================================

def construct_R_and_amplitude(
    z
):
    """
    Input:
        z : [T, NX, D]

    Returns:
        R : [T, NX, D, D]
        a : [T, NX]

    R = z z^T / Tr(z z^T)
    a = ||z||
    """

    amplitude = np.linalg.norm(
        z,
        axis=-1
    )

    outer = (
        z[
            ...,
            :,
            None
        ]
        *
        z[
            ...,
            None,
            :
        ]
    )

    trace = np.trace(
        outer,
        axis1=-2,
        axis2=-1
    )

    R = (
        outer
        /
        (
            np.abs(
                trace
            )
            + 1e-12
        )[
            ...,
            None,
            None
        ]
    )

    return (
        R,
        amplitude
    )


R_open, A_open = (
    construct_R_and_amplitude(
        open_zn
    )
)

R_reference, A_reference = (
    construct_R_and_amplitude(
        reference_zn
    )
)

print(
    "\n[+] R shape:",
    R_open.shape
)

print(
    "[+] Amplitude shape:",
    A_open.shape
)

# ============================================================
# 8. STRUCTURE TEST
# ============================================================

def matrix_structure_test(
    R,
    label
):

    trace = np.trace(
        R,
        axis1=-2,
        axis2=-1
    )

    trace_error = float(
        np.max(
            np.abs(
                trace
                -
                1.0
            )
        )
    )

    symmetry_error = float(
        np.max(
            np.abs(
                R
                -
                np.swapaxes(
                    R,
                    -1,
                    -2
                )
            )
        )
    )

    subset = R[
        :min(
            T,
            32
        ),
        ::max(
            1,
            NX // 16
        )
    ]

    eigvals = np.linalg.eigvalsh(
        subset
    )

    min_eig = float(
        np.min(
            eigvals
        )
    )

    print(
        "\n" + "-" * 80
    )

    print(
        label
    )

    print(
        "-" * 80
    )

    print(
        "Trace max error:",
        f"{trace_error:.12e}"
    )

    print(
        "Symmetry max error:",
        f"{symmetry_error:.12e}"
    )

    print(
        "Minimum eigenvalue:",
        f"{min_eig:.12e}"
    )

    return (
        trace_error,
        symmetry_error,
        min_eig
    )


(
    Ropen_trace_error,
    Ropen_symmetry_error,
    Ropen_min_eig,
) = matrix_structure_test(
    R_open,
    "OPEN INFORMATION MATRIX"
)

(
    Rref_trace_error,
    Rref_symmetry_error,
    Rref_min_eig,
) = matrix_structure_test(
    R_reference,
    "REFERENCE INFORMATION MATRIX"
)

# ============================================================
# 9. AMPLITUDE STATISTICS
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "A. AMPLITUDE INFORMATION"
)

print(
    "=" * 100
)

print(
    "Open amplitude min :",
    f"{A_open.min():.12e}"
)

print(
    "Open amplitude max :",
    f"{A_open.max():.12e}"
)

print(
    "Open amplitude mean:",
    f"{A_open.mean():.12e}"
)

print(
    "Reference amplitude mean:",
    f"{A_reference.mean():.12e}"
)

# ============================================================
# 10. EXPLICIT SCALE-INVARIANCE TEST
#
# This is the reason amplitude exists.
#
# z -> c z:
#   R remains unchanged
#   a scales by c
# ============================================================

z_test = open_zn[
    0,
    0
].copy()

scale_factor = 2.0

z_scaled = (
    scale_factor
    *
    z_test
)

R1, A1 = construct_R_and_amplitude(
    z_test[
        None,
        None,
        :
    ]
)

R2, A2 = construct_R_and_amplitude(
    z_scaled[
        None,
        None,
        :
    ]
)

R_scale_error = float(
    np.max(
        np.abs(
            R1
            -
            R2
        )
    )
)

amplitude_ratio = float(
    A2[
        0,
        0
    ]
    /
    (
        A1[
            0,
            0
        ]
        +
        1e-30
    )
)

print(
    "\n" + "=" * 100
)

print(
    "B. SCALE-INVARIANCE / AMPLITUDE TEST"
)

print(
    "=" * 100
)

print(
    "R difference under x2 scaling:",
    f"{R_scale_error:.12e}"
)

print(
    "Amplitude ratio:",
    f"{amplitude_ratio:.12e}"
)

assert R_scale_error < 1e-6

assert abs(
    amplitude_ratio
    -
    2.0
) < 1e-6

print(
    "[PASS] R removes scale, while amplitude retains scale."
)

# ============================================================
# 11. ENVIRONMENT-INDUCED RESPONSE
# ============================================================

delta_R_env = (
    R_open
    -
    R_reference
)

delta_A_env = (
    A_open
    -
    A_reference
)

# IMPORTANT:
# delta_R_norm stays [T, NX].
# Do NOT apply another norm over axis=-1 later.

delta_R_norm = np.linalg.norm(
    delta_R_env.reshape(
        T,
        NX,
        -1
    ),
    axis=-1
)

delta_A_abs = np.abs(
    delta_A_env
)

mean_R_env = float(
    np.mean(
        delta_R_norm
    )
)

median_R_env = float(
    np.median(
        delta_R_norm
    )
)

p95_R_env = float(
    np.percentile(
        delta_R_norm,
        95
    )
)

max_R_env = float(
    np.max(
        delta_R_norm
    )
)

mean_A_env = float(
    np.mean(
        delta_A_abs
    )
)

median_A_env = float(
    np.median(
        delta_A_abs
    )
)

p95_A_env = float(
    np.percentile(
        delta_A_abs,
        95
    )
)

max_A_env = float(
    np.max(
        delta_A_abs
    )
)

print(
    "\n" + "=" * 100
)

print(
    "C. ENVIRONMENT-INDUCED INFORMATION RESPONSE"
)

print(
    "=" * 100
)

print(
    "Mean ||ΔR_env||:",
    f"{mean_R_env:.12e}"
)

print(
    "Median ||ΔR_env||:",
    f"{median_R_env:.12e}"
)

print(
    "95th percentile ||ΔR_env||:",
    f"{p95_R_env:.12e}"
)

print(
    "Maximum ||ΔR_env||:",
    f"{max_R_env:.12e}"
)

print(
    "Mean |Δamplitude_env|:",
    f"{mean_A_env:.12e}"
)

print(
    "Median |Δamplitude_env|:",
    f"{median_A_env:.12e}"
)

print(
    "95th percentile |Δamplitude_env|:",
    f"{p95_A_env:.12e}"
)

print(
    "Maximum |Δamplitude_env|:",
    f"{max_A_env:.12e}"
)

assert mean_R_env > 0.0

assert mean_A_env > 0.0

print(
    "[PASS] Environment produces measurable information response."
)

# ============================================================
# 12. SCALE-LOSS PRACTICAL TEST
#
# BOTH arrays are [T, NX].
#
# This section intentionally does NOT reduce
# delta_R_norm again.
# ============================================================

R_difference_field = (
    delta_R_norm
)

relative_amplitude_difference = (
    np.abs(
        delta_A_env
    )
    /
    (
        np.abs(
            A_reference
        )
        + 1e-12
    )
)

structure_threshold = np.percentile(
    R_difference_field,
    25
)

amplitude_threshold = np.percentile(
    relative_amplitude_difference,
    75
)

ambiguous_mask = (
    R_difference_field
    <= structure_threshold
) & (
    relative_amplitude_difference
    >= amplitude_threshold
)

ambiguous_fraction = float(
    np.mean(
        ambiguous_mask
    )
)

print(
    "\n" + "=" * 100
)

print(
    "D. PRACTICAL SCALE-LOSS TEST"
)

print(
    "=" * 100
)

print(
    "R difference shape:",
    R_difference_field.shape
)

print(
    "Amplitude difference shape:",
    relative_amplitude_difference.shape
)

print(
    "25th percentile R change:",
    f"{structure_threshold:.12e}"
)

print(
    "75th percentile relative amplitude change:",
    f"{amplitude_threshold:.12e}"
)

print(
    "Low-R-change / high-amplitude-change fraction:",
    f"{ambiguous_fraction:.6f}"
)

# ============================================================
# 13. TORCH LINDBLAD KERNEL
# ============================================================

class CompleteInformationLindbladKernel(
    nn.Module
):

    """
    Prototype state:

        (R, amplitude)

    Lindblad-form dissipator acts on R.

    Amplitude is carried separately so that global scale
    information is not discarded.

    IMPORTANT:
    This is a prototype dynamical block, not yet the final
    spatial LNO architecture.
    """

    def __init__(
        self,
        dim=6,
        channels=4,
    ):

        super().__init__()

        self.dim = dim

        self.channels = channels

        # ----------------------------------------------------
        # Learnable jump operators
        # ----------------------------------------------------

        self.jump_operators = nn.Parameter(
            0.05
            *
            torch.randn(
                channels,
                dim,
                dim
            )
        )

        # ----------------------------------------------------
        # Positive environment coupling multiplier
        # ----------------------------------------------------

        self.raw_alpha = nn.Parameter(
            torch.tensor(
                0.0
            )
        )

        # ----------------------------------------------------
        # Amplitude evolution parameter
        # ----------------------------------------------------

        self.amplitude_weight = nn.Parameter(
            torch.tensor(
                0.01
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

        total = torch.zeros_like(
            R
        )

        for k in range(
            self.channels
        ):

            L = self.jump_operators[
                k
            ]

            LT = L.transpose(
                -1,
                -2
            )

            A = (
                LT
                @
                L
            )

            jump_term = (
                L
                @
                R
                @
                LT
            )

            anticommutator = (
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
                jump_term
                -
                0.5
                *
                anticommutator
            )

        return total

    def forward(
        self,
        R,
        amplitude,
        gamma,
        dt=1e-3,
    ):

        alpha = self.positive_alpha()

        D_R = self.dissipator(
            R
        )

        # ----------------------------------------------------
        # Lindblad-form matrix evolution
        # ----------------------------------------------------

        R_next = (
            R
            +
            dt
            *
            alpha
            *
            gamma
            *
            D_R
        )

        # Symmetrize only against numerical round-off
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

        # ----------------------------------------------------
        # Amplitude evolution
        #
        # Positive parameterization through log(a)
        # ----------------------------------------------------

        log_a = torch.log(
            amplitude
            +
            1e-8
        )

        dissipative_activity = torch.linalg.vector_norm(
            D_R,
            dim=(-2, -1)
        )

        log_a_next = (
            log_a
            +
            dt
            *
            self.amplitude_weight
            *
            gamma
            *
            dissipative_activity
        )

        amplitude_next = torch.exp(
            log_a_next
        )

        return (
            R_next,
            amplitude_next
        )


# ============================================================
# 14. CREATE PROTOTYPE
# ============================================================

kernel = (
    CompleteInformationLindbladKernel(
        dim=6,
        channels=4
    )
    .to(device)
)

print(
    "\n[+] Complete-information Lindblad kernel created."
)

print(
    "[+] Jump operators:",
    tuple(
        kernel.jump_operators.shape
    )
)

print(
    "[+] Positive alpha:",
    kernel.positive_alpha().item()
)

# ============================================================
# 15. PROTOTYPE INPUT
# ============================================================

PROTOTYPE_SPACE = min(
    NX,
    16
)

R_input = torch.tensor(
    R_open[
        0,
        :PROTOTYPE_SPACE
    ],
    dtype=torch.float32,
    device=device
)

A_input = torch.tensor(
    A_open[
        0,
        :PROTOTYPE_SPACE
    ],
    dtype=torch.float32,
    device=device
)

gamma_value = float(
    np.mean(
        data[
            "open_gamma"
        ][0]
    )
)

gamma_tensor = torch.tensor(
    gamma_value,
    dtype=torch.float32,
    device=device
)

print(
    "\n[+] Prototype spatial points:",
    PROTOTYPE_SPACE
)

print(
    "[+] Prototype gamma:",
    gamma_value
)

# ============================================================
# 16. SINGLE-STEP FORWARD
# ============================================================

R_pred, A_pred = kernel(
    R_input,
    A_input,
    gamma_tensor,
    dt=1e-3
)

print(
    "\n" + "=" * 100
)

print(
    "E. SINGLE-STEP COMPLETE INFORMATION KERNEL"
)

print(
    "=" * 100
)

print(
    "Input R shape:",
    tuple(
        R_input.shape
    )
)

print(
    "Input amplitude shape:",
    tuple(
        A_input.shape
    )
)

print(
    "Output R shape:",
    tuple(
        R_pred.shape
    )
)

print(
    "Output amplitude shape:",
    tuple(
        A_pred.shape
    )
)

# ============================================================
# 17. STRUCTURAL FORWARD CHECK
# ============================================================

input_trace = torch.diagonal(
    R_input,
    dim1=-2,
    dim2=-1
).sum(
    dim=-1
)

output_trace = torch.diagonal(
    R_pred,
    dim1=-2,
    dim2=-1
).sum(
    dim=-1
)

trace_change = float(
    torch.max(
        torch.abs(
            output_trace
            -
            input_trace
        )
    ).item()
)

symmetry_error = float(
    torch.max(
        torch.abs(
            R_pred
            -
            R_pred.transpose(
                -1,
                -2
            )
        )
    ).item()
)

eigvals = torch.linalg.eigvalsh(
    R_pred
)

single_step_min_eig = float(
    torch.min(
        eigvals
    ).item()
)

amplitude_finite = bool(
    torch.isfinite(
        A_pred
    ).all().item()
)

print(
    "Max trace change:",
    f"{trace_change:.12e}"
)

print(
    "Max symmetry error:",
    f"{symmetry_error:.12e}"
)

print(
    "Minimum eigenvalue:",
    f"{single_step_min_eig:.12e}"
)

print(
    "Amplitude finite:",
    amplitude_finite
)

single_step_structure_pass = (
    trace_change < 1e-5
    and
    symmetry_error < 1e-6
    and
    single_step_min_eig > -1e-5
    and
    amplitude_finite
)

print(
    "[PASS] Single-step structural checks."
    if single_step_structure_pass
    else
    "[REVIEW] Single-step structural checks need inspection."
)

# ============================================================
# 18. ENVIRONMENT SWEEP
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "F. ENVIRONMENT COUPLING SWEEP"
)

print(
    "=" * 100
)

gamma_values = [
    0.0,
    0.01,
    0.05,
    0.15,
    0.35,
    0.50,
]

environment_rows = []

with torch.no_grad():

    for gamma_test_value in gamma_values:

        gamma_test = torch.tensor(
            gamma_test_value,
            dtype=torch.float32,
            device=device
        )

        R_test, A_test = kernel(
            R_input,
            A_input,
            gamma_test,
            dt=1e-3
        )

        R_change = float(
            torch.linalg.vector_norm(
                R_test
                -
                R_input
            ).item()
        )

        A_change = float(
            torch.linalg.vector_norm(
                A_test
                -
                A_input
            ).item()
        )

        environment_rows.append({

            "gamma":
                gamma_test_value,

            "R_change":
                R_change,

            "amplitude_change":
                A_change,
        })

environment_df = pd.DataFrame(
    environment_rows
)

display(
    environment_df.round(10)
)

R_changes = environment_df[
    "R_change"
].to_numpy()

A_changes = environment_df[
    "amplitude_change"
].to_numpy()

R_monotonic = bool(
    np.all(
        np.diff(
            R_changes
        )
        >=
        -1e-12
    )
)

A_monotonic = bool(
    np.all(
        np.diff(
            A_changes
        )
        >=
        -1e-12
    )
)

print(
    "\nR response monotonic:",
    R_monotonic
)

print(
    "Amplitude response monotonic:",
    A_monotonic
)

environment_nonzero = bool(
    R_changes[-1] > 0.0
    or
    A_changes[-1] > 0.0
)

print(
    "Non-zero strong-environment response:",
    environment_nonzero
)

# ============================================================
# 19. REPEATED ROLLOUT
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "G. REPEATED COMPLETE-INFORMATION ROLLOUT"
)

print(
    "=" * 100
)

R_roll = R_input.clone()

A_roll = A_input.clone()

rollout_rows = []

with torch.no_grad():

    for step in range(
        1,
        101
    ):

        R_roll, A_roll = kernel(
            R_roll,
            A_roll,
            gamma_tensor,
            dt=1e-3
        )

        trace_step = torch.diagonal(
            R_roll,
            dim1=-2,
            dim2=-1
        ).sum(
            dim=-1
        )

        eigvals_step = torch.linalg.eigvalsh(
            R_roll
        )

        min_eig_step = float(
            torch.min(
                eigvals_step
            ).item()
        )

        R_norm = float(
            torch.linalg.vector_norm(
                R_roll
            ).item()
        )

        A_norm = float(
            torch.linalg.vector_norm(
                A_roll
            ).item()
        )

        rollout_rows.append({

            "step":
                step,

            "R_norm":
                R_norm,

            "amplitude_norm":
                A_norm,

            "trace_mean":
                float(
                    trace_step.mean().item()
                ),

            "min_eigenvalue":
                min_eig_step,

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
            99,
        ]
    ].round(10)
)

rollout_min_eig = float(
    rollout_df[
        "min_eigenvalue"
    ].min()
)

rollout_max_trace_error = float(
    np.max(
        np.abs(
            rollout_df[
                "trace_mean"
            ].to_numpy()
            -
            1.0
        )
    )
)

rollout_max_R_norm = float(
    rollout_df[
        "R_norm"
    ].max()
)

rollout_max_A_norm = float(
    rollout_df[
        "amplitude_norm"
    ].max()
)

rollout_finite = bool(
    np.isfinite(
        rollout_df[
            [
                "R_norm",
                "amplitude_norm",
                "trace_mean",
                "min_eigenvalue",
            ]
        ].to_numpy()
    ).all()
)

print(
    "\nMax R norm:",
    f"{rollout_max_R_norm:.12e}"
)

print(
    "Max amplitude norm:",
    f"{rollout_max_A_norm:.12e}"
)

print(
    "Minimum rollout eigenvalue:",
    f"{rollout_min_eig:.12e}"
)

print(
    "Max rollout trace error:",
    f"{rollout_max_trace_error:.12e}"
)

print(
    "Rollout finite:",
    rollout_finite
)

rollout_stability_pass = (
    rollout_min_eig > -1e-5
    and
    rollout_max_trace_error < 1e-4
    and
    np.isfinite(
        rollout_max_R_norm
    )
    and
    np.isfinite(
        rollout_max_A_norm
    )
    and
    rollout_finite
)

print(
    "[PASS] Repeated rollout stable."
    if rollout_stability_pass
    else
    "[REVIEW] Repeated rollout needs inspection."
)

# ============================================================
# 20. GRADIENT TEST
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "H. GRADIENT / LEARNING SIGNAL"
)

print(
    "=" * 100
)

kernel.zero_grad(
    set_to_none=True
)

R_grad = (
    R_input
    .detach()
    .clone()
    .requires_grad_()
)

A_grad = (
    A_input
    .detach()
    .clone()
    .requires_grad_()
)

R_target = torch.tensor(
    R_open[
        1,
        :PROTOTYPE_SPACE
    ],
    dtype=torch.float32,
    device=device
)

A_target = torch.tensor(
    A_open[
        1,
        :PROTOTYPE_SPACE
    ],
    dtype=torch.float32,
    device=device
)

R_out, A_out = kernel(
    R_grad,
    A_grad,
    gamma_tensor,
    dt=1e-3
)

loss_R = F.mse_loss(
    R_out,
    R_target
)

loss_A = F.mse_loss(
    A_out,
    A_target
)

total_loss = (
    loss_R
    +
    loss_A
)

total_loss.backward()

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

amp_grad = float(
    torch.abs(
        kernel.amplitude_weight.grad
    ).item()
)

R_input_grad_norm = float(
    torch.linalg.vector_norm(
        R_grad.grad
    ).item()
)

A_input_grad_norm = float(
    torch.linalg.vector_norm(
        A_grad.grad
    ).item()
)

print(
    "R loss:",
    f"{loss_R.item():.12e}"
)

print(
    "Amplitude loss:",
    f"{loss_A.item():.12e}"
)

print(
    "Total loss:",
    f"{total_loss.item():.12e}"
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
    "Amplitude-weight gradient:",
    f"{amp_grad:.12e}"
)

print(
    "R input gradient L2:",
    f"{R_input_grad_norm:.12e}"
)

print(
    "Amplitude input gradient L2:",
    f"{A_input_grad_norm:.12e}"
)

gradient_pass = all([

    np.isfinite(
        jump_grad_norm
    ),

    np.isfinite(
        alpha_grad
    ),

    np.isfinite(
        amp_grad
    ),

    jump_grad_norm > 0.0,

    alpha_grad > 0.0,

    amp_grad > 0.0,
])

print(
    "[PASS] Non-zero learning signal."
    if gradient_pass
    else
    "[REVIEW] One or more trainable paths have zero/invalid gradient."
)

# ============================================================
# 21. GLOBAL DECISION
# ============================================================

representation_structure_pass = (
    Ropen_trace_error < 1e-5
    and
    Ropen_symmetry_error < 1e-6
    and
    Ropen_min_eig > -1e-5
    and
    Rref_trace_error < 1e-5
    and
    Rref_symmetry_error < 1e-6
    and
    Rref_min_eig > -1e-5
)

environment_pass = (
    np.isfinite(
        mean_R_env
    )
    and
    np.isfinite(
        mean_A_env
    )
    and
    mean_R_env > 0.0
    and
    mean_A_env > 0.0
)

# ============================================================
# 22. SAVE RESULTS
# ============================================================

OUT_DIR = (
    ROOT
    / "results"
    / "step5b_complete_information_state"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Environment sensitivity
environment_df.to_csv(
    OUT_DIR
    / "environment_sensitivity.csv",
    index=False
)

# Rollout
rollout_df.to_csv(
    OUT_DIR
    / "rollout_stability.csv",
    index=False
)

# Save core representations
np.save(
    OUT_DIR
    / "R_open.npy",
    R_open.astype(
        np.float32
    )
)

np.save(
    OUT_DIR
    / "R_reference.npy",
    R_reference.astype(
        np.float32
    )
)

np.save(
    OUT_DIR
    / "amplitude_open.npy",
    A_open.astype(
        np.float32
    )
)

np.save(
    OUT_DIR
    / "amplitude_reference.npy",
    A_reference.astype(
        np.float32
    )
)

np.save(
    OUT_DIR
    / "delta_R_environment.npy",
    delta_R_env.astype(
        np.float32
    )
)

np.save(
    OUT_DIR
    / "delta_amplitude_environment.npy",
    delta_A_env.astype(
        np.float32
    )
)

# ============================================================
# 23. SUMMARY JSON
# ============================================================

summary = {

    "representation":
        "S = (R, amplitude)",

    "R_definition":
        "R = z z^T / Tr(z z^T)",

    "amplitude_definition":
        "a = ||z||",

    "feature_names":
        FEATURE_NAMES,

    "timesteps":
        int(T),

    "spatial_points":
        int(NX),

    "R_open_shape":
        list(
            R_open.shape
        ),

    "R_reference_shape":
        list(
            R_reference.shape
        ),

    "scale_invariance_R_error":
        R_scale_error,

    "scale_amplitude_ratio":
        amplitude_ratio,

    "mean_environment_R_change":
        mean_R_env,

    "median_environment_R_change":
        median_R_env,

    "p95_environment_R_change":
        p95_R_env,

    "max_environment_R_change":
        max_R_env,

    "mean_environment_amplitude_change":
        mean_A_env,

    "median_environment_amplitude_change":
        median_A_env,

    "p95_environment_amplitude_change":
        p95_A_env,

    "max_environment_amplitude_change":
        max_A_env,

    "ambiguous_scale_loss_fraction":
        ambiguous_fraction,

    "single_step_trace_change":
        trace_change,

    "single_step_symmetry_error":
        symmetry_error,

    "single_step_minimum_eigenvalue":
        single_step_min_eig,

    "R_environment_monotonic":
        R_monotonic,

    "amplitude_environment_monotonic":
        A_monotonic,

    "representation_structure_pass":
        representation_structure_pass,

    "kernel_structure_pass":
        single_step_structure_pass,

    "environment_pass":
        environment_pass,

    "rollout_stability_pass":
        rollout_stability_pass,

    "gradient_pass":
        gradient_pass,

    "jump_gradient_l2":
        jump_grad_norm,

    "alpha_gradient":
        alpha_grad,

    "amplitude_weight_gradient":
        amp_grad,

    "rollout_minimum_eigenvalue":
        rollout_min_eig,

    "rollout_max_trace_error":
        rollout_max_trace_error,

    "rollout_max_R_norm":
        rollout_max_R_norm,

    "rollout_max_amplitude_norm":
        rollout_max_A_norm,
}

SUMMARY_PATH = (
    OUT_DIR
    / "step5b_summary.json"
)

with open(
    SUMMARY_PATH,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

# ============================================================
# 24. FINAL DECISION
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "STEP 5B FINAL DECISION"
)

print(
    "=" * 100
)

print(
    "Information structure valid:",
    representation_structure_pass
)

print(
    "Single-step kernel structure valid:",
    single_step_structure_pass
)

print(
    "Environment sensitivity:",
    environment_pass
)

print(
    "Repeated rollout stable:",
    rollout_stability_pass
)

print(
    "Non-zero learning signal:",
    gradient_pass
)

print(
    "Results saved:",
    OUT_DIR
)

if all([
    representation_structure_pass,
    single_step_structure_pass,
    environment_pass,
    rollout_stability_pass,
    gradient_pass,
]):

    print(
        "\n[PASS] STEP 5B PASSED."
    )

    print(
        "[+] Candidate complete information state:"
    )

    print(
        "    S(x,t) = {R(x,t), a(x,t)}"
    )

    print(
        "[+] Ready for STEP 6:"
    )

    print(
        "    spatial neural-operator integration"
    )

else:

    print(
        "\n[REVIEW] STEP 5B requires refinement."
    )

print("=" * 100)
