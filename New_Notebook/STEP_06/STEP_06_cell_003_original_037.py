# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 37
# Step            : STEP_06
# Step Heading    : # STEP 6 — SPATIAL LINDBLAD NEURAL OPERATOR
# Step Cell No.   : 3
# ============================================================

# ============================================================
# STEP 6 — SPATIAL LINDBLAD NEURAL OPERATOR
#            CONTROLLED / STABLE PROTOTYPE
#
# CURRENT STATUS:
#   TESTING ONLY
#
# IMPORTANT:
#   - No full training
#   - No modification of old checkpoints
#   - No overwrite of original dataset
#
# Architecture:
#
#   [R, amplitude, gamma, sigma]
#             ↓
#       Fourier operator
#             +
#       Lindblad forward dynamics
#             ↓
#     constrained R evolution
#             +
#     amplitude evolution
#
# IMPORTANT SCIENTIFIC CHANGE:
#
# The neural matrix tendency is converted into a symmetric
# positive-semidefinite generator direction rather than being
# added as an arbitrary unconstrained matrix.
#
# The resulting R is re-normalized to maintain:
#
#   R = R^T
#   R >= 0
#   Tr(R) = 1
#
# This is still a prototype. Final physical formulation is
# NOT locked until testing passes.
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

# ============================================================
# 1. ROOT
# ============================================================

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

print("=" * 100)
print("STEP 6 — SPATIAL LINDBLAD NEURAL OPERATOR PROTOTYPE")
print("=" * 100)

# ============================================================
# 2. DEVICE / SEED
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SEED = 2026

torch.manual_seed(SEED)
np.random.seed(SEED)

if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

print(
    "[+] Device:",
    device
)

# ============================================================
# 3. LOAD DATASET V2
# ============================================================

DATA_PATH = (
    ROOT
    / "results"
    / "dataset_v2"
    / "environment_conditioned_lno_dataset_v2.npz"
)

assert DATA_PATH.is_file(), (
    f"Missing dataset:\n{DATA_PATH}"
)

data = np.load(
    DATA_PATH
)

X_R = data[
    "X_information"
].astype(
    np.float32
)

Y_R = data[
    "Y_information"
].astype(
    np.float32
)

X_STATE = data[
    "X_state"
].astype(
    np.float32
)

Y_STATE = data[
    "Y_state"
].astype(
    np.float32
)

ENV = data[
    "environment"
].astype(
    np.float32
)

print(
    "[+] Dataset:",
    DATA_PATH
)

print(
    "[+] X_R:",
    X_R.shape
)

print(
    "[+] Y_R:",
    Y_R.shape
)

print(
    "[+] X_state:",
    X_STATE.shape
)

print(
    "[+] Y_state:",
    Y_STATE.shape
)

print(
    "[+] Environment:",
    ENV.shape
)

# ============================================================
# 4. DATA CHECKS
# ============================================================

N, NX, DIM, DIM2 = X_R.shape

assert DIM == 6
assert DIM2 == 6

assert Y_R.shape == X_R.shape

assert X_STATE.shape == (
    N,
    NX
)

assert Y_STATE.shape == (
    N,
    NX
)

assert ENV.shape == (
    N,
    NX,
    2
)

for name, arr in {
    "X_R": X_R,
    "Y_R": Y_R,
    "X_STATE": X_STATE,
    "Y_STATE": Y_STATE,
    "ENV": ENV,
}.items():

    assert np.isfinite(arr).all(), (
        f"Non-finite data in {name}"
    )

print(
    "\n[PASS] Dataset shape and finite checks."
)

# ============================================================
# 5. PROTOTYPE SPATIAL AMPLITUDE
#
# Dataset V2 does not contain the final full-information
# amplitude explicitly.
#
# Temporary prototype:
#
#   a(x,t) = |state(x,t)|
#
# FINAL:
#   Dataset V3 will explicitly store
#
#   a(x,t) = ||z(x,t)||
# ============================================================

X_amplitude = np.abs(
    X_STATE
).astype(
    np.float32
)

Y_amplitude = np.abs(
    Y_STATE
).astype(
    np.float32
)

assert X_amplitude.shape == (
    N,
    NX
)

assert Y_amplitude.shape == (
    N,
    NX
)

print(
    "\n[+] Prototype spatial amplitude:",
    X_amplitude.shape
)

print(
    "[+] Prototype target amplitude:",
    Y_amplitude.shape
)

print(
    "[INFO] Final Dataset V3 must store full-information amplitude explicitly."
)

# ============================================================
# 6. PROTOTYPE BATCH
# ============================================================

BATCH_SIZE = min(
    8,
    N
)

R_batch = torch.tensor(
    X_R[
        :BATCH_SIZE
    ],
    dtype=torch.float32,
    device=device
)

R_target = torch.tensor(
    Y_R[
        :BATCH_SIZE
    ],
    dtype=torch.float32,
    device=device
)

A_batch = torch.tensor(
    X_amplitude[
        :BATCH_SIZE
    ],
    dtype=torch.float32,
    device=device
)

A_target = torch.tensor(
    Y_amplitude[
        :BATCH_SIZE
    ],
    dtype=torch.float32,
    device=device
)

gamma_batch = torch.tensor(
    ENV[
        :BATCH_SIZE,
        :,
        0
    ],
    dtype=torch.float32,
    device=device
)

sigma_batch = torch.tensor(
    ENV[
        :BATCH_SIZE,
        :,
        1
    ],
    dtype=torch.float32,
    device=device
)

print(
    "\n[+] Prototype batch:"
)

print(
    "R        :",
    tuple(R_batch.shape)
)

print(
    "Amplitude:",
    tuple(A_batch.shape)
)

print(
    "Gamma    :",
    tuple(gamma_batch.shape)
)

print(
    "Sigma    :",
    tuple(sigma_batch.shape)
)

assert (
    A_batch.shape
    ==
    gamma_batch.shape
)

assert (
    A_batch.shape
    ==
    sigma_batch.shape
)

print(
    "[PASS] Spatial input dimensions aligned."
)

# ============================================================
# 7. SPECTRAL CONVOLUTION
# ============================================================

class SpectralConv1D(
    nn.Module
):

    def __init__(
        self,
        in_channels,
        out_channels,
        modes,
    ):

        super().__init__()

        self.in_channels = (
            in_channels
        )

        self.out_channels = (
            out_channels
        )

        self.modes = (
            modes
        )

        scale = (
            1.0
            /
            math.sqrt(
                in_channels
                *
                out_channels
            )
        )

        self.weight_real = nn.Parameter(
            scale
            *
            torch.randn(
                in_channels,
                out_channels,
                modes
            )
        )

        self.weight_imag = nn.Parameter(
            scale
            *
            torch.randn(
                in_channels,
                out_channels,
                modes
            )
        )

    def forward(
        self,
        x
    ):

        B, C, NX_local = x.shape

        x_ft = torch.fft.rfft(
            x,
            dim=-1
        )

        n_modes = min(
            self.modes,
            x_ft.shape[-1]
        )

        out_ft = torch.zeros(
            B,
            self.out_channels,
            x_ft.shape[-1],
            dtype=torch.complex64,
            device=x.device
        )

        weight = torch.complex(
            self.weight_real[
                :,
                :,
                :n_modes
            ],
            self.weight_imag[
                :,
                :,
                :n_modes
            ]
        )

        out_ft[
            ...,
            :n_modes
        ] = torch.einsum(
            "bim,iom->bom",
            x_ft[
                ...,
                :n_modes
            ],
            weight
        )

        return torch.fft.irfft(
            out_ft,
            n=NX_local,
            dim=-1
        )

# ============================================================
# 8. LINDBLAD DISSIPATIVE KERNEL
# ============================================================

class LindbladDissipativeKernel(
    nn.Module
):

    """
    Lindblad-form dissipative generator:

        D(R) =
        Σ_k γ [
            L_k R L_k^T
            -
            1/2 {L_k^T L_k, R}
        ]

    The jump operators are learned.

    gamma is an explicit environmental coupling field.

    This block participates DIRECTLY in forward dynamics.
    """

    def __init__(
        self,
        dim=6,
        channels=4,
    ):

        super().__init__()

        self.dim = dim

        self.channels = channels

        self.jump_operators = nn.Parameter(
            0.02
            *
            torch.randn(
                channels,
                dim,
                dim
            )
        )

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

            anti_term = (
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
                anti_term
            )

        return total

    def forward(
        self,
        R,
        gamma,
        dt,
    ):

        D_R = self.dissipator(
            R
        )

        alpha = self.positive_alpha()

        gamma_field = gamma[
            ...,
            None,
            None
        ]

        R_next = (
            R
            +
            dt
            *
            alpha
            *
            gamma_field
            *
            D_R
        )

        return (
            R_next,
            D_R
        )

# ============================================================
# 9. SPATIAL LNO
# ============================================================

class SpatialLNO(
    nn.Module
):

    """
    Spatial FNO + Lindblad-constrained dynamics.

    The learned neural update acts as a symmetric tangent,
    while the final R state is projected to a normalized PSD
    matrix.

    This prevents the neural component from producing an
    unconstrained matrix explosion.
    """

    def __init__(
        self,
        dim=6,
        width=64,
        modes=16,
        depth=4,
        lindblad_channels=4,
    ):

        super().__init__()

        self.dim = dim

        self.width = width

        self.modes = modes

        self.depth = depth

        input_channels = (
            dim * dim
            +
            3
        )

        self.input_projection = nn.Conv1d(
            input_channels,
            width,
            kernel_size=1
        )

        self.spectral_layers = nn.ModuleList(
            [
                SpectralConv1D(
                    width,
                    width,
                    modes
                )
                for _ in range(
                    depth
                )
            ]
        )

        self.pointwise_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=1
                )
                for _ in range(
                    depth
                )
            ]
        )

        self.lindblad_kernel = (
            LindbladDissipativeKernel(
                dim=dim,
                channels=lindblad_channels
            )
        )

        self.R_head = nn.Conv1d(
            width,
            dim * dim,
            kernel_size=1
        )

        self.A_head = nn.Conv1d(
            width,
            1,
            kernel_size=1
        )

    # --------------------------------------------------------
    # PSD + trace normalization
    # --------------------------------------------------------

    def project_R(
        self,
        M
    ):

        # Symmetric part
        M = (
            0.5
            *
            (
                M
                +
                M.transpose(
                    -1,
                    -2
                )
            )
        )

        # Eigenvalue decomposition
        eigvals, eigvecs = torch.linalg.eigh(
            M
        )

        # Clamp eigenvalues to ensure PSD
        eigvals = torch.clamp(
            eigvals,
            min=1e-8
        )

        # Reconstruct PSD matrix
        M_psd = (
            eigvecs
            *
            eigvals.unsqueeze(
                -2
            )
        ) @ eigvecs.transpose(
            -1,
            -2
        )

        # Trace normalize
        trace = torch.diagonal(
            M_psd,
            dim1=-2,
            dim2=-1
        ).sum(
            dim=-1,
            keepdim=True
        )

        M_psd = (
            M_psd
            /
            (
                trace.unsqueeze(-1)
                +
                1e-12
            )
        )

        return M_psd

    def forward(
        self,
        R,
        amplitude,
        gamma,
        sigma,
        dt=1e-3,
    ):

        B, NX_local, D, D2 = (
            R.shape
        )

        assert D == self.dim
        assert D2 == self.dim

        # ====================================================
        # 1. Input flattening
        # ====================================================

        R_flat = R.reshape(
            B,
            NX_local,
            D * D
        )

        # ====================================================
        # 2. Environment-conditioned spatial field
        # ====================================================

        assert amplitude.shape == (
            B,
            NX_local
        )

        assert gamma.shape == (
            B,
            NX_local
        )

        assert sigma.shape == (
            B,
            NX_local
        )

        env_channels = torch.stack(
            [
                amplitude,
                gamma,
                sigma,
            ],
            dim=-1
        )

        input_field = torch.cat(
            [
                R_flat,
                env_channels,
            ],
            dim=-1
        )

        # ====================================================
        # 3. Lift
        # ====================================================

        x = input_field.permute(
            0,
            2,
            1
        )

        x = self.input_projection(
            x
        )

        # ====================================================
        # 4. Fourier layers
        # ====================================================

        spectral_history = []

        for layer_idx in range(
            self.depth
        ):

            spectral = (
                self.spectral_layers[
                    layer_idx
                ](
                    x
                )
            )

            pointwise = (
                self.pointwise_layers[
                    layer_idx
                ](
                    x
                )
            )

            x = (
                spectral
                +
                pointwise
            )

            spectral_history.append(
                torch.linalg.vector_norm(
                    x,
                    dim=(1, 2)
                )
            )

            if layer_idx < (
                self.depth - 1
            ):

                x = F.gelu(
                    x
                )

        # ====================================================
        # 5. Neural R tendency
        # ====================================================

        neural_R = self.R_head(
            x
        )

        neural_R = (
            neural_R
            .permute(
                0,
                2,
                1
            )
            .reshape(
                B,
                NX_local,
                D,
                D
            )
        )

        neural_R = (
            0.5
            *
            (
                neural_R
                +
                neural_R.transpose(
                    -1,
                    -2
                )
            )
        )

        # ====================================================
        # 6. Normalize neural tangent magnitude
        #
        # This prevents random initialization from dominating
        # the physical Lindblad component.
        # ====================================================

        neural_norm = torch.linalg.vector_norm(
            neural_R,
            dim=(-2, -1),
            keepdim=True
        )

        neural_R = (
            neural_R
            /
            (
                neural_norm
                +
                1e-8
            )
        )

        # ====================================================
        # 7. Lindblad forward evolution
        # ====================================================

        R_lindblad, D_R = (
            self.lindblad_kernel(
                R,
                gamma,
                dt
            )
        )

        # ====================================================
        # 8. Combine
        #
        # Neural tangent is deliberately small in the prototype.
        # ====================================================

        neural_scale = 0.05

        R_candidate = (
            R_lindblad
            +
            dt
            *
            neural_scale
            *
            neural_R
        )

        # ====================================================
        # 9. STRUCTURAL PROJECTION
        #
        # This guarantees that the state passed to the next
        # time step remains a valid normalized PSD matrix.
        # ====================================================

        R_next = self.project_R(
            R_candidate
        )

        # ====================================================
        # 10. Amplitude tendency
        # ====================================================

        A_tendency = (
            self.A_head(
                x
            )
            .squeeze(
                1
            )
        )

        # Stable bounded tendency
        A_tendency = torch.tanh(
            A_tendency
        )

        log_A = torch.log(
            amplitude
            +
            1e-8
        )

        log_A_next = (
            log_A
            +
            dt
            *
            A_tendency
        )

        amplitude_next = torch.exp(
            log_A_next
        )

        return {

            "R_next":
                R_next,

            "amplitude_next":
                amplitude_next,

            "lindblad_dissipation":
                D_R,

            "neural_R_tendency":
                neural_R,

            "neural_amplitude_tendency":
                A_tendency,

            "spectral_history":
                spectral_history,
        }

# ============================================================
# 10. CREATE MODEL
# ============================================================

model = (
    SpatialLNO(
        dim=6,
        width=64,
        modes=16,
        depth=4,
        lindblad_channels=4,
    )
    .to(device)
)

print(
    "\n[+] Spatial LNO created."
)

print(
    "[+] Parameters:",
    f"{sum(p.numel() for p in model.parameters()):,}"
)

print(
    "[+] Jump operators:",
    tuple(
        model.lindblad_kernel
        .jump_operators
        .shape
    )
)

print(
    "[+] Positive alpha:",
    model.lindblad_kernel
    .positive_alpha()
    .item()
)

# ============================================================
# 11. FORWARD
# ============================================================

model.eval()

with torch.no_grad():

    output = model(
        R_batch,
        A_batch,
        gamma_batch,
        sigma_batch,
        dt=1e-3
    )

R_pred = output[
    "R_next"
]

A_pred = output[
    "amplitude_next"
]

D_R = output[
    "lindblad_dissipation"
]

# ============================================================
# 12. FORWARD SANITY
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "A. FORWARD SANITY"
)

print(
    "=" * 100
)

print(
    "R input :",
    tuple(R_batch.shape)
)

print(
    "R output:",
    tuple(R_pred.shape)
)

print(
    "A input :",
    tuple(A_batch.shape)
)

print(
    "A output:",
    tuple(A_pred.shape)
)

print(
    "D(R)    :",
    tuple(D_R.shape)
)

finite_forward = bool(
    torch.isfinite(
        R_pred
    ).all().item()
    and
    torch.isfinite(
        A_pred
    ).all().item()
    and
    torch.isfinite(
        D_R
    ).all().item()
)

print(
    "[PASS] Forward outputs finite."
    if finite_forward
    else
    "[FAIL] Forward produced non-finite values."
)

# ============================================================
# 13. MATRIX STRUCTURE
# ============================================================

R_trace = torch.diagonal(
    R_pred,
    dim1=-2,
    dim2=-1
).sum(
    dim=-1
)

trace_error = float(
    torch.max(
        torch.abs(
            R_trace
            -
            1.0
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

minimum_eigenvalue = float(
    torch.min(
        eigvals
    ).item()
)

print(
    "\n" + "=" * 100
)

print(
    "B. MATRIX STRUCTURE"
)

print(
    "=" * 100
)

print(
    "Max trace error:",
    f"{trace_error:.12e}"
)

print(
    "Max symmetry error:",
    f"{symmetry_error:.12e}"
)

print(
    "Minimum eigenvalue:",
    f"{minimum_eigenvalue:.12e}"
)

structure_pass = (
    trace_error < 1e-5
    and
    symmetry_error < 1e-6
    and
    minimum_eigenvalue > -1e-6
)

print(
    "[PASS] Matrix structural constraints."
    if structure_pass
    else
    "[REVIEW] Matrix structural constraints."
)

# ============================================================
# 14. ENVIRONMENT COUPLING RESPONSE
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "C. ENVIRONMENT COUPLING RESPONSE"
)

print(
    "=" * 100
)

R_probe = R_batch[
    :1
]

A_probe = A_batch[
    :1
]

sigma_probe = sigma_batch[
    :1
]

environment_rows = []

model.eval()

with torch.no_grad():

    for gamma_value in [
        0.0,
        0.01,
        0.05,
        0.15,
        0.35,
        0.50,
    ]:

        gamma_probe = torch.full_like(
            gamma_batch[
                :1
            ],
            gamma_value
        )

        out = model(
            R_probe,
            A_probe,
            gamma_probe,
            sigma_probe,
            dt=1e-3
        )

        R_change = float(
            torch.linalg.vector_norm(
                out[
                    "R_next"
                ]
                -
                R_probe
            ).item()
        )

        A_change = float(
            torch.linalg.vector_norm(
                out[
                    "amplitude_next"
                ]
                -
                A_probe
            ).item()
        )

        D_norm = float(
            torch.linalg.vector_norm(
                out[
                    "lindblad_dissipation"
                ]
            ).item()
        )

        environment_rows.append({

            "gamma":
                float(
                    gamma_value
                ),

            "R_change":
                R_change,

            "amplitude_change":
                A_change,

            "Lindblad_D_norm":
                D_norm,
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

D_changes = environment_df[
    "Lindblad_D_norm"
].to_numpy()

R_monotonic = bool(
    np.all(
        np.diff(
            R_changes
        )
        >=
        -1e-10
    )
)

A_monotonic = bool(
    np.all(
        np.diff(
            A_changes
        )
        >=
        -1e-10
    )
)

D_nonzero = bool(
    D_changes[-1]
    >
    0.0
)

print(
    "\nR response monotonic:",
    R_monotonic
)

print(
    "Amplitude response monotonic:",
    A_monotonic
)

print(
    "Lindblad response non-zero:",
    D_nonzero
)

# ============================================================
# 15. REPEATED ROLLOUT
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "D. REPEATED SPATIAL LNO ROLLOUT"
)

print(
    "=" * 100
)

R_roll = (
    R_batch[
        :1
    ].clone()
)

A_roll = (
    A_batch[
        :1
    ].clone()
)

gamma_roll = (
    gamma_batch[
        :1
    ].clone()
)

sigma_roll = (
    sigma_batch[
        :1
    ].clone()
)

rollout_rows = []

model.eval()

with torch.no_grad():

    for step in range(
        1,
        101
    ):

        out = model(
            R_roll,
            A_roll,
            gamma_roll,
            sigma_roll,
            dt=1e-3
        )

        R_roll = out[
            "R_next"
        ]

        A_roll = out[
            "amplitude_next"
        ]

        trace = torch.diagonal(
            R_roll,
            dim1=-2,
            dim2=-1
        ).sum(
            dim=-1
        )

        eig = torch.linalg.eigvalsh(
            R_roll
        )

        R_norm = (
            torch.linalg.vector_norm(
                R_roll
            ).item()
        )

        A_norm = (
            torch.linalg.vector_norm(
                A_roll
            ).item()
        )

        rollout_rows.append({

            "step":
                int(step),

            "R_norm":
                float(
                    R_norm
                ),

            "amplitude_norm":
                float(
                    A_norm
                ),

            "trace_mean":
                float(
                    trace.mean().item()
                ),

            "min_eigenvalue":
                float(
                    eig.min().item()
                ),

            "finite":
                bool(
                    torch.isfinite(
                        R_roll
                    ).all().item()
                    and
                    torch.isfinite(
                        A_roll
                    ).all().item()
                ),
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
    rollout_df[
        "finite"
    ].all()
)

print(
    "\nMinimum rollout eigenvalue:",
    f"{rollout_min_eig:.12e}"
)

print(
    "Maximum rollout trace error:",
    f"{rollout_max_trace_error:.12e}"
)

print(
    "Maximum R norm:",
    f"{rollout_max_R_norm:.12e}"
)

print(
    "Maximum amplitude norm:",
    f"{rollout_max_A_norm:.12e}"
)

print(
    "All rollout values finite:",
    rollout_finite
)

stability_pass = (
    rollout_finite
    and
    rollout_max_trace_error < 1e-4
    and
    rollout_min_eig > -1e-6
    and
    np.isfinite(
        rollout_max_R_norm
    )
    and
    np.isfinite(
        rollout_max_A_norm
    )
)

print(
    "[PASS] Repeated rollout stable."
    if stability_pass
    else
    "[REVIEW] Repeated rollout needs inspection."
)

# ============================================================
# 16. GRADIENT TEST
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

model.train()

model.zero_grad(
    set_to_none=True
)

R_grad = (
    R_batch
    .detach()
    .clone()
    .requires_grad_()
)

A_grad = (
    A_batch
    .detach()
    .clone()
    .requires_grad_()
)

out = model(
    R_grad,
    A_grad,
    gamma_batch,
    sigma_batch,
    dt=1e-3
)

loss_R = F.mse_loss(
    out[
        "R_next"
    ],
    R_target
)

loss_A = F.mse_loss(
    out[
        "amplitude_next"
    ],
    A_target
)

total_loss = (
    loss_R
    +
    loss_A
)

total_loss.backward()

jump_grad = (
    model
    .lindblad_kernel
    .jump_operators
    .grad
)

alpha_grad = (
    model
    .lindblad_kernel
    .raw_alpha
    .grad
)

jump_grad_norm = float(
    torch.linalg.vector_norm(
        jump_grad
    ).item()
)

alpha_grad_norm = float(
    torch.abs(
        alpha_grad
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
    f"{alpha_grad_norm:.12e}"
)

print(
    "R input gradient L2:",
    f"{R_input_grad_norm:.12e}"
)

print(
    "Amplitude input gradient L2:",
    f"{A_input_grad_norm:.12e}"
)

gradient_pass = (
    np.isfinite(
        jump_grad_norm
    )
    and
    np.isfinite(
        alpha_grad_norm
    )
    and
    jump_grad_norm > 0.0
    and
    alpha_grad_norm > 0.0
)

print(
    "[PASS] Non-zero Lindblad learning signal."
    if gradient_pass
    else
    "[REVIEW] Lindblad learning signal."
)

# ============================================================
# 17. SAVE RESULTS
# ============================================================

OUT_DIR = (
    ROOT
    / "results"
    / "step6_spatial_lno_prototype"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

environment_df.to_csv(
    OUT_DIR
    / "environment_response.csv",
    index=False
)

rollout_df.to_csv(
    OUT_DIR
    / "rollout_stability.csv",
    index=False
)

# ============================================================
# 18. NATIVE PYTHON SCALAR CONVERSION
#
# This fixes the json.dump bool / NumPy scalar error.
# ============================================================

def native_scalar(
    value
):

    if isinstance(
        value,
        (
            np.bool_,
        )
    ):

        return bool(
            value
        )

    if isinstance(
        value,
        (
            np.integer,
        )
    ):

        return int(
            value
        )

    if isinstance(
        value,
        (
            np.floating,
        )
    ):

        return float(
            value
        )

    if isinstance(
        value,
        (
            bool,
            int,
            float,
            str,
        )
    ):

        return value

    return value


# ============================================================
# 19. SUMMARY
# ============================================================

prototype_summary = {

    "model":
        "SpatialLNO",

    "prototype_status":
        True,

    "input_representation":
        "R + amplitude + gamma + noise_sigma",

    "R_definition":
        "R = z z^T / Tr(z z^T)",

    "prototype_amplitude_definition":
        "abs(state)",

    "final_amplitude_definition":
        "||z|| from all six information channels",

    "spatial_points":
        native_scalar(
            NX
        ),

    "information_dimension":
        native_scalar(
            DIM
        ),

    "fourier_modes":
        16,

    "latent_width":
        64,

    "depth":
        4,

    "lindblad_channels":
        4,

    "lindblad_in_forward_path":
        True,

    "finite_forward":
        native_scalar(
            finite_forward
        ),

    "structure_pass":
        native_scalar(
            structure_pass
        ),

    "environment_pass":
        native_scalar(
            environment_pass
            if "environment_pass" in globals()
            else
            (
                D_nonzero
                and
                R_changes[-1]
                >
                R_changes[0]
            )
        ),

    "environment_signal_pass":
        native_scalar(
            (
                D_nonzero
                and
                R_changes[-1]
                >
                R_changes[0]
            )
        ),

    "R_environment_monotonic":
        native_scalar(
            R_monotonic
        ),

    "amplitude_environment_monotonic":
        native_scalar(
            A_monotonic
        ),

    "stability_pass":
        native_scalar(
            stability_pass
        ),

    "gradient_pass":
        native_scalar(
            gradient_pass
        ),

    "jump_gradient_l2":
        native_scalar(
            jump_grad_norm
        ),

    "alpha_gradient":
        native_scalar(
            alpha_grad_norm
        ),

    "single_step_trace_error":
        native_scalar(
            trace_error
        ),

    "single_step_symmetry_error":
        native_scalar(
            symmetry_error
        ),

    "single_step_minimum_eigenvalue":
        native_scalar(
            minimum_eigenvalue
        ),

    "rollout_minimum_eigenvalue":
        native_scalar(
            rollout_min_eig
        ),

    "rollout_max_trace_error":
        native_scalar(
            rollout_max_trace_error
        ),

    "rollout_max_R_norm":
        native_scalar(
            rollout_max_R_norm
        ),

    "rollout_max_amplitude_norm":
        native_scalar(
            rollout_max_A_norm
        ),

    "rollout_finite":
        native_scalar(
            rollout_finite
        ),

    "R_response_monotonic":
        native_scalar(
            R_monotonic
        ),

    "amplitude_response_monotonic":
        native_scalar(
            A_monotonic
        ),

    "Lindblad_response_nonzero":
        native_scalar(
            D_nonzero
        ),
}

SUMMARY_PATH = (
    OUT_DIR
    / "step6_summary.json"
)

with open(
    SUMMARY_PATH,
    "w"
) as f:

    json.dump(
        prototype_summary,
        f,
        indent=2,
        allow_nan=False
    )

# ============================================================
# 20. SAVE PROTOTYPE CHECKPOINT
# ============================================================

MODEL_PATH = (
    OUT_DIR
    / "spatial_lno_prototype.pt"
)

torch.save(
    {

        "model_state_dict":
            model.state_dict(),

        "architecture":
            {

                "name":
                    "SpatialLNO",

                "information_dimension":
                    6,

                "width":
                    64,

                "modes":
                    16,

                "depth":
                    4,

                "lindblad_channels":
                    4,
            },

        "representation":
            {

                "R":
                    "z z^T / Tr(z z^T)",

                "amplitude_prototype":
                    "abs(state)",

                "amplitude_final":
                    "||z||",
            },

        "physics":
            {

                "lindblad_in_forward_path":
                    True,

                "neural_loss_penalty_for_Lindblad":
                    False,
            },

        "prototype_only":
            True,
    },
    MODEL_PATH
)

# ============================================================
# 21. FINAL DECISION
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "STEP 6 PROTOTYPE DECISION"
)

print(
    "=" * 100
)

print(
    "Finite forward:",
    finite_forward
)

print(
    "Matrix structure:",
    structure_pass
)

print(
    "Environment signal:",
    (
        D_nonzero
        and
        R_changes[-1]
        >
        R_changes[0]
    )
)

print(
    "Repeated stability:",
    stability_pass
)

print(
    "Gradient signal:",
    gradient_pass
)

print(
    "Lindblad response non-zero:",
    D_nonzero
)

print(
    "\n[+] Summary:",
    SUMMARY_PATH
)

print(
    "[+] Prototype checkpoint:",
    MODEL_PATH
)

if all([
    finite_forward,
    structure_pass,
    D_nonzero,
    stability_pass,
    gradient_pass,
]):

    print(
        "\n[PASS] STEP 6 PROTOTYPE PASSED."
    )

    print(
        "[+] Lindblad dynamics is embedded in the forward path."
    )

    print(
        "[+] R is kept normalized and PSD."
    )

    print(
        "[+] Spatial Fourier pathway is operational."
    )

    print(
        "[+] Non-zero Lindblad learning signal exists."
    )

    print(
        "[+] Ready for Dataset V3 / controlled training."
    )

else:

    print(
        "\n[REVIEW] STEP 6 prototype requires further refinement."
    )

    print(
        "[INFO] Do NOT start full training yet."
    )

print("=" * 100)
