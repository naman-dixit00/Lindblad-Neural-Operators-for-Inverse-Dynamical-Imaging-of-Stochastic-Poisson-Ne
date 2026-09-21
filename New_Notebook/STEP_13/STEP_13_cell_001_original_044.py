# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 44
# Step            : STEP_13
# Step Heading    : # STEP 13 — LNO DYNAMICAL COUPLING CALIBRATION
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 13 — LNO DYNAMICAL COUPLING CALIBRATION
#
# PURPOSE:
#   Determine how strongly the learned neural evolution should
#   contribute relative to the Lindblad dissipative evolution.
#
# NO TRAINING
#
# We measure:
#
#   ΔR_target    = Y_R - X_R
#
#   ΔR_Lindblad  = R_Lindblad - R
#
#   ΔR_neural    = neural tendency
#
# and sweep:
#
#   beta in:
#       0.01
#       0.05
#       0.10
#       0.25
#       0.50
#       1.00
#       2.00
#       5.00
#
# Candidate:
#
#   ΔR_candidate =
#       ΔR_Lindblad
#       +
#       dt * beta * neural_direction
#
# Then compare candidate against ΔR_target.
#
# IMPORTANT:
#   This is a CALIBRATION experiment.
#   Best beta is NOT yet a final scientific result.
#
# CURRENT DATA:
#   Dataset V3 prototype
#   495 transitions
#
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
print("STEP 13 — LNO DYNAMICAL COUPLING CALIBRATION")
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
# 3. PATHS
# ============================================================

SPLIT_PATH = (
    ROOT
    / "results"
    / "dataset_v3"
    / "step9_splits"
    / "prototype_train_val_test_split.npz"
)

CHECKPOINT_PATH = (
    ROOT
    / "results"
    / "step12_ablation"
    / "D_Full_LNO.pt"
)

OUT_DIR = (
    ROOT
    / "results"
    / "step13_dynamical_coupling"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. CHECK FILES
# ============================================================

if not SPLIT_PATH.is_file():

    raise FileNotFoundError(
        "\nStep-9 split is missing.\n"
        f"Expected:\n{SPLIT_PATH}\n\n"
        "Recreate the Step-9 split or rerun Step-9 first."
    )

if not CHECKPOINT_PATH.is_file():

    raise FileNotFoundError(
        "\nStep-11 LNO checkpoint is missing.\n"
        f"Expected:\n{CHECKPOINT_PATH}\n\n"
        "The checkpoint may have been lost after the Colab "
        "runtime reset. Re-download/recover the committed "
        "checkpoint and place it at the expected path."
    )


# ============================================================
# 5. LOAD STEP-9 SPLIT
# ============================================================

data = np.load(
    SPLIT_PATH
)

X_R_train = data[
    "X_R_train"
].astype(
    np.float32
)

Y_R_train = data[
    "Y_R_train"
].astype(
    np.float32
)

X_A_train = data[
    "X_amplitude_train"
].astype(
    np.float32
)

Y_A_train = data[
    "Y_amplitude_train"
].astype(
    np.float32
)

ENV_train = data[
    "environment_train"
].astype(
    np.float32
)

X_R_val = data[
    "X_R_val"
].astype(
    np.float32
)

Y_R_val = data[
    "Y_R_val"
].astype(
    np.float32
)

X_A_val = data[
    "X_amplitude_val"
].astype(
    np.float32
)

Y_A_val = data[
    "Y_amplitude_val"
].astype(
    np.float32
)

ENV_val = data[
    "environment_val"
].astype(
    np.float32
)

X_R_test = data[
    "X_R_test"
].astype(
    np.float32
)

Y_R_test = data[
    "Y_R_test"
].astype(
    np.float32
)

X_A_test = data[
    "X_amplitude_test"
].astype(
    np.float32
)

Y_A_test = data[
    "Y_amplitude_test"
].astype(
    np.float32
)

ENV_test = data[
    "environment_test"
].astype(
    np.float32
)


# ============================================================
# 6. DIMENSIONS
# ============================================================

N_train, NX, DIM, DIM2 = X_R_train.shape

assert DIM == 6
assert DIM2 == 6

N_test = X_R_test.shape[0]

print(
    "\n[+] Train:",
    N_train
)

print(
    "[+] Test:",
    N_test
)

print(
    "[+] NX:",
    NX
)

print(
    "[+] DIM:",
    DIM
)


# ============================================================
# 7. TORCH TEST BATCH
#
# Use a small representative batch so calibration remains
# lightweight.
# ============================================================

BATCH_SIZE = min(
    16,
    N_test
)

X_R = torch.tensor(
    X_R_test[
        :BATCH_SIZE
    ],
    dtype=torch.float32,
    device=device
)

Y_R = torch.tensor(
    Y_R_test[
        :BATCH_SIZE
    ],
    dtype=torch.float32,
    device=device
)

X_A = torch.tensor(
    X_A_test[
        :BATCH_SIZE
    ],
    dtype=torch.float32,
    device=device
)

Y_A = torch.tensor(
    Y_A_test[
        :BATCH_SIZE
    ],
    dtype=torch.float32,
    device=device
)

ENV = torch.tensor(
    ENV_test[
        :BATCH_SIZE
    ],
    dtype=torch.float32,
    device=device
)

gamma = ENV[
    ...,
    0
]

sigma = ENV[
    ...,
    1
]

print(
    "\n[+] Calibration batch:",
    BATCH_SIZE
)

print(
    "[+] R:",
    tuple(X_R.shape)
)

print(
    "[+] amplitude:",
    tuple(X_A.shape)
)

print(
    "[+] gamma:",
    tuple(gamma.shape)
)

print(
    "[+] sigma:",
    tuple(sigma.shape)
)


# ============================================================
# 8. SPECTRAL CONVOLUTION
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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

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
                ...,
                :n_modes
            ],
            self.weight_imag[
                ...,
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
# 9. LINDBLAD KERNEL
# ============================================================

class LindbladDissipativeKernel(
    nn.Module
):

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

            jump = (
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
                jump
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
        dt
    ):

        D_R = self.dissipator(
            R
        )

        alpha = self.positive_alpha()

        gamma_field = (
            gamma[
                ...,
                None,
                None
            ]
        )

        R_lindblad = (
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
            R_lindblad,
            D_R
        )


# ============================================================
# 10. LNO MODEL
# ============================================================

class LNOPrototype(
    nn.Module
):

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
            1
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
                    1
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
            1
        )

        self.A_head = nn.Conv1d(
            width,
            1,
            1
        )

    def neural_tendency(
        self,
        R,
        amplitude,
        gamma,
        sigma
    ):

        B, NX_local, D, D2 = R.shape

        R_flat = R.reshape(
            B,
            NX_local,
            D * D
        )

        env = torch.stack(
            [
                amplitude,
                gamma,
                sigma,
            ],
            dim=-1
        )

        inp = torch.cat(
            [
                R_flat,
                env,
            ],
            dim=-1
        )

        x = inp.permute(
            0,
            2,
            1
        )

        x = self.input_projection(
            x
        )

        for k in range(
            self.depth
        ):

            x = (
                self.spectral_layers[k](
                    x
                )
                +
                self.pointwise_layers[k](
                    x
                )
            )

            if k < self.depth - 1:

                x = F.gelu(
                    x
                )

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

        # Direction only for calibration.
        norm = torch.linalg.vector_norm(
            neural_R,
            dim=(-2, -1),
            keepdim=True
        )

        neural_direction = (
            neural_R
            /
            (
                norm
                +
                1e-8
            )
        )

        return neural_direction

    def forward_components(
        self,
        R,
        amplitude,
        gamma,
        sigma,
        dt
    ):

        neural_direction = (
            self.neural_tendency(
                R,
                amplitude,
                gamma,
                sigma
            )
        )

        R_lindblad, D_R = (
            self.lindblad_kernel(
                R,
                gamma,
                dt
            )
        )

        delta_lindblad = (
            R_lindblad
            -
            R
        )

        return (
            delta_lindblad,
            neural_direction,
            D_R
        )


# ============================================================
# 11. CREATE / LOAD LNO
# ============================================================

model = (
    LNOPrototype(
        dim=6,
        width=64,
        modes=16,
        depth=4,
        lindblad_channels=4,
    )
    .to(device)
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

state_dict = checkpoint.get(
    "model_state_dict",
    checkpoint
)

missing, unexpected = (
    model.load_state_dict(
        state_dict,
        strict=False
    )
)

print(
    "\n[+] Checkpoint loaded:",
    CHECKPOINT_PATH
)

print(
    "[+] Missing keys:",
    len(missing)
)

print(
    "[+] Unexpected keys:",
    len(unexpected)
)

if missing or unexpected:

    print(
        "[WARNING] Checkpoint/model architecture mismatch."
    )

else:

    print(
        "[PASS] Checkpoint architecture matches."
    )

model.eval()


# ============================================================
# 12. TARGET DYNAMICS
# ============================================================

dt = 1e-3

delta_target = (
    Y_R
    -
    X_R
)

target_norm = torch.linalg.vector_norm(
    delta_target
)

print(
    "\n" + "=" * 100
)

print(
    "A. TARGET DYNAMICS"
)

print(
    "=" * 100
)

print(
    "Mean target ΔR norm:",
    f"{torch.mean(torch.linalg.vector_norm(delta_target, dim=(-2,-1))).item():.12e}"
)

print(
    "Global target ΔR norm:",
    f"{target_norm.item():.12e}"
)


# ============================================================
# 13. GET TWO DYNAMICAL COMPONENTS
# ============================================================

with torch.no_grad():

    (
        delta_lindblad,
        neural_direction,
        D_R
    ) = model.forward_components(
        X_R,
        X_A,
        gamma,
        sigma,
        dt
    )

# ============================================================
# 14. COMPONENT NORMS
# ============================================================

lindblad_norm = torch.linalg.vector_norm(
    delta_lindblad
)

neural_norm = torch.linalg.vector_norm(
    dt * neural_direction
)

print(
    "\n" + "=" * 100
)

print(
    "B. COMPONENT MAGNITUDES"
)

print(
    "=" * 100
)

print(
    "Target ΔR norm:",
    f"{target_norm.item():.12e}"
)

print(
    "Lindblad ΔR norm:",
    f"{lindblad_norm.item():.12e}"
)

print(
    "Neural ΔR norm @ beta=1:",
    f"{neural_norm.item():.12e}"
)

print(
    "Lindblad / target:",
    f"{(lindblad_norm/(target_norm+1e-12)).item():.12e}"
)

print(
    "Neural / target:",
    f"{(neural_norm/(target_norm+1e-12)).item():.12e}"
)


# ============================================================
# 15. SWEEP BETA
# ============================================================

BETAS = [
    0.01,
    0.025,
    0.05,
    0.10,
    0.25,
    0.50,
    1.00,
    2.00,
    5.00,
    10.00,
]

calibration_rows = []

with torch.no_grad():

    for beta in BETAS:

        delta_candidate = (
            delta_lindblad
            +
            dt
            *
            beta
            *
            neural_direction
        )

        candidate_error = (
            delta_candidate
            -
            delta_target
        )

        mse = torch.mean(
            candidate_error ** 2
        )

        rmse = torch.sqrt(
            mse
            +
            1e-12
        )

        mae = torch.mean(
            torch.abs(
                candidate_error
            )
        )

        relative_l2 = (
            torch.linalg.vector_norm(
                candidate_error
            )
            /
            (
                torch.linalg.vector_norm(
                    delta_target
                )
                +
                1e-12
            )
        )

        candidate_norm = (
            torch.linalg.vector_norm(
                delta_candidate
            )
        )

        # Cosine between candidate transition and target
        dot = torch.sum(
            delta_candidate
            *
            delta_target
        )

        cosine = (
            dot
            /
            (
                torch.linalg.vector_norm(
                    delta_candidate
                )
                *
                torch.linalg.vector_norm(
                    delta_target
                )
                +
                1e-12
            )
        )

        calibration_rows.append({

            "beta":
                float(beta),

            "MSE":
                float(
                    mse.item()
                ),

            "RMSE":
                float(
                    rmse.item()
                ),

            "MAE":
                float(
                    mae.item()
                ),

            "Relative_L2":
                float(
                    relative_l2.item()
                ),

            "Candidate_deltaR_norm":
                float(
                    candidate_norm.item()
                ),

            "Target_deltaR_norm":
                float(
                    target_norm.item()
                ),

            "Transition_cosine":
                float(
                    cosine.item()
                ),
        })

calibration_df = pd.DataFrame(
    calibration_rows
)

print(
    "\n" + "=" * 100
)

print(
    "C. DYNAMICAL COUPLING SWEEP"
)

print(
    "=" * 100
)

display(
    calibration_df.round(8)
)


# ============================================================
# 16. BEST BETA
# ============================================================

best_idx = calibration_df[
    "Relative_L2"
].idxmin()

best_row = calibration_df.loc[
    best_idx
]

BEST_BETA = float(
    best_row[
        "beta"
    ]
)

BEST_REL_L2 = float(
    best_row[
        "Relative_L2"
    ]
)

BEST_RMSE = float(
    best_row[
        "RMSE"
    ]
)

print(
    "\n" + "=" * 100
)

print(
    "D. BEST CALIBRATED COUPLING"
)

print(
    "=" * 100
)

print(
    "Best beta:",
    BEST_BETA
)

print(
    "Best Relative-L2:",
    f"{BEST_REL_L2:.12e}"
)

print(
    "Best RMSE:",
    f"{BEST_RMSE:.12e}"
)


# ============================================================
# 17. FIT A CONTINUOUS SCALAR FOR REFERENCE
#
# This is NOT the final architecture parameter.
# It answers:
# "What scalar combination of Lindblad and neural direction
# best approximates the target transition?"
#
# We solve:
#
#   target ≈ delta_L + beta * neural
#
# in least-squares sense.
# ============================================================

with torch.no_grad():

    neural_component = (
        dt
        *
        neural_direction
    )

    residual_after_lindblad = (
        delta_target
        -
        delta_lindblad
    )

    numerator = torch.sum(
        neural_component
        *
        residual_after_lindblad
    )

    denominator = torch.sum(
        neural_component
        *
        neural_component
    )

    continuous_beta = (
        numerator
        /
        (
            denominator
            +
            1e-12
        )
    )

    continuous_beta_value = float(
        continuous_beta.item()
    )

print(
    "\nContinuous least-squares beta:",
    f"{continuous_beta_value:.12e}"
)


# ============================================================
# 18. ENVIRONMENT-STRATIFIED CALIBRATION
#
# Determine whether beta behaves differently under
# low / moderate / strong environment.
# ============================================================

gamma_np = (
    gamma[
        :,
        :
    ]
    .detach()
    .cpu()
    .numpy()
)

# Representative gamma per sample.
sample_gamma = (
    np.mean(
        gamma_np,
        axis=1
    )
)

sample_target = (
    delta_target
    .detach()
    .cpu()
    .numpy()
)

sample_lindblad = (
    delta_lindblad
    .detach()
    .cpu()
    .numpy()
)

sample_neural = (
    neural_component
    .detach()
    .cpu()
    .numpy()
)

sample_rel_rows = []

gamma_bins = [
    (
        "low",
        0.0,
        0.10
    ),
    (
        "moderate",
        0.10,
        0.30
    ),
    (
        "strong",
        0.30,
        0.60
    ),
]

for name, lo, hi in gamma_bins:

    mask = (
        (sample_gamma >= lo)
        &
        (sample_gamma < hi)
    )

    if not np.any(mask):

        continue

    t = sample_target[
        mask
    ]

    l = sample_lindblad[
        mask
    ]

    n = sample_neural[
        mask
    ]

    best_beta_local = None
    best_error_local = np.inf

    for beta in BETAS:

        candidate = (
            l
            +
            beta
            *
            n
        )

        error = (
            candidate
            -
            t
        )

        rel = (
            np.linalg.norm(
                error.ravel()
            )
            /
            (
                np.linalg.norm(
                    t.ravel()
                )
                +
                1e-12
            )
        )

        if rel < best_error_local:

            best_error_local = (
                rel
            )

            best_beta_local = (
                beta
            )

    sample_rel_rows.append({

        "environment_region":
            name,

        "gamma_lower":
            float(lo),

        "gamma_upper":
            float(hi),

        "samples":
            int(mask.sum()),

        "best_beta":
            float(
                best_beta_local
            ),

        "best_relative_L2":
            float(
                best_error_local
            ),
    })

environment_calibration_df = pd.DataFrame(
    sample_rel_rows
)

print(
    "\n" + "=" * 100
)

print(
    "E. ENVIRONMENT-STRATIFIED BETA"
)

print(
    "=" * 100
)

if len(
    environment_calibration_df
) > 0:

    display(
        environment_calibration_df.round(8)
    )

else:

    print(
        "[INFO] No samples fell into the predefined bins."
    )


# ============================================================
# 19. SAVE RESULTS
# ============================================================

calibration_df.to_csv(
    OUT_DIR
    / "beta_sweep.csv",
    index=False
)

environment_calibration_df.to_csv(
    OUT_DIR
    / "environment_stratified_beta.csv",
    index=False
)

summary = {

    "step":
        13,

    "purpose":
        "Dynamical coupling calibration",

    "prototype":
        True,

    "batch_size":
        int(BATCH_SIZE),

    "dt":
        float(dt),

    "betas_tested":
        [
            float(x)
            for x in BETAS
        ],

    "best_beta":
        BEST_BETA,

    "best_relative_L2":
        BEST_REL_L2,

    "best_RMSE":
        BEST_RMSE,

    "continuous_least_squares_beta":
        continuous_beta_value,

    "target_transition_norm":
        float(
            target_norm.item()
        ),

    "lindblad_transition_norm":
        float(
            lindblad_norm.item()
        ),

    "neural_transition_norm_beta_1":
        float(
            neural_norm.item()
        ),

    "checkpoint":
        str(
            CHECKPOINT_PATH
        ),

    "environment_stratified_results":
        environment_calibration_df
        .to_dict(
            orient="records"
        ),

    "next_step":
        "Use calibrated coupling in a controlled retraining "
        "experiment, then expand to independent trajectories.",
}

SUMMARY_PATH = (
    OUT_DIR
    / "step13_summary.json"
)

with open(
    SUMMARY_PATH,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        allow_nan=False
    )


# ============================================================
# 20. FINAL
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "STEP 13 COMPLETE"
)

print(
    "=" * 100
)

print(
    "[+] Best tested beta:",
    BEST_BETA
)

print(
    "[+] Continuous LS beta:",
    continuous_beta_value
)

print(
    "[+] Best calibrated Relative-L2:",
    f"{BEST_REL_L2:.12e}"
)

print(
    "[+] Results:",
    OUT_DIR
)

print(
    "\n[INFO] This does NOT change the checkpoint."
)

print(
    "[INFO] This does NOT constitute final LNO training."
)

print(
    "[INFO] Beta is only being calibrated against the target"
)

print(
    "       transition before the next controlled experiment."
)

print("=" * 100)
