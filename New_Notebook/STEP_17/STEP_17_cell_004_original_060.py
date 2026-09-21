# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 60
# Step            : STEP_17
# Step Heading    : # RECOVER STEP 17 FNO TEST PREDICTIONS
# Step Cell No.   : 4
# ============================================================

# ============================================================
# RECOVER STEP 17 FNO TEST PREDICTIONS
# No retraining
# ============================================================

import os
import math
import json
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# ROOT
# ============================================================

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 100)
print("RECOVERING STEP 17 FNO TEST PREDICTIONS")
print("=" * 100)

print(
    "[+] Device:",
    device
)


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = (
    ROOT
    / "results"
    / "dataset_v3_expanded"
    / "environment_conditioned_lno_dataset_v3_expanded.npz"
)

SPLIT_PATH = (
    ROOT
    / "results"
    / "step16_final_splits"
    / "final_trajectory_level_split.npz"
)

FNO_CHECKPOINT = (
    ROOT
    / "results"
    / "step17_final_fno"
    / "fno_final_best.pt"
)

FNO_SUMMARY = (
    ROOT
    / "results"
    / "step17_final_fno"
    / "fno_final_summary.json"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "step17_final_fno"
    / "fno_final_test_predictions.npz"
)


# ============================================================
# CHECK FILES
# ============================================================

for name, path in {

    "Dataset":
        DATASET_PATH,

    "Split":
        SPLIT_PATH,

    "FNO checkpoint":
        FNO_CHECKPOINT,

    "FNO summary":
        FNO_SUMMARY,

}.items():

    print(
        f"{name:20s}:",
        path.is_file()
    )

    if not path.is_file():

        raise FileNotFoundError(
            f"Missing {name}:\n{path}"
        )

print(
    "\n[PASS] All required recovery files found."
)


# ============================================================
# LOAD DATASET
# ============================================================

data = np.load(
    DATASET_PATH
)

X_R = data[
    "X_R"
].astype(
    np.float32
)

Y_R = data[
    "Y_R"
].astype(
    np.float32
)

X_A = data[
    "X_amplitude"
].astype(
    np.float32
)

Y_A = data[
    "Y_amplitude"
].astype(
    np.float32
)

ENV = data[
    "environment"
].astype(
    np.float32
)

print(
    "\nDataset shape:"
)

print(
    "X_R:",
    X_R.shape
)

print(
    "X_A:",
    X_A.shape
)

print(
    "Environment:",
    ENV.shape
)


# ============================================================
# LOAD SPLIT
# ============================================================

split = np.load(
    SPLIT_PATH
)

test_indices = split[
    "test_indices"
].astype(
    np.int64
)

assert len(
    test_indices
) == 1485

print(
    "\n[+] Test transitions:",
    len(test_indices)
)


# ============================================================
# EXTRACT TEST
# ============================================================

X_R_test = X_R[
    test_indices
]

Y_R_test = Y_R[
    test_indices
]

X_A_test = X_A[
    test_indices
]

Y_A_test = Y_A[
    test_indices
]

ENV_test = ENV[
    test_indices
]


# ============================================================
# EXACT FNO NORMALIZATION
# ============================================================

with open(
    FNO_SUMMARY,
    "r"
) as f:

    fno_summary = json.load(
        f
    )

norm = fno_summary[
    "amplitude_normalization"
]

amplitude_mean = float(
    norm["mean"]
)

amplitude_std = float(
    norm["std"]
)

print(
    "\n[+] FNO amplitude mean:",
    amplitude_mean
)

print(
    "[+] FNO amplitude std:",
    amplitude_std
)

X_A_test_norm = (
    X_A_test
    -
    amplitude_mean
) / amplitude_std


# ============================================================
# FNO SPECTRAL CONVOLUTION
# ============================================================

class SpectralConv1D(
    nn.Module
):

    def __init__(
        self,
        in_channels,
        out_channels,
        modes
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

        B, C, NX = x.shape

        x_ft = torch.fft.rfft(
            x,
            dim=-1
        )

        n_modes = min(
            self.modes,
            x_ft.shape[-1]
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

        out_ft = torch.zeros(
            B,
            self.out_channels,
            x_ft.shape[-1],
            dtype=torch.complex64,
            device=x.device
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
            n=NX,
            dim=-1
        )


# ============================================================
# EXACT STEP-17 FNO ARCHITECTURE
# ============================================================

class FinalFNO(
    nn.Module
):

    def __init__(
        self,
        dim=6,
        width=64,
        modes=16,
        depth=4
    ):

        super().__init__()

        input_channels = (
            dim * dim
            +
            3
        )

        self.dim = dim
        self.width = width
        self.modes = modes
        self.depth = depth

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
                for _ in range(depth)
            ]
        )

        self.pointwise_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    width,
                    width,
                    1
                )
                for _ in range(depth)
            ]
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

    def forward(
        self,
        R,
        amplitude,
        gamma,
        sigma
    ):

        B, NX, D, D2 = R.shape

        assert D == self.dim
        assert D2 == self.dim

        R_flat = R.reshape(
            B,
            NX,
            D * D
        )

        env = torch.stack(
            [
                amplitude,
                gamma,
                sigma
            ],
            dim=-1
        )

        inp = torch.cat(
            [
                R_flat,
                env
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

            spectral = self.spectral_layers[k](
                x
            )

            pointwise = self.pointwise_layers[k](
                x
            )

            x = (
                spectral
                +
                pointwise
            )

            if k < self.depth - 1:

                x = torch.nn.functional.gelu(
                    x
                )

        R_next = self.R_head(
            x
        )

        R_next = (
            R_next
            .permute(
                0,
                2,
                1
            )
            .reshape(
                B,
                NX,
                D,
                D
            )
        )

        # EXACT Step-17 behavior:
        # symmetric output, but NO PSD projection
        # and NO trace normalization.

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

        A_next = self.A_head(
            x
        ).squeeze(
            1
        )

        return (
            R_next,
            A_next
        )


# ============================================================
# CREATE MODEL
# ============================================================

model = FinalFNO(
    dim=6,
    width=64,
    modes=16,
    depth=4
).to(
    device
)

print(
    "\n[+] FNO model recreated."
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    FNO_CHECKPOINT,
    map_location=device
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()

print(
    "[PASS] Step-17 FNO checkpoint loaded."
)

print(
    "[+] Best epoch:",
    checkpoint.get(
        "epoch",
        "unknown"
    )
)


# ============================================================
# TEST FORWARD
# ============================================================

X_R_test_t = torch.from_numpy(
    X_R_test
).to(
    device
)

X_A_test_t = torch.from_numpy(
    X_A_test_norm.astype(
        np.float32
    )
).to(
    device
)

ENV_test_t = torch.from_numpy(
    ENV_test
).to(
    device
)

gamma_test = ENV_test_t[
    ...,
    0
]

sigma_test = ENV_test_t[
    ...,
    1
]


# ============================================================
# BATCHED INFERENCE
# ============================================================

BATCH_SIZE = 32

R_predictions = []
A_predictions = []

print(
    "\n" + "=" * 100
)

print(
    "GENERATING TEST PREDICTIONS"
)

print(
    "=" * 100
)

with torch.no_grad():

    for start in range(
        0,
        len(test_indices),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(test_indices)
        )

        R_batch = X_R_test_t[
            start:end
        ]

        A_batch = X_A_test_t[
            start:end
        ]

        gamma_batch = gamma_test[
            start:end
        ]

        sigma_batch = sigma_test[
            start:end
        ]

        R_out, A_out = model(
            R_batch,
            A_batch,
            gamma_batch,
            sigma_batch
        )

        R_predictions.append(
            R_out
            .cpu()
            .numpy()
        )

        A_predictions.append(
            A_out
            .cpu()
            .numpy()
        )

R_prediction = np.concatenate(
    R_predictions,
    axis=0
)

A_prediction_norm = np.concatenate(
    A_predictions,
    axis=0
)


# ============================================================
# RESTORE PHYSICAL AMPLITUDE
# ============================================================

A_prediction = (
    A_prediction_norm
    *
    amplitude_std
    +
    amplitude_mean
)


# ============================================================
# FINITE CHECK
# ============================================================

assert np.isfinite(
    R_prediction
).all()

assert np.isfinite(
    A_prediction
).all()

print(
    "[PASS] Predictions finite."
)

print(
    "R prediction:",
    R_prediction.shape
)

print(
    "Amplitude prediction:",
    A_prediction.shape
)


# ============================================================
# SAVE EXACT FILE EXPECTED BY STEP 19
# ============================================================

np.savez_compressed(

    OUTPUT_PATH,

    R_prediction=
        R_prediction,

    R_target=
        Y_R_test,

    amplitude_prediction=
        A_prediction,

    amplitude_target=
        Y_A_test,

    environment=
        ENV_test,

    test_indices=
        test_indices,
)


# ============================================================
# VERIFY SAVED FILE
# ============================================================

assert OUTPUT_PATH.is_file()

saved = np.load(
    OUTPUT_PATH
)

for key in [

    "R_prediction",
    "R_target",
    "amplitude_prediction",
    "amplitude_target",
    "environment",
    "test_indices",

]:

    assert key in saved

assert np.array_equal(
    saved[
        "test_indices"
    ],
    test_indices
)

assert np.allclose(
    saved[
        "R_target"
    ],
    Y_R_test,
    atol=1e-7
)

assert np.allclose(
    saved[
        "amplitude_target"
    ],
    Y_A_test,
    atol=1e-7
)


# ============================================================
# RECALCULATE METRICS
# ============================================================

R_error = (
    R_prediction
    -
    Y_R_test
)

A_error = (
    A_prediction
    -
    Y_A_test
)

R_mse = float(
    np.mean(
        R_error ** 2
    )
)

R_rmse = float(
    np.sqrt(
        R_mse
    )
)

R_mae = float(
    np.mean(
        np.abs(
            R_error
        )
    )
)

R_relative_l2 = float(
    np.linalg.norm(
        R_error
    )
    /
    (
        np.linalg.norm(
            Y_R_test
        )
        +
        1e-12
    )
)

A_mse = float(
    np.mean(
        A_error ** 2
    )
)

A_rmse = float(
    np.sqrt(
        A_mse
    )
)

A_mae = float(
    np.mean(
        np.abs(
            A_error
        )
    )
)

A_relative_l2 = float(
    np.linalg.norm(
        A_error
    )
    /
    (
        np.linalg.norm(
            Y_A_test
        )
        +
        1e-12
    )
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "STEP 17 FNO PREDICTION RECOVERY COMPLETE"
)

print(
    "=" * 100
)

print(
    "Test transitions:",
    len(test_indices)
)

print(
    "\nR metrics:"
)

print(
    "  MSE:",
    f"{R_mse:.12e}"
)

print(
    "  RMSE:",
    f"{R_rmse:.12e}"
)

print(
    "  MAE:",
    f"{R_mae:.12e}"
)

print(
    "  Relative-L2:",
    f"{R_relative_l2:.12e}"
)

print(
    "\nAmplitude metrics:"
)

print(
    "  MSE:",
    f"{A_mse:.12e}"
)

print(
    "  RMSE:",
    f"{A_rmse:.12e}"
)

print(
    "  MAE:",
    f"{A_mae:.12e}"
)

print(
    "  Relative-L2:",
    f"{A_relative_l2:.12e}"
)

print(
    "\nRecovered file:"
)

print(
    OUTPUT_PATH
)

print(
    "Size:",
    f"{OUTPUT_PATH.stat().st_size / (1024**2):.3f} MB"
)

print(
    "\n[PASS] fno_final_test_predictions.npz recreated."
)

print(
    "[INFO] FNO was NOT retrained."
)

print(
    "[INFO] Original Step-17 checkpoint was used."
)

print("=" * 100)
