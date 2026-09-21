# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 41
# Step            : STEP_10
# Step Heading    : # STEP 10 — FNO BASELINE PROTOTYPE TRAINING
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 10 — FNO BASELINE PROTOTYPE TRAINING
#
# SAME:
#   Dataset V3
#   Train / Val / Test split
#   Input representation
#   Environment conditioning
#   Target
#
# PURPOSE:
#   Establish a fair, properly trained FNO baseline.
#
# INPUT:
#   R(t), amplitude(t), gamma, noise_sigma
#
# TARGET:
#   R(t+1), amplitude(t+1)
#
# IMPORTANT:
#   This is ONLY a small prototype training run.
#   Final training happens after dataset expansion.
#
# NO LNO TRAINING HERE.
# ============================================================

import os
import json
import math
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

# ============================================================
# 1. ROOT
# ============================================================

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

print("=" * 100)
print("STEP 10 — FNO BASELINE PROTOTYPE TRAINING")
print("=" * 100)

# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

SEED = 2026

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# 3. DEVICE
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
# 4. LOAD SPLIT DATA
# ============================================================

SPLIT_PATH = (
    ROOT
    / "results"
    / "dataset_v3"
    / "step9_splits"
    / "prototype_train_val_test_split.npz"
)

SPLIT_META_PATH = (
    ROOT
    / "results"
    / "dataset_v3"
    / "step9_splits"
    / "prototype_split_metadata.csv"
)

assert SPLIT_PATH.is_file(), SPLIT_PATH
assert SPLIT_META_PATH.is_file(), SPLIT_META_PATH

data = np.load(
    SPLIT_PATH
)

split_meta = pd.read_csv(
    SPLIT_META_PATH
)

# ============================================================
# 5. LOAD TRAIN
# ============================================================

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

# ============================================================
# 6. LOAD VALIDATION
# ============================================================

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

# ============================================================
# 7. LOAD TEST
# ============================================================

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
# 8. BASIC SHAPE CHECKS
# ============================================================

N_train, NX, D, D2 = X_R_train.shape
N_val = X_R_val.shape[0]
N_test = X_R_test.shape[0]

assert D == 6
assert D2 == 6

assert X_R_train.shape == Y_R_train.shape
assert X_R_val.shape == Y_R_val.shape
assert X_R_test.shape == Y_R_test.shape

assert X_A_train.shape == (
    N_train,
    NX
)

assert X_A_val.shape == (
    N_val,
    NX
)

assert X_A_test.shape == (
    N_test,
    NX
)

assert ENV_train.shape == (
    N_train,
    NX,
    2
)

assert ENV_val.shape == (
    N_val,
    NX,
    2
)

assert ENV_test.shape == (
    N_test,
    NX,
    2
)

print(
    "\n[+] Train:",
    N_train
)

print(
    "[+] Validation:",
    N_val
)

print(
    "[+] Test:",
    N_test
)

print(
    "[+] Spatial points:",
    NX
)

print(
    "[+] Information dimension:",
    D
)

# ============================================================
# 9. CONVERT TO TORCH
# ============================================================

def to_tensor(
    arr
):

    return torch.tensor(
        arr,
        dtype=torch.float32,
        device=device
    )


X_R_train_t = to_tensor(
    X_R_train
)

Y_R_train_t = to_tensor(
    Y_R_train
)

X_A_train_t = to_tensor(
    X_A_train
)

Y_A_train_t = to_tensor(
    Y_A_train
)

ENV_train_t = to_tensor(
    ENV_train
)

X_R_val_t = to_tensor(
    X_R_val
)

Y_R_val_t = to_tensor(
    Y_R_val
)

X_A_val_t = to_tensor(
    X_A_val
)

Y_A_val_t = to_tensor(
    Y_A_val
)

ENV_val_t = to_tensor(
    ENV_val
)

X_R_test_t = to_tensor(
    X_R_test
)

Y_R_test_t = to_tensor(
    Y_R_test
)

X_A_test_t = to_tensor(
    X_A_test
)

Y_A_test_t = to_tensor(
    Y_A_test
)

ENV_test_t = to_tensor(
    ENV_test
)

# ============================================================
# 10. DATASET / DATALOADER
# ============================================================

# Keep the prototype run lightweight.
BATCH_SIZE = 8

train_dataset = TensorDataset(
    X_R_train_t,
    X_A_train_t,
    ENV_train_t,
    Y_R_train_t,
    Y_A_train_t,
)

val_dataset = TensorDataset(
    X_R_val_t,
    X_A_val_t,
    ENV_val_t,
    Y_R_val_t,
    Y_A_val_t,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

print(
    "\n[+] Train batches:",
    len(train_loader)
)

print(
    "[+] Validation batches:",
    len(val_loader)
)

# ============================================================
# 11. SPECTRAL CONVOLUTION
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

        scale = 1.0 / math.sqrt(
            in_channels
            *
            out_channels
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
# 12. FNO BASELINE
# ============================================================

class FNOBaseline(
    nn.Module
):

    """
    Fair unconstrained Fourier neural operator baseline.

    INPUT:
        R          = 36 channels
        amplitude  = 1 channel
        gamma      = 1 channel
        sigma      = 1 channel

    TOTAL:
        39 channels

    OUTPUT:
        R_next
        amplitude_next

    IMPORTANT:
        No Lindblad block.
        No dissipative constraint.
        No physics-specific stabilization.

    This is the control baseline against which LNO will be
    compared.
    """

    def __init__(
        self,
        dim=6,
        width=64,
        modes=16,
        depth=4,
    ):

        super().__init__()

        self.dim = dim
        self.width = width
        self.modes = modes
        self.depth = depth

        input_channels = (
            dim * dim
            + 3
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

    def forward(
        self,
        R,
        amplitude,
        gamma,
        sigma,
    ):

        B, NX_local, D, D2 = (
            R.shape
        )

        assert D == self.dim
        assert D2 == self.dim

        # ----------------------------------------------------
        # Flatten R
        # ----------------------------------------------------

        R_flat = R.reshape(
            B,
            NX_local,
            D * D
        )

        # ----------------------------------------------------
        # Environment-conditioned channels
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Fourier layers
        # ----------------------------------------------------

        for k in range(
            self.depth
        ):

            spectral = (
                self.spectral_layers[k](
                    x
                )
            )

            pointwise = (
                self.pointwise_layers[k](
                    x
                )
            )

            x = (
                spectral
                +
                pointwise
            )

            if k < (
                self.depth - 1
            ):

                x = F.gelu(
                    x
                )

        # ----------------------------------------------------
        # R output
        # ----------------------------------------------------

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
                NX_local,
                D,
                D
            )
        )

        # Keep symmetry only for a clean common output format.
        # This is NOT a physical PSD constraint.
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
        # Amplitude output
        # ----------------------------------------------------

        A_next = self.A_head(
            x
        ).squeeze(
            1
        )

        return {
            "R_next":
                R_next,

            "amplitude_next":
                A_next,
        }

# ============================================================
# 13. MODEL
# ============================================================

model = (
    FNOBaseline(
        dim=6,
        width=64,
        modes=16,
        depth=4,
    )
    .to(device)
)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "\n[+] FNO parameters:",
    f"{parameter_count:,}"
)

# ============================================================
# 14. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)

# Small prototype run
EPOCHS = 5

print(
    "[+] Optimizer: AdamW"
)

print(
    "[+] Learning rate: 1e-3"
)

print(
    "[+] Weight decay: 1e-4"
)

print(
    "[+] Prototype epochs:",
    EPOCHS
)

# ============================================================
# 15. METRICS
# ============================================================

def compute_batch_metrics(
    prediction,
    target,
):

    mse = torch.mean(
        (
            prediction
            -
            target
        ) ** 2
    )

    rmse = torch.sqrt(
        mse
        +
        1e-12
    )

    mae = torch.mean(
        torch.abs(
            prediction
            -
            target
        )
    )

    denom = torch.linalg.vector_norm(
        target
    )

    rel_l2 = (
        torch.linalg.vector_norm(
            prediction
            -
            target
        )
        /
        (
            denom
            +
            1e-12
        )
    )

    return (
        mse,
        rmse,
        mae,
        rel_l2
    )

# ============================================================
# 16. TRAIN / VALIDATION
# ============================================================

history = []

best_val = float(
    "inf"
)

best_epoch = None

OUT_DIR = (
    ROOT
    / "results"
    / "step10_fno_baseline"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    OUT_DIR
    / "fno_prototype_best.pt"
)

print(
    "\n" + "=" * 100
)

print(
    "STARTING FNO PROTOTYPE TRAINING"
)

print(
    "=" * 100
)

for epoch in range(
    1,
    EPOCHS + 1
):

    start_time = time.time()

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_loss_sum = 0.0

    train_R_mse_sum = 0.0
    train_A_mse_sum = 0.0

    train_count = 0

    for (
        R_x,
        A_x,
        env_x,
        R_y,
        A_y,
    ) in train_loader:

        optimizer.zero_grad(
            set_to_none=True
        )

        gamma_x = env_x[
            ...,
            0
        ]

        sigma_x = env_x[
            ...,
            1
        ]

        out = model(
            R_x,
            A_x,
            gamma_x,
            sigma_x,
        )

        R_pred = out[
            "R_next"
        ]

        A_pred = out[
            "amplitude_next"
        ]

        R_loss = F.mse_loss(
            R_pred,
            R_y
        )

        A_loss = F.mse_loss(
            A_pred,
            A_y
        )

        loss = (
            R_loss
            +
            A_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_n = R_x.shape[0]

        train_loss_sum += (
            loss.item()
            *
            batch_n
        )

        train_R_mse_sum += (
            R_loss.item()
            *
            batch_n
        )

        train_A_mse_sum += (
            A_loss.item()
            *
            batch_n
        )

        train_count += batch_n

    train_loss = (
        train_loss_sum
        /
        train_count
    )

    train_R_mse = (
        train_R_mse_sum
        /
        train_count
    )

    train_A_mse = (
        train_A_mse_sum
        /
        train_count
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss_sum = 0.0

    val_R_mse_sum = 0.0
    val_A_mse_sum = 0.0

    val_R_abs_error_sum = 0.0
    val_A_abs_error_sum = 0.0

    val_count = 0

    with torch.no_grad():

        for (
            R_x,
            A_x,
            env_x,
            R_y,
            A_y,
        ) in val_loader:

            gamma_x = env_x[
                ...,
                0
            ]

            sigma_x = env_x[
                ...,
                1
            ]

            out = model(
                R_x,
                A_x,
                gamma_x,
                sigma_x,
            )

            R_pred = out[
                "R_next"
            ]

            A_pred = out[
                "amplitude_next"
            ]

            R_loss = F.mse_loss(
                R_pred,
                R_y
            )

            A_loss = F.mse_loss(
                A_pred,
                A_y
            )

            loss = (
                R_loss
                +
                A_loss
            )

            batch_n = R_x.shape[0]

            val_loss_sum += (
                loss.item()
                *
                batch_n
            )

            val_R_mse_sum += (
                R_loss.item()
                *
                batch_n
            )

            val_A_mse_sum += (
                A_loss.item()
                *
                batch_n
            )

            val_R_abs_error_sum += (
                torch.mean(
                    torch.abs(
                        R_pred
                        -
                        R_y
                    )
                ).item()
                *
                batch_n
            )

            val_A_abs_error_sum += (
                torch.mean(
                    torch.abs(
                        A_pred
                        -
                        A_y
                    )
                ).item()
                *
                batch_n
            )

            val_count += batch_n

    val_loss = (
        val_loss_sum
        /
        val_count
    )

    val_R_mse = (
        val_R_mse_sum
        /
        val_count
    )

    val_A_mse = (
        val_A_mse_sum
        /
        val_count
    )

    val_R_mae = (
        val_R_abs_error_sum
        /
        val_count
    )

    val_A_mae = (
        val_A_abs_error_sum
        /
        val_count
    )

    elapsed = (
        time.time()
        -
        start_time
    )

    history.append({

        "epoch":
            int(epoch),

        "train_loss":
            float(train_loss),

        "train_R_mse":
            float(train_R_mse),

        "train_amplitude_mse":
            float(train_A_mse),

        "val_loss":
            float(val_loss),

        "val_R_mse":
            float(val_R_mse),

        "val_amplitude_mse":
            float(val_A_mse),

        "val_R_mae":
            float(val_R_mae),

        "val_amplitude_mae":
            float(val_A_mae),

        "time_seconds":
            float(elapsed),
    })

    print(
        f"\nEpoch {epoch:02d}/{EPOCHS}"
    )

    print(
        f"  Train loss      : {train_loss:.12e}"
    )

    print(
        f"  Train R MSE     : {train_R_mse:.12e}"
    )

    print(
        f"  Train A MSE     : {train_A_mse:.12e}"
    )

    print(
        f"  Val loss        : {val_loss:.12e}"
    )

    print(
        f"  Val R MSE       : {val_R_mse:.12e}"
    )

    print(
        f"  Val amplitude   : {val_A_mse:.12e}"
    )

    print(
        f"  Time            : {elapsed:.1f}s"
    )

    # ========================================================
    # BEST CHECKPOINT
    # ========================================================

    if val_loss < best_val:

        best_val = (
            val_loss
        )

        best_epoch = (
            epoch
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "epoch":
                    int(epoch),

                "val_loss":
                    float(val_loss),

                "architecture":
                    {
                        "type":
                            "FNOBaseline",

                        "dim":
                            6,

                        "width":
                            64,

                        "modes":
                            16,

                        "depth":
                            4,
                    },

                "prototype":
                    True,
            },
            BEST_MODEL_PATH
        )

        print(
            "  >>> NEW BEST FNO CHECKPOINT SAVED"
        )

# ============================================================
# 17. SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_path = (
    OUT_DIR
    / "training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)

# ============================================================
# 18. LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()

print(
    "\n[+] Best epoch:",
    best_epoch
)

print(
    "[+] Best validation loss:",
    best_val
)

# ============================================================
# 19. TEST EVALUATION
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "F. FNO PROTOTYPE TEST EVALUATION"
)

print(
    "=" * 100
)

X_R_test_t = X_R_test_t
X_A_test_t = X_A_test_t
ENV_test_t = ENV_test_t

with torch.no_grad():

    gamma_test = ENV_test_t[
        ...,
        0
    ]

    sigma_test = ENV_test_t[
        ...,
        1
    ]

    out = model(
        X_R_test_t,
        X_A_test_t,
        gamma_test,
        sigma_test,
    )

    R_test_pred = out[
        "R_next"
    ]

    A_test_pred = out[
        "amplitude_next"
    ]

test_R_mse, test_R_rmse, test_R_mae, test_R_rel_l2 = (
    compute_batch_metrics(
        R_test_pred,
        Y_R_test_t
    )
)

test_A_mse, test_A_rmse, test_A_mae, test_A_rel_l2 = (
    compute_batch_metrics(
        A_test_pred,
        Y_A_test_t
    )
)

print(
    "R MSE:",
    f"{test_R_mse.item():.12e}"
)

print(
    "R RMSE:",
    f"{test_R_rmse.item():.12e}"
)

print(
    "R MAE:",
    f"{test_R_mae.item():.12e}"
)

print(
    "R Relative-L2:",
    f"{test_R_rel_l2.item():.12e}"
)

print(
    "\nAmplitude MSE:",
    f"{test_A_mse.item():.12e}"
)

print(
    "Amplitude RMSE:",
    f"{test_A_rmse.item():.12e}"
)

print(
    "Amplitude MAE:",
    f"{test_A_mae.item():.12e}"
)

print(
    "Amplitude Relative-L2:",
    f"{test_A_rel_l2.item():.12e}"
)

# ============================================================
# 20. TEST FINITE CHECK
# ============================================================

test_finite = bool(
    torch.isfinite(
        R_test_pred
    ).all().item()
    and
    torch.isfinite(
        A_test_pred
    ).all().item()
)

print(
    "\nTest predictions finite:",
    test_finite
)

# ============================================================
# 21. FNO STRUCTURE DIAGNOSTIC
#
# FNO is deliberately NOT physically constrained.
# We record its output properties for later comparison.
# ============================================================

R_test_trace = torch.diagonal(
    R_test_pred,
    dim1=-2,
    dim2=-1
).sum(
    dim=-1
)

fno_trace_deviation = float(
    torch.mean(
        torch.abs(
            R_test_trace
            -
            1.0
        )
    ).item()
)

fno_trace_max_deviation = float(
    torch.max(
        torch.abs(
            R_test_trace
            -
            1.0
        )
    ).item()
)

fno_symmetry_error = float(
    torch.max(
        torch.abs(
            R_test_pred
            -
            R_test_pred.transpose(
                -1,
                -2
            )
        )
    ).item()
)

fno_eigenvalues = torch.linalg.eigvalsh(
    R_test_pred
)

fno_min_eigenvalue = float(
    torch.min(
        fno_eigenvalues
    ).item()
)

print(
    "\n" + "=" * 100
)

print(
    "G. FNO OUTPUT STRUCTURE"
)

print(
    "=" * 100
)

print(
    "Mean |trace - 1|:",
    f"{fno_trace_deviation:.12e}"
)

print(
    "Max  |trace - 1|:",
    f"{fno_trace_max_deviation:.12e}"
)

print(
    "Max symmetry error:",
    f"{fno_symmetry_error:.12e}"
)

print(
    "Minimum eigenvalue:",
    f"{fno_min_eigenvalue:.12e}"
)

# ============================================================
# 22. SAVE TEST PREDICTIONS
# ============================================================

np.savez_compressed(

    OUT_DIR
    / "fno_prototype_test_predictions.npz",

    R_prediction=
        R_test_pred.cpu().numpy(),

    R_target=
        Y_R_test_t.cpu().numpy(),

    amplitude_prediction=
        A_test_pred.cpu().numpy(),

    amplitude_target=
        Y_A_test_t.cpu().numpy(),

    environment=
        ENV_test_t.cpu().numpy(),
)

# ============================================================
# 23. SAVE SUMMARY
# ============================================================

summary = {

    "model":
        "FNOBaseline",

    "prototype":
        True,

    "train_transitions":
        int(N_train),

    "validation_transitions":
        int(N_val),

    "test_transitions":
        int(N_test),

    "spatial_points":
        int(NX),

    "information_dimension":
        int(D),

    "parameters":
        int(parameter_count),

    "epochs":
        int(EPOCHS),

    "best_epoch":
        int(best_epoch),

    "best_validation_loss":
        float(best_val),

    "test_R_MSE":
        float(
            test_R_mse.item()
        ),

    "test_R_RMSE":
        float(
            test_R_rmse.item()
        ),

    "test_R_MAE":
        float(
            test_R_mae.item()
        ),

    "test_R_relative_L2":
        float(
            test_R_rel_l2.item()
        ),

    "test_amplitude_MSE":
        float(
            test_A_mse.item()
        ),

    "test_amplitude_RMSE":
        float(
            test_A_rmse.item()
        ),

    "test_amplitude_MAE":
        float(
            test_A_mae.item()
        ),

    "test_amplitude_relative_L2":
        float(
            test_A_rel_l2.item()
        ),

    "test_predictions_finite":
        bool(
            test_finite
        ),

    "output_mean_trace_deviation":
        float(
            fno_trace_deviation
        ),

    "output_max_trace_deviation":
        float(
            fno_trace_max_deviation
        ),

    "output_max_symmetry_error":
        float(
            fno_symmetry_error
        ),

    "output_minimum_eigenvalue":
        float(
            fno_min_eigenvalue
        ),

    "physics_constraint":
        False,

    "lindblad_kernel":
        False,

    "environment_inputs":
        [
            "gamma",
            "noise_sigma",
        ],

    "checkpoint":
        str(
            BEST_MODEL_PATH
        ),
}

summary_path = (
    OUT_DIR
    / "fno_prototype_summary.json"
)

with open(
    summary_path,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        allow_nan=False
    )

# ============================================================
# 24. FINAL
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "STEP 10 COMPLETE — FNO BASELINE PROTOTYPE"
)

print(
    "=" * 100
)

print(
    "[+] Best epoch:",
    best_epoch
)

print(
    "[+] Best validation loss:",
    f"{best_val:.12e}"
)

print(
    "[+] Test R RMSE:",
    f"{test_R_rmse.item():.12e}"
)

print(
    "[+] Test R Relative-L2:",
    f"{test_R_rel_l2.item():.12e}"
)

print(
    "[+] Test amplitude RMSE:",
    f"{test_A_rmse.item():.12e}"
)

print(
    "[+] Test amplitude Relative-L2:",
    f"{test_A_rel_l2.item():.12e}"
)

print(
    "[+] Test finite:",
    test_finite
)

print(
    "[+] Checkpoint:",
    BEST_MODEL_PATH
)

print(
    "[+] History:",
    history_path
)

print(
    "[+] Summary:",
    summary_path
)

print(
    "\n[INFO] This FNO is the FAIR unconstrained baseline."
)

print(
    "[INFO] It uses the same task and environment conditioning"
)

print(
    "       that the LNO prototype will receive."
)

print(
    "[INFO] Final sample expansion is still pending."
)

print("=" * 100)
