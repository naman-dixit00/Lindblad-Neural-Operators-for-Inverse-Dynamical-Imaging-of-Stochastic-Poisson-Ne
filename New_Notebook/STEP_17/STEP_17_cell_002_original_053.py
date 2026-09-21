# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 53
# Step            : STEP_17
# Step Heading    : # STEP 17 — FINAL FNO TRAINING
# Step Cell No.   : 2
# ============================================================

# ============================================================
# STEP 17 — FINAL FNO TRAINING
#
# DATASET:
#   Dataset V3 Expanded
#   9,900 transitions
#
# SPLIT:
#   6,930 train
#   1,485 validation
#   1,485 test
#
# TASK:
#   Input:
#       R(t)
#       amplitude(t)
#       gamma
#       noise_sigma
#
#   Target:
#       R(t+1)
#       amplitude(t+1)
#
# MODEL:
#   Unconstrained Fourier Neural Operator baseline
#
# IMPORTANT:
#   - Same task as repaired LNO
#   - Same split
#   - No Lindblad physics
#   - No PSD projection
#   - No trace constraint
#   - No physics penalty
#   - Best validation checkpoint saved
#   - Final test evaluation performed ONCE after
#     best checkpoint is restored
#
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
# 1. ROOT / SEED
# ============================================================

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

SEED = 2026

random.seed(
    SEED
)

np.random.seed(
    SEED
)

torch.manual_seed(
    SEED
)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if device.type == "cuda":

    torch.cuda.manual_seed_all(
        SEED
    )

print("=" * 100)
print("STEP 17 — FINAL FNO TRAINING")
print("=" * 100)

print(
    "[+] Device:",
    device
)


# ============================================================
# 3. PATHS
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

TRAJECTORY_MANIFEST = (
    ROOT
    / "results"
    / "step16_final_splits"
    / "final_trajectory_manifest.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "step17_final_fno"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

assert DATASET_PATH.is_file(), (
    f"Expanded dataset missing:\n{DATASET_PATH}"
)

assert SPLIT_PATH.is_file(), (
    f"Step-16 split missing:\n{SPLIT_PATH}"
)

assert TRAJECTORY_MANIFEST.is_file(), (
    f"Step-16 trajectory manifest missing:\n{TRAJECTORY_MANIFEST}"
)


# ============================================================
# 4. LOAD DATASET
# ============================================================

print(
    "\n[+] Loading expanded dataset..."
)

dataset = np.load(
    DATASET_PATH
)

# ============================================================
# 5. REQUIRED ARRAYS
# ============================================================

required_dataset_keys = [

    "X_R",
    "Y_R",

    "X_amplitude",
    "Y_amplitude",

    "environment",
]

for key in required_dataset_keys:

    assert key in dataset, (
        f"Missing dataset key: {key}"
    )

X_R_all = dataset[
    "X_R"
].astype(
    np.float32
)

Y_R_all = dataset[
    "Y_R"
].astype(
    np.float32
)

X_A_all = dataset[
    "X_amplitude"
].astype(
    np.float32
)

Y_A_all = dataset[
    "Y_amplitude"
].astype(
    np.float32
)

ENV_all = dataset[
    "environment"
].astype(
    np.float32
)

N_TOTAL, NX, DIM, DIM2 = (
    X_R_all.shape
)

assert DIM == 6
assert DIM2 == 6

assert Y_R_all.shape == (
    N_TOTAL,
    NX,
    DIM,
    DIM
)

assert X_A_all.shape == (
    N_TOTAL,
    NX
)

assert Y_A_all.shape == (
    N_TOTAL,
    NX
)

assert ENV_all.shape == (
    N_TOTAL,
    NX,
    2
)

print(
    "[+] Total transitions:",
    N_TOTAL
)

print(
    "[+] Spatial points:",
    NX
)

print(
    "[+] Information dimension:",
    DIM
)


# ============================================================
# 6. LOAD FINAL SPLIT
# ============================================================

split = np.load(
    SPLIT_PATH
)

train_indices = split[
    "train_indices"
].astype(
    np.int64
)

val_indices = split[
    "val_indices"
].astype(
    np.int64
)

test_indices = split[
    "test_indices"
].astype(
    np.int64
)

print(
    "\n" + "=" * 100
)

print(
    "FINAL SPLIT"
)

print(
    "=" * 100
)

print(
    "Train:",
    len(train_indices)
)

print(
    "Validation:",
    len(val_indices)
)

print(
    "Test:",
    len(test_indices)
)

assert len(
    train_indices
) == 6930

assert len(
    val_indices
) == 1485

assert len(
    test_indices
) == 1485

assert (
    len(
        np.unique(
            np.concatenate(
                [
                    train_indices,
                    val_indices,
                    test_indices,
                ]
            )
        )
    )
    ==
    N_TOTAL
)

print(
    "[PASS] Expected 6930 / 1485 / 1485 split."
)


# ============================================================
# 7. INDEX OVERLAP CHECK
# ============================================================

train_set = set(
    train_indices.tolist()
)

val_set = set(
    val_indices.tolist()
)

test_set = set(
    test_indices.tolist()
)

assert train_set.isdisjoint(
    val_set
)

assert train_set.isdisjoint(
    test_set
)

assert val_set.isdisjoint(
    test_set
)

print(
    "[PASS] No transition-index overlap."
)


# ============================================================
# 8. EXTRACT SPLITS
# ============================================================

X_R_train = X_R_all[
    train_indices
]

Y_R_train = Y_R_all[
    train_indices
]

X_A_train = X_A_all[
    train_indices
]

Y_A_train = Y_A_all[
    train_indices
]

ENV_train = ENV_all[
    train_indices
]


X_R_val = X_R_all[
    val_indices
]

Y_R_val = Y_R_all[
    val_indices
]

X_A_val = X_A_all[
    val_indices
]

Y_A_val = Y_A_all[
    val_indices
]

ENV_val = ENV_all[
    val_indices
]


X_R_test = X_R_all[
    test_indices
]

Y_R_test = Y_R_all[
    test_indices
]

X_A_test = X_A_all[
    test_indices
]

Y_A_test = Y_A_all[
    test_indices
]

ENV_test = ENV_all[
    test_indices
]


# ============================================================
# 9. FINITE CHECK
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "FINITE CHECK"
)

print(
    "=" * 100
)

arrays_to_check = {

    "X_R_train":
        X_R_train,

    "Y_R_train":
        Y_R_train,

    "X_A_train":
        X_A_train,

    "Y_A_train":
        Y_A_train,

    "ENV_train":
        ENV_train,

    "X_R_val":
        X_R_val,

    "Y_R_val":
        Y_R_val,

    "X_A_val":
        X_A_val,

    "Y_A_val":
        Y_A_val,

    "ENV_val":
        ENV_val,

    "X_R_test":
        X_R_test,

    "Y_R_test":
        Y_R_test,

    "X_A_test":
        X_A_test,

    "Y_A_test":
        Y_A_test,

    "ENV_test":
        ENV_test,
}

for name, arr in arrays_to_check.items():

    ok = bool(
        np.isfinite(
            arr
        ).all()
    )

    print(
        f"{name:20s}:",
        ok
    )

    assert ok


# ============================================================
# 10. AMPLITUDE NORMALIZATION
#
# IMPORTANT:
# R and amplitude have very different scales.
#
# For a fair multi-output optimization, amplitude is normalized
# using TRAINING DATA ONLY.
#
# This normalization MUST be reused unchanged for the LNO.
# ============================================================

amplitude_mean = float(
    np.mean(
        X_A_train
    )
)

amplitude_std = float(
    np.std(
        X_A_train
    )
)

if amplitude_std < 1e-8:

    amplitude_std = 1.0

X_A_train_n = (
    X_A_train
    -
    amplitude_mean
) / amplitude_std

Y_A_train_n = (
    Y_A_train
    -
    amplitude_mean
) / amplitude_std

X_A_val_n = (
    X_A_val
    -
    amplitude_mean
) / amplitude_std

Y_A_val_n = (
    Y_A_val
    -
    amplitude_mean
) / amplitude_std

X_A_test_n = (
    X_A_test
    -
    amplitude_mean
) / amplitude_std

Y_A_test_n = (
    Y_A_test
    -
    amplitude_mean
) / amplitude_std

print(
    "\n" + "=" * 100
)

print(
    "AMPLITUDE NORMALIZATION"
)

print(
    "=" * 100
)

print(
    "Training amplitude mean:",
    amplitude_mean
)

print(
    "Training amplitude std:",
    amplitude_std
)

print(
    "[PASS] Validation/test normalization uses TRAIN statistics only."
)


# ============================================================
# 11. TORCH DATASETS
# ============================================================

train_dataset = TensorDataset(

    torch.from_numpy(
        X_R_train
    ),

    torch.from_numpy(
        X_A_train_n.astype(
            np.float32
        )
    ),

    torch.from_numpy(
        ENV_train
    ),

    torch.from_numpy(
        Y_R_train
    ),

    torch.from_numpy(
        Y_A_train_n.astype(
            np.float32
        )
    ),
)

val_dataset = TensorDataset(

    torch.from_numpy(
        X_R_val
    ),

    torch.from_numpy(
        X_A_val_n.astype(
            np.float32
        )
    ),

    torch.from_numpy(
        ENV_val
    ),

    torch.from_numpy(
        Y_R_val
    ),

    torch.from_numpy(
        Y_A_val_n.astype(
            np.float32
        )
    ),
)


# ============================================================
# 12. DATALOADERS
# ============================================================

BATCH_SIZE = 16

loader_kwargs = {

    "batch_size":
        BATCH_SIZE,

    "shuffle":
        True,

    "num_workers":
        0,
}

if device.type == "cuda":

    loader_kwargs[
        "pin_memory"
    ] = True

train_loader = DataLoader(
    train_dataset,
    **loader_kwargs
)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=(
        device.type == "cuda"
    ),
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
# 13. SPECTRAL CONVOLUTION
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

        B, C, NX_local = (
            x.shape
        )

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
# 14. FINAL FNO MODEL
# ============================================================

class FinalFNO(
    nn.Module
):

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

        # R = 36
        # amplitude = 1
        # gamma = 1
        # noise_sigma = 1
        #
        # total = 39 channels

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
        sigma,
    ):

        B, NX_local, D, D2 = (
            R.shape
        )

        assert D == self.dim
        assert D2 == self.dim

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

        spectral_norms = []

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

            spectral_norms.append(
                torch.linalg.vector_norm(
                    x,
                    dim=(1, 2)
                )
            )

            if k < (
                self.depth
                -
                1
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

        # Symmetric representation only.
        #
        # IMPORTANT:
        # No PSD projection.
        # No trace normalization.
        # This remains the unconstrained FNO baseline.

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

            "spectral_norms":
                spectral_norms,
        }


# ============================================================
# 15. CREATE MODEL
# ============================================================

model = FinalFNO(
    dim=6,
    width=64,
    modes=16,
    depth=4,
).to(
    device
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
# 16. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=1e-3,

    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=4,
)

# ------------------------------------------------------------
# Final training budget
# ------------------------------------------------------------

MAX_EPOCHS = 40

EARLY_STOPPING_PATIENCE = 8

MIN_DELTA = 1e-6

print(
    "[+] Optimizer: AdamW"
)

print(
    "[+] Initial LR: 1e-3"
)

print(
    "[+] Weight decay: 1e-4"
)

print(
    "[+] Maximum epochs:",
    MAX_EPOCHS
)

print(
    "[+] Early stopping patience:",
    EARLY_STOPPING_PATIENCE
)


# ============================================================
# 17. CHECKPOINT
# ============================================================

BEST_CHECKPOINT = (
    OUT_DIR
    / "fno_final_best.pt"
)

HISTORY_PATH = (
    OUT_DIR
    / "fno_final_training_history.csv"
)


# ============================================================
# 18. TRAINING
# ============================================================

history = []

best_val_loss = float(
    "inf"
)

best_epoch = None

epochs_without_improvement = 0

print(
    "\n" + "=" * 100
)

print(
    "STARTING FINAL FNO TRAINING"
)

print(
    "=" * 100
)

for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    start_time = time.time()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss_sum = 0.0
    train_R_sum = 0.0
    train_A_sum = 0.0

    train_count = 0

    for (
        R_x,
        A_x,
        env_x,
        R_y,
        A_y,
    ) in train_loader:

        R_x = R_x.to(
            device,
            non_blocking=(
                device.type == "cuda"
            )
        )

        A_x = A_x.to(
            device,
            non_blocking=(
                device.type == "cuda"
            )
        )

        env_x = env_x.to(
            device,
            non_blocking=(
                device.type == "cuda"
            )
        )

        R_y = R_y.to(
            device,
            non_blocking=(
                device.type == "cuda"
            )
        )

        A_y = A_y.to(
            device,
            non_blocking=(
                device.type == "cuda"
            )
        )

        gamma_x = env_x[
            ...,
            0
        ]

        sigma_x = env_x[
            ...,
            1
        ]

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            R_x,
            A_x,
            gamma_x,
            sigma_x
        )

        R_pred = output[
            "R_next"
        ]

        A_pred = output[
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

        n_batch = (
            R_x.shape[0]
        )

        train_loss_sum += (
            loss.item()
            *
            n_batch
        )

        train_R_sum += (
            R_loss.item()
            *
            n_batch
        )

        train_A_sum += (
            A_loss.item()
            *
            n_batch
        )

        train_count += (
            n_batch
        )

    train_loss = (
        train_loss_sum
        /
        train_count
    )

    train_R_loss = (
        train_R_sum
        /
        train_count
    )

    train_A_loss = (
        train_A_sum
        /
        train_count
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_sum = 0.0
    val_R_sum = 0.0
    val_A_sum = 0.0

    val_count = 0

    with torch.no_grad():

        for (
            R_x,
            A_x,
            env_x,
            R_y,
            A_y,
        ) in val_loader:

            R_x = R_x.to(
                device,
                non_blocking=(
                    device.type == "cuda"
                )
            )

            A_x = A_x.to(
                device,
                non_blocking=(
                    device.type == "cuda"
                )
            )

            env_x = env_x.to(
                device,
                non_blocking=(
                    device.type == "cuda"
                )
            )

            R_y = R_y.to(
                device,
                non_blocking=(
                    device.type == "cuda"
                )
            )

            A_y = A_y.to(
                device,
                non_blocking=(
                    device.type == "cuda"
                )
            )

            gamma_x = env_x[
                ...,
                0
            ]

            sigma_x = env_x[
                ...,
                1
            ]

            output = model(
                R_x,
                A_x,
                gamma_x,
                sigma_x
            )

            R_pred = output[
                "R_next"
            ]

            A_pred = output[
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

            n_batch = (
                R_x.shape[0]
            )

            val_loss_sum += (
                loss.item()
                *
                n_batch
            )

            val_R_sum += (
                R_loss.item()
                *
                n_batch
            )

            val_A_sum += (
                A_loss.item()
                *
                n_batch
            )

            val_count += (
                n_batch
            )

    val_loss = (
        val_loss_sum
        /
        val_count
    )

    val_R_loss = (
        val_R_sum
        /
        val_count
    )

    val_A_loss = (
        val_A_sum
        /
        val_count
    )

    scheduler.step(
        val_loss
    )

    current_lr = float(
        optimizer.param_groups[
            0
        ][
            "lr"
        ]
    )

    elapsed = (
        time.time()
        -
        start_time
    )

    improved = bool(
        val_loss
        <
        (
            best_val_loss
            -
            MIN_DELTA
        )
    )

    if improved:

        best_val_loss = (
            val_loss
        )

        best_epoch = (
            epoch
        )

        epochs_without_improvement = 0

        torch.save(
            {

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "epoch":
                    int(epoch),

                "best_validation_loss":
                    float(
                        best_val_loss
                    ),

                "architecture":
                    {

                        "name":
                            "FinalFNO",

                        "dim":
                            6,

                        "width":
                            64,

                        "modes":
                            16,

                        "depth":
                            4,
                    },

                "training":
                    {

                        "batch_size":
                            BATCH_SIZE,

                        "max_epochs":
                            MAX_EPOCHS,

                        "learning_rate":
                            1e-3,

                        "weight_decay":
                            1e-4,
                    },

                "amplitude_normalization":
                    {

                        "mean":
                            amplitude_mean,

                        "std":
                            amplitude_std,
                    },

                "physics_constraint":
                    False,

                "lindblad_kernel":
                    False,

                "prototype":
                    False,

                "dataset":
                    "Dataset V3 Expanded",
            },

            BEST_CHECKPOINT
        )

    else:

        epochs_without_improvement += 1

    history.append({

        "epoch":
            int(epoch),

        "train_loss":
            float(train_loss),

        "train_R_loss":
            float(train_R_loss),

        "train_amplitude_loss":
            float(train_A_loss),

        "val_loss":
            float(val_loss),

        "val_R_loss":
            float(val_R_loss),

        "val_amplitude_loss":
            float(val_A_loss),

        "learning_rate":
            current_lr,

        "time_seconds":
            float(elapsed),

        "improved":
            bool(improved),
    })

    print(
        f"\nEpoch {epoch:02d}/{MAX_EPOCHS}"
    )

    print(
        f"  Train total : {train_loss:.10e}"
    )

    print(
        f"  Train R     : {train_R_loss:.10e}"
    )

    print(
        f"  Train Amp   : {train_A_loss:.10e}"
    )

    print(
        f"  Val total   : {val_loss:.10e}"
    )

    print(
        f"  Val R       : {val_R_loss:.10e}"
    )

    print(
        f"  Val Amp     : {val_A_loss:.10e}"
    )

    print(
        f"  LR          : {current_lr:.3e}"
    )

    print(
        f"  Time        : {elapsed:.2f}s"
    )

    if improved:

        print(
            "  >>> NEW BEST CHECKPOINT"
        )

    else:

        print(
            f"  No improvement: "
            f"{epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )

    if (
        epochs_without_improvement
        >=
        EARLY_STOPPING_PATIENCE
    ):

        print(
            "\n[INFO] Early stopping triggered."
        )

        break


# ============================================================
# 19. SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# ============================================================
# 20. RESTORE BEST MODEL
# ============================================================

assert BEST_CHECKPOINT.is_file()

checkpoint = torch.load(
    BEST_CHECKPOINT,
    map_location=device
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

model.eval()

print(
    "\n" + "=" * 100
)

print(
    "BEST FNO CHECKPOINT RESTORED"
)

print(
    "=" * 100
)

print(
    "Best epoch:",
    checkpoint[
        "epoch"
    ]
)

print(
    "Best validation loss:",
    checkpoint[
        "best_validation_loss"
    ]
)


# ============================================================
# 21. FINAL TEST EVALUATION
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "FINAL FNO TEST EVALUATION"
)

print(
    "=" * 100
)

X_R_test_t = torch.from_numpy(
    X_R_test
).to(
    device
)

Y_R_test_t = torch.from_numpy(
    Y_R_test
).to(
    device
)

X_A_test_t = torch.from_numpy(
    X_A_test_n.astype(
        np.float32
    )
).to(
    device
)

Y_A_test_t = torch.from_numpy(
    Y_A_test_n.astype(
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

with torch.no_grad():

    gamma_test = ENV_test_t[
        ...,
        0
    ]

    sigma_test = ENV_test_t[
        ...,
        1
    ]

    output = model(
        X_R_test_t,
        X_A_test_t,
        gamma_test,
        sigma_test
    )

    R_pred = output[
        "R_next"
    ]

    A_pred_normalized = output[
        "amplitude_next"
    ]

# ============================================================
# 22. RESTORE AMPLITUDE SCALE
# ============================================================

A_pred = (
    A_pred_normalized
    *
    amplitude_std
    +
    amplitude_mean
)

A_target = (
    Y_A_test_t
    *
    amplitude_std
    +
    amplitude_mean
)

# ============================================================
# 23. METRICS
# ============================================================

R_error = (
    R_pred
    -
    Y_R_test_t
)

A_error = (
    A_pred
    -
    A_target
)

R_MSE = torch.mean(
    R_error ** 2
)

R_RMSE = torch.sqrt(
    R_MSE
    +
    1e-12
)

R_MAE = torch.mean(
    torch.abs(
        R_error
    )
)

R_relative_L2 = (
    torch.linalg.vector_norm(
        R_error
    )
    /
    (
        torch.linalg.vector_norm(
            Y_R_test_t
        )
        +
        1e-12
    )
)

A_MSE = torch.mean(
    A_error ** 2
)

A_RMSE = torch.sqrt(
    A_MSE
    +
    1e-12
)

A_MAE = torch.mean(
    torch.abs(
        A_error
    )
)

A_relative_L2 = (
    torch.linalg.vector_norm(
        A_error
    )
    /
    (
        torch.linalg.vector_norm(
            A_target
        )
        +
        1e-12
    )
)

print(
    "\nR metrics:"
)

print(
    "  MSE:",
    f"{R_MSE.item():.12e}"
)

print(
    "  RMSE:",
    f"{R_RMSE.item():.12e}"
)

print(
    "  MAE:",
    f"{R_MAE.item():.12e}"
)

print(
    "  Relative-L2:",
    f"{R_relative_L2.item():.12e}"
)

print(
    "\nAmplitude metrics:"
)

print(
    "  MSE:",
    f"{A_MSE.item():.12e}"
)

print(
    "  RMSE:",
    f"{A_RMSE.item():.12e}"
)

print(
    "  MAE:",
    f"{A_MAE.item():.12e}"
)

print(
    "  Relative-L2:",
    f"{A_relative_L2.item():.12e}"
)


# ============================================================
# 24. FNO STRUCTURAL DIAGNOSTICS
#
# These are diagnostics ONLY.
# FNO is not constrained to satisfy them.
# ============================================================

R_trace = torch.diagonal(
    R_pred,
    dim1=-2,
    dim2=-1
).sum(
    dim=-1
)

trace_abs_error = torch.abs(
    R_trace
    -
    1.0
)

mean_trace_error = float(
    trace_abs_error.mean().item()
)

max_trace_error = float(
    trace_abs_error.max().item()
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

# Eigenvalues can be expensive but dimension is only 6.
eigvalues = torch.linalg.eigvalsh(
    R_pred
)

minimum_eigenvalue = float(
    eigvalues.min().item()
)

negative_eigenvalue_fraction = float(
    (
        eigvalues
        <
        0
    )
    .float()
    .mean()
    .item()
)

test_predictions_finite = bool(
    torch.isfinite(
        R_pred
    ).all().item()
    and
    torch.isfinite(
        A_pred
    ).all().item()
)

print(
    "\n" + "=" * 100
)

print(
    "FNO OUTPUT STRUCTURE DIAGNOSTIC"
)

print(
    "=" * 100
)

print(
    "Mean |trace - 1|:",
    f"{mean_trace_error:.12e}"
)

print(
    "Max |trace - 1|:",
    f"{max_trace_error:.12e}"
)

print(
    "Max symmetry error:",
    f"{symmetry_error:.12e}"
)

print(
    "Minimum eigenvalue:",
    f"{minimum_eigenvalue:.12e}"
)

print(
    "Negative-eigenvalue fraction:",
    f"{negative_eigenvalue_fraction:.6f}"
)

print(
    "Predictions finite:",
    test_predictions_finite
)


# ============================================================
# 25. REGIME-WISE TEST METRICS
#
# Use the Step-16 transition manifest.
# ============================================================

manifest = pd.read_csv(
    ROOT
    / "results"
    / "step16_final_splits"
    / "final_transition_manifest.csv"
)

test_manifest = manifest.iloc[
    test_indices
].copy()

test_manifest = (
    test_manifest
    .reset_index(
        drop=True
    )
)

assert len(
    test_manifest
) == len(
    test_indices
)

R_error_np = (
    R_error
    .detach()
    .cpu()
    .numpy()
)

A_error_np = (
    A_error
    .detach()
    .cpu()
    .numpy()
)

regime_rows = []

for regime in sorted(
    test_manifest[
        "regime"
    ].unique()
):

    mask = (
        test_manifest[
            "regime"
        ]
        .to_numpy()
        ==
        regime
    )

    regime_R_error = (
        R_error_np[
            mask
        ]
    )

    regime_A_error = (
        A_error_np[
            mask
        ]
    )

    regime_R_target = (
        Y_R_test[
            mask
        ]
    )

    regime_A_target = (
        A_target
        .detach()
        .cpu()
        .numpy()[
            mask
        ]
    )

    r_rel = (
        np.linalg.norm(
            regime_R_error
        )
        /
        (
            np.linalg.norm(
                regime_R_target
            )
            +
            1e-12
        )
    )

    a_rel = (
        np.linalg.norm(
            regime_A_error
        )
        /
        (
            np.linalg.norm(
                regime_A_target
            )
            +
            1e-12
        )
    )

    regime_rows.append({

        "regime":
            regime,

        "transitions":
            int(
                mask.sum()
            ),

        "R_RMSE":
            float(
                np.sqrt(
                    np.mean(
                        regime_R_error ** 2
                    )
                )
            ),

        "R_MAE":
            float(
                np.mean(
                    np.abs(
                        regime_R_error
                    )
                )
            ),

        "R_relative_L2":
            float(
                r_rel
            ),

        "amplitude_RMSE":
            float(
                np.sqrt(
                    np.mean(
                        regime_A_error ** 2
                    )
                )
            ),

        "amplitude_MAE":
            float(
                np.mean(
                    np.abs(
                        regime_A_error
                    )
                )
            ),

        "amplitude_relative_L2":
            float(
                a_rel
            ),
    })

regime_df = pd.DataFrame(
    regime_rows
)

print(
    "\n" + "=" * 100
)

print(
    "REGIME-WISE FNO TEST PERFORMANCE"
)

print(
    "=" * 100
)

display(
    regime_df.round(8)
)


# ============================================================
# 26. SAVE TEST PREDICTIONS
# ============================================================

PREDICTIONS_PATH = (
    OUT_DIR
    / "fno_final_test_predictions.npz"
)

np.savez_compressed(

    PREDICTIONS_PATH,

    R_prediction=
        R_pred
        .detach()
        .cpu()
        .numpy(),

    R_target=
        Y_R_test,

    amplitude_prediction=
        A_pred
        .detach()
        .cpu()
        .numpy(),

    amplitude_target=
        A_target
        .detach()
        .cpu()
        .numpy(),

    environment=
        ENV_test,

    test_indices=
        test_indices,
)


# ============================================================
# 27. SAVE REGIME METRICS
# ============================================================

REGIME_METRICS_PATH = (
    OUT_DIR
    / "fno_final_regime_metrics.csv"
)

regime_df.to_csv(
    REGIME_METRICS_PATH,
    index=False
)


# ============================================================
# 28. FINAL SUMMARY
# ============================================================

summary = {

    "step":
        17,

    "model":
        "FinalFNO",

    "prototype":
        False,

    "dataset":
        "Dataset V3 Expanded",

    "total_transitions":
        int(N_TOTAL),

    "train_transitions":
        int(len(train_indices)),

    "validation_transitions":
        int(len(val_indices)),

    "test_transitions":
        int(len(test_indices)),

    "spatial_points":
        int(NX),

    "information_dimension":
        int(DIM),

    "parameters":
        int(parameter_count),

    "best_epoch":
        int(best_epoch),

    "best_validation_loss":
        float(best_val_loss),

    "max_epochs":
        int(MAX_EPOCHS),

    "batch_size":
        int(BATCH_SIZE),

    "learning_rate":
        1e-3,

    "weight_decay":
        1e-4,

    "test_metrics":
        {

            "R_MSE":
                float(
                    R_MSE.item()
                ),

            "R_RMSE":
                float(
                    R_RMSE.item()
                ),

            "R_MAE":
                float(
                    R_MAE.item()
                ),

            "R_relative_L2":
                float(
                    R_relative_L2.item()
                ),

            "amplitude_MSE":
                float(
                    A_MSE.item()
                ),

            "amplitude_RMSE":
                float(
                    A_RMSE.item()
                ),

            "amplitude_MAE":
                float(
                    A_MAE.item()
                ),

            "amplitude_relative_L2":
                float(
                    A_relative_L2.item()
                ),
        },

    "structural_diagnostics":
        {

            "mean_trace_error":
                mean_trace_error,

            "max_trace_error":
                max_trace_error,

            "symmetry_error":
                symmetry_error,

            "minimum_eigenvalue":
                minimum_eigenvalue,

            "negative_eigenvalue_fraction":
                negative_eigenvalue_fraction,

            "predictions_finite":
                test_predictions_finite,
        },

    "amplitude_normalization":
        {

            "mean":
                amplitude_mean,

            "std":
                amplitude_std,
        },

    "physics_constraint":
        False,

    "lindblad_kernel":
        False,

    "split":
        str(
            SPLIT_PATH
        ),

    "checkpoint":
        str(
            BEST_CHECKPOINT
        ),

    "history":
        str(
            HISTORY_PATH
        ),

    "predictions":
        str(
            PREDICTIONS_PATH
        ),

    "regime_metrics":
        str(
            REGIME_METRICS_PATH
        ),
}

SUMMARY_PATH = (
    OUT_DIR
    / "fno_final_summary.json"
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
# 29. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "STEP 17 COMPLETE — FINAL FNO"
)

print(
    "=" * 100
)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best validation loss:",
    f"{best_val_loss:.12e}"
)

print(
    "\nFINAL TEST"
)

print(
    "R RMSE:",
    f"{R_RMSE.item():.12e}"
)

print(
    "R Relative-L2:",
    f"{R_relative_L2.item():.12e}"
)

print(
    "Amplitude RMSE:",
    f"{A_RMSE.item():.12e}"
)

print(
    "Amplitude Relative-L2:",
    f"{A_relative_L2.item():.12e}"
)

print(
    "\nSTRUCTURAL DIAGNOSTICS"
)

print(
    "Mean trace error:",
    f"{mean_trace_error:.12e}"
)

print(
    "Max trace error:",
    f"{max_trace_error:.12e}"
)

print(
    "Minimum eigenvalue:",
    f"{minimum_eigenvalue:.12e}"
)

print(
    "Negative-eigenvalue fraction:",
    f"{negative_eigenvalue_fraction:.6f}"
)

print(
    "\nCheckpoint:",
    BEST_CHECKPOINT
)

print(
    "History:",
    HISTORY_PATH
)

print(
    "Predictions:",
    PREDICTIONS_PATH
)

print(
    "Regime metrics:",
    REGIME_METRICS_PATH
)

print(
    "Summary:",
    SUMMARY_PATH
)

print(
    "\n[INFO] This is the final FNO baseline training run."
)

print(
    "[INFO] The same amplitude normalization must be reused"
)

print(
    "       for the final repaired-LNO training."
)

print(
    "[INFO] Final LNO comparison comes next."
)

print("=" * 100)
