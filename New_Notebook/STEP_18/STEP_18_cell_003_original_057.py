# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 57
# Step            : STEP_18
# Step Heading    : # STEP 18 — FINAL REPAIRED LNO TRAINING
# Step Cell No.   : 3
# ============================================================

# ============================================================
# STEP 18 — FINAL REPAIRED LNO TRAINING
# COMPLETE SINGLE-CELL VERSION
# ============================================================
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
# SAME TASK AS FNO:
#   INPUT:
#       R(t)
#       amplitude(t)
#       gamma
#       noise_sigma
#
#   TARGET:
#       R(t+1)
#       amplitude(t+1)
#
# LNO:
#   Spatial Fourier pathway
#   Lindblad dissipative generator in forward path
#   Positive trainable Lindblad coupling
#   Positive trainable neural coupling
#   Symmetry + PSD + trace-normalized R output
#
# FAIRNESS:
#   Same dataset
#   Same split
#   Same target
#   Same amplitude normalization as final FNO
#   Fresh final-LNO initialization
#
# ============================================================

import os
import json
import math
import time
import random

from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 1. ROOT / REPRODUCIBILITY
# ============================================================

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

SEED = 2026

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 100)
print("STEP 18 — FINAL REPAIRED LNO TRAINING")
print("=" * 100)

print("[+] Device:", device)


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

FNO_SUMMARY_PATH = (
    ROOT
    / "results"
    / "step17_final_fno"
    / "fno_final_summary.json"
)

STEP14_CHECKPOINT = (
    ROOT
    / "results"
    / "step14_repaired_lno"
    / "repaired_lno_best.pt"
)

OUT_DIR = (
    ROOT
    / "results"
    / "step18_final_lno"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_CHECKPOINT = (
    OUT_DIR
    / "lno_final_best.pt"
)

LAST_CHECKPOINT = (
    OUT_DIR
    / "lno_final_last.pt"
)

HISTORY_PATH = (
    OUT_DIR
    / "lno_final_training_history.csv"
)

PREDICTIONS_PATH = (
    OUT_DIR
    / "lno_final_test_predictions.npz"
)

ENVIRONMENT_PATH = (
    OUT_DIR
    / "lno_final_environment_response.csv"
)

ROLLOUT_PATH = (
    OUT_DIR
    / "lno_final_rollout_stability.csv"
)

REGIME_PATH = (
    OUT_DIR
    / "lno_final_regime_metrics.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "lno_final_summary.json"
)


# ============================================================
# 4. REQUIRED FILE CHECK
# ============================================================

required_paths = {

    "Dataset V3 Expanded":
        DATASET_PATH,

    "Step 16 split":
        SPLIT_PATH,

    "Step 17 FNO summary":
        FNO_SUMMARY_PATH,

    "Step 14 repaired LNO":
        STEP14_CHECKPOINT,
}

print("\n" + "=" * 100)
print("REQUIRED FILE CHECK")
print("=" * 100)

for name, path in required_paths.items():

    exists = path.is_file()

    print(
        f"{name:30s}:",
        exists,
        "|",
        path
    )

    if not exists:
        raise FileNotFoundError(
            f"\nRequired file missing:\n{path}"
        )

print("\n[PASS] All required files available.")


# ============================================================
# 5. LOAD EXPANDED DATASET
# ============================================================

print("\n" + "=" * 100)
print("LOADING DATASET")
print("=" * 100)

data = np.load(
    DATASET_PATH
)

X_R_all = data[
    "X_R"
].astype(
    np.float32
)

Y_R_all = data[
    "Y_R"
].astype(
    np.float32
)

X_A_all = data[
    "X_amplitude"
].astype(
    np.float32
)

Y_A_all = data[
    "Y_amplitude"
].astype(
    np.float32
)

ENV_all = data[
    "environment"
].astype(
    np.float32
)

N_TOTAL, NX, DIM, DIM2 = X_R_all.shape

assert N_TOTAL == 9900
assert NX == 128
assert DIM == 6
assert DIM2 == 6

assert Y_R_all.shape == (
    N_TOTAL,
    NX,
    6,
    6
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

print("[+] Dataset:", DATASET_PATH)
print("[+] Total transitions:", N_TOTAL)
print("[+] Spatial points:", NX)
print("[+] Information dimension:", DIM)


# ============================================================
# 6. LOAD FINAL SPLIT
# ============================================================

print("\n" + "=" * 100)
print("LOADING STEP 16 FINAL SPLIT")
print("=" * 100)

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

assert len(train_indices) == 6930
assert len(val_indices) == 1485
assert len(test_indices) == 1485

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

print("[+] Train:", len(train_indices))
print("[+] Validation:", len(val_indices))
print("[+] Test:", len(test_indices))
print("[PASS] Split = 6930 / 1485 / 1485")
print("[PASS] No index overlap")


# ============================================================
# 7. EXTRACT SPLITS
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
# 8. FINITE AUDIT
# ============================================================

print("\n" + "=" * 100)
print("FINITE AUDIT")
print("=" * 100)

for name, arr in {

    "X_R_train": X_R_train,
    "Y_R_train": Y_R_train,
    "X_A_train": X_A_train,
    "Y_A_train": Y_A_train,
    "ENV_train": ENV_train,

    "X_R_val": X_R_val,
    "Y_R_val": Y_R_val,
    "X_A_val": X_A_val,
    "Y_A_val": Y_A_val,
    "ENV_val": ENV_val,

    "X_R_test": X_R_test,
    "Y_R_test": Y_R_test,
    "X_A_test": X_A_test,
    "Y_A_test": Y_A_test,
    "ENV_test": ENV_test,

}.items():

    ok = bool(
        np.isfinite(arr).all()
    )

    print(
        f"{name:20s}:",
        ok
    )

    assert ok

print("[PASS] All split tensors finite.")


# ============================================================
# 9. LOAD EXACT FNO AMPLITUDE NORMALIZATION
# ============================================================

print("\n" + "=" * 100)
print("FNO-CONSISTENT AMPLITUDE NORMALIZATION")
print("=" * 100)

with open(
    FNO_SUMMARY_PATH,
    "r"
) as f:

    fno_summary = json.load(
        f
    )

fno_norm = fno_summary.get(
    "amplitude_normalization"
)

if fno_norm is None:

    raise RuntimeError(
        "Step 17 FNO summary does not contain "
        "amplitude_normalization."
    )

amplitude_mean = float(
    fno_norm["mean"]
)

amplitude_std = float(
    fno_norm["std"]
)

assert amplitude_std > 0.0

print(
    "Mean:",
    f"{amplitude_mean:.12e}"
)

print(
    "Std :",
    f"{amplitude_std:.12e}"
)


# ============================================================
# 10. AMPLITUDE NORMALIZATION
# ============================================================

def normalize_amplitude(
    x
):

    return (
        x
        -
        amplitude_mean
    ) / amplitude_std


X_A_train_n = normalize_amplitude(
    X_A_train
)

Y_A_train_n = normalize_amplitude(
    Y_A_train
)

X_A_val_n = normalize_amplitude(
    X_A_val
)

Y_A_val_n = normalize_amplitude(
    Y_A_val
)

X_A_test_n = normalize_amplitude(
    X_A_test
)

Y_A_test_n = normalize_amplitude(
    Y_A_test
)

print(
    "[PASS] Same normalization as FNO."
)


# ============================================================
# 11. PYTORCH DATASETS
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
    )
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
    )
)


# ============================================================
# 12. DATALOADERS
# ============================================================

BATCH_SIZE = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(
        device.type == "cuda"
    )
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(
        device.type == "cuda"
    )
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
        modes
    ):

        super().__init__()

        self.in_channels = (
            in_channels
        )

        self.out_channels = (
            out_channels
        )

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
            n=NX_local,
            dim=-1
        )


# ============================================================
# 14. LINDBLAD KERNEL
# ============================================================

class LindbladKernel(
    nn.Module
):

    def __init__(
        self,
        dim=6,
        channels=4
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

    def forward(
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


# ============================================================
# 15. FINAL REPAIRED LNO
# ============================================================

class FinalRepairedLNO(
    nn.Module
):

    def __init__(
        self,
        dim=6,
        width=64,
        modes=16,
        depth=4,
        lindblad_channels=4
    ):

        super().__init__()

        self.dim = dim
        self.width = width
        self.depth = depth

        # ----------------------------------------------------
        # R = 36 channels
        # amplitude = 1
        # gamma = 1
        # sigma = 1
        #
        # total = 39
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Lindblad forward-path generator
        # ----------------------------------------------------

        self.lindblad = LindbladKernel(
            dim=dim,
            channels=lindblad_channels
        )

        # Positive trainable dynamical couplings
        self.raw_kappa_lindblad = nn.Parameter(
            torch.tensor(
                0.0
            )
        )

        self.raw_kappa_neural = nn.Parameter(
            torch.tensor(
                0.0
            )
        )


    # --------------------------------------------------------
    # Positive couplings
    # --------------------------------------------------------

    def kappa_lindblad(
        self
    ):

        return F.softplus(
            self.raw_kappa_lindblad
        )


    def kappa_neural(
        self
    ):

        return F.softplus(
            self.raw_kappa_neural
        )


    # --------------------------------------------------------
    # PSD + trace-one projection
    # --------------------------------------------------------

    def project_R(
        self,
        M
    ):

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

        eigvals, eigvecs = torch.linalg.eigh(
            M
        )

        eigvals = torch.clamp(
            eigvals,
            min=1e-8
        )

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
                trace.unsqueeze(
                    -1
                )
                +
                1e-12
            )
        )

        return M_psd


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        R,
        amplitude,
        gamma,
        sigma
    ):

        B, NX_local, D, D2 = R.shape

        assert D == self.dim
        assert D2 == self.dim

        # ----------------------------------------------------
        # Flatten information matrix
        # ----------------------------------------------------

        R_flat = R.reshape(
            B,
            NX_local,
            D * D
        )

        # ----------------------------------------------------
        # Environment conditioning
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Input projection
        # ----------------------------------------------------

        x = self.input_projection(
            x
        )

        # ----------------------------------------------------
        # Spatial Fourier operator
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
        # Neural dynamical generator
        # ----------------------------------------------------

        G_N = self.R_head(
            x
        )

        G_N = (
            G_N
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

        G_N = (
            0.5
            *
            (
                G_N
                +
                G_N.transpose(
                    -1,
                    -2
                )
            )
        )

        G_N = torch.tanh(
            G_N
        )

        # ----------------------------------------------------
        # Lindblad dynamical generator
        # ----------------------------------------------------

        G_L = self.lindblad(
            R
        )

        kL = self.kappa_lindblad()
        kN = self.kappa_neural()

        gamma_field = gamma[
            ...,
            None,
            None
        ]

        delta_R_lindblad = (
            kL
            *
            gamma_field
            *
            G_L
        )

        delta_R_neural = (
            kN
            *
            G_N
        )

        delta_R_total = (
            delta_R_lindblad
            +
            delta_R_neural
        )

        # ----------------------------------------------------
        # Candidate next information state
        # ----------------------------------------------------

        R_candidate = (
            R
            +
            delta_R_total
        )

        # ----------------------------------------------------
        # Structural information-state enforcement
        # ----------------------------------------------------

        R_next = self.project_R(
            R_candidate
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

            "G_lindblad":
                G_L,

            "G_neural":
                G_N,

            "delta_R_lindblad":
                delta_R_lindblad,

            "delta_R_neural":
                delta_R_neural,

            "delta_R_total":
                delta_R_total,

            "kappa_lindblad":
                kL,

            "kappa_neural":
                kN,
        }


# ============================================================
# 16. MODEL
# ============================================================

print("\n" + "=" * 100)
print("MODEL INITIALIZATION")
print("=" * 100)

model = FinalRepairedLNO(
    dim=6,
    width=64,
    modes=16,
    depth=4,
    lindblad_channels=4
).to(
    device
)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "[+] Model: FinalRepairedLNO"
)

print(
    "[+] Parameters:",
    f"{parameter_count:,}"
)

print(
    "[+] Expected prototype architecture:",
    "546,039 parameters"
)

print(
    "[+] Initial kappa_L:",
    f"{model.kappa_lindblad().item():.8e}"
)

print(
    "[+] Initial kappa_N:",
    f"{model.kappa_neural().item():.8e}"
)

print(
    "[+] Fresh initialization:",
    True
)


# ============================================================
# 17. DO NOT LOAD STEP-14 WEIGHTS
# ============================================================

print(
    "\n[INFO] Step-14 checkpoint preserved:",
    STEP14_CHECKPOINT.is_file()
)

print(
    "[INFO] Step-14 checkpoint is NOT used for final initialization."
)

print(
    "[INFO] Final LNO starts fresh for a fair training comparison."
)


# ============================================================
# 18. OPTIMIZER / SCHEDULER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=4
)

MAX_EPOCHS = 40

EARLY_STOPPING_PATIENCE = 8

MIN_DELTA = 1e-6

print(
    "\n[+] Optimizer: AdamW"
)

print(
    "[+] Learning rate: 1e-3"
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
# 19. TRAINING
# ============================================================

history = []

best_val_loss = float(
    "inf"
)

best_epoch = None

epochs_without_improvement = 0

print("\n" + "=" * 100)
print("STARTING FINAL LNO TRAINING")
print("=" * 100)

for epoch in range(
    1,
    MAX_EPOCHS + 1
):

    start_time = time.time()

    # ========================================================
    # TRAIN
    # ========================================================

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
        A_y
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

        n_batch = R_x.shape[0]

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

        train_count += n_batch

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

    # ========================================================
    # VALIDATION
    # ========================================================

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
            A_y
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

            n_batch = R_x.shape[0]

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

            val_count += n_batch

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

    kappa_L = float(
        model
        .kappa_lindblad()
        .item()
    )

    kappa_N = float(
        model
        .kappa_neural()
        .item()
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
                            "FinalRepairedLNO",

                        "dim":
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

                "physics":
                    {
                        "lindblad_in_forward_path":
                            True,

                        "physics_loss_penalty":
                            False,

                        "positive_lindblad_coupling":
                            True,

                        "positive_neural_coupling":
                            True,

                        "fresh_initialization":
                            True,
                    },

                "dataset":
                    "Dataset V3 Expanded",

                "prototype":
                    False,
            },

            BEST_CHECKPOINT
        )

    else:

        epochs_without_improvement += 1

    # --------------------------------------------------------
    # ALWAYS SAVE LAST CHECKPOINT
    # --------------------------------------------------------

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

            "validation_loss":
                float(val_loss),

            "amplitude_normalization":
                {
                    "mean":
                        amplitude_mean,

                    "std":
                        amplitude_std,
                },
        },

        LAST_CHECKPOINT
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

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
            float(current_lr),

        "kappa_lindblad":
            float(kappa_L),

        "kappa_neural":
            float(kappa_N),

        "time_seconds":
            float(elapsed),

        "improved":
            bool(improved),
    })

    # Save history EVERY epoch
    pd.DataFrame(
        history
    ).to_csv(
        HISTORY_PATH,
        index=False
    )

    # ========================================================
    # LOG
    # ========================================================

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
        f"  kappa_L     : {kappa_L:.8e}"
    )

    print(
        f"  kappa_N     : {kappa_N:.8e}"
    )

    print(
        f"  LR          : {current_lr:.3e}"
    )

    print(
        f"  Time        : {elapsed:.2f}s"
    )

    if improved:

        print(
            "  >>> NEW BEST LNO CHECKPOINT"
        )

    else:

        print(
            "  No improvement:"
            f" {epochs_without_improvement}/"
            f"{EARLY_STOPPING_PATIENCE}"
        )

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

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
# 20. TRAINING COMPLETE
# ============================================================

if not BEST_CHECKPOINT.is_file():

    raise RuntimeError(
        "Final LNO training ended without producing "
        "a best checkpoint."
    )

print(
    "\n" + "=" * 100
)

print(
    "TRAINING COMPLETE"
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
    "Best checkpoint:",
    BEST_CHECKPOINT
)

print(
    "Last checkpoint:",
    LAST_CHECKPOINT
)

print(
    "History:",
    HISTORY_PATH
)


# ============================================================
# 21. RESTORE BEST CHECKPOINT
# ============================================================

best_checkpoint = torch.load(
    BEST_CHECKPOINT,
    map_location=device
)

model.load_state_dict(
    best_checkpoint[
        "model_state_dict"
    ]
)

model.eval()

print(
    "\n[PASS] Best LNO checkpoint restored."
)


# ============================================================
# 22. FINAL TEST FORWARD
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "FINAL LNO TEST EVALUATION"
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

    delta_L = output[
        "delta_R_lindblad"
    ]

    delta_N = output[
        "delta_R_neural"
    ]

    delta_total = output[
        "delta_R_total"
    ]


# ============================================================
# 23. RESTORE PHYSICAL AMPLITUDE
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
# 24. METRICS
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
    "R MSE:",
    f"{R_MSE.item():.12e}"
)

print(
    "R RMSE:",
    f"{R_RMSE.item():.12e}"
)

print(
    "R MAE:",
    f"{R_MAE.item():.12e}"
)

print(
    "R Relative-L2:",
    f"{R_relative_L2.item():.12e}"
)

print(
    "\nAmplitude MSE:",
    f"{A_MSE.item():.12e}"
)

print(
    "Amplitude RMSE:",
    f"{A_RMSE.item():.12e}"
)

print(
    "Amplitude MAE:",
    f"{A_MAE.item():.12e}"
)

print(
    "Amplitude Relative-L2:",
    f"{A_relative_L2.item():.12e}"
)


# ============================================================
# 25. STRUCTURE
# ============================================================

trace = torch.diagonal(
    R_pred,
    dim1=-2,
    dim2=-1
).sum(
    dim=-1
)

trace_error = torch.abs(
    trace
    -
    1.0
)

mean_trace_error = float(
    trace_error.mean().item()
)

max_trace_error = float(
    trace_error.max().item()
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

predictions_finite = bool(
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
    "LNO OUTPUT STRUCTURE"
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
    "Symmetry error:",
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
    predictions_finite
)


# ============================================================
# 26. ENVIRONMENT RESPONSE
# ============================================================

probe_R = X_R_test_t[
    :1
].clone()

probe_A = torch.from_numpy(
    X_A_test_n[
        :1
    ].astype(
        np.float32
    )
).to(
    device
)

probe_sigma = ENV_test_t[
    :1,
    :,
    1
]

environment_rows = []

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
            ENV_test_t[
                :1,
                :,
                0
            ],
            gamma_value
        )

        env_output = model(
            probe_R,
            probe_A,
            gamma_probe,
            probe_sigma
        )

        environment_rows.append({

            "gamma":
                float(
                    gamma_value
                ),

            "R_change":
                float(
                    torch.linalg.vector_norm(
                        env_output[
                            "R_next"
                        ]
                        -
                        probe_R
                    ).item()
                ),

            "amplitude_change":
                float(
                    torch.linalg.vector_norm(
                        env_output[
                            "amplitude_next"
                        ]
                        -
                        probe_A
                    ).item()
                ),

            "Lindblad_delta_norm":
                float(
                    torch.linalg.vector_norm(
                        env_output[
                            "delta_R_lindblad"
                        ]
                    ).item()
                ),

            "Neural_delta_norm":
                float(
                    torch.linalg.vector_norm(
                        env_output[
                            "delta_R_neural"
                        ]
                    ).item()
                ),
        })

environment_df = pd.DataFrame(
    environment_rows
)

print(
    "\n" + "=" * 100
)

print(
    "ENVIRONMENT COUPLING RESPONSE"
)

print(
    "=" * 100
)

display(
    environment_df.round(8)
)

environment_df.to_csv(
    ENVIRONMENT_PATH,
    index=False
)


# ============================================================
# 27. 40-STEP ROLLOUT
# ============================================================

R_roll = X_R_test_t[
    :1
].clone()

A_roll = torch.from_numpy(
    X_A_test_n[
        :1
    ].astype(
        np.float32
    )
).to(
    device
)

gamma_roll = ENV_test_t[
    :1,
    :,
    0
].clone()

sigma_roll = ENV_test_t[
    :1,
    :,
    1
].clone()

rollout_rows = []

with torch.no_grad():

    for step in range(
        1,
        41
    ):

        out = model(
            R_roll,
            A_roll,
            gamma_roll,
            sigma_roll
        )

        R_roll = out[
            "R_next"
        ]

        A_roll = out[
            "amplitude_next"
        ]

        tr = torch.diagonal(
            R_roll,
            dim1=-2,
            dim2=-1
        ).sum(
            dim=-1
        )

        eig = torch.linalg.eigvalsh(
            R_roll
        )

        finite_step = bool(
            torch.isfinite(
                R_roll
            ).all().item()
            and
            torch.isfinite(
                A_roll
            ).all().item()
        )

        rollout_rows.append({

            "step":
                int(step),

            "R_norm":
                float(
                    torch.linalg.vector_norm(
                        R_roll
                    ).item()
                ),

            "amplitude_norm":
                float(
                    torch.linalg.vector_norm(
                        A_roll
                    ).item()
                ),

            "trace_mean":
                float(
                    tr.mean().item()
                ),

            "min_eigenvalue":
                float(
                    eig.min().item()
                ),

            "finite":
                finite_step,
        })

rollout_df = pd.DataFrame(
    rollout_rows
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

rollout_finite = bool(
    rollout_df[
        "finite"
    ].all()
)

rollout_stable = bool(
    rollout_finite
    and
    rollout_max_trace_error < 1e-4
    and
    rollout_min_eig > -1e-6
)

print(
    "\n" + "=" * 100
)

print(
    "40-STEP ROLLOUT"
)

print(
    "=" * 100
)

display(
    rollout_df.iloc[
        [
            0,
            9,
            19,
            29,
            39
        ]
    ].round(8)
)

print(
    "Minimum rollout eigenvalue:",
    f"{rollout_min_eig:.12e}"
)

print(
    "Maximum rollout trace error:",
    f"{rollout_max_trace_error:.12e}"
)

print(
    "Rollout finite:",
    rollout_finite
)

print(
    "Rollout stable:",
    rollout_stable
)

rollout_df.to_csv(
    ROLLOUT_PATH,
    index=False
)


# ============================================================
# 28. DYNAMICAL COMPONENT MAGNITUDES
# ============================================================

target_delta_norm = float(
    torch.linalg.vector_norm(
        Y_R_test_t
        -
        X_R_test_t
    ).item()
)

lindblad_delta_norm = float(
    torch.linalg.vector_norm(
        delta_L
    ).item()
)

neural_delta_norm = float(
    torch.linalg.vector_norm(
        delta_N
    ).item()
)

combined_delta_norm = float(
    torch.linalg.vector_norm(
        delta_total
    ).item()
)

final_kappa_L = float(
    model
    .kappa_lindblad()
    .item()
)

final_kappa_N = float(
    model
    .kappa_neural()
    .item()
)

print(
    "\n" + "=" * 100
)

print(
    "LEARNED DYNAMICAL COMPONENTS"
)

print(
    "=" * 100
)

print(
    "Target ΔR norm:",
    f"{target_delta_norm:.12e}"
)

print(
    "Lindblad ΔR norm:",
    f"{lindblad_delta_norm:.12e}"
)

print(
    "Neural ΔR norm:",
    f"{neural_delta_norm:.12e}"
)

print(
    "Combined ΔR norm:",
    f"{combined_delta_norm:.12e}"
)

print(
    "Final kappa_L:",
    f"{final_kappa_L:.12e}"
)

print(
    "Final kappa_N:",
    f"{final_kappa_N:.12e}"
)


# ============================================================
# 29. REGIME-WISE TEST METRICS
# ============================================================

manifest_path = (
    ROOT
    / "results"
    / "step16_final_splits"
    / "final_transition_manifest.csv"
)

manifest = pd.read_csv(
    manifest_path
)

assert len(
    manifest
) == N_TOTAL

test_manifest = manifest.iloc[
    test_indices
].reset_index(
    drop=True
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

A_target_np = (
    A_target
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
        ].to_numpy()
        ==
        regime
    )

    r_err = R_error_np[
        mask
    ]

    a_err = A_error_np[
        mask
    ]

    r_target = Y_R_test[
        mask
    ]

    a_target = A_target_np[
        mask
    ]

    r_rel = (
        np.linalg.norm(
            r_err
        )
        /
        (
            np.linalg.norm(
                r_target
            )
            +
            1e-12
        )
    )

    a_rel = (
        np.linalg.norm(
            a_err
        )
        /
        (
            np.linalg.norm(
                a_target
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
                        r_err ** 2
                    )
                )
            ),

        "R_MAE":
            float(
                np.mean(
                    np.abs(
                        r_err
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
                        a_err ** 2
                    )
                )
            ),

        "amplitude_MAE":
            float(
                np.mean(
                    np.abs(
                        a_err
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
    "REGIME-WISE FINAL LNO PERFORMANCE"
)

print(
    "=" * 100
)

display(
    regime_df.round(8)
)

regime_df.to_csv(
    REGIME_PATH,
    index=False
)


# ============================================================
# 30. SAVE TEST PREDICTIONS
# ============================================================

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

    delta_R_lindblad=
        delta_L
        .detach()
        .cpu()
        .numpy(),

    delta_R_neural=
        delta_N
        .detach()
        .cpu()
        .numpy(),

    delta_R_total=
        delta_total
        .detach()
        .cpu()
        .numpy(),
)

print(
    "\n[+] Test predictions saved:",
    PREDICTIONS_PATH
)


# ============================================================
# 31. FINAL SUMMARY
# ============================================================

summary = {

    "step":
        18,

    "model":
        "FinalRepairedLNO",

    "prototype":
        False,

    "fresh_initialization":
        True,

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

    "training":
        {

            "max_epochs":
                int(MAX_EPOCHS),

            "batch_size":
                int(BATCH_SIZE),

            "learning_rate":
                1e-3,

            "weight_decay":
                1e-4,

            "early_stopping_patience":
                int(
                    EARLY_STOPPING_PATIENCE
                ),
        },

    "amplitude_normalization":
        {

            "mean":
                float(
                    amplitude_mean
                ),

            "std":
                float(
                    amplitude_std
                ),
        },

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
                float(
                    mean_trace_error
                ),

            "max_trace_error":
                float(
                    max_trace_error
                ),

            "symmetry_error":
                float(
                    symmetry_error
                ),

            "minimum_eigenvalue":
                float(
                    minimum_eigenvalue
                ),

            "negative_eigenvalue_fraction":
                float(
                    negative_eigenvalue_fraction
                ),

            "predictions_finite":
                bool(
                    predictions_finite
                ),
        },

    "dynamical_components":
        {

            "target_delta_R_norm":
                float(
                    target_delta_norm
                ),

            "lindblad_delta_R_norm":
                float(
                    lindblad_delta_norm
                ),

            "neural_delta_R_norm":
                float(
                    neural_delta_norm
                ),

            "combined_delta_R_norm":
                float(
                    combined_delta_norm
                ),

            "kappa_lindblad":
                float(
                    final_kappa_L
                ),

            "kappa_neural":
                float(
                    final_kappa_N
                ),
        },

    "rollout":
        {

            "steps":
                40,

            "minimum_eigenvalue":
                float(
                    rollout_min_eig
                ),

            "maximum_trace_error":
                float(
                    rollout_max_trace_error
                ),

            "finite":
                bool(
                    rollout_finite
                ),

            "stable":
                bool(
                    rollout_stable
                ),
        },

    "physics":
        {

            "lindblad_in_forward_path":
                True,

            "physics_loss_penalty":
                False,

            "positive_lindblad_coupling":
                True,

            "positive_neural_coupling":
                True,

        },

    "checkpoints":
        {

            "best":
                str(
                    BEST_CHECKPOINT
                ),

            "last":
                str(
                    LAST_CHECKPOINT
                ),

        },

    "outputs":
        {

            "history":
                str(
                    HISTORY_PATH
                ),

            "predictions":
                str(
                    PREDICTIONS_PATH
                ),

            "environment_response":
                str(
                    ENVIRONMENT_PATH
                ),

            "rollout":
                str(
                    ROLLOUT_PATH
                ),

            "regime_metrics":
                str(
                    REGIME_PATH
                ),

            "summary":
                str(
                    SUMMARY_PATH
                ),
        },
}

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
# 32. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "STEP 18 COMPLETE — FINAL REPAIRED LNO"
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
    "\nSTRUCTURE"
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
    "Predictions finite:",
    predictions_finite
)

print(
    "\nROLLOUT"
)

print(
    "Minimum eigenvalue:",
    f"{rollout_min_eig:.12e}"
)

print(
    "Maximum trace error:",
    f"{rollout_max_trace_error:.12e}"
)

print(
    "Rollout finite:",
    rollout_finite
)

print(
    "Rollout stable:",
    rollout_stable
)

print(
    "\nDYNAMICAL COUPLINGS"
)

print(
    "kappa_L:",
    f"{final_kappa_L:.12e}"
)

print(
    "kappa_N:",
    f"{final_kappa_N:.12e}"
)

print(
    "\nCHECKPOINTS"
)

print(
    "Best:",
    BEST_CHECKPOINT
)

print(
    "Last:",
    LAST_CHECKPOINT
)

print(
    "\nOUTPUTS"
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
    "Environment:",
    ENVIRONMENT_PATH
)

print(
    "Rollout:",
    ROLLOUT_PATH
)

print(
    "Regime metrics:",
    REGIME_PATH
)

print(
    "Summary:",
    SUMMARY_PATH
)

print(
    "\n[PASS] STEP 18 EXECUTION COMPLETE."
)

print(
    "[PASS] Same dataset / split / target / FNO normalization."
)

print(
    "[PASS] Lindblad dynamics embedded in forward path."
)

print(
    "[PASS] Final LNO artifacts saved."
)

print("=" * 100)
