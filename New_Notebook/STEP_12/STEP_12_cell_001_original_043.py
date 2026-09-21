# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 43
# Step            : STEP_12
# Step Heading    : # STEP 12 — TRAINING BEHAVIOR / ABLATION DIAGNOSTIC
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 12 — TRAINING BEHAVIOR / ABLATION DIAGNOSTIC
#
# A. FNO BASELINE
# B. LNO WITHOUT LINDBLAD
# C. LNO WITH LINDBLAD
# D. FULL LNO
#       = Lindblad + explicit environment conditioning
#
# SAME:
#   Dataset
#   Split
#   Targets
#   Optimizer
#   Learning rate
#   Batch size
#   Epochs
#
# MEASURE:
#   prediction error
#   matrix constraint behavior
#   gradient flow
#   environment sensitivity
#
# CURRENT:
#   495-transition prototype dataset
#
# IMPORTANT:
#   If Step-9 split artifact is missing, this cell
#   reconstructs it automatically from Dataset V3.
#
# NO FINAL TRAINING
# NO OLD CHECKPOINT OVERWRITE
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

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

print("=" * 100)
print("STEP 12 — TRAINING BEHAVIOR / ABLATION DIAGNOSTIC")
print("=" * 100)

print(
    "[+] Device:",
    device
)


# ============================================================
# 2. LOAD DATASET V3
# ============================================================

DATASET_V3 = (
    ROOT
    / "results"
    / "dataset_v3"
    / "environment_conditioned_lno_dataset_v3.npz"
)

METADATA_V3 = (
    ROOT
    / "results"
    / "dataset_v3"
    / "dataset_v3_metadata.csv"
)

assert DATASET_V3.is_file(), (
    f"Dataset V3 not found:\n{DATASET_V3}"
)

assert METADATA_V3.is_file(), (
    f"Dataset V3 metadata not found:\n{METADATA_V3}"
)

v3 = np.load(
    DATASET_V3
)

metadata = pd.read_csv(
    METADATA_V3
)

print(
    "[+] Dataset V3 loaded."
)

# ============================================================
# 3. LOAD V3 ARRAYS
# ============================================================

X_R = v3[
    "X_R"
].astype(
    np.float32
)

Y_R = v3[
    "Y_R"
].astype(
    np.float32
)

X_A = v3[
    "X_amplitude"
].astype(
    np.float32
)

Y_A = v3[
    "Y_amplitude"
].astype(
    np.float32
)

ENV = v3[
    "environment"
].astype(
    np.float32
)

N, NX, DIM, DIM2 = X_R.shape

assert DIM == 6
assert DIM2 == 6

assert metadata.shape[0] == N

assert X_R.shape == Y_R.shape
assert X_A.shape == Y_A.shape
assert ENV.shape == (
    N,
    NX,
    2
)

print(
    "[+] Total transitions:",
    N
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
# 4. FINITE CHECK
# ============================================================

for name, arr in {

    "X_R":
        X_R,

    "Y_R":
        Y_R,

    "X_A":
        X_A,

    "Y_A":
        Y_A,

    "ENV":
        ENV,

}.items():

    finite = bool(
        np.isfinite(
            arr
        ).all()
    )

    print(
        f"{name:12s}:",
        finite
    )

    assert finite


# ============================================================
# 5. STEP-9 SPLIT PATH
# ============================================================

SPLIT_DIR = (
    ROOT
    / "results"
    / "dataset_v3"
    / "step9_splits"
)

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SPLIT_PATH = (
    SPLIT_DIR
    / "prototype_train_val_test_split.npz"
)

SPLIT_META_PATH = (
    SPLIT_DIR
    / "prototype_split_metadata.csv"
)


# ============================================================
# 6. REBUILD SPLIT IF MISSING
# ============================================================

if not SPLIT_PATH.is_file():

    print(
        "\n[WARNING] Step-9 split artifact missing."
    )

    print(
        "[+] Reconstructing leakage-safe split from Dataset V3..."
    )

    # --------------------------------------------------------
    # Current prototype has exactly one trajectory per regime.
    # Keep temporal order inside each regime.
    # --------------------------------------------------------

    train_indices = []
    val_indices = []
    test_indices = []

    split_rows = []

    regimes = sorted(
        metadata[
            "regime"
        ].unique()
    )

    for regime in regimes:

        regime_mask = (
            metadata[
                "regime"
            ].to_numpy()
            ==
            regime
        )

        indices = np.where(
            regime_mask
        )[0]

        # Current prototype expectation:
        # 99 transitions / trajectory
        if len(indices) != 99:

            raise RuntimeError(
                f"Unexpected transition count for "
                f"{regime}: {len(indices)}"
            )

        n_total = len(
            indices
        )

        n_train = int(
            np.floor(
                n_total
                *
                0.60
            )
        )

        n_val = int(
            np.floor(
                n_total
                *
                0.20
            )
        )

        train_part = indices[
            :n_train
        ]

        val_part = indices[
            n_train:
            n_train + n_val
        ]

        test_part = indices[
            n_train + n_val:
        ]

        train_indices.extend(
            train_part.tolist()
        )

        val_indices.extend(
            val_part.tolist()
        )

        test_indices.extend(
            test_part.tolist()
        )

        for idx in train_part:

            row = metadata.iloc[
                int(idx)
            ]

            split_rows.append({

                "dataset_index":
                    int(idx),

                "regime":
                    str(
                        row["regime"]
                    ),

                "timestep":
                    int(
                        row["timestep"]
                    ),

                "split":
                    "train",

                "gamma":
                    float(
                        row["gamma"]
                    ),

                "noise_sigma":
                    float(
                        row["noise_sigma"]
                    ),
            })

        for idx in val_part:

            row = metadata.iloc[
                int(idx)
            ]

            split_rows.append({

                "dataset_index":
                    int(idx),

                "regime":
                    str(
                        row["regime"]
                    ),

                "timestep":
                    int(
                        row["timestep"]
                    ),

                "split":
                    "validation",

                "gamma":
                    float(
                        row["gamma"]
                    ),

                "noise_sigma":
                    float(
                        row["noise_sigma"]
                    ),
            })

        for idx in test_part:

            row = metadata.iloc[
                int(idx)
            ]

            split_rows.append({

                "dataset_index":
                    int(idx),

                "regime":
                    str(
                        row["regime"]
                    ),

                "timestep":
                    int(
                        row["timestep"]
                    ),

                "split":
                    "test",

                "gamma":
                    float(
                        row["gamma"]
                    ),

                "noise_sigma":
                    float(
                        row["noise_sigma"]
                    ),
            })

    train_indices = np.asarray(
        train_indices,
        dtype=np.int64
    )

    val_indices = np.asarray(
        val_indices,
        dtype=np.int64
    )

    test_indices = np.asarray(
        test_indices,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # No-overlap checks
    # --------------------------------------------------------

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

    assert (
        len(train_set)
        +
        len(val_set)
        +
        len(test_set)
        ==
        N
    )

    # --------------------------------------------------------
    # Save reconstructed split
    # --------------------------------------------------------

    np.savez_compressed(

        SPLIT_PATH,

        train_indices=
            train_indices,

        val_indices=
            val_indices,

        test_indices=
            test_indices,

        X_R_train=
            X_R[
                train_indices
            ],

        Y_R_train=
            Y_R[
                train_indices
            ],

        X_amplitude_train=
            X_A[
                train_indices
            ],

        Y_amplitude_train=
            Y_A[
                train_indices
            ],

        environment_train=
            ENV[
                train_indices
            ],

        X_R_val=
            X_R[
                val_indices
            ],

        Y_R_val=
            Y_R[
                val_indices
            ],

        X_amplitude_val=
            X_A[
                val_indices
            ],

        Y_amplitude_val=
            Y_A[
                val_indices
            ],

        environment_val=
            ENV[
                val_indices
            ],

        X_R_test=
            X_R[
                test_indices
            ],

        Y_R_test=
            Y_R[
                test_indices
            ],

        X_amplitude_test=
            X_A[
                test_indices
            ],

        Y_amplitude_test=
            Y_A[
                test_indices
            ],

        environment_test=
            ENV[
                test_indices
            ],
    )

    reconstructed_metadata = pd.DataFrame(
        split_rows
    )

    reconstructed_metadata.to_csv(
        SPLIT_META_PATH,
        index=False
    )

    print(
        "[PASS] Step-9 split reconstructed and saved."
    )

else:

    print(
        "\n[+] Existing Step-9 split found."
    )


# ============================================================
# 7. LOAD SPLIT ARTIFACT
# ============================================================

split = np.load(
    SPLIT_PATH
)

required_keys = [

    "train_indices",
    "val_indices",
    "test_indices",

    "X_R_train",
    "Y_R_train",
    "X_amplitude_train",
    "Y_amplitude_train",
    "environment_train",

    "X_R_val",
    "Y_R_val",
    "X_amplitude_val",
    "Y_amplitude_val",
    "environment_val",

    "X_R_test",
    "Y_R_test",
    "X_amplitude_test",
    "Y_amplitude_test",
    "environment_test",
]

for key in required_keys:

    assert key in split, (
        f"Missing split array: {key}"
    )


# ============================================================
# 8. LOAD TRAIN / VAL / TEST
# ============================================================

X_R_train = split[
    "X_R_train"
].astype(
    np.float32
)

Y_R_train = split[
    "Y_R_train"
].astype(
    np.float32
)

X_A_train = split[
    "X_amplitude_train"
].astype(
    np.float32
)

Y_A_train = split[
    "Y_amplitude_train"
].astype(
    np.float32
)

ENV_train = split[
    "environment_train"
].astype(
    np.float32
)


X_R_val = split[
    "X_R_val"
].astype(
    np.float32
)

Y_R_val = split[
    "Y_R_val"
].astype(
    np.float32
)

X_A_val = split[
    "X_amplitude_val"
].astype(
    np.float32
)

Y_A_val = split[
    "Y_amplitude_val"
].astype(
    np.float32
)

ENV_val = split[
    "environment_val"
].astype(
    np.float32
)


X_R_test = split[
    "X_R_test"
].astype(
    np.float32
)

Y_R_test = split[
    "Y_R_test"
].astype(
    np.float32
)

X_A_test = split[
    "X_amplitude_test"
].astype(
    np.float32
)

Y_A_test = split[
    "Y_amplitude_test"
].astype(
    np.float32
)

ENV_test = split[
    "environment_test"
].astype(
    np.float32
)


# ============================================================
# 9. SPLIT REPORT
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "SPLIT"
)

print(
    "=" * 100
)

print(
    "Train:",
    X_R_train.shape[0]
)

print(
    "Validation:",
    X_R_val.shape[0]
)

print(
    "Test:",
    X_R_test.shape[0]
)

print(
    "Train R:",
    X_R_train.shape
)

print(
    "Validation R:",
    X_R_val.shape
)

print(
    "Test R:",
    X_R_test.shape
)

# ============================================================
# 10. DATA LOADERS
# ============================================================

BATCH_SIZE = 8

train_dataset = TensorDataset(

    torch.tensor(
        X_R_train,
        dtype=torch.float32
    ),

    torch.tensor(
        X_A_train,
        dtype=torch.float32
    ),

    torch.tensor(
        ENV_train,
        dtype=torch.float32
    ),

    torch.tensor(
        Y_R_train,
        dtype=torch.float32
    ),

    torch.tensor(
        Y_A_train,
        dtype=torch.float32
    ),
)

val_dataset = TensorDataset(

    torch.tensor(
        X_R_val,
        dtype=torch.float32
    ),

    torch.tensor(
        X_A_val,
        dtype=torch.float32
    ),

    torch.tensor(
        ENV_val,
        dtype=torch.float32
    ),

    torch.tensor(
        Y_R_val,
        dtype=torch.float32
    ),

    torch.tensor(
        Y_A_val,
        dtype=torch.float32
    ),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
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
# 12. LINDBLAD KERNEL
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
        dt
    ):

        D_R = self.dissipator(
            R
        )

        alpha = (
            self.positive_alpha()
        )

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
# 13. SHARED PSD / TRACE PROJECTION
# ============================================================

def project_R(
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


# ============================================================
# 14. A — TRUE FNO BASELINE
# ============================================================

class FNOBaseline(
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

        # Same task inputs:
        # R + amplitude + gamma + sigma

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
        dt=1e-3,
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

        # Common symmetric representation.
        # No PSD / trace projection here.
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

        return {

            "R_next":
                R_next,

            "amplitude_next":
                A_next,

            "D_R":
                torch.zeros_like(
                    R
                ),
        }


# ============================================================
# 15. B/C/D LNO ABLATION MODEL
# ============================================================

class AblationLNO(
    nn.Module
):

    def __init__(
        self,
        mode,
        dim=6,
        width=64,
        modes=16,
        depth=4,
        lindblad_channels=4,
    ):

        super().__init__()

        assert mode in {

            "B_no_lindblad",

            "C_lindblad",

            "D_full_lno",
        }

        self.mode = mode

        self.dim = dim
        self.width = width
        self.depth = depth

        # ----------------------------------------------------
        # B / C:
        #   R + amplitude
        #
        # D:
        #   R + amplitude + gamma + sigma
        # ----------------------------------------------------

        if mode == "D_full_lno":

            input_channels = (
                dim * dim
                +
                3
            )

        else:

            input_channels = (
                dim * dim
                +
                1
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

        if mode in {

            "C_lindblad",

            "D_full_lno",

        }:

            self.lindblad_kernel = (
                LindbladDissipativeKernel(
                    dim=dim,
                    channels=lindblad_channels
                )
            )

        else:

            self.lindblad_kernel = None

    def forward(
        self,
        R,
        amplitude,
        gamma,
        sigma,
        dt=1e-3,
    ):

        B, NX_local, D, D2 = R.shape

        R_flat = R.reshape(
            B,
            NX_local,
            D * D
        )

        if self.mode == "D_full_lno":

            env = torch.stack(
                [
                    amplitude,
                    gamma,
                    sigma,
                ],
                dim=-1
            )

        else:

            env = amplitude[
                ...,
                None
            ]

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

        # ----------------------------------------------------
        # Lindblad branch
        # ----------------------------------------------------

        if self.lindblad_kernel is not None:

            R_lindblad, D_R = (
                self.lindblad_kernel(
                    R,
                    gamma,
                    dt
                )
            )

            candidate = (
                R_lindblad
                +
                dt
                *
                0.05
                *
                neural_R
            )

        else:

            D_R = torch.zeros_like(
                R
            )

            candidate = (
                R
                +
                dt
                *
                0.05
                *
                neural_R
            )

        R_next = project_R(
            candidate
        )

        # ----------------------------------------------------
        # Amplitude
        # ----------------------------------------------------

        A_tendency = self.A_head(
            x
        ).squeeze(
            1
        )

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

        A_next = torch.exp(
            log_A_next
        )

        return {

            "R_next":
                R_next,

            "amplitude_next":
                A_next,

            "D_R":
                D_R,
        }


# ============================================================
# 16. FACTORY
# ============================================================

def build_model(
    label
):

    if label == "A_FNO":

        return FNOBaseline(
            dim=6,
            width=64,
            modes=16,
            depth=4,
        ).to(device)

    if label == "B_LNO_no_Lindblad":

        return AblationLNO(
            mode="B_no_lindblad",
            dim=6,
            width=64,
            modes=16,
            depth=4,
        ).to(device)

    if label == "C_LNO_Lindblad":

        return AblationLNO(
            mode="C_lindblad",
            dim=6,
            width=64,
            modes=16,
            depth=4,
            lindblad_channels=4,
        ).to(device)

    if label == "D_Full_LNO":

        return AblationLNO(
            mode="D_full_lno",
            dim=6,
            width=64,
            modes=16,
            depth=4,
            lindblad_channels=4,
        ).to(device)

    raise ValueError(
        f"Unknown model: {label}"
    )


# ============================================================
# 17. TRAIN ONE MODEL
# ============================================================

def train_one_model(
    model,
    epochs=5,
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    best_val = float(
        "inf"
    )

    best_epoch = None

    best_state = None

    history = []

    for epoch in range(
        1,
        epochs + 1
    ):

        start_time = time.time()

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        total_sum = 0.0
        R_sum = 0.0
        A_sum = 0.0
        count = 0

        for (
            R_x,
            A_x,
            env_x,
            R_y,
            A_y,
        ) in train_loader:

            R_x = R_x.to(
                device
            )

            A_x = A_x.to(
                device
            )

            env_x = env_x.to(
                device
            )

            R_y = R_y.to(
                device
            )

            A_y = A_y.to(
                device
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

            out = model(
                R_x,
                A_x,
                gamma_x,
                sigma_x,
                dt=1e-3
            )

            R_loss = F.mse_loss(
                out[
                    "R_next"
                ],
                R_y
            )

            A_loss = F.mse_loss(
                out[
                    "amplitude_next"
                ],
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

            total_sum += (
                loss.item()
                *
                n_batch
            )

            R_sum += (
                R_loss.item()
                *
                n_batch
            )

            A_sum += (
                A_loss.item()
                *
                n_batch
            )

            count += n_batch

        train_loss = (
            total_sum
            /
            count
        )

        train_R = (
            R_sum
            /
            count
        )

        train_A = (
            A_sum
            /
            count
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_sum = 0.0
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
                    device
                )

                A_x = A_x.to(
                    device
                )

                env_x = env_x.to(
                    device
                )

                R_y = R_y.to(
                    device
                )

                A_y = A_y.to(
                    device
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
                    dt=1e-3
                )

                R_loss = F.mse_loss(
                    out[
                        "R_next"
                    ],
                    R_y
                )

                A_loss = F.mse_loss(
                    out[
                        "amplitude_next"
                    ],
                    A_y
                )

                loss = (
                    R_loss
                    +
                    A_loss
                )

                n_batch = R_x.shape[0]

                val_sum += (
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
            val_sum
            /
            val_count
        )

        val_R = (
            val_R_sum
            /
            val_count
        )

        val_A = (
            val_A_sum
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

            "train_R_loss":
                float(train_R),

            "train_amplitude_loss":
                float(train_A),

            "val_loss":
                float(val_loss),

            "val_R_loss":
                float(val_R),

            "val_amplitude_loss":
                float(val_A),

            "time_seconds":
                float(elapsed),
        })

        print(
            f"    Epoch {epoch:02d}/{epochs}"
            f" | train={train_loss:.6e}"
            f" | val={val_loss:.6e}"
            f" | time={elapsed:.2f}s"
        )

        # ----------------------------------------------------
        # Best state
        # ----------------------------------------------------

        if val_loss < best_val:

            best_val = (
                val_loss
            )

            best_epoch = (
                epoch
            )

            best_state = {

                key:
                    value.detach()
                    .cpu()
                    .clone()

                for key, value
                in model.state_dict().items()
            }

    # Restore best state
    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return (
        history,
        best_val,
        best_epoch
    )


# ============================================================
# 18. EVALUATION
# ============================================================

def evaluate_model(
    model,
):

    model.eval()

    with torch.no_grad():

        R_x = X_R_test_t.to(
            device
        )

        A_x = X_A_test_t.to(
            device
        )

        env_x = ENV_test_t.to(
            device
        )

        R_y = Y_R_test_t.to(
            device
        )

        A_y = Y_A_test_t.to(
            device
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
            dt=1e-3
        )

        R_pred = out[
            "R_next"
        ]

        A_pred = out[
            "amplitude_next"
        ]

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    R_error = (
        R_pred
        -
        R_y
    )

    A_error = (
        A_pred
        -
        A_y
    )

    R_mse = torch.mean(
        R_error ** 2
    )

    R_rmse = torch.sqrt(
        R_mse
        +
        1e-12
    )

    R_mae = torch.mean(
        torch.abs(
            R_error
        )
    )

    R_rel_l2 = (
        torch.linalg.vector_norm(
            R_error
        )
        /
        (
            torch.linalg.vector_norm(
                R_y
            )
            +
            1e-12
        )
    )

    A_mse = torch.mean(
        A_error ** 2
    )

    A_rmse = torch.sqrt(
        A_mse
        +
        1e-12
    )

    A_mae = torch.mean(
        torch.abs(
            A_error
        )
    )

    A_rel_l2 = (
        torch.linalg.vector_norm(
            A_error
        )
        /
        (
            torch.linalg.vector_norm(
                A_y
            )
            +
            1e-12
        )
    )

    # --------------------------------------------------------
    # Structural output
    # --------------------------------------------------------

    trace = torch.diagonal(
        R_pred,
        dim1=-2,
        dim2=-1
    ).sum(
        dim=-1
    )

    trace_mean_error = float(
        torch.mean(
            torch.abs(
                trace
                -
                1.0
            )
        ).item()
    )

    trace_max_error = float(
        torch.max(
            torch.abs(
                trace
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

    eigenvalues = torch.linalg.eigvalsh(
        R_pred
    )

    min_eigenvalue = float(
        eigenvalues.min().item()
    )

    finite = bool(
        torch.isfinite(
            R_pred
        ).all().item()
        and
        torch.isfinite(
            A_pred
        ).all().item()
    )

    # --------------------------------------------------------
    # Environment response
    # --------------------------------------------------------

    probe_R = R_x[
        :1
    ]

    probe_A = A_x[
        :1
    ]

    probe_sigma = sigma_x[
        :1
    ]

    env_rows = []

    for gamma_value in [
        0.0,
        0.01,
        0.05,
        0.15,
        0.35,
        0.50,
    ]:

        gamma_probe = torch.full_like(
            gamma_x[
                :1
            ],
            gamma_value
        )

        out_env = model(
            probe_R,
            probe_A,
            gamma_probe,
            probe_sigma,
            dt=1e-3
        )

        r_change = float(
            torch.linalg.vector_norm(
                out_env[
                    "R_next"
                ]
                -
                probe_R
            ).item()
        )

        a_change = float(
            torch.linalg.vector_norm(
                out_env[
                    "amplitude_next"
                ]
                -
                probe_A
            ).item()
        )

        d_norm = float(
            torch.linalg.vector_norm(
                out_env[
                    "D_R"
                ]
            ).item()
        )

        env_rows.append({

            "gamma":
                float(
                    gamma_value
                ),

            "R_change":
                r_change,

            "amplitude_change":
                a_change,

            "Lindblad_D_norm":
                d_norm,
        })

    env_df = pd.DataFrame(
        env_rows
    )

    return {

        "R_MSE":
            float(
                R_mse.item()
            ),

        "R_RMSE":
            float(
                R_rmse.item()
            ),

        "R_MAE":
            float(
                R_mae.item()
            ),

        "R_relative_L2":
            float(
                R_rel_l2.item()
            ),

        "A_MSE":
            float(
                A_mse.item()
            ),

        "A_RMSE":
            float(
                A_rmse.item()
            ),

        "A_MAE":
            float(
                A_mae.item()
            ),

        "A_relative_L2":
            float(
                A_rel_l2.item()
            ),

        "trace_mean_error":
            trace_mean_error,

        "trace_max_error":
            trace_max_error,

        "symmetry_error":
            symmetry_error,

        "minimum_eigenvalue":
            min_eigenvalue,

        "finite":
            finite,

        "environment":
            env_df,

        "R_prediction":
            R_pred.detach()
            .cpu()
            .numpy(),

        "A_prediction":
            A_pred.detach()
            .cpu()
            .numpy(),
    }


# ============================================================
# 19. TORCH TEST TENSORS
# ============================================================

X_R_test_t = torch.tensor(
    X_R_test,
    dtype=torch.float32
)

Y_R_test_t = torch.tensor(
    Y_R_test,
    dtype=torch.float32
)

X_A_test_t = torch.tensor(
    X_A_test,
    dtype=torch.float32
)

Y_A_test_t = torch.tensor(
    Y_A_test,
    dtype=torch.float32
)

ENV_test_t = torch.tensor(
    ENV_test,
    dtype=torch.float32
)

X_R_train_t = torch.tensor(
    X_R_train,
    dtype=torch.float32
)

Y_R_train_t = torch.tensor(
    Y_R_train,
    dtype=torch.float32
)

X_A_train_t = torch.tensor(
    X_A_train,
    dtype=torch.float32
)

Y_A_train_t = torch.tensor(
    Y_A_train,
    dtype=torch.float32
)

ENV_train_t = torch.tensor(
    ENV_train,
    dtype=torch.float32
)


# ============================================================
# 20. GRADIENT DIAGNOSTIC
# ============================================================

def gradient_diagnostic(
    model
):

    model.train()

    model.zero_grad(
        set_to_none=True
    )

    R_sample = (
        X_R_train_t[
            :BATCH_SIZE
        ]
        .to(device)
        .detach()
        .clone()
        .requires_grad_()
    )

    A_sample = (
        X_A_train_t[
            :BATCH_SIZE
        ]
        .to(device)
        .detach()
        .clone()
        .requires_grad_()
    )

    env_sample = (
        ENV_train_t[
            :BATCH_SIZE
        ]
        .to(device)
    )

    gamma_sample = env_sample[
        ...,
        0
    ]

    sigma_sample = env_sample[
        ...,
        1
    ]

    R_target_sample = (
        Y_R_train_t[
            :BATCH_SIZE
        ]
        .to(device)
    )

    A_target_sample = (
        Y_A_train_t[
            :BATCH_SIZE
        ]
        .to(device)
    )

    out = model(
        R_sample,
        A_sample,
        gamma_sample,
        sigma_sample,
        dt=1e-3
    )

    loss = (
        F.mse_loss(
            out[
                "R_next"
            ],
            R_target_sample
        )
        +
        F.mse_loss(
            out[
                "amplitude_next"
            ],
            A_target_sample
        )
    )

    loss.backward()

    result = {

        "gradient_loss":
            float(
                loss.item()
            ),

        "input_R_gradient":
            float(
                torch.linalg.vector_norm(
                    R_sample.grad
                ).item()
            ),

        "input_A_gradient":
            float(
                torch.linalg.vector_norm(
                    A_sample.grad
                ).item()
            ),

    }

    if hasattr(
        model,
        "lindblad_kernel"
    ):

        if model.lindblad_kernel is not None:

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

            result[
                "jump_gradient_L2"
            ] = float(
                torch.linalg.vector_norm(
                    jump_grad
                ).item()
            )

            result[
                "alpha_gradient"
            ] = float(
                torch.abs(
                    alpha_grad
                ).item()
            )

        else:

            result[
                "jump_gradient_L2"
            ] = 0.0

            result[
                "alpha_gradient"
            ] = 0.0

    else:

        result[
            "jump_gradient_L2"
        ] = 0.0

        result[
            "alpha_gradient"
        ] = 0.0

    return result


# ============================================================
# 21. RUN A/B/C/D
# ============================================================

MODELS = [

    "A_FNO",

    "B_LNO_no_Lindblad",

    "C_LNO_Lindblad",

    "D_Full_LNO",
]

EPOCHS = 5

OUT_DIR = (
    ROOT
    / "results"
    / "step12_ablation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

results = []
histories = {}
env_tables = {}

print(
    "\n" + "=" * 100
)

print(
    "STARTING STEP 12 ABLATION"
)

print(
    "=" * 100
)

for label in MODELS:

    print(
        "\n" + "-" * 90
    )

    print(
        f"MODEL: {label}"
    )

    print(
        "-" * 90
    )

    model = build_model(
        label
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "[+] Parameters:",
        f"{parameter_count:,}"
    )

    history, best_val, best_epoch = (
        train_one_model(
            model,
            epochs=EPOCHS
        )
    )

    metrics = evaluate_model(
        model
    )

    gradients = gradient_diagnostic(
        model
    )

    histories[
        label
    ] = history

    env_tables[
        label
    ] = metrics[
        "environment"
    ]

    result = {

        "model":
            label,

        "parameters":
            int(
                parameter_count
            ),

        "best_epoch":
            int(
                best_epoch
            ),

        "best_val_loss":
            float(
                best_val
            ),

        "test_R_MSE":
            float(
                metrics[
                    "R_MSE"
                ]
            ),

        "test_R_RMSE":
            float(
                metrics[
                    "R_RMSE"
                ]
            ),

        "test_R_MAE":
            float(
                metrics[
                    "R_MAE"
                ]
            ),

        "test_R_relative_L2":
            float(
                metrics[
                    "R_relative_L2"
                ]
            ),

        "test_A_MSE":
            float(
                metrics[
                    "A_MSE"
                ]
            ),

        "test_A_RMSE":
            float(
                metrics[
                    "A_RMSE"
                ]
            ),

        "test_A_MAE":
            float(
                metrics[
                    "A_MAE"
                ]
            ),

        "test_A_relative_L2":
            float(
                metrics[
                    "A_relative_L2"
                ]
            ),

        "trace_mean_error":
            float(
                metrics[
                    "trace_mean_error"
                ]
            ),

        "trace_max_error":
            float(
                metrics[
                    "trace_max_error"
                ]
            ),

        "symmetry_error":
            float(
                metrics[
                    "symmetry_error"
                ]
            ),

        "minimum_eigenvalue":
            float(
                metrics[
                    "minimum_eigenvalue"
                ]
            ),

        "finite":
            bool(
                metrics[
                    "finite"
                ]
            ),

        "gradient_loss":
            float(
                gradients[
                    "gradient_loss"
                ]
            ),

        "input_R_gradient":
            float(
                gradients[
                    "input_R_gradient"
                ]
            ),

        "input_A_gradient":
            float(
                gradients[
                    "input_A_gradient"
                ]
            ),

        "jump_gradient_L2":
            float(
                gradients[
                    "jump_gradient_L2"
                ]
            ),

        "alpha_gradient":
            float(
                gradients[
                    "alpha_gradient"
                ]
            ),
    }

    results.append(
        result
    )

    # Save model
    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "label":
                label,

            "prototype":
                True,

            "best_epoch":
                int(
                    best_epoch
                ),

            "best_validation_loss":
                float(
                    best_val
                ),
        },

        OUT_DIR
        /
        f"{label}.pt"
    )

    np.savez_compressed(

        OUT_DIR
        /
        f"{label}_test_predictions.npz",

        R_prediction=
            metrics[
                "R_prediction"
            ],

        A_prediction=
            metrics[
                "A_prediction"
            ],

        R_target=
            Y_R_test,

        A_target=
            Y_A_test,

        environment=
            ENV_test,
    )

    print(
        "\nRESULT:"
    )

    print(
        "  R Relative-L2:",
        f"{result['test_R_relative_L2']:.8f}"
    )

    print(
        "  Amplitude Relative-L2:",
        f"{result['test_A_relative_L2']:.8f}"
    )

    print(
        "  Trace max error:",
        f"{result['trace_max_error']:.8e}"
    )

    print(
        "  Minimum eigenvalue:",
        f"{result['minimum_eigenvalue']:.8e}"
    )

    print(
        "  Jump gradient:",
        f"{result['jump_gradient_L2']:.8e}"
    )

    print(
        "  Alpha gradient:",
        f"{result['alpha_gradient']:.8e}"
    )


# ============================================================
# 22. SUMMARY TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

print(
    "\n" + "=" * 100
)

print(
    "STEP 12 — ABLATION SUMMARY"
)

print(
    "=" * 100
)

display(
    results_df.round(8)
)

results_df.to_csv(
    OUT_DIR
    / "ablation_summary.csv",
    index=False
)


# ============================================================
# 23. ENVIRONMENT RESPONSE TABLE
# ============================================================

environment_rows = []

for label in MODELS:

    table = env_tables[
        label
    ]

    for _, row in table.iterrows():

        environment_rows.append({

            "model":
                label,

            "gamma":
                float(
                    row[
                        "gamma"
                    ]
                ),

            "R_change":
                float(
                    row[
                        "R_change"
                    ]
                ),

            "amplitude_change":
                float(
                    row[
                        "amplitude_change"
                    ]
                ),

            "Lindblad_D_norm":
                float(
                    row[
                        "Lindblad_D_norm"
                    ]
                ),
        })

environment_df = pd.DataFrame(
    environment_rows
)

environment_df.to_csv(
    OUT_DIR
    / "environment_response.csv",
    index=False
)

print(
    "\n" + "=" * 100
)

print(
    "ENVIRONMENT RESPONSE"
)

print(
    "=" * 100
)

display(
    environment_df.round(10)
)


# ============================================================
# 24. TRAINING HISTORY
# ============================================================

history_rows = []

for label in MODELS:

    for item in histories[
        label
    ]:

        history_rows.append({

            "model":
                label,

            "epoch":
                int(
                    item[
                        "epoch"
                    ]
                ),

            "train_loss":
                float(
                    item[
                        "train_loss"
                    ]
                ),

            "train_R_loss":
                float(
                    item[
                        "train_R_loss"
                    ]
                ),

            "train_amplitude_loss":
                float(
                    item[
                        "train_amplitude_loss"
                    ]
                ),

            "val_loss":
                float(
                    item[
                        "val_loss"
                    ]
                ),

            "val_R_loss":
                float(
                    item[
                        "val_R_loss"
                    ]
                ),

            "val_amplitude_loss":
                float(
                    item[
                        "val_amplitude_loss"
                    ]
                ),

            "time_seconds":
                float(
                    item[
                        "time_seconds"
                    ]
                ),
        })

history_df = pd.DataFrame(
    history_rows
)

history_df.to_csv(
    OUT_DIR
    / "training_histories.csv",
    index=False
)


# ============================================================
# 25. ABLATION EFFECT CALCULATIONS
# ============================================================

lookup = results_df.set_index(
    "model"
)

B = lookup.loc[
    "B_LNO_no_Lindblad"
]

C = lookup.loc[
    "C_LNO_Lindblad"
]

D = lookup.loc[
    "D_Full_LNO"
]

A = lookup.loc[
    "A_FNO"
]

# Positive improvement means error decreased.
lindblad_R_improvement = (
    B[
        "test_R_relative_L2"
    ]
    -
    C[
        "test_R_relative_L2"
    ]
)

environment_R_improvement = (
    C[
        "test_R_relative_L2"
    ]
    -
    D[
        "test_R_relative_L2"
    ]
)

fno_to_full_lno_R_improvement = (
    A[
        "test_R_relative_L2"
    ]
    -
    D[
        "test_R_relative_L2"
    ]
)

# ============================================================
# 26. SAVE FINAL JSON
# ============================================================

summary = {

    "step":
        12,

    "prototype":
        True,

    "dataset":
        str(
            DATASET_V3
        ),

    "transitions":
        int(N),

    "train_transitions":
        int(
            X_R_train.shape[0]
        ),

    "validation_transitions":
        int(
            X_R_val.shape[0]
        ),

    "test_transitions":
        int(
            X_R_test.shape[0]
        ),

    "epochs":
        int(EPOCHS),

    "batch_size":
        int(BATCH_SIZE),

    "models":
        MODELS,

    "results":
        results,

    "ablation_effects":
        {

            "Lindblad_R_relative_L2_improvement":
                float(
                    lindblad_R_improvement
                ),

            "environment_conditioning_R_relative_L2_improvement":
                float(
                    environment_R_improvement
                ),

            "FNO_to_full_LNO_R_relative_L2_difference":
                float(
                    fno_to_full_lno_R_improvement
                ),
        },

    "artifacts":
        {

            "summary_csv":
                str(
                    OUT_DIR
                    / "ablation_summary.csv"
                ),

            "environment_csv":
                str(
                    OUT_DIR
                    / "environment_response.csv"
                ),

            "history_csv":
                str(
                    OUT_DIR
                    / "training_histories.csv"
                ),

            "split_file":
                str(
                    SPLIT_PATH
                ),
        },
}

SUMMARY_PATH = (
    OUT_DIR
    / "step12_summary.json"
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
# 27. FINAL INTERPRETATION
# ============================================================

best_model = results_df.loc[
    results_df[
        "test_R_relative_L2"
    ].idxmin(),
    "model"
]

best_structural_model = results_df.loc[
    results_df[
        "trace_max_error"
    ].idxmin(),
    "model"
]

print(
    "\n" + "=" * 100
)

print(
    "STEP 12 COMPLETE"
)

print(
    "=" * 100
)

print(
    "Best predictive R model:",
    best_model
)

print(
    "Best trace preservation:",
    best_structural_model
)

print(
    "\nLindblad contribution:"
)

print(
    "B → C R Relative-L2 improvement:",
    f"{lindblad_R_improvement:.8f}"
)

print(
    "\nEnvironment conditioning contribution:"
)

print(
    "C → D R Relative-L2 improvement:",
    f"{environment_R_improvement:.8f}"
)

print(
    "\nFNO → Full LNO difference:"
)

print(
    fno_to_full_lno_R_improvement
)

print(
    "\n[INFO] Positive improvement = lower R Relative-L2."
)

print(
    "[INFO] Do NOT interpret this prototype as the final paper result."
)

print(
    "[INFO] Final independent-trajectory dataset expansion is still pending."
)

print(
    "\nArtifacts:",
    OUT_DIR
)

print("=" * 100)
