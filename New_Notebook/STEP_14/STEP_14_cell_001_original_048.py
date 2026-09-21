# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 48
# Step            : STEP_14
# Step Heading    : # STEP 14 — REPAIRED LNO DYNAMICAL PARAMETERIZATION
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 14 — REPAIRED LNO DYNAMICAL PARAMETERIZATION
#
# GOAL:
#   Repair the scale mismatch discovered in Step 13.
#
# OLD:
#   R_next = project(
#       R_lindblad + dt * 0.05 * neural_direction
#   )
#
# PROBLEM:
#   neural contribution was artificially suppressed.
#
# NEW:
#
#   G_L = Lindblad dissipative generator
#   G_N = learned neural generator
#
#   R_candidate =
#       R
#       + kappa_L * gamma * G_L
#       + kappa_N * G_N
#
#   where kappa_L and kappa_N are POSITIVE TRAINABLE
#   coupling strengths.
#
#   Then:
#
#       R_next = PSD + trace-normalized projection
#
# CURRENT:
#   495-transition prototype dataset
#
# THIS CELL:
#   trains a repaired LNO prototype
#
# NOT FINAL PAPER TRAINING
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
print("STEP 14 — REPAIRED LNO DYNAMICAL PARAMETERIZATION")
print("=" * 100)

print(
    "[+] Device:",
    device
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

DATASET_PATH = (
    ROOT
    / "results"
    / "dataset_v3"
    / "environment_conditioned_lno_dataset_v3.npz"
)

assert DATASET_PATH.is_file(), (
    f"Dataset V3 missing:\n{DATASET_PATH}"
)

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

N, NX, DIM, DIM2 = X_R.shape

assert DIM == 6
assert DIM2 == 6

assert X_R.shape == Y_R.shape
assert X_A.shape == Y_A.shape
assert ENV.shape == (
    N,
    NX,
    2
)

print(
    "[+] Dataset:",
    DATASET_PATH
)

print(
    "[+] Transitions:",
    N
)

print(
    "[+] Spatial points:",
    NX
)


# ============================================================
# 3. LOAD / REBUILD STEP-9 SPLIT
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

if not SPLIT_PATH.is_file():

    print(
        "\n[WARNING] Step-9 split missing."
    )

    print(
        "[+] Rebuilding split from Dataset V3..."
    )

    metadata_path = (
        ROOT
        / "results"
        / "dataset_v3"
        / "dataset_v3_metadata.csv"
    )

    assert metadata_path.is_file()

    metadata = pd.read_csv(
        metadata_path
    )

    train_indices = []
    val_indices = []
    test_indices = []

    for regime in sorted(
        metadata[
            "regime"
        ].unique()
    ):

        indices = np.where(
            metadata[
                "regime"
            ].to_numpy()
            ==
            regime
        )[0]

        assert len(indices) == 99

        n_train = 59
        n_val = 19

        train_indices.extend(
            indices[
                :n_train
            ].tolist()
        )

        val_indices.extend(
            indices[
                n_train:
                n_train + n_val
            ].tolist()
        )

        test_indices.extend(
            indices[
                n_train + n_val:
            ].tolist()
        )

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

    print(
        "[PASS] Split rebuilt."
    )

else:

    print(
        "\n[+] Step-9 split found."
    )


split = np.load(
    SPLIT_PATH
)

# ============================================================
# 4. LOAD SPLIT
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

print(
    "\n[+] Train:",
    X_R_train.shape[0]
)

print(
    "[+] Validation:",
    X_R_val.shape[0]
)

print(
    "[+] Test:",
    X_R_test.shape[0]
)


# ============================================================
# 5. DATALOADERS
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
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
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
# 6. SPECTRAL CONVOLUTION
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
# 7. LINDBLAD DYNAMICS
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
# 8. REPAIRED LNO
# ============================================================

class RepairedLNO(
    nn.Module
):

    """
    Repaired spatial LNO.

    Instead of a fixed:

        dt * 0.05 * neural_direction

    the model learns TWO POSITIVE couplings:

        kappa_L
        kappa_N

    and combines:

        G = kappa_L * gamma * G_L
            +
            kappa_N * G_N

    Then the result is converted into a valid normalized
    PSD information state.

    This keeps the Lindblad mechanism in the FORWARD PATH.
    """

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
        # R = 36
        # amplitude = 1
        # gamma = 1
        # sigma = 1
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

        self.lindblad = LindbladKernel(
            dim=dim,
            channels=lindblad_channels
        )

        # ----------------------------------------------------
        # POSITIVE trainable coupling strengths
        #
        # softplus(0) ~= 0.693
        # ----------------------------------------------------

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

    def forward(
        self,
        R,
        amplitude,
        gamma,
        sigma,
        return_components=False
    ):

        B, NX_local, D, D2 = R.shape

        # ----------------------------------------------------
        # INPUT
        # ----------------------------------------------------

        R_flat = R.reshape(
            B,
            NX_local,
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

        # ----------------------------------------------------
        # FOURIER BACKBONE
        # ----------------------------------------------------

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

            if k < (
                self.depth - 1
            ):

                x = F.gelu(
                    x
                )

        # ----------------------------------------------------
        # LEARNED NEURAL GENERATOR
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

        # ----------------------------------------------------
        # NO UNIT-NORM SUPPRESSION
        #
        # Let the model learn the correct magnitude.
        # Only bound pathological initialization.
        # ----------------------------------------------------

        G_N = torch.tanh(
            G_N
        )

        # ----------------------------------------------------
        # LINDBLAD GENERATOR
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

        # ----------------------------------------------------
        # COMBINED DYNAMICAL GENERATOR
        # ----------------------------------------------------

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
        # DIRECT STATE EVOLUTION
        #
        # No arbitrary dt * 0.05.
        #
        # Learnable couplings determine contribution size.
        # ----------------------------------------------------

        R_candidate = (
            R
            +
            delta_R_total
        )

        R_next = self.project_R(
            R_candidate
        )

        # ----------------------------------------------------
        # AMPLITUDE
        # ----------------------------------------------------

        A_tendency = self.A_head(
            x
        ).squeeze(
            1
        )

        # Stable bounded amplitude evolution
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
            A_tendency
        )

        amplitude_next = torch.exp(
            log_A_next
        )

        result = {

            "R_next":
                R_next,

            "amplitude_next":
                amplitude_next,

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

        if return_components:

            return result

        return result


# ============================================================
# 9. MODEL
# ============================================================

model = RepairedLNO(
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
    "\n[+] Repaired LNO created."
)

print(
    "[+] Parameters:",
    f"{parameter_count:,}"
)

print(
    "[+] Initial kappa_L:",
    f"{model.kappa_lindblad().item():.8e}"
)

print(
    "[+] Initial kappa_N:",
    f"{model.kappa_neural().item():.8e}"
)


# ============================================================
# 10. LOAD STEP-12 FULL LNO WEIGHTS WHERE COMPATIBLE
#
# We intentionally do NOT require this checkpoint.
# If compatible keys exist, load them.
# ============================================================

OLD_CKPT = (
    ROOT
    / "results"
    / "step12_ablation"
    / "D_Full_LNO.pt"
)

if OLD_CKPT.is_file():

    old = torch.load(
        OLD_CKPT,
        map_location=device
    )

    old_state = old.get(
        "model_state_dict",
        old
    )

    current_state = model.state_dict()

    compatible = {}

    for key, value in old_state.items():

        if (
            key in current_state
            and
            current_state[
                key
            ].shape
            ==
            value.shape
        ):

            compatible[
                key
            ] = value

    if compatible:

        current_state.update(
            compatible
        )

        model.load_state_dict(
            current_state
        )

        print(
            "[PASS] Compatible Step-12 weights loaded:",
            len(compatible)
        )

    else:

        print(
            "[INFO] No compatible Step-12 weights found."
        )

else:

    print(
        "[INFO] Step-12 checkpoint unavailable; "
        "using fresh initialization."
    )


# ============================================================
# 11. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

EPOCHS = 10

print(
    "[+] Optimizer: AdamW"
)

print(
    "[+] Learning rate: 1e-3"
)

print(
    "[+] Epochs:",
    EPOCHS
)


# ============================================================
# 12. TRAINING
# ============================================================

OUT_DIR = (
    ROOT
    / "results"
    / "step14_repaired_lno"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_PATH = (
    OUT_DIR
    / "repaired_lno_best.pt"
)

history = []

best_val = float(
    "inf"
)

best_epoch = None

print(
    "\n" + "=" * 100
)

print(
    "STARTING STEP 14 TRAINING"
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

    train_total = 0.0
    train_R = 0.0
    train_A = 0.0

    count = 0

    for (
        R_x,
        A_x,
        env_x,
        R_y,
        A_y
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
            sigma_x
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

        n_batch = (
            R_x.shape[0]
        )

        train_total += (
            loss.item()
            *
            n_batch
        )

        train_R += (
            R_loss.item()
            *
            n_batch
        )

        train_A += (
            A_loss.item()
            *
            n_batch
        )

        count += n_batch

    train_loss = (
        train_total
        /
        count
    )

    train_R_loss = (
        train_R
        /
        count
    )

    train_A_loss = (
        train_A
        /
        count
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_total = 0.0
    val_R = 0.0
    val_A = 0.0

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
                sigma_x
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

            n_batch = (
                R_x.shape[0]
            )

            val_total += (
                loss.item()
                *
                n_batch
            )

            val_R += (
                R_loss.item()
                *
                n_batch
            )

            val_A += (
                A_loss.item()
                *
                n_batch
            )

            val_count += (
                n_batch
            )

    val_loss = (
        val_total
        /
        val_count
    )

    val_R_loss = (
        val_R
        /
        val_count
    )

    val_A_loss = (
        val_A
        /
        val_count
    )

    kL = float(
        model
        .kappa_lindblad()
        .item()
    )

    kN = float(
        model
        .kappa_neural()
        .item()
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
            float(train_R_loss),

        "train_amplitude_loss":
            float(train_A_loss),

        "val_loss":
            float(val_loss),

        "val_R_loss":
            float(val_R_loss),

        "val_amplitude_loss":
            float(val_A_loss),

        "kappa_lindblad":
            kL,

        "kappa_neural":
            kN,

        "time_seconds":
            float(elapsed)
    })

    print(
        f"\nEpoch {epoch:02d}/{EPOCHS}"
    )

    print(
        f"  Train loss : {train_loss:.12e}"
    )

    print(
        f"  Train R    : {train_R_loss:.12e}"
    )

    print(
        f"  Train A    : {train_A_loss:.12e}"
    )

    print(
        f"  Val loss   : {val_loss:.12e}"
    )

    print(
        f"  Val R      : {val_R_loss:.12e}"
    )

    print(
        f"  Val A      : {val_A_loss:.12e}"
    )

    print(
        f"  kappa_L    : {kL:.8e}"
    )

    print(
        f"  kappa_N    : {kN:.8e}"
    )

    print(
        f"  Time       : {elapsed:.2f}s"
    )

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
                        "name":
                            "RepairedLNO",

                        "dim":
                            6,

                        "width":
                            64,

                        "modes":
                            16,

                        "depth":
                            4,

                        "lindblad_channels":
                            4
                    },

                "dynamical_parameterization":
                    {
                        "learnable_lindblad_coupling":
                            True,

                        "learnable_neural_coupling":
                            True,

                        "fixed_neural_0_05":
                            False,

                        "lindblad_in_forward_path":
                            True
                    }
            },

            BEST_PATH
        )

        print(
            "  >>> NEW BEST CHECKPOINT SAVED"
        )


# ============================================================
# 13. RESTORE BEST
# ============================================================

best_checkpoint = torch.load(
    BEST_PATH,
    map_location=device
)

model.load_state_dict(
    best_checkpoint[
        "model_state_dict"
    ]
)

model.eval()


# ============================================================
# 14. FINAL TEST
# ============================================================

X_R_test_t = torch.tensor(
    X_R_test,
    dtype=torch.float32,
    device=device
)

Y_R_test_t = torch.tensor(
    Y_R_test,
    dtype=torch.float32,
    device=device
)

X_A_test_t = torch.tensor(
    X_A_test,
    dtype=torch.float32,
    device=device
)

Y_A_test_t = torch.tensor(
    Y_A_test,
    dtype=torch.float32,
    device=device
)

ENV_test_t = torch.tensor(
    ENV_test,
    dtype=torch.float32,
    device=device
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

    out = model(
        X_R_test_t,
        X_A_test_t,
        gamma_test,
        sigma_test
    )

    R_pred = out[
        "R_next"
    ]

    A_pred = out[
        "amplitude_next"
    ]

    dL = out[
        "delta_R_lindblad"
    ]

    dN = out[
        "delta_R_neural"
    ]

    dTotal = out[
        "delta_R_total"
    ]


# ============================================================
# 15. TEST METRICS
# ============================================================

R_error = (
    R_pred
    -
    Y_R_test_t
)

A_error = (
    A_pred
    -
    Y_A_test_t
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

R_Relative_L2 = (
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

A_Relative_L2 = (
    torch.linalg.vector_norm(
        A_error
    )
    /
    (
        torch.linalg.vector_norm(
            Y_A_test_t
        )
        +
        1e-12
    )
)


# ============================================================
# 16. STRUCTURAL TEST
# ============================================================

trace = torch.diagonal(
    R_pred,
    dim1=-2,
    dim2=-1
).sum(
    dim=-1
)

trace_error = float(
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

eigvals = torch.linalg.eigvalsh(
    R_pred
)

minimum_eigenvalue = float(
    eigvals.min().item()
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


# ============================================================
# 17. COMPONENT MAGNITUDES
# ============================================================

lindblad_norm = float(
    torch.linalg.vector_norm(
        dL
    ).item()
)

neural_norm = float(
    torch.linalg.vector_norm(
        dN
    ).item()
)

total_norm = float(
    torch.linalg.vector_norm(
        dTotal
    ).item()
)

target_delta_norm = float(
    torch.linalg.vector_norm(
        Y_R_test_t
        -
        X_R_test_t
    ).item()
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


# ============================================================
# 18. ENVIRONMENT RESPONSE
# ============================================================

probe_R = X_R_test_t[
    :1
]

probe_A = X_A_test_t[
    :1
]

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
        0.50
    ]:

        gamma_probe = torch.full_like(
            ENV_test_t[
                :1,
                :,
                0
            ],
            gamma_value
        )

        env_out = model(
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
                        env_out[
                            "R_next"
                        ]
                        -
                        probe_R
                    ).item()
                ),

            "amplitude_change":
                float(
                    torch.linalg.vector_norm(
                        env_out[
                            "amplitude_next"
                        ]
                        -
                        probe_A
                    ).item()
                ),

            "Lindblad_delta_norm":
                float(
                    torch.linalg.vector_norm(
                        env_out[
                            "delta_R_lindblad"
                        ]
                    ).item()
                ),

            "Neural_delta_norm":
                float(
                    torch.linalg.vector_norm(
                        env_out[
                            "delta_R_neural"
                        ]
                    ).item()
                ),
        })

environment_df = pd.DataFrame(
    environment_rows
)


# ============================================================
# 19. REPEATED ROLLOUT
# ============================================================

R_roll = (
    X_R_test_t[
        :1
    ].clone()
)

A_roll = (
    X_A_test_t[
        :1
    ].clone()
)

gamma_roll = (
    ENV_test_t[
        :1,
        :,
        0
    ].clone()
)

sigma_roll = (
    ENV_test_t[
        :1,
        :,
        1
    ].clone()
)

rollout_rows = []

with torch.no_grad():

    for step in range(
        1,
        101
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

rollout_pass = bool(
    rollout_finite
    and
    rollout_max_trace_error < 1e-4
    and
    rollout_min_eig > -1e-6
)


# ============================================================
# 20. SAVE
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    OUT_DIR
    / "training_history.csv",
    index=False
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

np.savez_compressed(

    OUT_DIR
    / "test_predictions.npz",

    R_prediction=
        R_pred
        .detach()
        .cpu()
        .numpy(),

    R_target=
        Y_R_test_t
        .detach()
        .cpu()
        .numpy(),

    amplitude_prediction=
        A_pred
        .detach()
        .cpu()
        .numpy(),

    amplitude_target=
        Y_A_test_t
        .detach()
        .cpu()
        .numpy(),

    environment=
        ENV_test_t
        .detach()
        .cpu()
        .numpy(),

    delta_R_lindblad=
        dL
        .detach()
        .cpu()
        .numpy(),

    delta_R_neural=
        dN
        .detach()
        .cpu()
        .numpy()
)


# ============================================================
# 21. SUMMARY JSON
# ============================================================

summary = {

    "step":
        14,

    "prototype":
        True,

    "dataset":
        str(
            DATASET_PATH
        ),

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

    "best_epoch":
        int(
            best_epoch
        ),

    "best_validation_loss":
        float(
            best_val
        ),

    "test_R_MSE":
        float(
            R_MSE.item()
        ),

    "test_R_RMSE":
        float(
            R_RMSE.item()
        ),

    "test_R_MAE":
        float(
            R_MAE.item()
        ),

    "test_R_relative_L2":
        float(
            R_Relative_L2.item()
        ),

    "test_amplitude_MSE":
        float(
            A_MSE.item()
        ),

    "test_amplitude_RMSE":
        float(
            A_RMSE.item()
        ),

    "test_amplitude_MAE":
        float(
            A_MAE.item()
        ),

    "test_amplitude_relative_L2":
        float(
            A_Relative_L2.item()
        ),

    "trace_error":
        trace_error,

    "symmetry_error":
        symmetry_error,

    "minimum_eigenvalue":
        minimum_eigenvalue,

    "finite":
        finite,

    "kappa_lindblad":
        kappa_L,

    "kappa_neural":
        kappa_N,

    "target_delta_R_norm":
        target_delta_norm,

    "learned_lindblad_delta_R_norm":
        lindblad_norm,

    "learned_neural_delta_R_norm":
        neural_norm,

    "combined_delta_R_norm":
        total_norm,

    "rollout_minimum_eigenvalue":
        rollout_min_eig,

    "rollout_max_trace_error":
        rollout_max_trace_error,

    "rollout_max_R_norm":
        rollout_max_R_norm,

    "rollout_max_amplitude_norm":
        rollout_max_A_norm,

    "rollout_finite":
        rollout_finite,

    "rollout_pass":
        rollout_pass,

    "lindblad_in_forward_path":
        True,

    "fixed_neural_0_05":
        False,

    "trainable_lindblad_coupling":
        True,

    "trainable_neural_coupling":
        True,

    "checkpoint":
        str(
            BEST_PATH
        ),
}

SUMMARY_PATH = (
    OUT_DIR
    / "step14_summary.json"
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
# 22. FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "STEP 14 COMPLETE — REPAIRED LNO"
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
    f"{best_val:.12e}"
)

print(
    "\nTEST"
)

print(
    "R RMSE:",
    f"{R_RMSE.item():.12e}"
)

print(
    "R Relative-L2:",
    f"{R_Relative_L2.item():.12e}"
)

print(
    "Amplitude RMSE:",
    f"{A_RMSE.item():.12e}"
)

print(
    "Amplitude Relative-L2:",
    f"{A_Relative_L2.item():.12e}"
)

print(
    "\nDYNAMICAL COUPLINGS"
)

print(
    "kappa_L:",
    f"{kappa_L:.12e}"
)

print(
    "kappa_N:",
    f"{kappa_N:.12e}"
)

print(
    "\nCOMPONENT MAGNITUDES"
)

print(
    "Target ΔR norm:",
    f"{target_delta_norm:.12e}"
)

print(
    "Lindblad ΔR norm:",
    f"{lindblad_norm:.12e}"
)

print(
    "Neural ΔR norm:",
    f"{neural_norm:.12e}"
)

print(
    "Combined ΔR norm:",
    f"{total_norm:.12e}"
)

print(
    "\nSTRUCTURE"
)

print(
    "Trace error:",
    f"{trace_error:.12e}"
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
    "\nROLLOUT"
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
    rollout_pass
)

print(
    "\nEnvironment response:"
)

display(
    environment_df.round(8)
)

print(
    "\nCheckpoint:",
    BEST_PATH
)

print(
    "Summary:",
    SUMMARY_PATH
)

print(
    "\n[INFO] Step 14 replaces the old fixed 0.05 neural coupling."
)

print(
    "[INFO] Lindblad remains part of the forward dynamics."
)

print(
    "[INFO] This is still a prototype; final expanded dataset remains pending."
)

print("=" * 100)
