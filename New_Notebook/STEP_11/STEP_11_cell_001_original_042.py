# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 42
# Step            : STEP_11
# Step Heading    : # STEP 11 — LNO PROTOTYPE TRAINING
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 11 — LNO PROTOTYPE TRAINING
#
# SAME DATA / SAME SPLIT / SAME TARGET AS FNO
#
# INPUT:
#   R(t), amplitude(t), gamma, noise_sigma
#
# TARGET:
#   R(t+1), amplitude(t+1)
#
# DIFFERENCE:
#   FNO = unconstrained Fourier operator
#   LNO = Fourier operator + Lindblad forward kernel
#
# CURRENT:
#   495-transition prototype dataset
#
# FINAL:
#   Expanded independent-trajectory Dataset V3
#
# NO OLD LNO CHECKPOINT MODIFICATION
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

# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

print("=" * 100)
print("STEP 11 — LNO PROTOTYPE TRAINING")
print("=" * 100)

print(
    "[+] Device:",
    device
)

# ============================================================
# 3. LOAD STEP-9 SPLIT
# ============================================================

SPLIT_PATH = (
    ROOT
    / "results"
    / "dataset_v3"
    / "step9_splits"
    / "prototype_train_val_test_split.npz"
)

assert SPLIT_PATH.is_file(), (
    f"Missing split file:\n{SPLIT_PATH}"
)

data = np.load(
    SPLIT_PATH
)

# ============================================================
# 4. LOAD TRAIN / VAL / TEST
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
# 5. SHAPE CHECK
# ============================================================

N_train, NX, DIM, DIM2 = X_R_train.shape
N_val = X_R_val.shape[0]
N_test = X_R_test.shape[0]

assert DIM == 6
assert DIM2 == 6

assert X_R_train.shape == Y_R_train.shape
assert X_R_val.shape == Y_R_val.shape
assert X_R_test.shape == Y_R_test.shape

assert X_A_train.shape == (
    N_train,
    NX
)

assert Y_A_train.shape == (
    N_train,
    NX
)

assert X_A_val.shape == (
    N_val,
    NX
)

assert Y_A_val.shape == (
    N_val,
    NX
)

assert X_A_test.shape == (
    N_test,
    NX
)

assert Y_A_test.shape == (
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
    "[+] NX:",
    NX
)

print(
    "[+] Information dimension:",
    DIM
)

# ============================================================
# 6. FINITE CHECK
# ============================================================

for name, arr in {

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

}.items():

    assert np.isfinite(
        arr
    ).all(), (
        f"Non-finite values: {name}"
    )

print(
    "[PASS] Dataset finite."
)

# ============================================================
# 7. TORCH TENSORS
# ============================================================

def tensor(
    arr
):

    return torch.tensor(
        arr,
        dtype=torch.float32,
        device=device
    )

X_R_train_t = tensor(
    X_R_train
)

Y_R_train_t = tensor(
    Y_R_train
)

X_A_train_t = tensor(
    X_A_train
)

Y_A_train_t = tensor(
    Y_A_train
)

ENV_train_t = tensor(
    ENV_train
)

X_R_val_t = tensor(
    X_R_val
)

Y_R_val_t = tensor(
    Y_R_val
)

X_A_val_t = tensor(
    X_A_val
)

Y_A_val_t = tensor(
    Y_A_val
)

ENV_val_t = tensor(
    ENV_val
)

X_R_test_t = tensor(
    X_R_test
)

Y_R_test_t = tensor(
    Y_R_test
)

X_A_test_t = tensor(
    X_A_test
)

Y_A_test_t = tensor(
    Y_A_test
)

ENV_test_t = tensor(
    ENV_test
)

# ============================================================
# 8. DATALOADERS
# ============================================================

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
# 9. SPECTRAL CONVOLUTION
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
# 10. LINDBLAD KERNEL
# ============================================================

class LindbladDissipativeKernel(
    nn.Module
):

    """
    Learnable Lindblad-form dissipative forward operator.

    D(R) =
        Σ_k [
            L_k R L_k^T
            -
            1/2{
                L_k^T L_k,
                R
            }
        ]

    Environmental coupling gamma(x) explicitly modulates
    the dissipative evolution.

    This is NOT a loss-function regularizer.
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
        dt
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
# 11. LNO MODEL
# ============================================================

class LNOPrototype(
    nn.Module
):

    """
    Spatial Lindblad Neural Operator.

    Fourier layers learn global spatial transformations.

    Lindblad kernel supplies structured dissipative dynamics
    directly inside the forward evolution.
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

    # --------------------------------------------------------
    # PSD / trace projection
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
        # A. Build input field
        # ====================================================

        R_flat = R.reshape(
            B,
            NX_local,
            D * D
        )

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

        # ====================================================
        # B. Fourier representation
        # ====================================================

        x = inp.permute(
            0,
            2,
            1
        )

        x = self.input_projection(
            x
        )

        spectral_history = []

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

            spectral_history.append(
                torch.linalg.vector_norm(
                    x,
                    dim=(1, 2)
                )
            )

            if k < (
                self.depth - 1
            ):

                x = F.gelu(
                    x
                )

        # ====================================================
        # C. Neural R tendency
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

        # Normalize neural tendency
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
        # D. Lindblad forward dynamics
        # ====================================================

        R_lindblad, D_R = (
            self.lindblad_kernel(
                R,
                gamma,
                dt
            )
        )

        # Small neural correction during prototype
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
        # E. Constrained state projection
        # ====================================================

        R_next = self.project_R(
            R_candidate
        )

        # ====================================================
        # F. Amplitude evolution
        # ====================================================

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

            "spectral_history":
                spectral_history,
        }

# ============================================================
# 12. CREATE MODEL
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

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "\n[+] LNO parameters:",
    f"{parameter_count:,}"
)

print(
    "[+] Lindblad jump operators:",
    tuple(
        model
        .lindblad_kernel
        .jump_operators
        .shape
    )
)

print(
    "[+] Initial alpha:",
    model
    .lindblad_kernel
    .positive_alpha()
    .item()
)

# ============================================================
# 13. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)

EPOCHS = 5

print(
    "[+] Optimizer: AdamW"
)

print(
    "[+] LR: 1e-3"
)

print(
    "[+] Weight decay: 1e-4"
)

print(
    "[+] Prototype epochs:",
    EPOCHS
)

# ============================================================
# 14. OUTPUT DIRECTORY
# ============================================================

OUT_DIR = (
    ROOT
    / "results"
    / "step11_lno_prototype"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    OUT_DIR
    / "lno_prototype_best.pt"
)

# ============================================================
# 15. TRAINING HISTORY
# ============================================================

history = []

best_val_loss = float(
    "inf"
)

best_epoch = None

# ============================================================
# 16. TRAIN
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "STARTING LNO PROTOTYPE TRAINING"
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
    train_R_total = 0.0
    train_A_total = 0.0

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

        output = model(
            R_x,
            A_x,
            gamma_x,
            sigma_x,
            dt=1e-3
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

        batch_n = R_x.shape[0]

        train_total += (
            loss.item()
            *
            batch_n
        )

        train_R_total += (
            R_loss.item()
            *
            batch_n
        )

        train_A_total += (
            A_loss.item()
            *
            batch_n
        )

        train_count += batch_n

    train_loss = (
        train_total
        /
        train_count
    )

    train_R_loss = (
        train_R_total
        /
        train_count
    )

    train_A_loss = (
        train_A_total
        /
        train_count
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_total = 0.0
    val_R_total = 0.0
    val_A_total = 0.0

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

            output = model(
                R_x,
                A_x,
                gamma_x,
                sigma_x,
                dt=1e-3
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

            batch_n = R_x.shape[0]

            val_total += (
                loss.item()
                *
                batch_n
            )

            val_R_total += (
                R_loss.item()
                *
                batch_n
            )

            val_A_total += (
                A_loss.item()
                *
                batch_n
            )

            val_count += batch_n

    val_loss = (
        val_total
        /
        val_count
    )

    val_R_loss = (
        val_R_total
        /
        val_count
    )

    val_A_loss = (
        val_A_total
        /
        val_count
    )

    elapsed = (
        time.time()
        -
        start_time
    )

    alpha_value = (
        model
        .lindblad_kernel
        .positive_alpha()
        .item()
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

        "alpha":
            float(alpha_value),

        "time_seconds":
            float(elapsed),
    })

    print(
        f"\nEpoch {epoch:02d}/{EPOCHS}"
    )

    print(
        f"  Train loss       : {train_loss:.12e}"
    )

    print(
        f"  Train R loss     : {train_R_loss:.12e}"
    )

    print(
        f"  Train amplitude  : {train_A_loss:.12e}"
    )

    print(
        f"  Val loss         : {val_loss:.12e}"
    )

    print(
        f"  Val R loss       : {val_R_loss:.12e}"
    )

    print(
        f"  Val amplitude    : {val_A_loss:.12e}"
    )

    print(
        f"  Alpha            : {alpha_value:.8e}"
    )

    print(
        f"  Time             : {elapsed:.2f}s"
    )

    # ========================================================
    # BEST CHECKPOINT
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = (
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
                            "LNOPrototype",

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

                "representation":
                    {

                        "R":
                            "z z^T / Tr(z z^T)",

                        "amplitude":
                            "full-information amplitude in Dataset V3",
                    },

                "lindblad_in_forward_path":
                    True,

                "prototype":
                    True,
            },
            BEST_MODEL_PATH
        )

        print(
            "  >>> NEW BEST LNO CHECKPOINT SAVED"
        )

# ============================================================
# 17. SAVE HISTORY
# ============================================================

history_df = pd.DataFrame(
    history
)

HISTORY_PATH = (
    OUT_DIR
    / "training_history.csv"
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)

# ============================================================
# 18. LOAD BEST
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
    best_val_loss
)

# ============================================================
# 19. TEST EVALUATION
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "F. LNO PROTOTYPE TEST EVALUATION"
)

print(
    "=" * 100
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
        sigma_test,
        dt=1e-3
    )

    R_test_pred = output[
        "R_next"
    ]

    A_test_pred = output[
        "amplitude_next"
    ]

    D_test = output[
        "lindblad_dissipation"
    ]

# ============================================================
# 20. TEST METRICS
# ============================================================

R_error = (
    R_test_pred
    -
    Y_R_test_t
)

A_error = (
    A_test_pred
    -
    Y_A_test_t
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

R_relative_l2 = (
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

A_relative_l2 = (
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

print(
    "R MSE:",
    f"{R_mse.item():.12e}"
)

print(
    "R RMSE:",
    f"{R_rmse.item():.12e}"
)

print(
    "R MAE:",
    f"{R_mae.item():.12e}"
)

print(
    "R Relative-L2:",
    f"{R_relative_l2.item():.12e}"
)

print(
    "\nAmplitude MSE:",
    f"{A_mse.item():.12e}"
)

print(
    "Amplitude RMSE:",
    f"{A_rmse.item():.12e}"
)

print(
    "Amplitude MAE:",
    f"{A_mae.item():.12e}"
)

print(
    "Amplitude Relative-L2:",
    f"{A_relative_l2.item():.12e}"
)

# ============================================================
# 21. LNO STRUCTURAL TEST
# ============================================================

R_trace = torch.diagonal(
    R_test_pred,
    dim1=-2,
    dim2=-1
).sum(
    dim=-1
)

trace_error = torch.max(
    torch.abs(
        R_trace
        -
        1.0
    )
).item()

symmetry_error = torch.max(
    torch.abs(
        R_test_pred
        -
        R_test_pred.transpose(
            -1,
            -2
        )
    )
).item()

eigvals = torch.linalg.eigvalsh(
    R_test_pred
)

minimum_eigenvalue = torch.min(
    eigvals
).item()

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
    "\n" + "=" * 100
)

print(
    "G. LNO OUTPUT STRUCTURE"
)

print(
    "=" * 100
)

print(
    "Max |trace - 1|:",
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

print(
    "Test predictions finite:",
    test_finite
)

# ============================================================
# 22. LNO GRADIENT DIAGNOSTIC
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "H. FINAL LNO LEARNING SIGNAL"
)

print(
    "=" * 100
)

model.train()

model.zero_grad(
    set_to_none=True
)

sample_R = (
    X_R_train_t[
        :BATCH_SIZE
    ]
    .detach()
    .clone()
    .requires_grad_()
)

sample_A = (
    X_A_train_t[
        :BATCH_SIZE
    ]
    .detach()
    .clone()
    .requires_grad_()
)

sample_ENV = (
    ENV_train_t[
        :BATCH_SIZE
    ]
    .detach()
)

sample_gamma = (
    sample_ENV[
        ...,
        0
    ]
)

sample_sigma = (
    sample_ENV[
        ...,
        1
    ]
)

sample_target_R = (
    Y_R_train_t[
        :BATCH_SIZE
    ]
)

sample_target_A = (
    Y_A_train_t[
        :BATCH_SIZE
    ]
)

output = model(
    sample_R,
    sample_A,
    sample_gamma,
    sample_sigma,
    dt=1e-3
)

grad_loss = (
    F.mse_loss(
        output[
            "R_next"
        ],
        sample_target_R
    )
    +
    F.mse_loss(
        output[
            "amplitude_next"
        ],
        sample_target_A
    )
)

grad_loss.backward()

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

input_R_grad = float(
    torch.linalg.vector_norm(
        sample_R.grad
    ).item()
)

input_A_grad = float(
    torch.linalg.vector_norm(
        sample_A.grad
    ).item()
)

print(
    "Gradient-test loss:",
    f"{grad_loss.item():.12e}"
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
    "Input R gradient:",
    f"{input_R_grad:.12e}"
)

print(
    "Input amplitude gradient:",
    f"{input_A_grad:.12e}"
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
    "[PASS] Lindblad parameters receive learning signal."
    if gradient_pass
    else
    "[REVIEW] Lindblad parameters have weak/zero learning signal."
)

# ============================================================
# 23. ENVIRONMENT RESPONSE
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "I. LNO ENVIRONMENT RESPONSE"
)

print(
    "=" * 100
)

model.eval()

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
        0.50,
    ]:

        probe_gamma = torch.full_like(
            ENV_test_t[
                :1,
                :,
                0
            ],
            gamma_value
        )

        output = model(
            probe_R,
            probe_A,
            probe_gamma,
            probe_sigma,
            dt=1e-3
        )

        R_change = (
            torch.linalg.vector_norm(
                output[
                    "R_next"
                ]
                -
                probe_R
            )
            .item()
        )

        A_change = (
            torch.linalg.vector_norm(
                output[
                    "amplitude_next"
                ]
                -
                probe_A
            )
            .item()
        )

        D_norm = (
            torch.linalg.vector_norm(
                output[
                    "lindblad_dissipation"
                ]
            )
            .item()
        )

        environment_rows.append({

            "gamma":
                float(
                    gamma_value
                ),

            "R_change":
                float(
                    R_change
                ),

            "amplitude_change":
                float(
                    A_change
                ),

            "Lindblad_D_norm":
                float(
                    D_norm
                ),
        })

environment_df = pd.DataFrame(
    environment_rows
)

display(
    environment_df.round(10)
)

# ============================================================
# 24. SAVE TEST RESULTS
# ============================================================

np.savez_compressed(

    OUT_DIR
    / "lno_prototype_test_predictions.npz",

    R_prediction=
        R_test_pred
        .detach()
        .cpu()
        .numpy(),

    R_target=
        Y_R_test_t
        .detach()
        .cpu()
        .numpy(),

    amplitude_prediction=
        A_test_pred
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
)

environment_df.to_csv(
    OUT_DIR
    / "environment_response.csv",
    index=False
)

# ============================================================
# 25. SUMMARY
# ============================================================

summary = {

    "model":
        "LNOPrototype",

    "prototype":
        True,

    "dataset":
        "Dataset V3 — 495 transitions",

    "train_transitions":
        int(N_train),

    "validation_transitions":
        int(N_val),

    "test_transitions":
        int(N_test),

    "spatial_points":
        int(NX),

    "information_dimension":
        int(DIM),

    "parameters":
        int(parameter_count),

    "epochs":
        int(EPOCHS),

    "best_epoch":
        int(best_epoch),

    "best_validation_loss":
        float(best_val_loss),

    "test_R_MSE":
        float(R_mse.item()),

    "test_R_RMSE":
        float(R_rmse.item()),

    "test_R_MAE":
        float(R_mae.item()),

    "test_R_relative_L2":
        float(R_relative_l2.item()),

    "test_amplitude_MSE":
        float(A_mse.item()),

    "test_amplitude_RMSE":
        float(A_rmse.item()),

    "test_amplitude_MAE":
        float(A_mae.item()),

    "test_amplitude_relative_L2":
        float(A_relative_l2.item()),

    "test_predictions_finite":
        bool(test_finite),

    "output_trace_max_error":
        float(trace_error),

    "output_symmetry_max_error":
        float(symmetry_error),

    "output_minimum_eigenvalue":
        float(minimum_eigenvalue),

    "jump_gradient_L2":
        float(jump_grad_norm),

    "alpha_gradient":
        float(alpha_grad_norm),

    "gradient_pass":
        bool(gradient_pass),

    "lindblad_in_forward_path":
        True,

    "lindblad_as_loss_penalty":
        False,

    "environment_response_nonzero":
        bool(
            environment_df[
                "Lindblad_D_norm"
            ].iloc[-1]
            >
            0.0
        ),

    "checkpoint":
        str(BEST_MODEL_PATH),

    "history":
        str(HISTORY_PATH),
}

SUMMARY_PATH = (
    OUT_DIR
    / "lno_prototype_summary.json"
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
# 26. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "STEP 11 COMPLETE — LNO PROTOTYPE"
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
    f"{best_val_loss:.12e}"
)

print(
    "[+] Test R RMSE:",
    f"{R_rmse.item():.12e}"
)

print(
    "[+] Test R Relative-L2:",
    f"{R_relative_l2.item():.12e}"
)

print(
    "[+] Test amplitude RMSE:",
    f"{A_rmse.item():.12e}"
)

print(
    "[+] Test amplitude Relative-L2:",
    f"{A_relative_l2.item():.12e}"
)

print(
    "[+] Output trace error:",
    f"{trace_error:.12e}"
)

print(
    "[+] Output minimum eigenvalue:",
    f"{minimum_eigenvalue:.12e}"
)

print(
    "[+] Jump gradient:",
    f"{jump_grad_norm:.12e}"
)

print(
    "[+] Alpha gradient:",
    f"{alpha_grad_norm:.12e}"
)

print(
    "[+] Checkpoint:",
    BEST_MODEL_PATH
)

print(
    "[+] Summary:",
    SUMMARY_PATH
)

print(
    "\n[INFO] This is a PROTOTYPE comparison only."
)

print(
    "[INFO] Do not interpret it as the final paper result."
)

print(
    "[INFO] Final sample expansion is still pending."
)

print("=" * 100)
