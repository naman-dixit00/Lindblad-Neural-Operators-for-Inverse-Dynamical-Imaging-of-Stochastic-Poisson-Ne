# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 81
# Step            : STEP_26
# Step Heading    : # STEP 26 — R ACCURACY IMPROVEMENT ABLATION
# Step Cell No.   : 2
# ============================================================

# ======================================================================================
# STEP 26 — R ACCURACY IMPROVEMENT ABLATION
# ======================================================================================
#
# Scientific question:
#   Why is LNO's one-step R accuracy ~1% worse than FNO?
#
# Controlled factors:
#   - Same expanded dataset
#   - Same trajectory-level split
#   - Same architecture
#   - Same parameter count
#   - Fresh initialization
#   - Same optimizer
#   - Same amplitude normalization
#
# Variants:
#   A = baseline:              R_MSE + Amp_MSE
#   B = R-focused:             R_MSE + 0.25 * R_RelativeL2 + Amp_MSE
#   C = R-focused + mild
#       Lindblad coupling penalty
#
# No existing final checkpoint is modified.
# ======================================================================================

import os
import json
import math
import time
import random
import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader


# ======================================================================================
# 1. ROOT / REPRODUCIBILITY
# ======================================================================================

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

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 110)
print("STEP 26 — R ACCURACY IMPROVEMENT ABLATION")
print("=" * 110)
print("[+] Device:", device)


# ======================================================================================
# 2. PATHS — CURRENT COLAB SESSION
# ======================================================================================

DATASET_PATH = Path(
    "/content/environment_conditioned_lno_dataset_v3_expanded.npz"
)

SPLIT_PATH = Path(
    "/content/final_trajectory_level_split.npz"
)

OUT_DIR = (
    ROOT
    / "results"
    / "step26_r_accuracy_ablation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("[+] Dataset path:", DATASET_PATH)
print("[+] Split path  :", SPLIT_PATH)

assert DATASET_PATH.is_file(), f"Missing dataset: {DATASET_PATH}"
assert SPLIT_PATH.is_file(), f"Missing split: {SPLIT_PATH}"

OUT_DIR = (
    ROOT
    / "results"
    / "step26_r_accuracy_ablation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

assert DATASET_PATH.is_file()
assert SPLIT_PATH.is_file()


# ======================================================================================
# 3. LOAD DATA
# ======================================================================================

data = np.load(DATASET_PATH)
split = np.load(SPLIT_PATH)

X_R_all = data["X_R"].astype(np.float32)
Y_R_all = data["Y_R"].astype(np.float32)

X_A_all = data["X_amplitude"].astype(np.float32)
Y_A_all = data["Y_amplitude"].astype(np.float32)

ENV_all = data["environment"].astype(np.float32)

train_idx = split["train_indices"].astype(np.int64)
val_idx = split["val_indices"].astype(np.int64)
test_idx = split["test_indices"].astype(np.int64)

print("\nDATASET")
print("-" * 110)
print("X_R:", X_R_all.shape)
print("Y_R:", Y_R_all.shape)
print("X_A:", X_A_all.shape)
print("Y_A:", Y_A_all.shape)
print("ENV:", ENV_all.shape)

print("\nSPLIT")
print("-" * 110)
print("Train:", len(train_idx))
print("Val  :", len(val_idx))
print("Test :", len(test_idx))

assert len(train_idx) == 6930
assert len(val_idx) == 1485
assert len(test_idx) == 1485


# ======================================================================================
# 4. SAME AMPLITUDE NORMALIZATION AS FINAL FNO/LNO
# ======================================================================================

amplitude_mean = float(
    X_A_all[train_idx].mean()
)

amplitude_std = float(
    X_A_all[train_idx].std()
)

print("\nAMPLITUDE NORMALIZATION")
print("-" * 110)
print("Mean:", amplitude_mean)
print("Std :", amplitude_std)

X_A_norm = (
    X_A_all - amplitude_mean
) / max(amplitude_std, 1e-8)

Y_A_norm = (
    Y_A_all - amplitude_mean
) / max(amplitude_std, 1e-8)


# ======================================================================================
# 5. DATA LOADERS
# ======================================================================================

def make_loader(indices, shuffle):

    ds = TensorDataset(
        torch.from_numpy(X_R_all[indices]),
        torch.from_numpy(X_A_norm[indices]),
        torch.from_numpy(ENV_all[indices]),
        torch.from_numpy(Y_R_all[indices]),
        torch.from_numpy(Y_A_norm[indices]),
    )

    return DataLoader(
        ds,
        batch_size=16,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )


train_loader = make_loader(
    train_idx,
    True
)

val_loader = make_loader(
    val_idx,
    False
)

test_loader = make_loader(
    test_idx,
    False
)


# ======================================================================================
# 6. SPECTRAL CONVOLUTION
# ======================================================================================

class SpectralConv1D(nn.Module):

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

        scale = 1.0 / math.sqrt(
            in_channels * out_channels
        )

        self.weight_real = nn.Parameter(
            scale *
            torch.randn(
                in_channels,
                out_channels,
                modes
            )
        )

        self.weight_imag = nn.Parameter(
            scale *
            torch.randn(
                in_channels,
                out_channels,
                modes
            )
        )

    def forward(self, x):

        B, C, NX = x.shape

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
            self.weight_real[..., :n_modes],
            self.weight_imag[..., :n_modes]
        )

        out_ft[..., :n_modes] = torch.einsum(
            "bim,iom->bom",
            x_ft[..., :n_modes],
            weight
        )

        return torch.fft.irfft(
            out_ft,
            n=NX,
            dim=-1
        )


# ======================================================================================
# 7. LINDBLAD KERNEL
# ======================================================================================

class LindbladKernel(nn.Module):

    def __init__(
        self,
        dim=6,
        channels=4
    ):

        super().__init__()

        self.dim = dim
        self.channels = channels

        self.jump_operators = nn.Parameter(
            0.02 *
            torch.randn(
                channels,
                dim,
                dim
            )
        )

    def forward(self, R):

        total = torch.zeros_like(R)

        for k in range(self.channels):

            L = self.jump_operators[k]

            LT = L.transpose(
                -1,
                -2
            )

            A = (
                LT @ L
            )

            jump_term = (
                L @ R @ LT
            )

            anti_term = (
                A @ R +
                R @ A
            )

            total = (
                total +
                jump_term -
                0.5 * anti_term
            )

        return total


# ======================================================================================
# 8. FINAL REPAIRED LNO
# ======================================================================================

class FinalRepairedLNO(nn.Module):

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

        in_channels = (
            dim * dim + 3
        )

        self.input_projection = nn.Conv1d(
            in_channels,
            width,
            kernel_size=1
        )

        self.spectral_layers = nn.ModuleList([
            SpectralConv1D(
                width,
                width,
                modes
            )
            for _ in range(depth)
        ])

        self.pointwise_layers = nn.ModuleList([
            nn.Conv1d(
                width,
                width,
                kernel_size=1
            )
            for _ in range(depth)
        ])

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

        self.lindblad = LindbladKernel(
            dim=dim,
            channels=lindblad_channels
        )

        self.raw_kappa_lindblad = nn.Parameter(
            torch.tensor(-0.5)
        )

        self.raw_kappa_neural = nn.Parameter(
            torch.tensor(0.5)
        )


    def project_R(self, R):

        original_dtype = R.dtype

        # Float64 is used for numerical robustness.
        M = R.double()

        # Symmetrize.
        M = 0.5 * (
            M +
            M.transpose(-1, -2)
        )

        # Eigendecomposition.
        eigvals, eigvecs = torch.linalg.eigh(
            M
        )

        # PSD projection.
        eigvals = torch.clamp(
            eigvals,
            min=1e-8
        )

        M_psd = (
            eigvecs *
            eigvals.unsqueeze(-2)
        ) @ eigvecs.transpose(
            -1,
            -2
        )

        # Trace normalization.
        trace = torch.diagonal(
            M_psd,
            dim1=-2,
            dim2=-1
        ).sum(
            dim=-1
        )

        trace = torch.clamp(
            trace,
            min=1e-12
        )

        M_psd = (
            M_psd /
            trace.unsqueeze(-1).unsqueeze(-1)
        )

        # Final symmetrization.
        M_psd = 0.5 * (
            M_psd +
            M_psd.transpose(-1, -2)
        )

        return M_psd.to(
            original_dtype
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

        x = torch.cat(
            [
                R_flat,
                env
            ],
            dim=-1
        ).permute(
            0,
            2,
            1
        )

        x = self.input_projection(
            x
        )

        for k in range(
            len(self.spectral_layers)
        ):

            x = (
                self.spectral_layers[k](x)
                +
                self.pointwise_layers[k](x)
            )

            if k < len(
                self.spectral_layers
            ) - 1:

                x = F.gelu(x)

        G_N = torch.tanh(
            self.R_head(x)
            .permute(0, 2, 1)
            .reshape(
                B,
                NX,
                D,
                D
            )
        )

        G_L = self.lindblad(R)

        kappa_L = F.softplus(
            self.raw_kappa_lindblad
        )

        kappa_N = F.softplus(
            self.raw_kappa_neural
        )

        delta_R_lindblad = (
            kappa_L *
            gamma[..., None, None] *
            G_L
        )

        delta_R_neural = (
            kappa_N *
            G_N
        )

        delta_total = (
            delta_R_lindblad +
            delta_R_neural
        )

        R_next = self.project_R(
            R + delta_total
        )

        A_next = self.A_head(
            x
        ).squeeze(1)

        return {
            "R_next": R_next,
            "amplitude_next": A_next,
            "delta_R_lindblad":
                delta_R_lindblad,
            "delta_R_neural":
                delta_R_neural,
            "delta_R_total":
                delta_total
        }


# ======================================================================================
# 9. LOSSES
# ======================================================================================

def relative_l2_loss(pred, target):

    num = torch.linalg.vector_norm(
        pred - target
    )

    den = torch.linalg.vector_norm(
        target
    ).clamp_min(1e-8)

    return num / den


def compute_losses(
    output,
    R_target,
    A_target,
    variant
):

    R_pred = output["R_next"]

    A_pred = output["amplitude_next"]

    R_mse = F.mse_loss(
        R_pred,
        R_target
    )

    A_mse = F.mse_loss(
        A_pred,
        A_target
    )

    R_rel = relative_l2_loss(
        R_pred,
        R_target
    )

    # ------------------------------------------------------------------
    # Variant A
    # ------------------------------------------------------------------
    if variant == "A_baseline":

        loss = (
            R_mse +
            A_mse
        )

        physics_penalty = torch.tensor(
            0.0,
            device=R_pred.device
        )

    # ------------------------------------------------------------------
    # Variant B
    # ------------------------------------------------------------------
    elif variant == "B_R_focused":

        loss = (
            R_mse +
            0.25 * R_rel +
            A_mse
        )

        physics_penalty = torch.tensor(
            0.0,
            device=R_pred.device
        )

    # ------------------------------------------------------------------
    # Variant C
    # ------------------------------------------------------------------
    elif variant == "C_R_focused_soft_physics":

        kappa_L = F.softplus(
            model.raw_kappa_lindblad
        )

        physics_penalty = (
            kappa_L ** 2
        )

        loss = (
            R_mse +
            0.25 * R_rel +
            A_mse +
            1e-3 * physics_penalty
        )

    else:

        raise ValueError(
            f"Unknown variant: {variant}"
        )

    return {
        "loss": loss,
        "R_mse": R_mse,
        "R_rel": R_rel,
        "A_mse": A_mse,
        "physics_penalty":
            physics_penalty
    }


# ======================================================================================
# 10. TRAIN ONE VARIANT
# ======================================================================================

VARIANTS = [
    "A_baseline",
    "B_R_focused",
    "C_R_focused_soft_physics"
]

results = []

for variant in VARIANTS:

    print("\n")
    print("=" * 110)
    print("TRAINING VARIANT:", variant)
    print("=" * 110)

    # --------------------------------------------------------------
    # Fresh initialization for every variant.
    # --------------------------------------------------------------
    torch.manual_seed(SEED)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    model = FinalRepairedLNO(
        dim=6,
        width=64,
        modes=16,
        depth=4,
        lindblad_channels=4
    ).to(device)

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

    MAX_EPOCHS = 30
    PATIENCE = 7

    best_val = float("inf")
    best_epoch = None
    epochs_without = 0
    history = []

    best_state = None

    # --------------------------------------------------------------
    # Epoch loop.
    # --------------------------------------------------------------
    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        start_time = time.time()

        # ==========================================================
        # TRAIN
        # ==========================================================

        model.train()

        train_sum = 0.0
        train_R_mse = 0.0
        train_R_rel = 0.0
        train_A_mse = 0.0

        count = 0

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

            gamma = env_x[..., 0]
            sigma = env_x[..., 1]

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                R_x,
                A_x,
                gamma,
                sigma
            )

            losses = compute_losses(
                output,
                R_y,
                A_y,
                variant
            )

            loss = losses["loss"]

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            n = R_x.shape[0]

            train_sum += (
                losses["loss"].item() * n
            )

            train_R_mse += (
                losses["R_mse"].item() * n
            )

            train_R_rel += (
                losses["R_rel"].item() * n
            )

            train_A_mse += (
                losses["A_mse"].item() * n
            )

            count += n

        train_loss = train_sum / count
        train_R = train_R_mse / count
        train_rel = train_R_rel / count
        train_A = train_A_mse / count

        # ==========================================================
        # VALIDATION
        # ==========================================================

        model.eval()

        val_sum = 0.0
        val_R_mse = 0.0
        val_R_rel = 0.0
        val_A_mse = 0.0

        count = 0

        with torch.no_grad():

            for (
                R_x,
                A_x,
                env_x,
                R_y,
                A_y
            ) in val_loader:

                R_x = R_x.to(device)
                A_x = A_x.to(device)
                env_x = env_x.to(device)
                R_y = R_y.to(device)
                A_y = A_y.to(device)

                gamma = env_x[..., 0]
                sigma = env_x[..., 1]

                output = model(
                    R_x,
                    A_x,
                    gamma,
                    sigma
                )

                losses = compute_losses(
                    output,
                    R_y,
                    A_y,
                    variant
                )

                n = R_x.shape[0]

                val_sum += (
                    losses["loss"].item() * n
                )

                val_R_mse += (
                    losses["R_mse"].item() * n
                )

                val_R_rel += (
                    losses["R_rel"].item() * n
                )

                val_A_mse += (
                    losses["A_mse"].item() * n
                )

                count += n

        val_loss = val_sum / count
        val_R = val_R_mse / count
        val_rel = val_R_rel / count
        val_A = val_A_mse / count

        scheduler.step(
            val_loss
        )

        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = (
            time.time() - start_time
        )

        kappa_L = float(
            F.softplus(
                model.raw_kappa_lindblad
            ).item()
        )

        kappa_N = float(
            F.softplus(
                model.raw_kappa_neural
            ).item()
        )

        improved = (
            val_loss <
            best_val - 1e-6
        )

        if improved:

            best_val = val_loss
            best_epoch = epoch
            epochs_without = 0

            best_state = copy.deepcopy(
                model.state_dict()
            )

        else:

            epochs_without += 1

        history.append({

            "epoch": epoch,

            "train_loss":
                float(train_loss),

            "train_R_MSE":
                float(train_R),

            "train_R_relative_L2":
                float(train_rel),

            "train_Amplitude_MSE":
                float(train_A),

            "val_loss":
                float(val_loss),

            "val_R_MSE":
                float(val_R),

            "val_R_relative_L2":
                float(val_rel),

            "val_Amplitude_MSE":
                float(val_A),

            "learning_rate":
                float(current_lr),

            "kappa_lindblad":
                kappa_L,

            "kappa_neural":
                kappa_N,

            "seconds":
                float(elapsed)

        })

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Train={train_loss:.6e} | "
            f"Val={val_loss:.6e} | "
            f"Val R={val_rel:.6e} | "
            f"Val Amp={val_A:.6e} | "
            f"kL={kappa_L:.5f} | "
            f"kN={kappa_N:.5f}"
        )

        if improved:
            print("  >>> NEW BEST")

        if (
            epochs_without >= PATIENCE
        ):

            print(
                "[INFO] Early stopping."
            )

            break

    # ==============================================================
    # RESTORE BEST MODEL
    # ==============================================================

    assert best_state is not None

    model.load_state_dict(
        best_state
    )

    model.eval()

    checkpoint_path = (
        OUT_DIR /
        f"{variant}_best.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "variant":
                variant,
            "best_epoch":
                int(best_epoch),
            "best_validation_loss":
                float(best_val),
            "amplitude_mean":
                amplitude_mean,
            "amplitude_std":
                amplitude_std
        },
        checkpoint_path
    )

    history_path = (
        OUT_DIR /
        f"{variant}_history.csv"
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False
    )

    # ==============================================================
    # TEST
    # ==============================================================

    R_preds = []
    R_targets = []

    A_preds = []
    A_targets = []

    with torch.no_grad():

        for (
            R_x,
            A_x,
            env_x,
            R_y,
            A_y
        ) in test_loader:

            R_x = R_x.to(device)
            A_x = A_x.to(device)
            env_x = env_x.to(device)

            gamma = env_x[..., 0]
            sigma = env_x[..., 1]

            output = model(
                R_x,
                A_x,
                gamma,
                sigma
            )

            R_preds.append(
                output["R_next"].cpu()
            )

            R_targets.append(
                R_y
            )

            A_preds.append(
                output["amplitude_next"].cpu()
            )

            A_targets.append(
                A_y
            )

    R_pred = torch.cat(
        R_preds,
        dim=0
    )

    R_true = torch.cat(
        R_targets,
        dim=0
    )

    A_pred = torch.cat(
        A_preds,
        dim=0
    )

    A_true = torch.cat(
        A_targets,
        dim=0
    )

    test_R_mse = float(
        F.mse_loss(
            R_pred,
            R_true
        ).item()
    )

    test_R_rmse = math.sqrt(
        test_R_mse
    )

    test_R_rel = float(
        relative_l2_loss(
            R_pred,
            R_true
        ).item()
    )

    test_A_mse = float(
        F.mse_loss(
            A_pred,
            A_true
        ).item()
    )

    test_A_rmse = math.sqrt(
        test_A_mse
    )

    test_A_rel = float(
        relative_l2_loss(
            A_pred,
            A_true
        ).item()
    )

    # ==============================================================
    # STRUCTURAL METRICS
    # ==============================================================

    trace = torch.diagonal(
        R_pred,
        dim1=-2,
        dim2=-1
    ).sum(
        dim=-1
    )

    trace_error = torch.abs(
        trace - 1.0
    )

    eigvals = torch.linalg.eigvalsh(
        R_pred
    )

    mean_trace_error = float(
        trace_error.mean().item()
    )

    max_trace_error = float(
        trace_error.max().item()
    )

    minimum_eigenvalue = float(
        eigvals.min().item()
    )

    negative_fraction = float(
        (
            eigvals < -1e-6
        ).float().mean().item()
    )

    final_kappa_L = float(
        F.softplus(
            model.raw_kappa_lindblad
        ).item()
    )

    final_kappa_N = float(
        F.softplus(
            model.raw_kappa_neural
        ).item()
    )

    row = {

        "variant":
            variant,

        "best_epoch":
            int(best_epoch),

        "best_val_loss":
            float(best_val),

        "R_MSE":
            test_R_mse,

        "R_RMSE":
            test_R_rmse,

        "R_Relative_L2":
            test_R_rel,

        "Amplitude_MSE":
            test_A_mse,

        "Amplitude_RMSE":
            test_A_rmse,

        "Amplitude_Relative_L2":
            test_A_rel,

        "Mean_trace_error":
            mean_trace_error,

        "Max_trace_error":
            max_trace_error,

        "Minimum_eigenvalue":
            minimum_eigenvalue,

        "Negative_eigenvalue_fraction":
            negative_fraction,

        "kappa_lindblad":
            final_kappa_L,

        "kappa_neural":
            final_kappa_N

    }

    results.append(
        row
    )

    # ==============================================================
    # SAVE TEST PREDICTIONS
    # ==============================================================

    np.savez_compressed(
        OUT_DIR /
        f"{variant}_test_predictions.npz",

        R_prediction=
            R_pred.numpy(),

        R_target=
            R_true.numpy(),

        amplitude_prediction=
            A_pred.numpy(),

        amplitude_target=
            A_true.numpy(),

        test_indices=
            test_idx
    )

    print("\nTEST RESULT")
    print("-" * 110)
    print(
        "R RMSE:",
        f"{test_R_rmse:.8f}"
    )
    print(
        "R Relative-L2:",
        f"{test_R_rel:.8f}"
    )
    print(
        "Amplitude RMSE:",
        f"{test_A_rmse:.8f}"
    )
    print(
        "Amplitude Relative-L2:",
        f"{test_A_rel:.8f}"
    )
    print(
        "Mean trace error:",
        f"{mean_trace_error:.8e}"
    )
    print(
        "Minimum eigenvalue:",
        f"{minimum_eigenvalue:.8e}"
    )


# ======================================================================================
# 11. COMPARISON TABLE
# ======================================================================================

comparison_df = pd.DataFrame(
    results
)

comparison_path = (
    OUT_DIR /
    "step26_variant_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)

print("\n")
print("=" * 110)
print("STEP 26 — VARIANT COMPARISON")
print("=" * 110)

print(
    comparison_df[
        [
            "variant",
            "R_RMSE",
            "R_Relative_L2",
            "Amplitude_RMSE",
            "Amplitude_Relative_L2",
            "Mean_trace_error",
            "Minimum_eigenvalue",
            "Negative_eigenvalue_fraction",
            "kappa_lindblad",
            "kappa_neural"
        ]
    ].to_string(
        index=False
    )
)


# ======================================================================================
# 12. BASELINE DELTA
# ======================================================================================

baseline = comparison_df[
    comparison_df["variant"] ==
    "A_baseline"
].iloc[0]

comparison_df["R_Relative_L2_change_vs_A_percent"] = (
    (
        comparison_df["R_Relative_L2"]
        -
        baseline["R_Relative_L2"]
    )
    /
    baseline["R_Relative_L2"]
) * 100.0

comparison_df[
    [
        "variant",
        "R_Relative_L2",
        "R_Relative_L2_change_vs_A_percent"
    ]
].to_csv(
    OUT_DIR /
    "step26_R_accuracy_delta.csv",
    index=False
)


# ======================================================================================
# 13. FINAL JSON
# ======================================================================================

summary = {

    "step": 26,

    "title":
        "R Accuracy Improvement Ablation",

    "scientific_question":
        "Whether LNO one-step R accuracy can be improved by aligning the "
        "training objective more closely with the R evaluation metric and "
        "softly restraining Lindblad coupling.",

    "dataset":
        "Dataset V3 Expanded",

    "split":
        {
            "train": 6930,
            "validation": 1485,
            "test": 1485
        },

    "variants":
        [
            "A_baseline",
            "B_R_focused",
            "C_R_focused_soft_physics"
        ],

    "no_existing_checkpoint_modified":
        True,

    "fresh_initialization":
        True,

    "results":
        comparison_df.to_dict(
            orient="records"
        ),

    "outputs":
        {
            "comparison_csv":
                str(comparison_path),

            "R_delta_csv":
                str(
                    OUT_DIR /
                    "step26_R_accuracy_delta.csv"
                )
        }

}

with open(
    OUT_DIR /
    "step26_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ======================================================================================
# 14. PASS / FAIL
# ======================================================================================

best_variant = comparison_df.loc[
    comparison_df["R_Relative_L2"].idxmin()
]

print("\n")
print("=" * 110)
print("STEP 26 COMPLETE")
print("=" * 110)

print(
    "Best R Relative-L2 variant:",
    best_variant["variant"]
)

print(
    "Best R Relative-L2:",
    f"{best_variant['R_Relative_L2']:.8f}"
)

print(
    "\nResults:",
    OUT_DIR
)

print(
    "\n[INFO] This is a controlled LNO training ablation."
)

print(
    "[INFO] Existing final FNO/LNO checkpoints were not modified."
)

print(
    "[INFO] The purpose is to identify the cause of the small R-accuracy gap."
)

print("=" * 110)
