# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 83
# Step            : STEP_27
# Step Heading    : # STEP 27 — INDEPENDENT VALIDATION OF STEP-26 C-LNO
# Step Cell No.   : 1
# ============================================================

# ======================================================================================
# STEP 27 — INDEPENDENT VALIDATION OF STEP-26 C-LNO
# ======================================================================================
#
# Compare:
#   1. Final FNO (Step 17)
#   2. Original Final LNO (Step 18)
#   3. Step-26 C variant (R-focused + soft physics)
#
# C-LNO is evaluated from its SAVED CHECKPOINT.
#
# No training.
# No checkpoint modification.
# Fresh test inference only for C-LNO.
# Same 1,485 test transitions / same targets / same environment.
# ======================================================================================

import os
import math
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader


# ======================================================================================
# 1. ROOT
# ======================================================================================

CONTENT_ROOT = Path("/content")

repo_candidates = []

for p in CONTENT_ROOT.iterdir():
    if (
        p.is_dir()
        and (p / "results").is_dir()
        and (p / "physics").is_dir()
        and (p / "training").is_dir()
    ):
        repo_candidates.append(p)

assert repo_candidates, "Repository root not found."

ROOT = repo_candidates[0]
os.chdir(ROOT)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 110)
print("STEP 27 — INDEPENDENT VALIDATION OF STEP-26 C-LNO")
print("=" * 110)
print("[+] Root  :", ROOT)
print("[+] Device:", device)


# ======================================================================================
# 2. AUTO-DETECT FILES
# ======================================================================================

def find_file(filename):

    direct = CONTENT_ROOT / filename

    if direct.is_file():
        return direct

    matches = list(ROOT.rglob(filename))

    if matches:
        return matches[0]

    matches = [
        p for p in CONTENT_ROOT.rglob(filename)
        if p.is_file()
    ]

    if matches:
        return matches[0]

    return None


DATASET_PATH = find_file(
    "environment_conditioned_lno_dataset_v3_expanded.npz"
)

SPLIT_PATH = find_file(
    "final_trajectory_level_split.npz"
)

FNO_PRED_PATH = find_file(
    "fno_final_test_predictions.npz"
)

LNO_PRED_PATH = find_file(
    "lno_final_test_predictions.npz"
)

C_CHECKPOINT = (
    ROOT
    / "results"
    / "step26_r_accuracy_ablation"
    / "C_R_focused_soft_physics_best.pt"
)

assert DATASET_PATH is not None
assert SPLIT_PATH is not None
assert FNO_PRED_PATH is not None
assert LNO_PRED_PATH is not None
assert C_CHECKPOINT.is_file(), (
    f"C-LNO checkpoint not found: {C_CHECKPOINT}"
)

print("\nFILE LOCATIONS")
print("-" * 110)
print("Dataset :", DATASET_PATH)
print("Split   :", SPLIT_PATH)
print("FNO pred:", FNO_PRED_PATH)
print("LNO pred:", LNO_PRED_PATH)
print("C-LNO ckpt:", C_CHECKPOINT)


# ======================================================================================
# 3. LOAD DATA
# ======================================================================================

data = np.load(
    DATASET_PATH,
    allow_pickle=False
)

split = np.load(
    SPLIT_PATH,
    allow_pickle=False
)

fno = np.load(
    FNO_PRED_PATH,
    allow_pickle=False
)

lno = np.load(
    LNO_PRED_PATH,
    allow_pickle=False
)

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

assert len(train_idx) == 6930
assert len(val_idx) == 1485
assert len(test_idx) == 1485


# ======================================================================================
# 4. TEST IDENTITY AGAIN
# ======================================================================================

fno_test_idx = fno["test_indices"].astype(np.int64)
lno_test_idx = lno["test_indices"].astype(np.int64)

assert np.array_equal(
    fno_test_idx,
    test_idx
)

assert np.array_equal(
    lno_test_idx,
    test_idx
)

print("\n[PASS] FNO/LNO/test split identity confirmed.")


# ======================================================================================
# 5. SAME AMPLITUDE NORMALIZATION
# ======================================================================================

checkpoint_meta = torch.load(
    C_CHECKPOINT,
    map_location="cpu"
)

amplitude_mean = float(
    checkpoint_meta.get(
        "amplitude_mean",
        X_A_all[train_idx].mean()
    )
)

amplitude_std = float(
    checkpoint_meta.get(
        "amplitude_std",
        X_A_all[train_idx].std()
    )
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
# 6. C-LNO ARCHITECTURE — SAME AS STEP 26
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


class LindbladKernel(nn.Module):

    def __init__(
        self,
        dim=6,
        channels=4
    ):
        super().__init__()

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

        for k in range(4):

            L = self.jump_operators[k]

            LT = L.transpose(
                -1,
                -2
            )

            A = LT @ L

            jump_term = (
                L @ R @ LT
            )

            anti_term = (
                A @ R +
                R @ A
            )

            total = (
                total
                +
                jump_term
                -
                0.5 * anti_term
            )

        return total


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

        self.input_projection = nn.Conv1d(
            dim * dim + 3,
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

        M = R.double()

        M = 0.5 * (
            M +
            M.transpose(-1, -2)
        )

        eigvals, eigvecs = torch.linalg.eigh(
            M
        )

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

        x = self.input_projection(x)

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
            delta_R_lindblad
            +
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
# 7. LOAD C-LNO CHECKPOINT
# ======================================================================================

model = FinalRepairedLNO(
    dim=6,
    width=64,
    modes=16,
    depth=4,
    lindblad_channels=4
).to(device)

model.load_state_dict(
    checkpoint_meta["model_state_dict"]
)

model.eval()

print("\n[PASS] Step-26 C-LNO checkpoint loaded.")


# ======================================================================================
# 8. TEST DATALOADER
# ======================================================================================

test_ds = TensorDataset(
    torch.from_numpy(
        X_R_all[test_idx]
    ),
    torch.from_numpy(
        X_A_norm[test_idx]
    ),
    torch.from_numpy(
        ENV_all[test_idx]
    ),
    torch.from_numpy(
        Y_R_all[test_idx]
    ),
    torch.from_numpy(
        Y_A_norm[test_idx]
    )
)

test_loader = DataLoader(
    test_ds,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# ======================================================================================
# 9. C-LNO TEST INFERENCE
# ======================================================================================

C_R_preds = []
C_R_targets = []
C_A_preds = []
C_A_targets = []

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

        C_R_preds.append(
            output["R_next"].cpu()
        )

        C_R_targets.append(
            R_y
        )

        C_A_preds.append(
            output["amplitude_next"].cpu()
        )

        C_A_targets.append(
            A_y
        )

C_R_pred = torch.cat(
    C_R_preds,
    dim=0
)

C_R_true = torch.cat(
    C_R_targets,
    dim=0
)

C_A_pred_norm = torch.cat(
    C_A_preds,
    dim=0
)

C_A_true_norm = torch.cat(
    C_A_targets,
    dim=0
)

# Convert normalized C-LNO amplitude back to physical scale
C_A_pred = (
    C_A_pred_norm * amplitude_std
    + amplitude_mean
)

C_A_true = (
    C_A_true_norm * amplitude_std
    + amplitude_mean
)

print(
    "\n[PASS] C-LNO test inference complete:",
    C_R_pred.shape
)


# ======================================================================================
# 10. METRICS
# ======================================================================================

def metrics(pred, target):

    mse = F.mse_loss(
        pred,
        target
    ).item()

    rmse = math.sqrt(mse)

    mae = torch.mean(
        torch.abs(
            pred - target
        )
    ).item()

    rel = (
        torch.linalg.vector_norm(
            pred - target
        )
        /
        torch.linalg.vector_norm(
            target
        ).clamp_min(1e-8)
    ).item()

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Relative_L2": rel
    }


c_r = metrics(
    C_R_pred,
    C_R_true
)

c_a = metrics(
    C_A_pred,
    C_A_true
)


# ======================================================================================
# 11. STRUCTURAL METRICS
# ======================================================================================

def structural_metrics(R):

    trace = torch.diagonal(
        R,
        dim1=-2,
        dim2=-1
    ).sum(
        dim=-1
    )

    trace_error = torch.abs(
        trace - 1.0
    )

    eig = torch.linalg.eigvalsh(
        R
    )

    return {
        "Mean_trace_error":
            float(trace_error.mean().item()),

        "Max_trace_error":
            float(trace_error.max().item()),

        "Minimum_eigenvalue":
            float(eig.min().item()),

        "Negative_eigenvalue_fraction":
            float(
                (eig < -1e-6)
                .float()
                .mean()
                .item()
            )
    }


c_struct = structural_metrics(
    C_R_pred
)


# ======================================================================================
# 12. LOAD EXISTING FNO + ORIGINAL LNO METRICS DIRECTLY FROM PREDICTIONS
# ======================================================================================

FNO_R = torch.from_numpy(
    fno["R_prediction"]
)

FNO_R_target = torch.from_numpy(
    fno["R_target"]
)

FNO_A = torch.from_numpy(
    fno["amplitude_prediction"]
)

FNO_A_target = torch.from_numpy(
    fno["amplitude_target"]
)

LNO_R = torch.from_numpy(
    lno["R_prediction"]
)

LNO_R_target = torch.from_numpy(
    lno["R_target"]
)

LNO_A = torch.from_numpy(
    lno["amplitude_prediction"]
)

LNO_A_target = torch.from_numpy(
    lno["amplitude_target"]
)
# R targets should match exactly
assert torch.equal(
    FNO_R_target,
    C_R_true
)

assert torch.equal(
    LNO_R_target,
    C_R_true
)

# Amplitude targets should match numerically after
# normalization -> physical-scale conversion
assert torch.allclose(
    FNO_A_target,
    C_A_true,
    atol=1e-5,
    rtol=1e-5
)

assert torch.allclose(
    LNO_A_target,
    C_A_true,
    atol=1e-5,
    rtol=1e-5
)

print("[PASS] FNO/LNO/C-LNO targets are on the same physical scale.")

fno_r = metrics(
    FNO_R,
    FNO_R_target
)

fno_a = metrics(
    FNO_A,
    FNO_A_target
)

fno_struct = structural_metrics(
    FNO_R
)

lno_r = metrics(
    LNO_R,
    LNO_R_target
)

lno_a = metrics(
    LNO_A,
    LNO_A_target
)

lno_struct = structural_metrics(
    LNO_R
)


# ======================================================================================
# 13. FINAL THREE-MODEL TABLE
# ======================================================================================

rows = [

    {
        "model": "FNO",
        "R_RMSE": fno_r["RMSE"],
        "R_Relative_L2": fno_r["Relative_L2"],
        "Amplitude_RMSE": fno_a["RMSE"],
        "Amplitude_Relative_L2": fno_a["Relative_L2"],
        **fno_struct
    },

    {
        "model": "Final_LNO",
        "R_RMSE": lno_r["RMSE"],
        "R_Relative_L2": lno_r["Relative_L2"],
        "Amplitude_RMSE": lno_a["RMSE"],
        "Amplitude_Relative_L2": lno_a["Relative_L2"],
        **lno_struct
    },

    {
        "model": "Step26_C_LNO",
        "R_RMSE": c_r["RMSE"],
        "R_Relative_L2": c_r["Relative_L2"],
        "Amplitude_RMSE": c_a["RMSE"],
        "Amplitude_Relative_L2": c_a["Relative_L2"],
        **c_struct
    }
]

summary_df = pd.DataFrame(
    rows
)


# ======================================================================================
# 14. REGIME-WISE COMPARISON
# ======================================================================================

regimes = {
    "low_noise": (0.01, 0.02),
    "stochastic": (0.05, 0.45),
    "heavy_dissipation": (0.35, 0.15),
    "collapse": (0.50, 0.65),
    "metastable": (0.15, 0.30)
}

ENV_test = ENV_all[test_idx]

regime_rows = []

for regime_name, (gamma_val, sigma_val) in regimes.items():

    mask = np.all(
        np.isclose(
            ENV_test,
            np.array(
                [gamma_val, sigma_val],
                dtype=np.float32
            )[None, None, :],
            atol=1e-6
        ),
        axis=(1, 2)
    )

    # The above mask is trajectory-transition level.
    # Confirm expected 297 transitions.
    count = int(mask.sum())

    assert count == 297, (
        f"{regime_name}: expected 297, got {count}"
    )

    idx = torch.from_numpy(
        np.where(mask)[0]
    )

    fno_rel = metrics(
        FNO_R[idx],
        FNO_R_target[idx]
    )["Relative_L2"]

    lno_rel = metrics(
        LNO_R[idx],
        LNO_R_target[idx]
    )["Relative_L2"]

    c_rel = metrics(
        C_R_pred[idx],
        C_R_true[idx]
    )["Relative_L2"]

    fno_amp = metrics(
        FNO_A[idx],
        FNO_A_target[idx]
    )["Relative_L2"]

    lno_amp = metrics(
        LNO_A[idx],
        LNO_A_target[idx]
    )["Relative_L2"]

    c_amp = metrics(
        C_A_pred[idx],
        C_A_true[idx]
    )["Relative_L2"]

    regime_rows.append({

        "regime":
            regime_name,

        "transitions":
            count,

        "R_Relative_L2_FNO":
            fno_rel,

        "R_Relative_L2_Final_LNO":
            lno_rel,

        "R_Relative_L2_C_LNO":
            c_rel,

        "Amplitude_Relative_L2_FNO":
            fno_amp,

        "Amplitude_Relative_L2_Final_LNO":
            lno_amp,

        "Amplitude_Relative_L2_C_LNO":
            c_amp,

        "C_LNO_R_better_than_FNO":
            bool(c_rel < fno_rel),

        "C_LNO_R_better_than_Final_LNO":
            bool(c_rel < lno_rel)
    })


regime_df = pd.DataFrame(
    regime_rows
)


# ======================================================================================
# 15. OUTPUTS
# ======================================================================================

OUT_DIR = (
    ROOT
    / "results"
    / "step27_c_lno_independent_validation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

summary_df.to_csv(
    OUT_DIR / "three_model_comparison.csv",
    index=False
)

regime_df.to_csv(
    OUT_DIR / "regime_comparison.csv",
    index=False
)

np.savez_compressed(
    OUT_DIR / "c_lno_test_predictions.npz",

    R_prediction=C_R_pred.numpy(),

    R_target=C_R_true.numpy(),

    amplitude_prediction=C_A_pred.numpy(),

    amplitude_target=C_A_true.numpy(),

    test_indices=test_idx
)


# ======================================================================================
# 16. SUMMARY JSON
# ======================================================================================

summary_json = {

    "step": 27,

    "title":
        "Independent Validation of Step-26 C-LNO",

    "no_training":
        True,

    "fresh_C_LNO_inference":
        True,

    "test_transitions":
        1485,

    "models":
        [
            "FNO",
            "Final_LNO",
            "Step26_C_LNO"
        ],

    "overall":
        summary_df.to_dict(
            orient="records"
        ),

    "regimes":
        regime_df.to_dict(
            orient="records"
        )
}

with open(
    OUT_DIR / "step27_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary_json,
        f,
        indent=2
    )


# ======================================================================================
# 17. PRINT FINAL RESULTS
# ======================================================================================

print("\n")
print("=" * 110)
print("STEP 27 — THREE-MODEL FINAL VALIDATION")
print("=" * 110)

print(
    summary_df[
        [
            "model",
            "R_RMSE",
            "R_Relative_L2",
            "Amplitude_RMSE",
            "Amplitude_Relative_L2",
            "Mean_trace_error",
            "Minimum_eigenvalue",
            "Negative_eigenvalue_fraction"
        ]
    ].to_string(
        index=False
    )
)

print("\nREGIME-WISE R ACCURACY")
print("-" * 110)

print(
    regime_df[
        [
            "regime",
            "R_Relative_L2_FNO",
            "R_Relative_L2_Final_LNO",
            "R_Relative_L2_C_LNO",
            "C_LNO_R_better_than_FNO",
            "C_LNO_R_better_than_Final_LNO"
        ]
    ].to_string(
        index=False
    )
)

c_vs_fno = (
    (
        summary_df.loc[
            summary_df["model"] == "Step26_C_LNO",
            "R_Relative_L2"
        ].iloc[0]
        -
        summary_df.loc[
            summary_df["model"] == "FNO",
            "R_Relative_L2"
        ].iloc[0]
    )
    /
    summary_df.loc[
        summary_df["model"] == "FNO",
        "R_Relative_L2"
    ].iloc[0]
) * 100.0

print("\n")
print("C-LNO vs FNO R Relative-L2 change:")
print(f"{c_vs_fno:.6f}%")

print("\nResults:")
print(OUT_DIR)

print("\n[PASS] STEP 27 COMPLETE.")
print("[INFO] No model retraining performed.")
print("[INFO] C-LNO was evaluated from the saved Step-26 checkpoint.")
print("[INFO] FNO and original Final LNO use their existing final test predictions.")
print("[INFO] All three models use the identical 1,485-transition test target set.")

print("=" * 110)
