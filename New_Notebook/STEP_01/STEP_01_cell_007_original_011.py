# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 11
# Step            : STEP_01
# Step Heading    : # PHASE 1 — STEP 1: REPOSITORY + TENSOR GEOMETRY AUDIT
# Step Cell No.   : 7
# ============================================================

# ============================================================
# PHASE 2 — FNO TEST-SET EVALUATION
# Same test split + same metrics used for LNO
# READS checkpoint; DOES NOT modify model/checkpoint
# ============================================================

import os
import json
import csv
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

# ------------------------------------------------------------
# 1. Device
# ------------------------------------------------------------
device = torch.device("cpu")

print("=" * 78)
print("[FNO TEST-SET EVALUATION]")
print("=" * 78)
print("[+] Device:", device)

# ------------------------------------------------------------
# 2. Reproducibility
# ------------------------------------------------------------
SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------
# 3. Dataset paths
# ------------------------------------------------------------
assert "DATASET_ROOT" in globals(), (
    "DATASET_ROOT not found. Run the dataset-path cell first."
)

TEST_DIR = os.path.join(DATASET_ROOT, "test")
FNO_CKPT = "results/check_points/best_fno.pt"

assert os.path.isdir(TEST_DIR), (
    f"Missing test directory:\n{TEST_DIR}"
)

assert os.path.isfile(FNO_CKPT), (
    f"Missing FNO checkpoint:\n{FNO_CKPT}"
)

print("[+] Test dir :", TEST_DIR)
print("[+] FNO ckpt :", FNO_CKPT)

# ------------------------------------------------------------
# 4. Load test dataset
# ------------------------------------------------------------
from data.dataset_loader import IonTransportDataset

test_dataset = IonTransportDataset(
    root_dir=TEST_DIR
)

print(f"[+] Test samples: {len(test_dataset)}")

# ------------------------------------------------------------
# 5. Test geometry
# ------------------------------------------------------------
x0, y0 = test_dataset[0]

print("[+] Input geometry :", tuple(x0.shape))
print("[+] Target geometry:", tuple(y0.shape))

assert tuple(x0.shape) == (6, 128), (
    f"Unexpected test input shape: {tuple(x0.shape)}"
)

assert tuple(y0.shape) == (128,), (
    f"Unexpected test target shape: {tuple(y0.shape)}"
)

print("[+] Test geometry check: PASSED")

# ------------------------------------------------------------
# 6. DataLoader
# ------------------------------------------------------------
# Keep deterministic ordering for reproducible evaluation.
BATCH_SIZE = 32

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

print("[+] Test batches:", len(test_loader))

# ------------------------------------------------------------
# 7. Reconstruct EXACT FNO architecture
# ------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

from models.base_operator import BaseOperator
from models.spectral_kernel import SpectralConv1D


class FNOBaseline(BaseOperator):

    def __init__(self, modes=16, width=64, in_channels=6):
        super().__init__()

        self.modes = modes
        self.width = width

        self.input_projection = nn.Linear(
            in_channels,
            width
        )

        self.conv0 = SpectralConv1D(width, width, modes)
        self.conv1 = SpectralConv1D(width, width, modes)
        self.conv2 = SpectralConv1D(width, width, modes)
        self.conv3 = SpectralConv1D(width, width, modes)

        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(
        self,
        state,
        phi,
        flux,
        noise,
        dissipation,
        gamma
    ):

        if gamma.dim() == 1:
            gamma = gamma.unsqueeze(-1)

        x = torch.stack(
            [
                state,
                phi,
                flux,
                noise,
                dissipation,
                gamma.repeat(1, state.shape[-1]),
            ],
            dim=-1,
        )

        # [B, 128, 6]
        x = self.input_projection(x)

        # [B, 64, 128]
        x = x.permute(0, 2, 1)

        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)

        # [B, 128, 64]
        x = x.permute(0, 2, 1)

        x = F.gelu(self.fc1(x))
        out = self.fc2(x)

        return out.squeeze(-1)


model = FNOBaseline(
    modes=16,
    width=64,
    in_channels=6,
).to(device)

# ------------------------------------------------------------
# 8. Load BEST FNO checkpoint strictly
# ------------------------------------------------------------
ckpt = torch.load(
    FNO_CKPT,
    map_location=device,
)

assert "model_state_dict" in ckpt, (
    "Checkpoint does not contain model_state_dict."
)

model.load_state_dict(
    ckpt["model_state_dict"],
    strict=True,
)

model.eval()

saved_epoch = ckpt.get("epoch", None)
saved_val_mse = ckpt.get("best_val_mse", None)

print("\n[+] Loaded FNO checkpoint")
print("    Best epoch   :", saved_epoch)
print(
    "    Best Val MSE :",
    f"{float(saved_val_mse):.12e}"
    if saved_val_mse is not None
    else "N/A"
)

# ------------------------------------------------------------
# 9. Packed-input forward helper
# ------------------------------------------------------------
def fno_predict(model, inputs):

    state       = inputs[:, 0, :]
    phi         = inputs[:, 1, :]
    flux        = inputs[:, 2, :]
    noise       = inputs[:, 3, :]
    dissipation = inputs[:, 4, :]
    gamma       = inputs[:, 5, 0]

    return model(
        state=state,
        phi=phi,
        flux=flux,
        noise=noise,
        dissipation=dissipation,
        gamma=gamma,
    )

# ------------------------------------------------------------
# 10. Import EXACT project metrics
# ------------------------------------------------------------
from evaluation.metrics import (
    rmse,
    relative_l2_error,
    mae,
    mass_conservation_error,
    entropy_error,
    instability_index,
    dissipation_rate,
    spectral_energy,
)

# ------------------------------------------------------------
# 11. Full deterministic test inference
# ------------------------------------------------------------
all_predictions = []
all_targets = []
all_inputs = []

print("\n" + "-" * 78)
print("[+] Running FNO inference on COMPLETE test set")
print("-" * 78)

with torch.no_grad():

    for batch_idx, (inputs, targets) in enumerate(test_loader, start=1):

        inputs = inputs.to(device)
        targets = targets.to(device)

        predictions = fno_predict(
            model,
            inputs
        )

        # Numerical safety
        if not torch.all(torch.isfinite(predictions)):
            raise RuntimeError(
                f"Non-finite FNO prediction detected "
                f"at test batch {batch_idx}."
            )

        all_predictions.append(
            predictions.cpu()
        )

        all_targets.append(
            targets.cpu()
        )

        all_inputs.append(
            inputs.cpu()
        )

        if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == len(test_loader):
            print(
                f"    Batch {batch_idx:03d}/{len(test_loader)}"
            )

# ------------------------------------------------------------
# 12. Concatenate complete test set
# ------------------------------------------------------------
pred_all = torch.cat(
    all_predictions,
    dim=0
)

target_all = torch.cat(
    all_targets,
    dim=0
)

input_all = torch.cat(
    all_inputs,
    dim=0
)

print("\n[+] Prediction tensor :", tuple(pred_all.shape))
print("[+] Target tensor     :", tuple(target_all.shape))
print("[+] Input tensor      :", tuple(input_all.shape))

assert pred_all.shape == target_all.shape
assert pred_all.shape[1] == 128

assert torch.all(
    torch.isfinite(pred_all)
), "Predictions contain non-finite values."

# ------------------------------------------------------------
# 13. Compute EXACT test metrics
# ------------------------------------------------------------
test_metrics = {
    "RMSE": rmse(pred_all, target_all),

    "Relative_L2": relative_l2_error(
        pred_all,
        target_all
    ),

    "MAE": mae(
        pred_all,
        target_all
    ),

    "Mass_Error": mass_conservation_error(
        pred_all,
        target_all
    ),

    "Entropy_Error": entropy_error(
        pred_all,
        target_all
    ),

    "Instability_Index": instability_index(
        pred_all
    ),

    "Dissipation_Rate": dissipation_rate(
        pred_all
    ),

    "Spectral_Energy": spectral_energy(
        pred_all
    ),
}

# ------------------------------------------------------------
# 14. Print results
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("[+] FNO TEST-SET RESULTS")
print("=" * 78)

for name, value in test_metrics.items():

    assert math.isfinite(float(value)), (
        f"Metric '{name}' is non-finite: {value}"
    )

    print(
        f"{name:20s}: {float(value):.10f}"
    )

print("=" * 78)

# ------------------------------------------------------------
# 15. Save predictions / targets
# ------------------------------------------------------------
RESULTS_DIR = "results/phase1_zero_shot/fno_test"
os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

pred_path = os.path.join(
    RESULTS_DIR,
    "fno_test_predictions.npy"
)

target_path = os.path.join(
    RESULTS_DIR,
    "fno_test_targets.npy"
)

input_path = os.path.join(
    RESULTS_DIR,
    "fno_test_inputs.npy"
)

metrics_path = os.path.join(
    RESULTS_DIR,
    "fno_test_metrics.json"
)

np.save(
    pred_path,
    pred_all.numpy().astype(np.float32)
)

np.save(
    target_path,
    target_all.numpy().astype(np.float32)
)

np.save(
    input_path,
    input_all.numpy().astype(np.float32)
)

with open(metrics_path, "w") as f:

    json.dump(
        {
            "model": "FNOBaseline",
            "checkpoint": FNO_CKPT,
            "best_epoch": saved_epoch,
            "best_val_mse": (
                float(saved_val_mse)
                if saved_val_mse is not None
                else None
            ),
            "test_samples": int(len(test_dataset)),
            "input_shape": list(pred_all.shape),
            "target_shape": list(target_all.shape),
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "metrics": {
                k: float(v)
                for k, v in test_metrics.items()
            },
        },
        f,
        indent=2,
    )

print("\n[+] Saved artifacts:")
print("    Predictions :", pred_path)
print("    Targets     :", target_path)
print("    Inputs      :", input_path)
print("    Metrics     :", metrics_path)

# ------------------------------------------------------------
# 16. Final integrity checks
# ------------------------------------------------------------
loaded_pred = np.load(pred_path)
loaded_target = np.load(target_path)

assert loaded_pred.shape == (
    len(test_dataset),
    128
)

assert loaded_target.shape == (
    len(test_dataset),
    128
)

assert np.isfinite(loaded_pred).all()
assert np.isfinite(loaded_target).all()

print("\n[+] Saved-array integrity: PASSED")

# ------------------------------------------------------------
# 17. Final status
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("[+] FNO TEST EVALUATION COMPLETE")
print("=" * 78)
print("[+] Test samples evaluated :", len(test_dataset))
print("[+] Best training epoch    :", saved_epoch)
print("[+] Results directory      :", RESULTS_DIR)
print("[+] Ready for FNO vs LNO   : YES")
print("=" * 78)
