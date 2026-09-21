# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 10
# Step            : STEP_01
# Step Heading    : # PHASE 1 — STEP 1: REPOSITORY + TENSOR GEOMETRY AUDIT
# Step Cell No.   : 6
# ============================================================

# ============================================================
# PHASE 1B — FNO CHECKPOINT VERIFICATION
# READ-ONLY / NO TRAINING / NO FILE MODIFICATION
# ============================================================

import os
import math
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------
FNO_CKPT = "results/check_points/best_fno.pt"
LNO_CKPT = "results/check_points/best_lno.pt"

EXPECTED = {
    "name": "FNOBaseline",
    "modes": 16,
    "width": 64,
    "in_channels": 6,
    "grid_size": 128,
    "depth": 4,
    "expected_parameters": 287_681,
}

print("=" * 78)
print("[FNO CHECKPOINT VERIFICATION]")
print("=" * 78)

# ------------------------------------------------------------
# 2. Basic filesystem checks
# ------------------------------------------------------------
assert os.path.isfile(FNO_CKPT), (
    f"FNO checkpoint not found:\n{os.path.abspath(FNO_CKPT)}"
)

fno_size_mb = os.path.getsize(FNO_CKPT) / (1024 ** 2)

print(f"[+] FNO checkpoint : {os.path.abspath(FNO_CKPT)}")
print(f"[+] File size      : {fno_size_mb:.3f} MB")
print(f"[+] LNO checkpoint : {os.path.abspath(LNO_CKPT)}")
print(f"[+] LNO exists     : {os.path.isfile(LNO_CKPT)}")

# ------------------------------------------------------------
# 3. File integrity hash
#    Read-only SHA256 fingerprint
# ------------------------------------------------------------
sha256 = hashlib.sha256()

with open(FNO_CKPT, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        sha256.update(chunk)

checkpoint_sha256 = sha256.hexdigest()

print(f"[+] SHA256         : {checkpoint_sha256}")

# ------------------------------------------------------------
# 4. Load checkpoint
# ------------------------------------------------------------
try:
    ckpt = torch.load(
        FNO_CKPT,
        map_location="cpu",
    )
except Exception as e:
    raise RuntimeError(
        f"Failed to load FNO checkpoint: {type(e).__name__}: {e}"
    )

assert isinstance(ckpt, dict), (
    f"Unexpected checkpoint type: {type(ckpt)}"
)

print("[+] Checkpoint load : PASSED")

# ------------------------------------------------------------
# 5. Required top-level fields
# ------------------------------------------------------------
required_fields = [
    "model_state_dict",
    "epoch",
    "best_val_mse",
    "architecture",
    "training",
]

missing = [k for k in required_fields if k not in ckpt]

assert not missing, (
    f"Checkpoint missing required fields: {missing}"
)

print("[+] Checkpoint schema: PASSED")

# ------------------------------------------------------------
# 6. Validate checkpoint metadata
# ------------------------------------------------------------
architecture = ckpt["architecture"]
training = ckpt["training"]

print("\n[+] Saved architecture metadata:")
for k, v in architecture.items():
    print(f"    {k:16s}: {v}")

print("\n[+] Saved training metadata:")
for k, v in training.items():
    print(f"    {k:16s}: {v}")

# Exact architecture verification
for key in [
    "name",
    "modes",
    "width",
    "in_channels",
    "grid_size",
    "depth",
]:
    assert architecture.get(key) == EXPECTED[key], (
        f"Architecture mismatch for '{key}': "
        f"saved={architecture.get(key)}, expected={EXPECTED[key]}"
    )

print("\n[+] Architecture metadata: PASSED")

# ------------------------------------------------------------
# 7. Validate training metadata
# ------------------------------------------------------------
epoch = ckpt["epoch"]
best_val_mse = ckpt["best_val_mse"]

assert isinstance(epoch, int), (
    f"Invalid epoch type: {type(epoch)}"
)

assert 1 <= epoch <= 150, (
    f"Invalid best epoch: {epoch}"
)

assert isinstance(best_val_mse, (float, int)), (
    f"Invalid best_val_mse type: {type(best_val_mse)}"
)

assert math.isfinite(float(best_val_mse)), (
    f"Non-finite best_val_mse: {best_val_mse}"
)

assert float(best_val_mse) >= 0.0, (
    f"Negative best_val_mse: {best_val_mse}"
)

assert training.get("loss") == "MSE", (
    f"Unexpected loss: {training.get('loss')}"
)

assert int(training.get("batch_size")) == 32, (
    f"Unexpected batch size: {training.get('batch_size')}"
)

assert float(training.get("learning_rate")) == 1e-3, (
    f"Unexpected learning rate: {training.get('learning_rate')}"
)

assert float(training.get("weight_decay")) == 1e-4, (
    f"Unexpected weight decay: {training.get('weight_decay')}"
)

assert str(training.get("optimizer")).lower() == "adamw", (
    f"Unexpected optimizer: {training.get('optimizer')}"
)

assert int(training.get("epochs")) == 150, (
    f"Unexpected epoch budget: {training.get('epochs')}"
)

print("[+] Training metadata: PASSED")

# ------------------------------------------------------------
# 8. Reconstruct EXACT FNO architecture
# ------------------------------------------------------------
# Import the same operator components used by the project.
from models.base_operator import BaseOperator
from models.spectral_kernel import SpectralConv1D


class FNOBaselineVerified(BaseOperator):

    def __init__(self, modes=16, width=64, in_channels=6):
        super().__init__()

        self.modes = modes
        self.width = width

        self.input_projection = nn.Linear(
            in_channels,
            width
        )

        self.conv0 = SpectralConv1D(
            width, width, modes
        )
        self.conv1 = SpectralConv1D(
            width, width, modes
        )
        self.conv2 = SpectralConv1D(
            width, width, modes
        )
        self.conv3 = SpectralConv1D(
            width, width, modes
        )

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
        gamma,
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

        x = self.input_projection(x)
        x = x.permute(0, 2, 1)

        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = self.conv3(x) + self.w3(x)

        x = x.permute(0, 2, 1)

        x = F.gelu(self.fc1(x))
        out = self.fc2(x)

        return out.squeeze(-1)


verified_model = FNOBaselineVerified(
    modes=EXPECTED["modes"],
    width=EXPECTED["width"],
    in_channels=EXPECTED["in_channels"],
).cpu()

expected_param_count = sum(
    p.numel() for p in verified_model.parameters()
)

print("\n[+] Reconstructed parameter count:",
      f"{expected_param_count:,}")

assert expected_param_count == EXPECTED["expected_parameters"], (
    f"Parameter-count mismatch: "
    f"{expected_param_count:,} != "
    f"{EXPECTED['expected_parameters']:,}"
)

print("[+] Architecture parameter count: PASSED")

# ------------------------------------------------------------
# 9. Validate state_dict existence
# ------------------------------------------------------------
state_dict = ckpt["model_state_dict"]

assert isinstance(state_dict, dict), (
    f"model_state_dict has invalid type: {type(state_dict)}"
)

print("[+] State-dict entries:", len(state_dict))

# ------------------------------------------------------------
# 10. Compare exact parameter names
# ------------------------------------------------------------
model_state_dict = verified_model.state_dict()

saved_keys = set(state_dict.keys())
model_keys = set(model_state_dict.keys())

missing_keys = sorted(model_keys - saved_keys)
unexpected_keys = sorted(saved_keys - model_keys)

assert not missing_keys, (
    "Missing model parameters in checkpoint:\n"
    + "\n".join(missing_keys)
)

assert not unexpected_keys, (
    "Unexpected parameters in checkpoint:\n"
    + "\n".join(unexpected_keys)
)

print("[+] Parameter names: PASSED")

# ------------------------------------------------------------
# 11. Compare exact tensor shapes
# ------------------------------------------------------------
shape_mismatches = []

for key in model_state_dict.keys():

    saved_shape = tuple(state_dict[key].shape)
    expected_shape = tuple(model_state_dict[key].shape)

    if saved_shape != expected_shape:
        shape_mismatches.append(
            (key, saved_shape, expected_shape)
        )

assert not shape_mismatches, (
    "Tensor-shape mismatches found:\n"
    + "\n".join(map(str, shape_mismatches))
)

print("[+] Parameter shapes: PASSED")

# ------------------------------------------------------------
# 12. Numerical finiteness check
# ------------------------------------------------------------
nonfinite_tensors = []
nonfinite_values = 0

for key, tensor in state_dict.items():

    if not torch.is_tensor(tensor):
        raise TypeError(
            f"Checkpoint parameter '{key}' is not a tensor."
        )

    if not torch.is_floating_point(tensor) and not torch.is_complex(tensor):
        continue

    finite_mask = torch.isfinite(tensor)

    if not bool(torch.all(finite_mask)):
        bad_count = int((~finite_mask).sum().item())

        nonfinite_tensors.append(
            (key, bad_count)
        )

        nonfinite_values += bad_count

assert not nonfinite_tensors, (
    "Non-finite checkpoint values detected:\n"
    + "\n".join(map(str, nonfinite_tensors))
)

print("[+] Numerical finiteness: PASSED")
print(f"    Non-finite values: {nonfinite_values}")

# ------------------------------------------------------------
# 13. Load state_dict strictly
# ------------------------------------------------------------
try:
    verified_model.load_state_dict(
        state_dict,
        strict=True,
    )
except Exception as e:
    raise RuntimeError(
        f"Strict state_dict loading failed: "
        f"{type(e).__name__}: {e}"
    )

verified_model.eval()

print("[+] Strict model loading: PASSED")

# ------------------------------------------------------------
# 14. Optional forward-pass integrity test
# ------------------------------------------------------------
# Use the actual dataset when available.
forward_test_done = False

if "train_dataset" in globals():

    sample_input, sample_target = train_dataset[0]

    assert tuple(sample_input.shape) == (6, 128), (
        f"Unexpected sample input shape: "
        f"{tuple(sample_input.shape)}"
    )

    assert tuple(sample_target.shape) == (128,), (
        f"Unexpected sample target shape: "
        f"{tuple(sample_target.shape)}"
    )

    sample_input = sample_input.float().unsqueeze(0)

    with torch.no_grad():

        sample_pred = verified_model(
            state=sample_input[:, 0, :],
            phi=sample_input[:, 1, :],
            flux=sample_input[:, 2, :],
            noise=sample_input[:, 3, :],
            dissipation=sample_input[:, 4, :],
            gamma=sample_input[:, 5, 0],
        )

    assert tuple(sample_pred.shape) == (1, 128), (
        f"Unexpected prediction shape: "
        f"{tuple(sample_pred.shape)}"
    )

    assert bool(torch.all(torch.isfinite(sample_pred))), (
        "Forward-pass produced non-finite values."
    )

    forward_test_done = True

    print("[+] Forward-pass integrity: PASSED")
    print("    Input :", tuple(sample_input.shape))
    print("    Output:", tuple(sample_pred.shape))

else:
    print("[!] Forward-pass check skipped:")
    print("    train_dataset is not present in current runtime.")

# ------------------------------------------------------------
# 15. Runtime-history consistency check, when available
# ------------------------------------------------------------
history_check_done = False

if (
    "history" in globals()
    and isinstance(history, dict)
    and "val_mse" in history
    and len(history["val_mse"]) > 0
):

    val_history = np.asarray(
        history["val_mse"],
        dtype=np.float64
    )

    finite_history = np.isfinite(val_history)

    assert finite_history.all(), (
        "Runtime validation history contains non-finite values."
    )

    history_best_idx = int(np.argmin(val_history))
    history_best_epoch = history_best_idx + 1
    history_best_mse = float(val_history[history_best_idx])

    print("\n[+] Runtime history check:")
    print("    Runtime best epoch :", history_best_epoch)
    print("    Runtime best MSE   :", f"{history_best_mse:.8e}")
    print("    Saved best epoch   :", epoch)
    print("    Saved best MSE     :", f"{float(best_val_mse):.8e}")

    # Numerical equality should hold up to floating-point representation.
    assert abs(
        history_best_mse - float(best_val_mse)
    ) < 1e-12, (
        "Saved best_val_mse does not match runtime history."
    )

    assert history_best_epoch == epoch, (
        "Saved best epoch does not match runtime history."
    )

    history_check_done = True

    print("[+] Runtime history consistency: PASSED")

# ------------------------------------------------------------
# 16. Final verification report
# ------------------------------------------------------------
print("\n" + "=" * 78)
print("[+] FNO CHECKPOINT VERIFICATION COMPLETE")
print("=" * 78)

print(f"[+] Status             : VALID")
print(f"[+] Best epoch         : {epoch}")
print(f"[+] Best Val MSE       : {float(best_val_mse):.12e}")
print(f"[+] Parameter count    : {expected_param_count:,}")
print(f"[+] Architecture       : {architecture['name']}")
print(f"[+] Modes              : {architecture['modes']}")
print(f"[+] Width              : {architecture['width']}")
print(f"[+] Input channels     : {architecture['in_channels']}")
print(f"[+] Grid size          : {architecture['grid_size']}")
print(f"[+] Depth              : {architecture['depth']}")
print(f"[+] Forward check      : {forward_test_done}")
print(f"[+] History check      : {history_check_done}")
print(f"[+] SHA256             : {checkpoint_sha256}")

print("\n[+] LNO checkpoint untouched by this verification.")
print("=" * 78)
