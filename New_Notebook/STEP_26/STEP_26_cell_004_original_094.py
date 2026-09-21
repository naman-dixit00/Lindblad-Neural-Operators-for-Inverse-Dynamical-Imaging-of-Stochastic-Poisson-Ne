# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 94
# Step            : STEP_26
# Step Heading    : # Step 26 C-LNO checkpoint from the validated R-focused + soft-physics ablation.
# Step Cell No.   : 4
# ============================================================

# ======================================================================================
# EXPORT FINAL C-LNO MODEL FOR HUGGING FACE
# ======================================================================================
#
# Creates:
#   c_lno_final.pkl
#
# Contents:
#   - model_state_dict
#   - architecture configuration
#   - metadata
#   - parameter count
#
# NO TRAINING
# NO NEW MODEL INFERENCE
# NO CHECKPOINT MODIFICATION
# ======================================================================================

import os
import gc
import pickle
import hashlib
from pathlib import Path

import torch
import torch.nn as nn


# ======================================================================================
# 1. AUTO-DETECT REPOSITORY
# ======================================================================================

CONTENT_ROOT = Path("/content")

repo_candidates = [
    p for p in CONTENT_ROOT.iterdir()
    if (
        p.is_dir()
        and (p / "results").is_dir()
        and (p / "physics").is_dir()
        and (p / "training").is_dir()
    )
]

assert repo_candidates, "Repository root not found."

ROOT = repo_candidates[0]
os.chdir(ROOT)

print("=" * 110)
print("FINAL C-LNO → HUGGING FACE PICKLE EXPORT")
print("=" * 110)
print("[+] Repository:", ROOT)


# ======================================================================================
# 2. FLEXIBLE FILE LOCATOR
# ======================================================================================

def locate_file(filename, preferred_paths=None):
    preferred_paths = preferred_paths or []

    # Preferred exact paths
    for p in preferred_paths:
        p = Path(p)
        if p.is_file():
            return p

    # Repository recursive search
    matches = list(ROOT.rglob(filename))

    if matches:
        matches.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return matches[0]

    # Entire /content recursive search
    matches = []

    for p in Path("/content").rglob(filename):
        try:
            if p.is_file():
                matches.append(p)
        except Exception:
            pass

    if matches:
        matches.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return matches[0]

    raise FileNotFoundError(
        f"Could not locate required file: {filename}"
    )


# ======================================================================================
# 3. LOCATE FINAL C-LNO CHECKPOINT
# ======================================================================================
#
# Step 26 C-LNO checkpoint from the validated R-focused + soft-physics ablation.
#
# ======================================================================================

C_LNO_CHECKPOINT = locate_file(
    "C_R_focused_soft_physics_best.pt",
    preferred_paths=[
        ROOT
        / "results"
        / "step26_r_accuracy_ablation"
        / "C_R_focused_soft_physics_best.pt"
    ]
)

print("[PASS] C-LNO checkpoint found:")
print("       ", C_LNO_CHECKPOINT)


# ======================================================================================
# 4. CHECK MODEL CLASS
# ======================================================================================

if "FinalRepairedLNO" not in globals():

    raise RuntimeError(
        "\nFinalRepairedLNO is not defined in this Colab runtime.\n"
        "\n"
        "The exact validated architecture is:\n"
        "FinalRepairedLNO(dim=6, width=64, modes=16, depth=4, "
        "lindblad_channels=4)\n"
        "\n"
        "Please rerun the model-definition cell from the project "
        "before running this export cell."
    )

print("[PASS] FinalRepairedLNO class is available.")


# ======================================================================================
# 5. LOAD C-LNO
# ======================================================================================

device = torch.device("cpu")

model = FinalRepairedLNO(
    dim=6,
    width=64,
    modes=16,
    depth=4,
    lindblad_channels=4,
).to(device)

checkpoint = torch.load(
    C_LNO_CHECKPOINT,
    map_location=device
)

assert "model_state_dict" in checkpoint, (
    "Checkpoint does not contain 'model_state_dict'."
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("[PASS] C-LNO state_dict loaded.")


# ======================================================================================
# 6. PARAMETER COUNT
# ======================================================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "[INFO] Total parameters:",
    f"{total_parameters:,}"
)

print(
    "[INFO] Trainable parameters:",
    f"{trainable_parameters:,}"
)


# ======================================================================================
# 7. VERIFY EXPECTED ARCHITECTURE SIZE
# ======================================================================================

assert total_parameters == 546039, (
    "Unexpected C-LNO parameter count.\n"
    f"Expected: 546039\n"
    f"Found:    {total_parameters}"
)

print("[PASS] Parameter count matches validated C-LNO architecture.")


# ======================================================================================
# 8. STATE DICT → CPU + DETACHED
# ======================================================================================

state_dict_cpu = {
    key: value.detach().cpu().clone()
    for key, value in model.state_dict().items()
}


assert len(state_dict_cpu) > 0, (
    "Empty model state_dict."
)

assert all(
    torch.is_tensor(v)
    for v in state_dict_cpu.values()
), (
    "Non-tensor value found in model state_dict."
)


# ======================================================================================
# 9. CHECK FINITE WEIGHTS
# ======================================================================================

nonfinite_keys = []

for key, tensor in state_dict_cpu.items():

    if not torch.isfinite(tensor).all():

        nonfinite_keys.append(key)


assert not nonfinite_keys, (
    "Non-finite values found in C-LNO weights:\n"
    + "\n".join(nonfinite_keys)
)

print("[PASS] All exported weights are finite.")


# ======================================================================================
# 10. CHECKPOINT HASH
# ======================================================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


checkpoint_sha256 = sha256_file(
    C_LNO_CHECKPOINT
)


# ======================================================================================
# 11. CREATE PORTABLE HUGGING FACE PICKLE
# ======================================================================================

HF_DIR = (
    ROOT
    /
    "results"
    /
    "huggingface_c_lno"
)

HF_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PICKLE_PATH = (
    HF_DIR
    /
    "c_lno_final.pkl"
)


payload = {

    "format_version":
        "1.0",

    "model_name":
        "C-LNO",

    "model_description":
        (
            "Final C-LNO checkpoint for Lindblad Neural Operators "
            "for inverse dynamical imaging of stochastic "
            "Poisson-Nernst-Planck ion transport."
        ),

    "architecture":
        "FinalRepairedLNO",

    "architecture_config": {
        "dim": 6,
        "width": 64,
        "modes": 16,
        "depth": 4,
        "lindblad_channels": 4,
    },

    "parameter_count":
        total_parameters,

    "trainable_parameter_count":
        trainable_parameters,

    "model_state_dict":
        state_dict_cpu,

    "checkpoint_source":
        str(C_LNO_CHECKPOINT),

    "checkpoint_sha256":
        checkpoint_sha256,

    "training_performed_during_export":
        False,

    "new_inference_performed_during_export":
        False,

    "export_note":
        (
            "Weights exported from the validated C-LNO checkpoint "
            "without retraining or model modification."
        ),
}


with open(
    PICKLE_PATH,
    "wb"
) as f:

    pickle.dump(
        payload,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )


print("\n[PASS] Pickle exported:")
print("       ", PICKLE_PATH)


# ======================================================================================
# 12. RELOAD VERIFICATION
# ======================================================================================

with open(
    PICKLE_PATH,
    "rb"
) as f:

    reloaded_payload = pickle.load(f)


assert (
    reloaded_payload["model_name"]
    ==
    "C-LNO"
)

assert (
    reloaded_payload["architecture"]
    ==
    "FinalRepairedLNO"
)

assert (
    reloaded_payload["architecture_config"]
    ==
    {
        "dim": 6,
        "width": 64,
        "modes": 16,
        "depth": 4,
        "lindblad_channels": 4,
    }
)

assert (
    reloaded_payload["parameter_count"]
    ==
    546039
)

assert (
    len(
        reloaded_payload["model_state_dict"]
    )
    ==
    len(
        state_dict_cpu
    )
)

print("[PASS] Pickle reload verification passed.")


# ======================================================================================
# 13. EXACT WEIGHT CHECK
# ======================================================================================

for key in state_dict_cpu:

    a = state_dict_cpu[key]
    b = reloaded_payload["model_state_dict"][key]

    assert torch.equal(
        a,
        b
    ), (
        f"Weight mismatch after pickle reload: {key}"
    )


print("[PASS] Every tensor matches exactly after pickle reload.")


# ======================================================================================
# 14. FILE SIZE
# ======================================================================================

size_mb = (
    PICKLE_PATH.stat().st_size
    /
    (1024 ** 2)
)

print(
    "[INFO] Pickle size:",
    f"{size_mb:.2f} MB"
)


# ======================================================================================
# 15. FINAL
# ======================================================================================

gc.collect()

print("\n" + "=" * 110)
print("C-LNO HUGGING FACE EXPORT COMPLETE")
print("=" * 110)

print(
    "\nFile:"
)

print(
    PICKLE_PATH
)

print(
    "\nModel:",
    "C-LNO"
)

print(
    "Architecture:",
    "FinalRepairedLNO"
)

print(
    "Parameters:",
    f"{total_parameters:,}"
)

print(
    "\n[PASS] No training performed."
)

print(
    "[PASS] No new inference performed."
)

print(
    "[PASS] No checkpoint modified."
)

print(
    "[PASS] Pickle reload verified exactly."
)

print("=" * 110)
