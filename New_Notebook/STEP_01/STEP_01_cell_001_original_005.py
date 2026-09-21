# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 5
# Step            : STEP_01
# Step Heading    : # PHASE 1 — STEP 1: REPOSITORY + TENSOR GEOMETRY AUDIT
# Step Cell No.   : 1
# ============================================================

# ============================================================
# PHASE 1 — STEP 1: REPOSITORY + TENSOR GEOMETRY AUDIT
# DO NOT MODIFY ANY MODEL WEIGHTS OR FILES
# ============================================================

from pathlib import Path
import numpy as np
import torch

# Corrected ROOT path to point to the cloned repository
ROOT = Path.cwd() / "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"

print("=" * 80)
print("REPOSITORY ROOT")
print("=" * 80)
print(ROOT)

# Find dataset root (corrected to account for the trailing space in the directory name)
dataset_candidates = [
    ROOT / "dataset " / "train",
    ROOT / "dataset " / "val",
    ROOT / "dataset " / "test",
]

for d in dataset_candidates:
    print(f"{d}: {'FOUND' if d.exists() else 'MISSING'}")

print("\n" + "=" * 80)
print("DATASET SHAPES")
print("=" * 80)

for split in ["train", "val", "test"]:
    # Corrected split_dir to account for the trailing space in the directory name
    split_dir = ROOT / "dataset " / split
    print(f"\n[{split.upper()}]")

    for name in [
        "state_t.npy",
        "state_t1.npy",
        "phi.npy",
        "flux.npy",
        "noise.npy",
        "dissipation.npy",
        "gamma.npy",
        "instability.npy",
        "entropy.npy",
    ]:
        p = split_dir / name

        if p.exists():
            arr = np.load(p, mmap_mode="r")
            print(f"{name:20s} shape={arr.shape}, dtype={arr.dtype}")
        else:
            print(f"{name:20s} MISSING")

print("\n" + "=" * 80)
print("CHECKPOINT PATHS")
print("=" * 80)

for p in [
    ROOT / "results" / "check_points" / "best_lno.pt",
    ROOT / "results" / "checkpoints" / "best_lno.pt",
]:
    print(f"{p}: {'FOUND' if p.exists() else 'MISSING'}")

print("\n" + "=" * 80)
print("CUDA")
print("=" * 80)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
