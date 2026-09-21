# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 6
# Step            : STEP_01
# Step Heading    : # PHASE 1 — STEP 1: REPOSITORY + TENSOR GEOMETRY AUDIT
# Step Cell No.   : 2
# ============================================================

from pathlib import Path
import sys
import torch

ROOT = Path("/content/Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne")
sys.path.insert(0, str(ROOT))

# 1. Import actual dataset/model code
from data.dataset_loader import *
from models.lno_model import LindbladNeuralOperator

print("Imports: OK")

# 2. Inspect raw gamma spatial behavior
import numpy as np

gamma = np.load(ROOT / "dataset " / "val" / "gamma.npy", mmap_mode="r")

print("Gamma shape:", gamma.shape)
print("Gamma min:", gamma.min())
print("Gamma max:", gamma.max())
print("Gamma mean:", gamma.mean())
print("Gamma std:", gamma.std())

# 3. Load checkpoint
ckpt_path = ROOT / "results" / "check_points" / "best_lno.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

print("\nCheckpoint type:", type(ckpt))
if isinstance(ckpt, dict):
    print("Checkpoint keys:", ckpt.keys())

# 4. Instantiate the current model exactly as the existing pipeline expects
model = LindbladNeuralOperator(
    in_channels=6,
    width=64,
    depth=4,
    modes=16
)

print("\nModel created.")
print("Parameter count:", sum(p.numel() for p in model.parameters()))
