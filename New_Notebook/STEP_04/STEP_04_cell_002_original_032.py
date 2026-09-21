# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 32
# Step            : STEP_04
# Step Heading    : # STEP 4 — ENVIRONMENT-CONDITIONED DATASET V2
# Step Cell No.   : 2
# ============================================================

import os
from pathlib import Path

ROOT = Path("/content/Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne")
DATASET_PATH = ROOT / "results" / "dataset_v2" / "environment_conditioned_lno_dataset_v2.npz"

if DATASET_PATH.is_file():
    print(f"Dataset successfully created at: {DATASET_PATH}")
    print(f"File size: {DATASET_PATH.stat().st_size / (1024 * 1024):.2f} MB")
else:
    print(f"Dataset not found at: {DATASET_PATH}")
