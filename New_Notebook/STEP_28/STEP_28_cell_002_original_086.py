# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 86
# Step            : STEP_28
# Step Heading    : # STEP 28 — SAVE + PACKAGE COMPLETE C-LNO LONG-HORIZON VALIDATION
# Step Cell No.   : 2
# ============================================================

# ======================================================================================
# STEP 28 — SAVE + PACKAGE COMPLETE C-LNO LONG-HORIZON VALIDATION
# ======================================================================================

import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

SRC = (
    ROOT
    / "results"
    / "step28_c_lno_long_horizon"
)

SAVE_DIR = (
    ROOT
    / "results"
    / "STEP28_COMPLETE"
)

ZIP_PATH = (
    ROOT
    / "results"
    / "STEP28_COMPLETE.zip"
)

print("=" * 110)
print("STEP 28 — SAVE + PACKAGE")
print("=" * 110)

assert SRC.is_dir(), (
    f"Step 28 results folder not found: {SRC}"
)

# Remove old package
if SAVE_DIR.exists():
    shutil.rmtree(SAVE_DIR)

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

# Copy complete Step 28 results
shutil.copytree(
    SRC,
    SAVE_DIR
)

# Manifest
manifest = SAVE_DIR / "STEP28_MANIFEST.txt"

with open(
    manifest,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 28 — C-LNO LONG-HORIZON + REGIME STABILITY VALIDATION\n"
        "===========================================================\n\n"
        "Model:\n"
        "Step-26 C-LNO (R-focused + soft physics)\n\n"
        "Evaluation:\n"
        "25 representative test states\n"
        "5 representatives per regime\n"
        "100 rollout steps per state\n"
        "5 environment regimes\n\n"
        "Result:\n"
        "25/25 rollouts completed\n"
        "All values finite\n"
        "Zero meaningful PSD violations\n"
        "Trace error remained at numerical precision\n\n"
        "No model retraining performed.\n"
        "No checkpoint modified.\n"
        "No new ground-truth accuracy benchmark performed.\n\n"
        f"Source directory:\n{SRC}\n"
    )

# Create ZIP
with zipfile.ZipFile(
    ZIP_PATH,
    "w",
    zipfile.ZIP_DEFLATED
) as z:

    for root, dirs, files in os.walk(
        SAVE_DIR
    ):

        for file in files:

            full_path = os.path.join(
                root,
                file
            )

            arcname = os.path.relpath(
                full_path,
                SAVE_DIR.parent
            )

            z.write(
                full_path,
                arcname
            )

print("\n[PASS] Step 28 package folder:")
print(SAVE_DIR)

print("\n[PASS] Step 28 ZIP:")
print(ZIP_PATH)

print("\nContents:")
for p in sorted(
    SAVE_DIR.iterdir()
):
    print(" -", p.name)

print("\nZIP size:")
print(
    f"{ZIP_PATH.stat().st_size / (1024**2):.2f} MB"
)

print(
    "\n[PASS] STEP 28 PACKAGE COMPLETE"
)

print("=" * 110)
