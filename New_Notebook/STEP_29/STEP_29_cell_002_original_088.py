# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 88
# Step            : STEP_29
# Step Heading    : # STEP 29 — SAVE + PACKAGE COMPLETE FINAL C-LNO EVALUATION
# Step Cell No.   : 2
# ============================================================

# ======================================================================================
# STEP 29 — SAVE + PACKAGE COMPLETE FINAL C-LNO EVALUATION
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
    / "step29_final_c_lno_evaluation"
)

SAVE_DIR = (
    ROOT
    / "results"
    / "STEP29_COMPLETE"
)

ZIP_PATH = (
    ROOT
    / "results"
    / "STEP29_COMPLETE.zip"
)

print("=" * 110)
print("STEP 29 — SAVE + PACKAGE")
print("=" * 110)

assert SRC.is_dir(), (
    f"Step 29 results folder not found: {SRC}"
)

# Remove old package
if SAVE_DIR.exists():
    shutil.rmtree(SAVE_DIR)

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

# Copy complete Step 29 results
shutil.copytree(
    SRC,
    SAVE_DIR
)

# Manifest
manifest = SAVE_DIR / "STEP29_MANIFEST.txt"

with open(
    manifest,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 29 — FINAL C-LNO COMPREHENSIVE EVALUATION\n"
        "===============================================\n\n"
        "Models compared:\n"
        "1. FNO (Step 17)\n"
        "2. Original Final LNO (Step 18)\n"
        "3. Step-26 C-LNO\n\n"
        "Evidence used:\n"
        "- Step 27 independent C-LNO validation\n"
        "- Step 28 C-LNO 100-step long-horizon stability\n\n"
        "Key result:\n"
        "C-LNO is the R-accuracy winner.\n"
        "C-LNO beats FNO in R accuracy across 5/5 regimes.\n"
        "C-LNO beats original Final LNO in R accuracy across 5/5 regimes.\n"
        "C-LNO retains one-step structural validity.\n"
        "C-LNO retains 100-step long-horizon structural stability.\n\n"
        "No model retraining performed.\n"
        "No checkpoint modification performed.\n"
        "No new C-LNO inference performed in Step 29.\n"
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

print("\n[PASS] Step 29 package folder:")
print(SAVE_DIR)

print("\n[PASS] Step 29 ZIP:")
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

print("\n" + "=" * 110)
print("STEP 29 PACKAGE COMPLETE")
print("=" * 110)
