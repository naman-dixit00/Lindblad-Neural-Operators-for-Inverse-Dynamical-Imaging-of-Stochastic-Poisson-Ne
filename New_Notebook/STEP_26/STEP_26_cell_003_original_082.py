# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 82
# Step            : STEP_26
# Step Heading    : # STEP 26 — SAVE + PACKAGE COMPLETE R-ACCURACY ABLATION
# Step Cell No.   : 3
# ============================================================

# ======================================================================================
# STEP 26 — SAVE + PACKAGE COMPLETE R-ACCURACY ABLATION
# ======================================================================================

import os
import shutil
import zipfile
from pathlib import Path

REPO_ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

SRC = (
    REPO_ROOT
    / "results"
    / "step26_r_accuracy_ablation"
)

SAVE_DIR = (
    REPO_ROOT
    / "results"
    / "STEP26_COMPLETE"
)

ZIP_PATH = (
    REPO_ROOT
    / "results"
    / "STEP26_COMPLETE.zip"
)

print("=" * 110)
print("STEP 26 — SAVE + PACKAGE")
print("=" * 110)

assert SRC.is_dir(), f"Step 26 results folder not found: {SRC}"

# Remove old package
if SAVE_DIR.exists():
    shutil.rmtree(SAVE_DIR)

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

# Copy complete Step 26 results
shutil.copytree(SRC, SAVE_DIR)

# Manifest
manifest = SAVE_DIR / "STEP26_MANIFEST.txt"

with open(manifest, "w", encoding="utf-8") as f:
    f.write(
        "STEP 26 — R ACCURACY IMPROVEMENT ABLATION\n"
        "==========================================\n\n"
        "Variants:\n"
        "A_baseline\n"
        "B_R_focused\n"
        "C_R_focused_soft_physics\n\n"
        "Purpose:\n"
        "Controlled fresh LNO training ablation to test whether the small\n"
        "R-accuracy gap versus FNO can be reduced through an R-focused\n"
        "objective and restrained Lindblad coupling.\n\n"
        "Dataset:\n"
        "Dataset V3 Expanded — 9,900 transitions\n\n"
        "Split:\n"
        "Train 6930 / Validation 1485 / Test 1485\n\n"
        "Existing final FNO/LNO checkpoints were not modified.\n"
    )

# Create ZIP
with zipfile.ZipFile(
    ZIP_PATH,
    "w",
    zipfile.ZIP_DEFLATED
) as z:

    for root, dirs, files in os.walk(SAVE_DIR):

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

print("\n[PASS] Step 26 results saved:")
print(SAVE_DIR)

print("\n[PASS] Step 26 ZIP created:")
print(ZIP_PATH)

print("\nContents:")
for p in sorted(SAVE_DIR.iterdir()):
    print(" -", p.name)

print("\nZIP size:")
print(
    f"{ZIP_PATH.stat().st_size / (1024**2):.2f} MB"
)

print("\n" + "=" * 110)
print("STEP 26 PACKAGE COMPLETE")
print("=" * 110)
