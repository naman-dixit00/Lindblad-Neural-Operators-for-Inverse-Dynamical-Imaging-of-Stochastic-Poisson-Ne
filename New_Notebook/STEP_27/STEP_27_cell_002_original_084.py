# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 84
# Step            : STEP_27
# Step Heading    : # STEP 27 — SAVE + PACKAGE COMPLETE C-LNO VALIDATION
# Step Cell No.   : 2
# ============================================================

# ======================================================================================
# STEP 27 — SAVE + PACKAGE COMPLETE C-LNO VALIDATION
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
    / "step27_c_lno_independent_validation"
)

SAVE_DIR = (
    ROOT
    / "results"
    / "STEP27_COMPLETE"
)

ZIP_PATH = (
    ROOT
    / "results"
    / "STEP27_COMPLETE.zip"
)

print("=" * 110)
print("STEP 27 — SAVE + PACKAGE")
print("=" * 110)

assert SRC.is_dir(), f"Step 27 results folder not found: {SRC}"

# Remove previous package if present
if SAVE_DIR.exists():
    shutil.rmtree(SAVE_DIR)

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

# Copy complete results
shutil.copytree(SRC, SAVE_DIR)

# Manifest
manifest = SAVE_DIR / "STEP27_MANIFEST.txt"

with open(manifest, "w", encoding="utf-8") as f:
    f.write(
        "STEP 27 — INDEPENDENT VALIDATION OF STEP-26 C-LNO\n"
        "=================================================\n\n"
        "Compared models:\n"
        "1. Final FNO (Step 17)\n"
        "2. Original Final LNO (Step 18)\n"
        "3. Step-26 C-LNO (R-focused + soft physics)\n\n"
        "Test set:\n"
        "1,485 identical test transitions\n\n"
        "No model retraining performed.\n"
        "C-LNO evaluated from the saved Step-26 checkpoint.\n"
        "FNO and original Final LNO use their existing final predictions.\n\n"
        "Main result:\n"
        "C-LNO R Relative-L2 = 0.507714\n"
        "FNO R Relative-L2 = 0.516526\n"
        "C-LNO improvement vs FNO ≈ 1.71%\n\n"
        "C-LNO is better than FNO in R Relative-L2 across all 5 tested regimes.\n"
        "Structural preservation remains numerically strong.\n"
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

print("\n[PASS] Step 27 package folder:")
print(SAVE_DIR)

print("\n[PASS] Step 27 ZIP created:")
print(ZIP_PATH)

print("\nPackage contents:")
for p in sorted(SAVE_DIR.iterdir()):
    print(" -", p.name)

print("\nZIP size:")
print(f"{ZIP_PATH.stat().st_size / (1024**2):.2f} MB")

print("\n" + "=" * 110)
print("STEP 27 PACKAGE COMPLETE")
print("=" * 110)
