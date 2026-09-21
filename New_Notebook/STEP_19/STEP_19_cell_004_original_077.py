# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 77
# Step            : STEP_19
# Step Heading    : # STEP 19 — SAVE + PACKAGE COMPLETE RESULTS
# Step Cell No.   : 4
# ============================================================

# ======================================================================================
# STEP 19 — SAVE + PACKAGE COMPLETE RESULTS
# ======================================================================================

import os
import shutil
import zipfile

REPO_ROOT = "/content/Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
STEP19_SRC = os.path.join(REPO_ROOT, "results", "step19_final_comparison")

SAVE_ROOT = os.path.join(REPO_ROOT, "results", "STEP19_COMPLETE")
ZIP_PATH = os.path.join(REPO_ROOT, "results", "STEP19_COMPLETE.zip")

print("=" * 100)
print("STEP 19 — SAVE + PACKAGE")
print("=" * 100)

# Clean previous package if present
if os.path.exists(SAVE_ROOT):
    shutil.rmtree(SAVE_ROOT)

if os.path.exists(ZIP_PATH):
    os.remove(ZIP_PATH)

# Copy complete Step19 folder
shutil.copytree(STEP19_SRC, SAVE_ROOT)

# Also save a small manifest
manifest = os.path.join(SAVE_ROOT, "STEP19_MANIFEST.txt")
with open(manifest, "w") as f:
    f.write(
        "STEP 19 — FINAL FNO vs FINAL REPAIRED LNO COMPARISON\n"
        "====================================================\n"
        "Fresh comparison using current final prediction artifacts.\n"
        "No model retraining.\n"
        "No checkpoint modification.\n"
        "No new model inference.\n\n"
        "Contents copied from:\n"
        f"{STEP19_SRC}\n\n"
        "Main files:\n"
        "final_comparison.csv\n"
        "final_relative_change.csv\n"
        "final_regime_comparison.csv\n"
        "paired_test_comparison.csv\n"
        "final_comparison.json\n"
        "STEP19_MANIFEST.txt\n"
    )

# Create ZIP
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(SAVE_ROOT):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, os.path.dirname(SAVE_ROOT))
            z.write(full_path, arcname)

print("\n[PASS] STEP 19 folder saved:")
print(SAVE_ROOT)

print("\n[PASS] STEP 19 ZIP created:")
print(ZIP_PATH)

print("\nFiles inside package:")
for name in sorted(os.listdir(SAVE_ROOT)):
    print(" -", name)

print("\nZIP size:")
print(f"{os.path.getsize(ZIP_PATH) / (1024**2):.2f} MB")

print("\n" + "=" * 100)
print("STEP 19 PACKAGE COMPLETE")
print("=" * 100)
