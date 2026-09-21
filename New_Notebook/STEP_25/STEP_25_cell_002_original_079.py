# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 79
# Step            : STEP_25
# Step Heading    : # STEP 25 — SAVE + PACKAGE COMPLETE COMPARATIVE EVIDENCE
# Step Cell No.   : 2
# ============================================================

# ======================================================================================
# STEP 25 — SAVE + PACKAGE COMPLETE COMPARATIVE EVIDENCE
# ======================================================================================

import os
import shutil
import zipfile

REPO_ROOT = "/content/Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"

STEP25_SRC = os.path.join(
    REPO_ROOT,
    "results",
    "step25_final_comparative_evidence"
)

STEP25_SAVE = os.path.join(
    REPO_ROOT,
    "results",
    "STEP25_COMPLETE"
)

STEP25_ZIP = os.path.join(
    REPO_ROOT,
    "results",
    "STEP25_COMPLETE.zip"
)

print("=" * 100)
print("STEP 25 — SAVE + PACKAGE COMPLETE EVIDENCE")
print("=" * 100)

# Remove previous package if present
if os.path.exists(STEP25_SAVE):
    shutil.rmtree(STEP25_SAVE)

if os.path.exists(STEP25_ZIP):
    os.remove(STEP25_ZIP)

# Copy all Step 25 outputs
shutil.copytree(STEP25_SRC, STEP25_SAVE)

# Manifest
manifest_path = os.path.join(STEP25_SAVE, "STEP25_MANIFEST.txt")

with open(manifest_path, "w") as f:
    f.write(
        "STEP 25 — FINAL COMPARATIVE EVIDENCE CONSOLIDATION\n"
        "==================================================\n\n"
        "Purpose:\n"
        "Consolidate previously computed final evidence from Steps 19, 22, 23, and 24.\n\n"
        "No model retraining performed.\n"
        "No checkpoint modification performed.\n"
        "No new model inference performed.\n"
        "No new ground-truth accuracy benchmark performed.\n\n"
        "Evidence sources:\n"
        "- Step 19: Final FNO vs repaired LNO one-step accuracy + structural comparison\n"
        "- Step 22: Global 100-step long-horizon stability\n"
        "- Step 23: Regime-wise 100-step structural stability\n"
        "- Step 24: Stability visualization / completeness verification\n\n"
        f"Source directory:\n{STEP25_SRC}\n"
    )

# Create ZIP
with zipfile.ZipFile(STEP25_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(STEP25_SAVE):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(
                full_path,
                os.path.dirname(STEP25_SAVE)
            )
            z.write(full_path, arcname)

print("\n[PASS] Step 25 package folder:")
print(STEP25_SAVE)

print("\n[PASS] Step 25 ZIP:")
print(STEP25_ZIP)

print("\nPackage contents:")
for name in sorted(os.listdir(STEP25_SAVE)):
    print(" -", name)

print("\nZIP size:")
print(f"{os.path.getsize(STEP25_ZIP) / (1024**2):.2f} MB")

print("\n" + "=" * 100)
print("STEP 25 PACKAGE COMPLETE")
print("=" * 100)
