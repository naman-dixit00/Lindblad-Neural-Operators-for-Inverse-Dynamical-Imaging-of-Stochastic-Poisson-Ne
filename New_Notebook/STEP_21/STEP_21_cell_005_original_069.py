# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 69
# Step            : STEP_21
# Step Heading    : # SAVE / PACKAGE STEP 21 — COMPLETE RESULTS
# Step Cell No.   : 5
# ============================================================

# ============================================================
# SAVE / PACKAGE STEP 21 — COMPLETE RESULTS
# ============================================================

from pathlib import Path
import shutil
import json
import zipfile
from datetime import datetime

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

SOURCE_DIR = (
    ROOT
    / "results"
    / "step21_collapse_stress"
)

SAVE_DIR = (
    ROOT
    / "results"
    / "STEP21_COMPLETE"
)

ZIP_PATH = (
    ROOT
    / "results"
    / "STEP21_COMPLETE.zip"
)

# ------------------------------------------------------------
# 1. Verify Step 21 outputs
# ------------------------------------------------------------

required_files = [
    "collapse_stress_results.csv",
    "collapse_stress_rollout.csv",
    "collapse_stress_regime_summary.csv",
    "collapse_stress_response.png",
    "collapse_stress_summary.json",
]

print("=" * 110)
print("STEP 21 — SAVE / PACKAGE")
print("=" * 110)

print("\nChecking existing Step 21 artifacts...")

for fname in required_files:
    path = SOURCE_DIR / fname
    print(f"{fname:40s} -> {path.is_file()}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing Step 21 artifact:\n{path}")

print("\n[PASS] All Step 21 artifacts found.")

# ------------------------------------------------------------
# 2. Recreate clean archive folder
# ------------------------------------------------------------

if SAVE_DIR.exists():
    shutil.rmtree(SAVE_DIR)

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# 3. Copy all Step 21 results
# ------------------------------------------------------------

for item in SOURCE_DIR.iterdir():
    destination = SAVE_DIR / item.name

    if item.is_file():
        shutil.copy2(item, destination)

    elif item.is_dir():
        shutil.copytree(item, destination)

print("\n[PASS] Step 21 results copied.")

# ------------------------------------------------------------
# 4. Create human-readable README
# ------------------------------------------------------------

readme_text = f"""
STEP 21 — COLLAPSE STRESS TEST
================================

Project:
Lindblad Neural Operators for Inverse Dynamical Imaging
of Stochastic Poisson–Nernst–Planck Ion Transport

Step:
21 — Collapse Stress Test

Date packaged:
{datetime.now().isoformat(timespec="seconds")}

Test regime:
Collapse

Collapse baseline:
gamma = 0.50
sigma = 0.65

Progressive stress:
Level 0: gamma=0.50, sigma=0.65
Level 1: gamma=0.60, sigma=0.75
Level 2: gamma=0.70, sigma=0.85
Level 3: gamma=0.80, sigma=1.00
Level 4: gamma=0.90, sigma=1.15
Level 5: gamma=1.00, sigma=1.30

Collapse test transitions:
297

Diagnostics:
- Prediction response under progressively harder collapse
- Trace preservation
- Minimum eigenvalue
- PSD violation
- Rollout stability
- State/amplitude response
- Lindblad vs neural dynamical contribution

Important scientific note:
This is a counterfactual environment-response and
structural-stability stress test. It is NOT a new
ground-truth accuracy benchmark because no independently
simulated ground-truth trajectories were generated for
the stressed parameter values.

No model retraining was performed.
No checkpoint was modified.

Artifacts:
- collapse_stress_results.csv
- collapse_stress_rollout.csv
- collapse_stress_regime_summary.csv
- collapse_stress_response.png
- collapse_stress_summary.json
"""

with open(
    SAVE_DIR / "README_STEP21.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(readme_text)

# ------------------------------------------------------------
# 5. Create ZIP
# ------------------------------------------------------------

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

with zipfile.ZipFile(
    ZIP_PATH,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as z:

    for file_path in SAVE_DIR.rglob("*"):
        if file_path.is_file():
            z.write(
                file_path,
                arcname=file_path.relative_to(SAVE_DIR.parent)
            )

# ------------------------------------------------------------
# 6. Final verification
# ------------------------------------------------------------

print("\n" + "=" * 110)
print("STEP 21 SAVE COMPLETE")
print("=" * 110)

print("\nFolder:")
print(SAVE_DIR)

print("\nZIP:")
print(ZIP_PATH)

print("\nContents:")
for p in sorted(SAVE_DIR.rglob("*")):
    if p.is_file():
        print("  ", p.relative_to(SAVE_DIR))

print("\nZIP size (MB):", round(ZIP_PATH.stat().st_size / 1024**2, 3))

print("\n[PASS] STEP 21 safely packaged.")
print("[INFO] No retraining performed.")
print("[INFO] No checkpoint modified.")
