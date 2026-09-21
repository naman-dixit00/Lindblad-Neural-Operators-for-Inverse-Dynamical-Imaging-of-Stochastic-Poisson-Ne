# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 63
# Step            : STEP_18
# Step Heading    : #   Step 18 Final Repaired LNO artifacts
# Step Cell No.   : 7
# ============================================================

# ============================================================
# DOWNLOAD ALL STEP-18B PLOTS AS ONE ZIP
# ============================================================

from google.colab import files
from pathlib import Path
import zipfile
import os

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

FIG_DIR = (
    ROOT
    / "results"
    / "step18_final_lno"
    / "figures_square"
)

ZIP_PATH = (
    ROOT
    / "results"
    / "step18_final_lno"
    / "step18_square_figures_complete.zip"
)

print("=" * 100)
print("STEP 18B — ZIP ALL FIGURES")
print("=" * 100)

# ------------------------------------------------------------
# Check folder
# ------------------------------------------------------------

if not FIG_DIR.is_dir():

    raise FileNotFoundError(
        f"Figure directory not found:\n{FIG_DIR}"
    )

# ------------------------------------------------------------
# List files
# ------------------------------------------------------------

all_files = sorted(
    p for p in FIG_DIR.rglob("*")
    if p.is_file()
)

print(
    f"[+] Files found: {len(all_files)}"
)

for p in all_files:

    size_mb = (
        p.stat().st_size
        /
        (1024 ** 2)
    )

    print(
        f"  {p.relative_to(FIG_DIR)} "
        f"({size_mb:.3f} MB)"
    )

# ------------------------------------------------------------
# Create ZIP
# ------------------------------------------------------------

with zipfile.ZipFile(
    ZIP_PATH,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as zf:

    for p in all_files:

        zf.write(
            p,
            arcname=p.relative_to(
                FIG_DIR
            )
        )

print(
    "\n[PASS] ZIP created:"
)

print(
    ZIP_PATH
)

# ------------------------------------------------------------
# ZIP size
# ------------------------------------------------------------

zip_size_mb = (
    ZIP_PATH.stat().st_size
    /
    (1024 ** 2)
)

print(
    f"[+] ZIP size: {zip_size_mb:.3f} MB"
)

# ------------------------------------------------------------
# Browser download
# ------------------------------------------------------------

print(
    "\n[+] Starting browser download..."
)

files.download(
    str(ZIP_PATH)
)

print(
    "\n[PASS] Download initiated successfully."
)

print("=" * 100)
