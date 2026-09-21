# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 73
# Step            : STEP_23
# Step Heading    : # SAVE STEP 23 COMPLETE ARTIFACTS
# Step Cell No.   : 2
# ============================================================

# ============================================================
# SAVE STEP 23 COMPLETE ARTIFACTS
# ============================================================

from pathlib import Path
import shutil
import zipfile

STEP23_DIR = (
    ROOT
    / "results"
    / "step23_regime_analysis"
)

SAVE_DIR = (
    ROOT
    / "results"
    / "STEP23_COMPLETE"
)

ZIP_PATH = (
    ROOT
    / "results"
    / "STEP23_COMPLETE.zip"
)

assert STEP23_DIR.is_dir(), (
    f"Missing Step 23 directory:\n{STEP23_DIR}"
)

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Remove previous packaged copy if present
for item in SAVE_DIR.iterdir():
    if item.is_dir():
        shutil.rmtree(item)
    else:
        item.unlink()

# Copy complete Step 23 results
for item in STEP23_DIR.iterdir():

    target = SAVE_DIR / item.name

    if item.is_dir():

        shutil.copytree(
            item,
            target
        )

    else:

        shutil.copy2(
            item,
            target
        )

# Remove previous ZIP
if ZIP_PATH.exists():
    ZIP_PATH.unlink()

# Create ZIP
with zipfile.ZipFile(
    ZIP_PATH,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as z:

    for file_path in SAVE_DIR.rglob("*"):

        if file_path.is_file():

            z.write(
                file_path,
                file_path.relative_to(
                    SAVE_DIR
                )
            )

print("=" * 110)
print("STEP 23 SAVED")
print("=" * 110)

print(
    "Folder:",
    SAVE_DIR
)

print(
    "ZIP:",
    ZIP_PATH
)

print("\n[FILES]")

for p in sorted(
    SAVE_DIR.rglob("*")
):

    if p.is_file():

        print(
            "-",
            p.relative_to(
                SAVE_DIR
            )
        )

print(
    "\n[PASS] STEP 23 artifacts packaged successfully."
)
