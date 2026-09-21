# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 75
# Step            : STEP_24
# Step Heading    : # SAVE STEP 24 COMPLETE ARTIFACTS
# Step Cell No.   : 2
# ============================================================

# ============================================================
# SAVE STEP 24 COMPLETE ARTIFACTS
# ============================================================

from pathlib import Path
import shutil
import zipfile

STEP24_DIR = (
    ROOT
    / "results"
    / "step24_stability_visualization"
)

SAVE_DIR = (
    ROOT
    / "results"
    / "STEP24_COMPLETE"
)

ZIP_PATH = (
    ROOT
    / "results"
    / "STEP24_COMPLETE.zip"
)

assert STEP24_DIR.is_dir(), (
    f"Missing Step 24 directory:\n{STEP24_DIR}"
)

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Remove previous packaged copy
for item in SAVE_DIR.iterdir():
    if item.is_dir():
        shutil.rmtree(item)
    else:
        item.unlink()

# Copy all Step 24 artifacts
for item in STEP24_DIR.iterdir():

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

# Remove previous ZIP if present
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
print("STEP 24 SAVED")
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
    "\n[PASS] STEP 24 artifacts packaged successfully."
)
