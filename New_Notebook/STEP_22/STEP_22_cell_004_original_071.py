# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 71
# Step            : STEP_22
# Step Heading    : # SAVE STEP 22 COMPLETE ARTIFACTS
# Step Cell No.   : 4
# ============================================================

# ============================================================
# SAVE STEP 22 COMPLETE ARTIFACTS
# ============================================================

from pathlib import Path
import shutil
import zipfile

STEP22_DIR = ROOT / "results" / "step22_long_horizon"
SAVE_DIR = ROOT / "results" / "STEP22_COMPLETE"
ZIP_PATH = ROOT / "results" / "STEP22_COMPLETE.zip"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Copy complete Step 22 result folder
for item in STEP22_DIR.iterdir():
    target = SAVE_DIR / item.name

    if item.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(item, target)
    else:
        shutil.copy2(item, target)

# Create ZIP
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
                file_path.relative_to(SAVE_DIR)
            )

print("=" * 100)
print("STEP 22 SAVED")
print("=" * 100)
print("Folder:", SAVE_DIR)
print("ZIP   :", ZIP_PATH)

print("\n[FILES]")
for p in sorted(SAVE_DIR.rglob("*")):
    if p.is_file():
        print("-", p.relative_to(SAVE_DIR))

print("\n[PASS] STEP 22 artifacts packaged successfully.")
