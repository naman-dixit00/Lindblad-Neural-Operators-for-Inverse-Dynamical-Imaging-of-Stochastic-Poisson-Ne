# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 8
# Step            : STEP_01
# Step Heading    : # PHASE 1 — STEP 1: REPOSITORY + TENSOR GEOMETRY AUDIT
# Step Cell No.   : 4
# ============================================================

# ============================================================
# FIND ALL best_fno.pt FILES
# ============================================================

import os

print("=" * 80)
print("SEARCHING FOR best_fno.pt")
print("=" * 80)

found = []

# ------------------------------------------------------------
# Search Colab runtime
# ------------------------------------------------------------
for root, dirs, files in os.walk("/content"):
    if "best_fno.pt" in files:
        path = os.path.join(root, "best_fno.pt")
        found.append(path)

# ------------------------------------------------------------
# Search Google Drive explicitly
# ------------------------------------------------------------
drive_root = "/content/drive/MyDrive"

if os.path.exists(drive_root):

    for root, dirs, files in os.walk(drive_root):
        if "best_fno.pt" in files:
            path = os.path.join(root, "best_fno.pt")

            if path not in found:
                found.append(path)

# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------
if not found:

    print("❌ No best_fno.pt found.")

else:

    print(f"✅ Found {len(found)} file(s):\n")

    for i, path in enumerate(found, 1):

        size_mb = (
            os.path.getsize(path)
            / (1024 ** 2)
        )

        print(f"[{i}] {path}")
        print(f"    Size: {size_mb:.3f} MB")
        print()

print("=" * 80)
