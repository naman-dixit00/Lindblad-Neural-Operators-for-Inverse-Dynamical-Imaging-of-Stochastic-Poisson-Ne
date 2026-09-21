# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 9
# Step            : STEP_01
# Step Heading    : # PHASE 1 — STEP 1: REPOSITORY + TENSOR GEOMETRY AUDIT
# Step Cell No.   : 5
# ============================================================

import os

FNO_CKPT = "/content/Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne/results/check_points/best_fno.pt"

print("Exists:", os.path.isfile(FNO_CKPT))

if os.path.isfile(FNO_CKPT):
    print("Absolute path:", FNO_CKPT)
    print("Size (MB):", os.path.getsize(FNO_CKPT) / (1024**2))
else:
    print("\nSearching for best_fno.pt...")
    for root, dirs, files in os.walk("/content"):
        if "best_fno.pt" in files:
            p = os.path.join(root, "best_fno.pt")
            print("FOUND:", p)
