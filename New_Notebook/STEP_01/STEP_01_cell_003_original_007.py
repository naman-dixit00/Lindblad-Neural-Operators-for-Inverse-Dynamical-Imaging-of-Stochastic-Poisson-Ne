# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 7
# Step            : STEP_01
# Step Heading    : # PHASE 1 — STEP 1: REPOSITORY + TENSOR GEOMETRY AUDIT
# Step Cell No.   : 3
# ============================================================

from google.colab import drive
import os, shutil

drive.mount("/content/drive", force_remount=True)

SRC = "/content/Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne/results/check_points/best_fno.pt"

DST_DIR = "/content/drive/MyDrive/Lindblad_Neural_Operators/FNO_final"
DST = os.path.join(DST_DIR, "best_fno.pt")

os.makedirs(DST_DIR, exist_ok=True)
shutil.copy2(SRC, DST)

print("Backup exists:", os.path.isfile(DST))
print("Backup path:", DST)
