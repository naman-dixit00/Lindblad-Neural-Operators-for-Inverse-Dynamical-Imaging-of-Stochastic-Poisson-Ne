# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 51
# Step            : STEP_14
# Step Heading    : # STEP 14 — REPAIRED LNO DYNAMICAL PARAMETERIZATION
# Step Cell No.   : 4
# ============================================================

import os
import shutil
from google.colab import files

# Dataset directory path
ROOT = "/content/Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
dataset_path = os.path.join(ROOT, "results", "dataset_v3_expanded")
zip_filename = "dataset_v3_expanded"

if os.path.exists(dataset_path):
    print(f"Compressing {dataset_path}...")
    # Creating the zip file
    shutil.make_archive(zip_filename, 'zip', dataset_path)

    print(f"Download starting for {zip_filename}.zip...")
    # Triggering browser download
    files.download(f"{zip_filename}.zip")
else:
    print(f"Error: Dataset path not found at {dataset_path}")
