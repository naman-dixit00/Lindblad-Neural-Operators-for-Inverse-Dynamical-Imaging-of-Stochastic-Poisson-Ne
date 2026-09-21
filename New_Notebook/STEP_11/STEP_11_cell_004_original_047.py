# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 47
# Step            : STEP_11
# Step Heading    : # RECOVER STEP-11 LNO CHECKPOINT FROM GITHUB
# Step Cell No.   : 4
# ============================================================

import os

# Repository का रूट पाथ
ROOT = "/content/Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
dataset_v3_path = os.path.join(ROOT, "results", "dataset_v3")

os.chdir(ROOT)

print(f"Current directory: {os.getcwd()}")
print(f"Dataset V3 path to commit: {dataset_v3_path}")

# 1. Git कॉन्फिग सेटअप करें (यदि पहले से नहीं किया गया है)
!git config --global user.email "namandixit0000@gmail.com"
!git config --global user.name "naman-dixit00"

# 2. Dataset V3 फ़ाइलों को जोड़ें (add)
# सुनिश्चित करें कि git add में पूरा directory शामिल हो
!git add {dataset_v3_path}

# 3. परिवर्तनों को कमिट करें
commit_message = "Add generated Dataset V3 files"
!git commit -m "{commit_message}" || echo "Nothing to commit (or files already added)"

# 4. GitHub प्रमाणन (authentication) के लिए टोकन का उपयोग करें
# यह टोकन पिछले 'git push' कमांड से लिया गया है।
TOKEN = "[REDACTED_TOKEN]" # अपने टोकन से बदलें
USERNAME = "naman-dixit00"
REPO = "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"

remote_url = f"https://{TOKEN}@github.com/{USERNAME}/{REPO}.git"

# 5. परिवर्तनों को GitHub पर पुश करें
print("\n[+] Pushing changes to GitHub...")
!git push {remote_url} main

print("\n[+] Dataset V3 files successfully committed and pushed!")
