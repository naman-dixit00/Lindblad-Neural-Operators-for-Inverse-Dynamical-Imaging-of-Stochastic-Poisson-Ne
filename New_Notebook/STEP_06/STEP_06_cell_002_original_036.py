# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 36
# Step            : STEP_06
# Step Heading    : "[+] Ready for STEP 6:"
# Step Cell No.   : 2
# ============================================================

import os

# 1. Repo directory mein enter karein
repo_path = '/content/Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne'
os.chdir(repo_path)

# 2. Git config setup karein
!git config --global user.email "namandixit0000@gmail.com"
!git config --global user.name "naman-dixit00"

# 3. Saari files add aur commit karein
!git add .
!git commit -m "Update side panel files with Step 5B results" || echo "Nothing to commit"

# 4. Authenticated Remote URL set karein
TOKEN = "[REDACTED_TOKEN]"
USERNAME = "naman-dixit00"
REPO = "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"

remote_url = f"https://{TOKEN}@github.com/{USERNAME}/{REPO}.git"
!git remote set-url origin {remote_url}

# 5. Push karein
!git push origin main

print("\n[+] Push process completed!")
