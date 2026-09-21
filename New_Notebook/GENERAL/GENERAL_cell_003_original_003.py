# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 3
# Step            : GENERAL
# Step Heading    : No step heading detected
# Step Cell No.   : 3
# ============================================================

import os

# Ensure we are in the base content directory before listing the repo_dir
current_dir = os.getcwd()
if current_dir != '/content':
    %cd /content
    print(f'Changed directory to: {os.getcwd()}')

# List the contents of the repository directory recursively
repo_dir = 'Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne'
if os.path.exists(repo_dir):
    print(f'\nListing contents of {repo_dir}:')
    !ls -R {repo_dir}
else:
    print(f"Repository directory '{repo_dir}' not found.")
