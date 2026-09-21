# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 2
# Step            : GENERAL
# Step Heading    : No step heading detected
# Step Cell No.   : 2
# ============================================================

import os

# Change to the cloned repository directory
repo_dir = 'Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne'
if os.path.exists(repo_dir):
    %cd {repo_dir}
    print(f'Changed directory to: {os.getcwd()}')

    # Install dependencies from requirements.txt
    print('\nInstalling dependencies...')
    !pip install -r requirements.txt

    # Add the current directory to Python path to enable local imports
    import sys
    sys.path.insert(0, os.getcwd())

    # Try to import a module to verify loading
    print('\nVerifying module import...')
    try:
        from models import lno_model
        print("Successfully imported 'lno_model' from 'models'. Files and code appear to be loaded.")
    except ImportError as e:
        print(f"Failed to import a module. Error: {e}")
        print("This might indicate an issue with dependency installation or module paths.")

    # Change back to the original directory (optional, but good practice)
    %cd ..
    print(f'Changed back to original directory: {os.getcwd()}')

else:
    print(f"Repository directory '{repo_dir}' not found. Please ensure it was cloned successfully.")
