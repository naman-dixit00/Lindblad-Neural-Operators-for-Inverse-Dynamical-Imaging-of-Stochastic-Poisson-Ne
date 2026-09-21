# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 67
# Step            : STEP_21
# Step Heading    : print("✅ Ready to proceed to Step 21.")
# Step Cell No.   : 3
# ============================================================

# ============================================================
# VERIFY THE TWO UPLOADED NPZ FILES INTERNALLY (REPAIRED)
# ============================================================

from pathlib import Path
import numpy as np
import os

split_file = Path("/content/final_trajectory_level_split.npz")
fno_file   = Path("/content/fno_final_test_predictions.npz")

print("=" * 100)
print("NPZ INTERNAL VERIFICATION")
print("=" * 100)

def verify_npz(file_path, label):
    print(f"\n[{label}] {file_path.name}")
    print("-" * 100)

    if not file_path.is_file():
        print(f"ERROR: File not found at {file_path}")
        return

    size = file_path.stat().st_size
    print(f"File Size: {size / (1024**2):.3f} MB")

    if size == 0:
        print("ERROR: File is empty (0 bytes).")
        return

    try:
        # verify if it's a valid ZIP/NPZ file
        with np.load(file_path, allow_pickle=True) as data:
            print("Status: VALID NPZ")
            print("Keys:")
            for k in data.files:
                arr = data[k]
                # Handle cases where arr might not be a standard numpy array
                shape = getattr(arr, 'shape', 'N/A')
                dtype = getattr(arr, 'dtype', 'N/A')
                print(f"  {k:45s} shape={shape} dtype={dtype}")
    except Exception as e:
        print(f"ERROR: Failed to load {file_path.name}.")
        print(f"Reason: {e}")
        print("\nTip: If you see 'BadZipFile', the upload was likely interrupted. Please re-upload the file.")

# Run verification
verify_npz(split_file, "1")
verify_npz(fno_file, "2")

print("\n" + "=" * 100)
print("VERIFICATION COMPLETE")
print("=" * 100)
