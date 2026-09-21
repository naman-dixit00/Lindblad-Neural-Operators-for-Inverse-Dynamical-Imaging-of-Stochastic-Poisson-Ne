# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 66
# Step            : STEP_21
# Step Heading    : print("✅ Ready to proceed to Step 21.")
# Step Cell No.   : 2
# ============================================================

# ============================================================
# VERIFY UPLOADED LARGE ARTIFACTS
# final_trajectory_level_split.npz
# fno_final_test_predictions.npz
# ============================================================

from pathlib import Path
import os

# ------------------------------------------------------------
# 1. Repository root
# ------------------------------------------------------------
REPO = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

print("=" * 100)
print("LNO PROJECT — LARGE ARTIFACT VERIFICATION")
print("=" * 100)

print("\nRepository:")
print(REPO)
print("Exists:", REPO.is_dir())

if not REPO.is_dir():
    raise FileNotFoundError("Repository not found.")

# ------------------------------------------------------------
# 2. Expected locations
# ------------------------------------------------------------
expected = {
    "final_trajectory_level_split.npz":
        REPO / "results" / "step16_final_splits"
        / "final_trajectory_level_split.npz",

    "fno_final_test_predictions.npz":
        REPO / "results" / "step17_final_fno"
        / "fno_final_test_predictions.npz",
}

# ------------------------------------------------------------
# 3. Basic check
# ------------------------------------------------------------
print("\n" + "=" * 100)
print("EXPECTED PATH CHECK")
print("=" * 100)

for name, path in expected.items():
    print(f"\n{name}")
    print("-" * 100)
    print("Expected path:")
    print(path)

    if path.is_file():
        size_mb = path.stat().st_size / (1024 ** 2)
        print("STATUS : [FOUND]")
        print(f"SIZE   : {size_mb:.3f} MB")
    else:
        print("STATUS : [MISSING]")

# ------------------------------------------------------------
# 4. Search whole /content for duplicate copies
# ------------------------------------------------------------
print("\n" + "=" * 100)
print("SEARCHING /content FOR ALL COPIES")
print("=" * 100)

target_names = {
    "final_trajectory_level_split.npz",
    "fno_final_test_predictions.npz",
}

found = {name: [] for name in target_names}

for p in Path("/content").rglob("*"):
    if p.is_file() and p.name in target_names:
        found[p.name].append(p)

for name in target_names:
    print(f"\n{name}")
    print("-" * 100)

    if not found[name]:
        print("[NO COPY FOUND]")
        continue

    for p in found[name]:
        size_mb = p.stat().st_size / (1024 ** 2)
        print(f"[FOUND] {p}")
        print(f"        Size: {size_mb:.3f} MB")

# ------------------------------------------------------------
# 5. Verify required parent folders
# ------------------------------------------------------------
print("\n" + "=" * 100)
print("REQUIRED FOLDER CHECK")
print("=" * 100)

folders = [
    REPO / "results" / "step16_final_splits",
    REPO / "results" / "step17_final_fno",
    REPO / "results" / "step18_final_lno",
    REPO / "results" / "step19_final_comparison",
    REPO / "results" / "step20_ood_sigma",
    REPO / "results" / "dataset_v2",
    REPO / "results" / "dataset_v3",
    REPO / "results" / "dataset_v3_expanded",
]

for folder in folders:
    print(
        f"[{'FOUND' if folder.is_dir() else 'MISSING':7s}] "
        f"{folder}"
    )

# ------------------------------------------------------------
# 6. Final verdict
# ------------------------------------------------------------
print("\n" + "=" * 100)
print("FINAL VERDICT")
print("=" * 100)

all_expected_ok = all(path.is_file() for path in expected.values())

if all_expected_ok:
    print("✅ BOTH LARGE FILES ARE IN THE CORRECT REPOSITORY PATHS.")
    print("✅ Ready to proceed to Step 21.")
else:
    print("⚠️ One or more large files are NOT in the expected repository path.")
    print("Do NOT start Step 21 yet.")
    print("Use the 'SEARCHING /content FOR ALL COPIES' section above to locate them.")

print("=" * 100)
