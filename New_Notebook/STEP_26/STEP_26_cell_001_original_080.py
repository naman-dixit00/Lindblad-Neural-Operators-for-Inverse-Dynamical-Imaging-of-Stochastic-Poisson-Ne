# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 80
# Step            : STEP_26
# Step Heading    : # PRE-STEP 26 — FINAL ENVIRONMENT / ARTIFACT VERIFICATION
# Step Cell No.   : 1
# ============================================================

# ======================================================================================
# PRE-STEP 26 — FINAL ENVIRONMENT / ARTIFACT VERIFICATION
# ======================================================================================
#
# PURPOSE:
#   Verify that the new Colab session has the required repository + relocated files.
#
# IMPORTANT:
#   - No model training
#   - No model inference
#   - No checkpoint modification
#   - No files are modified
#
# Expected relocated root files:
#   environment_conditioned_lno_dataset_v3_expanded.npz
#   final_trajectory_level_split.npz
#   fno_final_test_predictions.npz
#   lno_final_test_predictions.npz
# ======================================================================================

import os
from pathlib import Path
import numpy as np

CONTENT_ROOT = Path("/content")

print("=" * 110)
print("PRE-STEP 26 — FINAL ARTIFACT VERIFICATION")
print("=" * 110)


# ======================================================================================
# 1. FIND REPOSITORY ROOT AUTOMATICALLY
# ======================================================================================

repo_candidates = []

for p in CONTENT_ROOT.iterdir():

    if not p.is_dir():
        continue

    if (
        (p / "results").is_dir()
        and
        (p / "physics").is_dir()
        and
        (p / "training").is_dir()
    ):
        repo_candidates.append(p)

if len(repo_candidates) == 0:

    raise FileNotFoundError(
        "Repository root not found. Expected a directory containing "
        "'results', 'physics', and 'training'."
    )

REPO_ROOT = repo_candidates[0]

print("\n[PASS] Repository root detected:")
print(REPO_ROOT)


# ======================================================================================
# 2. AUTO-DETECT REQUIRED FILES
# ======================================================================================

REQUIRED_FILES = [

    "environment_conditioned_lno_dataset_v3_expanded.npz",
    "final_trajectory_level_split.npz",
    "fno_final_test_predictions.npz",
    "lno_final_test_predictions.npz",
]


def find_file(filename):

    # First check /content root
    direct = CONTENT_ROOT / filename

    if direct.is_file():
        return direct

    # Then search repository
    repo_matches = list(
        REPO_ROOT.rglob(filename)
    )

    if len(repo_matches) > 0:
        return repo_matches[0]

    # Finally search entire /content
    content_matches = [
        p for p in CONTENT_ROOT.rglob(filename)
        if p.is_file()
    ]

    if len(content_matches) > 0:
        return content_matches[0]

    return None


FOUND = {}

print("\n" + "=" * 110)
print("REQUIRED FILE LOCATIONS")
print("=" * 110)

for filename in REQUIRED_FILES:

    path = find_file(filename)

    if path is None:

        print("[FAIL]", filename)
        continue

    FOUND[filename] = path

    print("[PASS]", filename)
    print("      ", path)


missing = [
    f for f in REQUIRED_FILES
    if f not in FOUND
]

if missing:

    raise FileNotFoundError(
        "\nMissing required files:\n"
        + "\n".join(
            f" - {x}" for x in missing
        )
    )


# ======================================================================================
# 3. DATASET VERIFICATION
# ======================================================================================

print("\n" + "=" * 110)
print("A. EXPANDED DATASET VERIFICATION")
print("=" * 110)

dataset_path = FOUND[
    "environment_conditioned_lno_dataset_v3_expanded.npz"
]

dataset = np.load(
    dataset_path,
    allow_pickle=False
)

required_dataset_keys = [
    "X_R",
    "Y_R",
    "X_amplitude",
    "Y_amplitude",
    "environment",
]

dataset_keys = set(dataset.files)

for key in required_dataset_keys:

    assert key in dataset_keys, (
        f"Missing dataset key: {key}"
    )

    print(
        f"[PASS] Dataset key: {key}"
    )

X_R = dataset["X_R"]
Y_R = dataset["Y_R"]
X_A = dataset["X_amplitude"]
Y_A = dataset["Y_amplitude"]
ENV = dataset["environment"]

print("\nShapes:")
print("X_R          :", X_R.shape)
print("Y_R          :", Y_R.shape)
print("X_amplitude  :", X_A.shape)
print("Y_amplitude  :", Y_A.shape)
print("Environment  :", ENV.shape)

assert X_R.shape == (9900, 128, 6, 6)
assert Y_R.shape == (9900, 128, 6, 6)
assert X_A.shape == (9900, 128)
assert Y_A.shape == (9900, 128)
assert ENV.shape == (9900, 128, 2)

print("\n[PASS] Expanded dataset shapes exactly match expected 9,900-transition dataset.")

for name, arr in [
    ("X_R", X_R),
    ("Y_R", Y_R),
    ("X_amplitude", X_A),
    ("Y_amplitude", Y_A),
    ("environment", ENV),
]:

    assert np.isfinite(arr).all(), (
        f"Non-finite values detected in {name}"
    )

print("[PASS] All expanded-dataset arrays are finite.")


# ======================================================================================
# 4. FINAL TRAJECTORY SPLIT VERIFICATION
# ======================================================================================

print("\n" + "=" * 110)
print("B. FINAL TRAJECTORY SPLIT VERIFICATION")
print("=" * 110)

split_path = FOUND[
    "final_trajectory_level_split.npz"
]

split = np.load(
    split_path,
    allow_pickle=False
)

print(
    "Split keys:",
    split.files
)

required_split_keys = [
    "train_indices",
    "val_indices",
    "test_indices",
]

for key in required_split_keys:

    assert key in split.files, (
        f"Missing split key: {key}"
    )

train_idx = split["train_indices"].astype(np.int64)
val_idx = split["val_indices"].astype(np.int64)
test_idx = split["test_indices"].astype(np.int64)

print("\nSplit sizes:")
print("Train:", len(train_idx))
print("Val  :", len(val_idx))
print("Test :", len(test_idx))

assert len(train_idx) == 6930
assert len(val_idx) == 1485
assert len(test_idx) == 1485

print(
    "[PASS] Split sizes = 6930 / 1485 / 1485."
)

# No overlap
assert len(
    np.intersect1d(train_idx, val_idx)
) == 0

assert len(
    np.intersect1d(train_idx, test_idx)
) == 0

assert len(
    np.intersect1d(val_idx, test_idx)
) == 0

print(
    "[PASS] No train/validation/test transition-index overlap."
)

# Bounds
assert train_idx.min() >= 0
assert val_idx.min() >= 0
assert test_idx.min() >= 0

assert train_idx.max() < 9900
assert val_idx.max() < 9900
assert test_idx.max() < 9900

print(
    "[PASS] All split indices are within 0..9899."
)


# ======================================================================================
# 5. FNO / LNO PREDICTION FILE VERIFICATION
# ======================================================================================

print("\n" + "=" * 110)
print("C. FINAL FNO / LNO PREDICTION VERIFICATION")
print("=" * 110)

fno_path = FOUND[
    "fno_final_test_predictions.npz"
]

lno_path = FOUND[
    "lno_final_test_predictions.npz"
]

fno = np.load(
    fno_path,
    allow_pickle=False
)

lno = np.load(
    lno_path,
    allow_pickle=False
)

prediction_keys = [
    "R_prediction",
    "R_target",
    "amplitude_prediction",
    "amplitude_target",
    "test_indices",
]

for key in prediction_keys:

    assert key in fno.files, (
        f"FNO missing key: {key}"
    )

    assert key in lno.files, (
        f"LNO missing key: {key}"
    )

print(
    "[PASS] Required prediction keys present in both FNO and LNO."
)


# Extract
fno_R = fno["R_prediction"]
fno_R_target = fno["R_target"]

fno_A = fno["amplitude_prediction"]
fno_A_target = fno["amplitude_target"]

fno_test_idx = fno["test_indices"].astype(np.int64)

lno_R = lno["R_prediction"]
lno_R_target = lno["R_target"]

lno_A = lno["amplitude_prediction"]
lno_A_target = lno["amplitude_target"]

lno_test_idx = lno["test_indices"].astype(np.int64)


print("\nFNO:")
print("R prediction       :", fno_R.shape)
print("R target           :", fno_R_target.shape)
print("Amplitude prediction:", fno_A.shape)
print("Amplitude target   :", fno_A_target.shape)

print("\nLNO:")
print("R prediction       :", lno_R.shape)
print("R target           :", lno_R_target.shape)
print("Amplitude prediction:", lno_A.shape)
print("Amplitude target   :", lno_A_target.shape)

assert fno_R.shape == (1485, 128, 6, 6)
assert lno_R.shape == (1485, 128, 6, 6)

assert fno_A.shape == (1485, 128)
assert lno_A.shape == (1485, 128)

assert fno_R_target.shape == lno_R_target.shape
assert fno_A_target.shape == lno_A_target.shape

print(
    "\n[PASS] FNO/LNO prediction dimensions are correct."
)


# ======================================================================================
# 6. FNO vs LNO TEST-SET IDENTITY
# ======================================================================================

print("\n" + "=" * 110)
print("D. FNO vs LNO TEST-SET IDENTITY")
print("=" * 110)

assert np.array_equal(
    fno_test_idx,
    lno_test_idx
)

assert np.array_equal(
    fno_test_idx,
    test_idx
)

print(
    "[PASS] FNO and LNO use exactly the same 1,485 test indices."
)

max_R_target_diff = float(
    np.max(
        np.abs(
            fno_R_target -
            lno_R_target
        )
    )
)

max_A_target_diff = float(
    np.max(
        np.abs(
            fno_A_target -
            lno_A_target
        )
    )
)

print(
    "Max R target difference:",
    f"{max_R_target_diff:.12e}"
)

print(
    "Max amplitude target difference:",
    f"{max_A_target_diff:.12e}"
)

assert max_R_target_diff == 0.0
assert max_A_target_diff == 0.0

print(
    "[PASS] FNO and LNO have identical targets."
)


# ======================================================================================
# 7. FINITE CHECK
# ======================================================================================

print("\n" + "=" * 110)
print("E. FINITE-VALUE VERIFICATION")
print("=" * 110)

arrays_to_check = {

    "FNO R prediction":
        fno_R,

    "LNO R prediction":
        lno_R,

    "FNO R target":
        fno_R_target,

    "LNO R target":
        lno_R_target,

    "FNO amplitude prediction":
        fno_A,

    "LNO amplitude prediction":
        lno_A,

    "FNO amplitude target":
        fno_A_target,

    "LNO amplitude target":
        lno_A_target,
}

for name, arr in arrays_to_check.items():

    ok = bool(
        np.isfinite(arr).all()
    )

    print(
        f"{name:30s}: {ok}"
    )

    assert ok

print(
    "[PASS] All final prediction artifacts are finite."
)


# ======================================================================================
# 8. CHECK EXISTING STEP 17 / STEP 18 RESULT DIRECTORIES
# ======================================================================================

print("\n" + "=" * 110)
print("F. EXISTING FINAL RESULT DIRECTORIES")
print("=" * 110)

step17 = (
    REPO_ROOT /
    "results" /
    "step17_final_fno"
)

step18 = (
    REPO_ROOT /
    "results" /
    "step18_final_lno"
)

step19 = (
    REPO_ROOT /
    "results" /
    "step19_final_comparison"
)

step25 = (
    REPO_ROOT /
    "results" /
    "step25_final_comparative_evidence"
)

for name, path in [
    ("Step17 FINAL FNO", step17),
    ("Step18 FINAL LNO", step18),
    ("Step19 FINAL comparison", step19),
    ("Step25 FINAL consolidation", step25),
]:

    print(
        f"{name:32s}: {path.is_dir()}"
    )


# ======================================================================================
# 9. FINAL STATUS
# ======================================================================================

print("\n" + "=" * 110)
print("PRE-STEP 26 VERIFICATION COMPLETE")
print("=" * 110)

print("\n[PASS] Repository structure detected.")
print("[PASS] Expanded dataset detected and validated.")
print("[PASS] Final trajectory-level split detected and validated.")
print("[PASS] FNO final predictions detected and validated.")
print("[PASS] LNO final predictions detected and validated.")
print("[PASS] FNO/LNO test-set identity confirmed.")
print("[PASS] Targets identical.")
print("[PASS] All prediction arrays finite.")

print(
    "\n[READY] STEP 26 CAN BE RUN."
)

print(
    "\n[INFO] This verification cell did NOT train or infer anything."
)

print("=" * 110)
