# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 56
# Step            : STEP_18
# Step Heading    : # FINAL PROJECT FILE CHECK — BEFORE STEP 18
# Step Cell No.   : 2
# ============================================================

# ============================================================
# FINAL PROJECT FILE CHECK — BEFORE STEP 18
# ============================================================

from pathlib import Path
import json
import os

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

print("=" * 110)
print("LNO PROJECT — FINAL FILE CHECK BEFORE STEP 18")
print("=" * 110)

print("\nROOT")
print(ROOT)
print("Exists:", ROOT.exists())


# ============================================================
# IMPORTANT FILES REQUIRED FOR STEP 18
# ============================================================

required = {

    "Dataset V3 Expanded":
        ROOT
        / "results"
        / "dataset_v3_expanded"
        / "environment_conditioned_lno_dataset_v3_expanded.npz",

    "Expanded Dataset Metadata":
        ROOT
        / "results"
        / "dataset_v3_expanded"
        / "dataset_v3_expanded_metadata.csv",

    "Expanded Dataset Summary":
        ROOT
        / "results"
        / "dataset_v3_expanded"
        / "dataset_v3_expanded_summary.json",

    "Step 16 Final Split NPZ":
        ROOT
        / "results"
        / "step16_final_splits"
        / "final_trajectory_level_split.npz",

    "Step 16 Trajectory Manifest":
        ROOT
        / "results"
        / "step16_final_splits"
        / "final_trajectory_manifest.csv",

    "Step 16 Transition Manifest":
        ROOT
        / "results"
        / "step16_final_splits"
        / "final_transition_manifest.csv",

    "Step 16 Summary":
        ROOT
        / "results"
        / "step16_final_splits"
        / "step16_summary.json",

    "Step 14 Repaired LNO Checkpoint":
        ROOT
        / "results"
        / "step14_repaired_lno"
        / "repaired_lno_best.pt",

    "Step 14 Summary":
        ROOT
        / "results"
        / "step14_repaired_lno"
        / "step14_summary.json",

    "Step 17 Final FNO Checkpoint":
        ROOT
        / "results"
        / "step17_final_fno"
        / "fno_final_best.pt",

    "Step 17 FNO Summary":
        ROOT
        / "results"
        / "step17_final_fno"
        / "fno_final_summary.json",

    "Step 17 FNO Training History":
        ROOT
        / "results"
        / "step17_final_fno"
        / "fno_final_training_history.csv",

    "Step 17 FNO Regime Metrics":
        ROOT
        / "results"
        / "step17_final_fno"
        / "fno_final_regime_metrics.csv",
}


# ============================================================
# PRINT STATUS
# ============================================================

print("\n" + "=" * 110)
print("REQUIRED ARTIFACTS")
print("=" * 110)

missing = []
found = []

for name, path in required.items():

    if path.is_file():

        size_mb = (
            path.stat().st_size
            /
            (1024 ** 2)
        )

        print(
            f"[FOUND]   {name:40s} | "
            f"{size_mb:.3f} MB"
        )

        found.append(name)

    else:

        print(
            f"[MISSING] {name:40s} | "
            f"{path}"
        )

        missing.append(name)


# ============================================================
# ALL STEP FOLDERS
# ============================================================

print("\n" + "=" * 110)
print("STEP FOLDERS PRESENT")
print("=" * 110)

results_dir = ROOT / "results"

step_dirs = sorted(
    [
        p
        for p in results_dir.iterdir()
        if p.is_dir()
        and p.name.startswith("step")
    ],
    key=lambda p: p.name
)

for p in step_dirs:
    print(
        "[FOUND]",
        p.name
    )


# ============================================================
# ALL CHECKPOINTS
# ============================================================

print("\n" + "=" * 110)
print("CHECKPOINT INVENTORY")
print("=" * 110)

pt_files = sorted(
    ROOT.rglob("*.pt")
)

for p in pt_files:

    size_mb = (
        p.stat().st_size
        /
        (1024 ** 2)
    )

    print(
        f"[PT] {p.relative_to(ROOT)} "
        f"| {size_mb:.3f} MB"
    )


# ============================================================
# VERIFY EXPANDED DATASET CONTENT
# ============================================================

dataset_path = required[
    "Dataset V3 Expanded"
]

print("\n" + "=" * 110)
print("EXPANDED DATASET CONTENT CHECK")
print("=" * 110)

if dataset_path.is_file():

    import numpy as np

    ds = np.load(
        dataset_path
    )

    expected_keys = [

        "X_R",
        "Y_R",

        "X_amplitude",
        "Y_amplitude",

        "X_information_z",
        "Y_information_z",

        "X_state",
        "Y_state",

        "environment",

        "delta_R_environment",
        "delta_amplitude_environment",
        "delta_state_environment",
    ]

    for key in expected_keys:

        if key in ds:

            arr = ds[key]

            print(
                f"[FOUND] {key:30s} "
                f"shape={arr.shape} "
                f"finite={bool(np.isfinite(arr).all())}"
            )

        else:

            print(
                f"[MISSING] dataset key: {key}"
            )

    assert ds["X_R"].shape[0] == 9900
    assert ds["Y_R"].shape[0] == 9900
    assert ds["environment"].shape[0] == 9900

    print(
        "\n[PASS] Expanded dataset contains 9900 transitions."
    )

else:

    print(
        "[SKIP] Expanded dataset file missing."
    )


# ============================================================
# VERIFY FINAL SPLIT CONTENT
# ============================================================

split_path = required[
    "Step 16 Final Split NPZ"
]

print("\n" + "=" * 110)
print("FINAL SPLIT CONTENT CHECK")
print("=" * 110)

if split_path.is_file():

    split = np.load(
        split_path
    )

    split_keys = [
        "train_indices",
        "val_indices",
        "test_indices",

        "X_R_train",
        "Y_R_train",

        "X_R_val",
        "Y_R_val",

        "X_R_test",
        "Y_R_test",
    ]

    for key in split_keys:

        if key in split:

            print(
                f"[FOUND] {key:25s} "
                f"shape={split[key].shape}"
            )

        else:

            print(
                f"[MISSING] split key: {key}"
            )

    print(
        "\nTrain:",
        len(split["train_indices"])
    )

    print(
        "Validation:",
        len(split["val_indices"])
    )

    print(
        "Test:",
        len(split["test_indices"])
    )

    assert len(
        split["train_indices"]
    ) == 6930

    assert len(
        split["val_indices"]
    ) == 1485

    assert len(
        split["test_indices"]
    ) == 1485

    print(
        "[PASS] Final split is 6930 / 1485 / 1485."
    )

else:

    print(
        "[MISSING] Final split NPZ."
    )

    print(
        "[INFO] Step 16 manifests are enough to reconstruct it."
    )


# ============================================================
# VERIFY FNO SUMMARY / NORMALIZATION
# ============================================================

fno_summary_path = required[
    "Step 17 FNO Summary"
]

print("\n" + "=" * 110)
print("FNO NORMALIZATION CHECK")
print("=" * 110)

if fno_summary_path.is_file():

    with open(
        fno_summary_path,
        "r"
    ) as f:

        fno_summary = json.load(
            f
        )

    normalization = (
        fno_summary.get(
            "amplitude_normalization"
        )
    )

    if normalization is not None:

        print(
            "[FOUND] FNO amplitude normalization:"
        )

        print(
            json.dumps(
                normalization,
                indent=2
            )
        )

        print(
            "[PASS] Normalization available for Step 18."
        )

    else:

        print(
            "[WARNING] FNO summary does not contain "
            "amplitude normalization."
        )

else:

    print(
        "[MISSING] FNO summary."
    )


# ============================================================
# VERIFY STEP 14 CHECKPOINT
# ============================================================

print("\n" + "=" * 110)
print("REPAIRED LNO CHECKPOINT")
print("=" * 110)

lno_ckpt = required[
    "Step 14 Repaired LNO Checkpoint"
]

if lno_ckpt.is_file():

    print(
        "[FOUND]",
        lno_ckpt
    )

    print(
        "Size:",
        f"{lno_ckpt.stat().st_size / (1024**2):.3f} MB"
    )

    print(
        "[PASS] Repaired LNO checkpoint available."
    )

else:

    print(
        "[MISSING] Repaired LNO checkpoint."
    )


# ============================================================
# FINAL READINESS
# ============================================================

print("\n" + "=" * 110)
print("STEP 18 READINESS")
print("=" * 110)

critical_names = [

    "Dataset V3 Expanded",
    "Step 14 Repaired LNO Checkpoint",
    "Step 17 Final FNO Checkpoint",
    "Step 17 FNO Summary",
]


critical_missing = [
    name
    for name in critical_names
    if name in missing
]

if not critical_missing:

    print(
        "[PASS] Critical Step-18 artifacts are present."
    )

    if (
        "Step 16 Final Split NPZ"
        not in missing
    ):

        print(
            "[PASS] Final split NPZ is also present."
        )

    else:

        print(
            "[INFO] Final split NPZ is missing, "
            "but can be reconstructed from Step-16 manifests."
        )

else:

    print(
        "[BLOCKED] Missing critical artifacts:"
    )

    for name in critical_missing:
        print(
            "   -",
            name
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 110)
print("FINAL SUMMARY")
print("=" * 110)

print(
    "Found:",
    len(found)
)

print(
    "Missing:",
    len(missing)
)

if missing:

    print(
        "\nMissing files:"
    )

    for name in missing:
        print(
            "  ❌",
            name
        )

else:

    print(
        "\n✅ All listed required artifacts are present."
    )

print("=" * 110)
