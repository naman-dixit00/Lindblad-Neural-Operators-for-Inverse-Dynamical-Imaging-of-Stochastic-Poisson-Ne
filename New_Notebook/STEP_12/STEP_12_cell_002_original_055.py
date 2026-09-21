# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 55
# Step            : STEP_12
# Step Heading    : # PROJECT ARTIFACT INVENTORY — STEP 12 ONWARD
# Step Cell No.   : 2
# ============================================================

# ============================================================
# PROJECT ARTIFACT INVENTORY — STEP 12 ONWARD
# ============================================================

from pathlib import Path

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

print("=" * 110)
print("LNO PROJECT — ARTIFACT INVENTORY")
print("=" * 110)

print("\nROOT:")
print(ROOT)
print("Exists:", ROOT.exists())


# ============================================================
# 1. ALL STEP FOLDERS
# ============================================================

print("\n" + "=" * 110)
print("STEP FOLDERS")
print("=" * 110)

step_dirs = sorted(
    [
        p for p in (ROOT / "results").iterdir()
        if p.is_dir() and p.name.startswith("step")
    ],
    key=lambda p: p.name
)

if not step_dirs:
    print("[NONE] No step folders found.")
else:
    for p in step_dirs:
        print(f"[FOUND] {p.name}")


# ============================================================
# 2. DETAILED FILE LIST
# ============================================================

print("\n" + "=" * 110)
print("DETAILED FILE INVENTORY")
print("=" * 110)

for step_dir in step_dirs:

    print("\n" + "-" * 100)
    print(f"FOLDER: {step_dir.name}")
    print("-" * 100)

    files = sorted(
        [
            p for p in step_dir.rglob("*")
            if p.is_file()
        ],
        key=lambda p: str(p.relative_to(step_dir))
    )

    if not files:
        print("  [EMPTY]")
        continue

    for f in files:

        size_mb = (
            f.stat().st_size
            /
            (1024 ** 2)
        )

        print(
            f"  [FILE] "
            f"{f.relative_to(step_dir)} "
            f"({size_mb:.3f} MB)"
        )


# ============================================================
# 3. IMPORTANT DATASET FILES
# ============================================================

print("\n" + "=" * 110)
print("IMPORTANT DATASET / SPLIT FILES")
print("=" * 110)

important_files = [

    ROOT
    / "results"
    / "dataset_v3"
    / "environment_conditioned_lno_dataset_v3.npz",

    ROOT
    / "results"
    / "dataset_v3_expanded"
    / "environment_conditioned_lno_dataset_v3_expanded.npz",

    ROOT
    / "results"
    / "step16_final_splits"
    / "final_trajectory_level_split.npz",

    ROOT
    / "results"
    / "step16_final_splits"
    / "final_trajectory_manifest.csv",

    ROOT
    / "results"
    / "step16_final_splits"
    / "final_transition_manifest.csv",
]

for f in important_files:

    if f.is_file():

        size_mb = (
            f.stat().st_size
            /
            (1024 ** 2)
        )

        print(
            f"[FOUND] {f.relative_to(ROOT)} "
            f"({size_mb:.3f} MB)"
        )

    else:

        print(
            f"[MISSING] {f.relative_to(ROOT)}"
        )


# ============================================================
# 4. CHECKPOINT INVENTORY
# ============================================================

print("\n" + "=" * 110)
print("ALL CHECKPOINTS (.pt)")
print("=" * 110)

checkpoints = sorted(
    ROOT.rglob("*.pt"),
    key=lambda p: str(p)
)

if not checkpoints:

    print("[NONE] No .pt files found.")

else:

    for f in checkpoints:

        size_mb = (
            f.stat().st_size
            /
            (1024 ** 2)
        )

        print(
            f"[PT] {f.relative_to(ROOT)} "
            f"({size_mb:.3f} MB)"
        )


# ============================================================
# 5. EXPECTED ARTIFACTS FOR NEXT STEPS
#
# Names can be adjusted later if we change the exact output
# convention, but this gives us an immediate inventory.
# ============================================================

expected_next = {

    "STEP 18 — FINAL REPAIRED LNO TRAINING": [

        "results/step18_final_lno/"
        "lno_final_best.pt",

        "results/step18_final_lno/"
        "lno_final_summary.json",

        "results/step18_final_lno/"
        "lno_final_training_history.csv",

        "results/step18_final_lno/"
        "lno_final_regime_metrics.csv",

        "results/step18_final_lno/"
        "lno_final_test_predictions.npz",
    ],

    "STEP 19 — FINAL FNO VS LNO COMPARISON": [

        "results/step19_final_comparison/"
        "final_comparison.csv",

        "results/step19_final_comparison/"
        "final_comparison.json",
    ],

    "STEP 20 — OOD SIGMA SWEEP": [

        "results/step20_ood_sigma/"
        "ood_sigma_results.csv",

        "results/step20_ood_sigma/"
        "ood_sigma_summary.json",
    ],

    "STEP 21 — COLLAPSE STRESS TEST": [

        "results/step21_collapse_stress/"
        "collapse_stress_results.csv",

        "results/step21_collapse_stress/"
        "collapse_stress_summary.json",
    ],

    "STEP 22 — LONG-HORIZON ROLLOUT": [

        "results/step22_long_horizon/"
        "long_horizon_results.csv",

        "results/step22_long_horizon/"
        "long_horizon_summary.json",
    ],

    "STEP 23 — SPECTRAL STABILITY": [

        "results/step23_spectral_stability/"
        "spectral_stability.csv",

        "results/step23_spectral_stability/"
        "spectral_stability_summary.json",
    ],

    "FINAL FIGURES / TABLES": [

        "results/final_figures/",
        "results/final_tables/",
    ],
}


print("\n" + "=" * 110)
print("NEXT-STEPS ARTIFACT AVAILABILITY")
print("=" * 110)

for section, paths in expected_next.items():

    print("\n" + section)
    print("-" * 90)

    for rel in paths:

        path = ROOT / rel

        if path.is_file():

            print(
                f"[FOUND FILE]    {rel}"
            )

        elif path.is_dir():

            print(
                f"[FOUND FOLDER]  {rel}"
            )

        else:

            print(
                f"[MISSING]       {rel}"
            )


# ============================================================
# 6. QUICK SUMMARY
# ============================================================

print("\n" + "=" * 110)
print("QUICK SUMMARY")
print("=" * 110)

all_files = [
    p for p in ROOT.rglob("*")
    if p.is_file()
]

print(
    "Total files:",
    len(all_files)
)

print(
    "Total checkpoints (.pt):",
    len(checkpoints)
)

print(
    "Step folders:",
    len(step_dirs)
)

print(
    "\n[COMPLETE] Files currently present in the Colab"
)

print(
    "filesystem have been enumerated above."
)

print("=" * 110)
