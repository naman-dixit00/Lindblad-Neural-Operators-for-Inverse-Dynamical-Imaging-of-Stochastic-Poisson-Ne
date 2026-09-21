# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 87
# Step            : STEP_29
# Step Heading    : # STEP 29 — FINAL C-LNO COMPREHENSIVE EVALUATION
# Step Cell No.   : 1
# ============================================================

# ======================================================================================
# STEP 29 — FINAL C-LNO COMPREHENSIVE EVALUATION
# ======================================================================================
#
# Models:
#   FNO             : Step 17
#   Final LNO       : Step 18
#   C-LNO           : Step 26 / independently validated in Step 27
#
# Evidence:
#   One-step accuracy       -> Step 27
#   Regime-wise accuracy    -> Step 27
#   C-LNO long-horizon      -> Step 28
#
# No training.
# No checkpoint modification.
# No new C-LNO inference.
# ======================================================================================

import os
import json
from pathlib import Path

import pandas as pd


# ======================================================================================
# 1. ROOT
# ======================================================================================

CONTENT_ROOT = Path("/content")

repo_candidates = []

for p in CONTENT_ROOT.iterdir():

    if (
        p.is_dir()
        and (p / "results").is_dir()
        and (p / "physics").is_dir()
        and (p / "training").is_dir()
    ):
        repo_candidates.append(p)

assert repo_candidates, "Repository root not found."

ROOT = repo_candidates[0]

os.chdir(ROOT)

print("=" * 110)
print("STEP 29 — FINAL C-LNO COMPREHENSIVE EVALUATION")
print("=" * 110)

print("[+] Root:", ROOT)


# ======================================================================================
# 2. REQUIRED STEP 27 / STEP 28 FILES
# ======================================================================================

STEP27_DIR = (
    ROOT
    / "results"
    / "step27_c_lno_independent_validation"
)

STEP28_DIR = (
    ROOT
    / "results"
    / "step28_c_lno_long_horizon"
)

assert STEP27_DIR.is_dir(), (
    f"Step 27 results not found: {STEP27_DIR}"
)

assert STEP28_DIR.is_dir(), (
    f"Step 28 results not found: {STEP28_DIR}"
)


# ======================================================================================
# 3. LOAD STEP 27
# ======================================================================================

STEP27_SUMMARY = (
    STEP27_DIR
    / "step27_summary.json"
)

STEP27_OVERALL = (
    STEP27_DIR
    / "three_model_comparison.csv"
)

STEP27_REGIME = (
    STEP27_DIR
    / "regime_comparison.csv"
)

assert STEP27_SUMMARY.is_file(), (
    f"Missing: {STEP27_SUMMARY}"
)

assert STEP27_OVERALL.is_file(), (
    f"Missing: {STEP27_OVERALL}"
)

assert STEP27_REGIME.is_file(), (
    f"Missing: {STEP27_REGIME}"
)

with open(
    STEP27_SUMMARY,
    "r",
    encoding="utf-8"
) as f:

    step27_json = json.load(f)

overall_df = pd.read_csv(
    STEP27_OVERALL
)

regime_df = pd.read_csv(
    STEP27_REGIME
)

print("\n[PASS] Step 27 evidence loaded.")


# ======================================================================================
# 4. LOAD STEP 28
# ======================================================================================

STEP28_SUMMARY = (
    STEP28_DIR
    / "step28_summary.json"
)

STEP28_GLOBAL = (
    STEP28_DIR
    / "c_lno_global_summary.csv"
)

STEP28_REGIME = (
    STEP28_DIR
    / "c_lno_regime_summary.csv"
)

assert STEP28_SUMMARY.is_file(), (
    f"Missing: {STEP28_SUMMARY}"
)

assert STEP28_GLOBAL.is_file(), (
    f"Missing: {STEP28_GLOBAL}"
)

assert STEP28_REGIME.is_file(), (
    f"Missing: {STEP28_REGIME}"
)

with open(
    STEP28_SUMMARY,
    "r",
    encoding="utf-8"
) as f:

    step28_json = json.load(f)

c_global = pd.read_csv(
    STEP28_GLOBAL
)

c_stability_df = pd.read_csv(
    STEP28_REGIME
)

print(
    "[PASS] Step 28 evidence loaded."
)


# ======================================================================================
# 5. VALIDATE STEP 27 OVERALL
# ======================================================================================

print("\n" + "=" * 110)
print("A. ONE-STEP ACCURACY")
print("=" * 110)

required_models = [
    "FNO",
    "Final_LNO",
    "Step26_C_LNO"
]

assert set(
    overall_df["model"]
) == set(
    required_models
)

accuracy_cols = [
    "model",
    "R_RMSE",
    "R_Relative_L2",
    "Amplitude_RMSE",
    "Amplitude_Relative_L2"
]

print(
    overall_df[
        accuracy_cols
    ].to_string(
        index=False
    )
)


# ======================================================================================
# 6. DETERMINE ACCURACY WINNERS
# ======================================================================================

best_R_row = overall_df.loc[
    overall_df["R_Relative_L2"].idxmin()
]

best_A_row = overall_df.loc[
    overall_df["Amplitude_Relative_L2"].idxmin()
]

best_R_model = str(
    best_R_row["model"]
)

best_A_model = str(
    best_A_row["model"]
)

print("\nR accuracy winner:")
print(
    best_R_model,
    f"({best_R_row['R_Relative_L2']:.8f})"
)

print("\nAmplitude accuracy winner:")
print(
    best_A_model,
    f"({best_A_row['Amplitude_Relative_L2']:.8f})"
)


# ======================================================================================
# 7. C-LNO IMPROVEMENT VS FNO
# ======================================================================================

fno_R = overall_df.loc[
    overall_df["model"] == "FNO",
    "R_Relative_L2"
].iloc[0]

c_R = overall_df.loc[
    overall_df["model"] == "Step26_C_LNO",
    "R_Relative_L2"
].iloc[0]

fno_A = overall_df.loc[
    overall_df["model"] == "FNO",
    "Amplitude_Relative_L2"
].iloc[0]

c_A = overall_df.loc[
    overall_df["model"] == "Step26_C_LNO",
    "Amplitude_Relative_L2"
].iloc[0]

c_R_change_vs_fno = (
    (c_R - fno_R)
    /
    fno_R
) * 100.0

c_A_change_vs_fno = (
    (c_A - fno_A)
    /
    fno_A
) * 100.0

print("\nC-LNO vs FNO")
print("-" * 110)

print(
    "R Relative-L2 change:",
    f"{c_R_change_vs_fno:.6f}%"
)

print(
    "Amplitude Relative-L2 change:",
    f"{c_A_change_vs_fno:.6f}%"
)


# ======================================================================================
# 8. REGIME-WISE R ACCURACY
# ======================================================================================

print("\n" + "=" * 110)
print("B. REGIME-WISE R ACCURACY")
print("=" * 110)

regime_accuracy_cols = [
    "regime",
    "R_Relative_L2_FNO",
    "R_Relative_L2_Final_LNO",
    "R_Relative_L2_C_LNO",
    "C_LNO_R_better_than_FNO",
    "C_LNO_R_better_than_Final_LNO"
]

print(
    regime_df[
        regime_accuracy_cols
    ].to_string(
        index=False
    )
)


c_better_fno_count = int(
    regime_df[
        "C_LNO_R_better_than_FNO"
    ].sum()
)

c_better_lno_count = int(
    regime_df[
        "C_LNO_R_better_than_Final_LNO"
    ].sum()
)

print(
    "\nC-LNO R better than FNO:",
    f"{c_better_fno_count}/5 regimes"
)

print(
    "C-LNO R better than original Final LNO:",
    f"{c_better_lno_count}/5 regimes"
)


# ======================================================================================
# 9. ONE-STEP STRUCTURAL COMPARISON
# ======================================================================================

print("\n" + "=" * 110)
print("C. ONE-STEP STRUCTURAL PRESERVATION")
print("=" * 110)

structural_cols = [
    "model",
    "Mean_trace_error",
    "Minimum_eigenvalue",
    "Negative_eigenvalue_fraction"
]

print(
    overall_df[
        structural_cols
    ].to_string(
        index=False
    )
)

c_structural_ok = bool(
    float(
        overall_df.loc[
            overall_df["model"] == "Step26_C_LNO",
            "Negative_eigenvalue_fraction"
        ].iloc[0]
    ) == 0.0
    and
    float(
        overall_df.loc[
            overall_df["model"] == "Step26_C_LNO",
            "Mean_trace_error"
        ].iloc[0]
    ) < 1e-5
)


# ======================================================================================
# 10. STEP 28 LONG-HORIZON
# ======================================================================================

print("\n" + "=" * 110)
print("D. C-LNO 100-STEP LONG-HORIZON")
print("=" * 110)

print(
    c_global.to_string(
        index=False
    )
)

c_long_horizon_finite = bool(
    c_global["all_finite"].iloc[0]
)

c_long_horizon_psd = bool(
    float(
        c_global[
            "worst_negative_eigenvalue_fraction"
        ].iloc[0]
    ) == 0.0
)

c_long_horizon_trace = bool(
    float(
        c_global[
            "worst_max_trace_error"
        ].iloc[0]
    ) < 1e-5
)


# ======================================================================================
# 11. REGIME-WISE STABILITY
# ======================================================================================

print("\n" + "=" * 110)
print("E. C-LNO REGIME-WISE STABILITY")
print("=" * 110)

stability_cols = [
    "regime",
    "completed_steps",
    "worst_minimum_eigenvalue",
    "worst_max_trace_error",
    "worst_psd_violation_fraction",
    "all_finite"
]

print(
    c_stability_df[
        stability_cols
    ].to_string(
        index=False
    )
)


regime_stability_pass = bool(
    (
        c_stability_df[
            "completed_steps"
        ] == 100
    ).all()
    and
    (
        c_stability_df[
            "worst_psd_violation_fraction"
        ] == 0.0
    ).all()
    and
    (
        c_stability_df[
            "worst_max_trace_error"
        ] < 1e-5
    ).all()
    and
    c_stability_df[
        "all_finite"
    ].all()
)


# ======================================================================================
# 12. FINAL SCIENTIFIC STATUS
# ======================================================================================

print("\n" + "=" * 110)
print("STEP 29 — FINAL SCIENTIFIC STATUS")
print("=" * 110)

print(
    "\nR accuracy winner:",
    best_R_model
)

print(
    "Amplitude accuracy winner:",
    best_A_model
)

print(
    "C-LNO beats FNO in R:",
    f"{c_better_fno_count}/5 regimes"
)

print(
    "C-LNO beats original Final LNO in R:",
    f"{c_better_lno_count}/5 regimes"
)

print(
    "C-LNO one-step structure valid:",
    c_structural_ok
)

print(
    "C-LNO long-horizon finite:",
    c_long_horizon_finite
)

print(
    "C-LNO long-horizon PSD stable:",
    c_long_horizon_psd
)

print(
    "C-LNO long-horizon trace stable:",
    c_long_horizon_trace
)

print(
    "C-LNO regime-wise stability:",
    regime_stability_pass
)


# ======================================================================================
# 13. FINAL CANDIDATE FLAG
# ======================================================================================

C_FINAL_CANDIDATE = bool(
    best_R_model == "Step26_C_LNO"
    and
    c_better_fno_count == 5
    and
    c_structural_ok
    and
    c_long_horizon_finite
    and
    c_long_horizon_psd
    and
    c_long_horizon_trace
    and
    regime_stability_pass
)

print(
    "\nC-LNO FINAL-CANDIDATE FLAG:",
    C_FINAL_CANDIDATE
)


# ======================================================================================
# 14. SAVE STEP 29
# ======================================================================================

OUT_DIR = (
    ROOT
    / "results"
    / "step29_final_c_lno_evaluation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

overall_df.to_csv(
    OUT_DIR / "final_three_model_accuracy.csv",
    index=False
)

regime_df.to_csv(
    OUT_DIR / "final_regime_accuracy.csv",
    index=False
)

c_stability_df.to_csv(
    OUT_DIR / "final_c_lno_regime_stability.csv",
    index=False
)

c_global.to_csv(
    OUT_DIR / "final_c_lno_global_stability.csv",
    index=False
)


# ======================================================================================
# 15. FINAL JSON
# ======================================================================================

final_json = {

    "step": 29,

    "title":
        "Final C-LNO Comprehensive Evaluation",

    "training_performed":
        False,

    "checkpoint_modified":
        False,

    "new_inference_performed":
        False,

    "test_transitions":
        1485,

    "models":
        [
            "FNO",
            "Final_LNO",
            "Step26_C_LNO"
        ],

    "accuracy":
        {
            "R_winner":
                best_R_model,

            "Amplitude_winner":
                best_A_model,

            "C_LNO_R_change_vs_FNO_percent":
                float(
                    c_R_change_vs_fno
                ),

            "C_LNO_Amplitude_change_vs_FNO_percent":
                float(
                    c_A_change_vs_fno
                ),

            "C_LNO_R_regime_wins_vs_FNO":
                c_better_fno_count,

            "C_LNO_R_regime_wins_vs_Final_LNO":
                c_better_lno_count
        },

    "structure":
        {
            "one_step_valid":
                c_structural_ok,

            "long_horizon_finite":
                c_long_horizon_finite,

            "long_horizon_psd_stable":
                c_long_horizon_psd,

            "long_horizon_trace_stable":
                c_long_horizon_trace,

            "regime_wise_stability":
                regime_stability_pass
        },

    "final_candidate":
        C_FINAL_CANDIDATE,

    "outputs":
        {
            "accuracy":
                str(
                    OUT_DIR /
                    "final_three_model_accuracy.csv"
                ),

            "regime_accuracy":
                str(
                    OUT_DIR /
                    "final_regime_accuracy.csv"
                ),

            "regime_stability":
                str(
                    OUT_DIR /
                    "final_c_lno_regime_stability.csv"
                ),

            "global_stability":
                str(
                    OUT_DIR /
                    "final_c_lno_global_stability.csv"
                ),

            "summary_json":
                str(
                    OUT_DIR /
                    "step29_final_summary.json"
                )
        }
}


with open(
    OUT_DIR / "step29_final_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_json,
        f,
        indent=2
    )


# ======================================================================================
# 16. FINAL STATUS
# ======================================================================================

print("\n")
print("=" * 110)
print("STEP 29 COMPLETE")
print("=" * 110)

print(
    "Results:",
    OUT_DIR
)

print(
    "\n[INFO] No model retraining performed."
)

print(
    "[INFO] No checkpoint modified."
)

print(
    "[INFO] No new C-LNO inference performed."
)

print(
    "[INFO] Step 29 consolidates independently validated evidence."
)

print("=" * 110)
