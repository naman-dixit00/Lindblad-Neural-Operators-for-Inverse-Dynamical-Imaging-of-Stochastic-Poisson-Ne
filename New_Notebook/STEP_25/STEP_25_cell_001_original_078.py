# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 78
# Step            : STEP_25
# Step Heading    : # STEP 25 — FINAL COMPARATIVE EVIDENCE CONSOLIDATION
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 25 — FINAL COMPARATIVE EVIDENCE CONSOLIDATION
# ============================================================
#
# Purpose:
#   Consolidate already-computed evidence from:
#
#     Step 19  -> one-step test accuracy / structure
#     Step 22  -> 100-step global stability
#     Step 23  -> regime-wise 100-step stability
#     Step 24  -> visualization-derived summaries
#
# Important:
#   - No retraining
#   - No checkpoint modification
#   - No new model inference
#   - No new ground-truth benchmark
# ============================================================

import os
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. ROOT
# ============================================================

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

OUT_DIR = (
    ROOT
    / "results"
    / "step25_final_comparative_evidence"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 110)
print("STEP 25 — FINAL COMPARATIVE EVIDENCE CONSOLIDATION")
print("=" * 110)


# ============================================================
# 2. SOURCE PATHS
# ============================================================

STEP19_DIR = (
    ROOT
    / "results"
    / "step19_final_comparison"
)

STEP22_DIR = (
    ROOT
    / "results"
    / "step22_long_horizon"
)

STEP23_DIR = (
    ROOT
    / "results"
    / "step23_regime_analysis"
)

STEP24_DIR = (
    ROOT
    / "results"
    / "step24_stability_visualization"
)


# ============================================================
# 3. REQUIRED FILES
# ============================================================

STEP19_COMPARISON = (
    STEP19_DIR
    / "final_comparison.csv"
)

STEP19_REGIME = (
    STEP19_DIR
    / "final_regime_comparison.csv"
)

STEP22_SUMMARY = (
    STEP22_DIR
    / "long_horizon_summary.csv"
)

STEP23_SUMMARY = (
    STEP23_DIR
    / "regime_long_horizon_summary.csv"
)

STEP24_SUMMARY = (
    STEP24_DIR
    / "stability_visualization_summary.csv"
)

required_sources = {
    "Step19 overall comparison":
        STEP19_COMPARISON,

    "Step19 regime comparison":
        STEP19_REGIME,

    "Step22 long-horizon summary":
        STEP22_SUMMARY,

    "Step23 regime long-horizon summary":
        STEP23_SUMMARY,

    "Step24 visualization summary":
        STEP24_SUMMARY,
}

for name, path in required_sources.items():

    assert path.is_file(), (
        f"Missing {name}:\n{path}"
    )

print(
    "\n[PASS] All required evidence sources located."
)


# ============================================================
# 4. LOAD SOURCE TABLES
# ============================================================

step19_df = pd.read_csv(
    STEP19_COMPARISON
)

step19_regime_df = pd.read_csv(
    STEP19_REGIME
)

step22_df = pd.read_csv(
    STEP22_SUMMARY
)

step23_df = pd.read_csv(
    STEP23_SUMMARY
)

step24_df = pd.read_csv(
    STEP24_SUMMARY
)

print(
    "[+] Step19 rows:",
    len(step19_df)
)

print(
    "[+] Step19 regime rows:",
    len(step19_regime_df)
)

print(
    "[+] Step22 rows:",
    len(step22_df)
)

print(
    "[+] Step23 rows:",
    len(step23_df)
)

print(
    "[+] Step24 rows:",
    len(step24_df)
)


# ============================================================
# 5. INSPECT COLUMN NAMES
# ============================================================

print(
    "\n[+] Step19 columns:",
    list(step19_df.columns)
)

print(
    "[+] Step19 regime columns:",
    list(step19_regime_df.columns)
)

print(
    "[+] Step22 columns:",
    list(step22_df.columns)
)

print(
    "[+] Step23 columns:",
    list(step23_df.columns)
)


# ============================================================
# 6. OVERALL ONE-STEP ACCURACY EVIDENCE
# ============================================================

accuracy_rows = []

# ------------------------------------------------------------
# Expected canonical Step-19 values already established
# ------------------------------------------------------------

accuracy_rows.append(
    {
        "model": "FNO",
        "R_RMSE": 0.0860810036,
        "R_Relative_L2": 0.51648602,
        "Amplitude_RMSE": 0.4143533045,
        "Amplitude_Relative_L2": 0.17607665,
    }
)

accuracy_rows.append(
    {
        "model": "LNO",
        "R_RMSE": 0.0869243741,
        "R_Relative_L2": 0.52154625,
        "Amplitude_RMSE": 0.4134957790,
        "Amplitude_Relative_L2": 0.17571226,
    }
)

accuracy_df = pd.DataFrame(
    accuracy_rows
)

accuracy_lno = accuracy_df[
    accuracy_df["model"] == "LNO"
].iloc[0]

accuracy_fno = accuracy_df[
    accuracy_df["model"] == "FNO"
].iloc[0]


# ============================================================
# 7. ACCURACY RELATIVE CHANGES
# ============================================================

accuracy_change = pd.DataFrame(
    [
        {
            "metric":
                "R Relative-L2",

            "FNO":
                accuracy_fno[
                    "R_Relative_L2"
                ],

            "LNO":
                accuracy_lno[
                    "R_Relative_L2"
                ],

            "LNO_vs_FNO_percent":
                (
                    (
                        accuracy_lno[
                            "R_Relative_L2"
                        ]
                        /
                        accuracy_fno[
                            "R_Relative_L2"
                        ]
                    )
                    -
                    1.0
                )
                *
                100.0,
        },

        {
            "metric":
                "Amplitude Relative-L2",

            "FNO":
                accuracy_fno[
                    "Amplitude_Relative_L2"
                ],

            "LNO":
                accuracy_lno[
                    "Amplitude_Relative_L2"
                ],

            "LNO_vs_FNO_percent":
                (
                    (
                        accuracy_lno[
                            "Amplitude_Relative_L2"
                        ]
                        /
                        accuracy_fno[
                            "Amplitude_Relative_L2"
                        ]
                    )
                    -
                    1.0
                )
                *
                100.0,
        },
    ]
)


# ============================================================
# 8. STRUCTURAL EVIDENCE
# ============================================================

structural_rows = [
    {
        "model": "FNO",
        "test_mean_trace_error": 1.149819e-2,
        "test_minimum_eigenvalue": -7.177095e-1,
        "test_negative_eigenvalue_fraction": 0.340796,
    },

    {
        "model": "LNO",
        "test_mean_trace_error": 2.838e-8,
        "test_minimum_eigenvalue": -3.285e-7,
        "test_negative_eigenvalue_fraction": 0.183762,
    },
]

structural_df = pd.DataFrame(
    structural_rows
)

STRUCTURAL_CSV = (
    OUT_DIR
    /
    "final_structural_comparison.csv"
)

structural_df.to_csv(
    STRUCTURAL_CSV,
    index=False
)


# ============================================================
# 9. GLOBAL LONG-HORIZON EVIDENCE
# ============================================================

long_horizon_df = step22_df.copy()

LONG_HORIZON_CSV = (
    OUT_DIR
    /
    "final_long_horizon_comparison.csv"
)

long_horizon_df.to_csv(
    LONG_HORIZON_CSV,
    index=False
)


# ============================================================
# 10. REGIME-WISE LONG-HORIZON EVIDENCE
# ============================================================

regime_long_df = step23_df.copy()

REGIME_CSV = (
    OUT_DIR
    /
    "final_regime_stability_comparison.csv"
)

regime_long_df.to_csv(
    REGIME_CSV,
    index=False
)


# ============================================================
# 11. REGIME ADVANTAGE SUMMARY
# ============================================================

regime_advantage_rows = []

for regime in [
    "low_noise",
    "stochastic",
    "heavy_dissipation",
    "collapse",
    "metastable",
]:

    fno = regime_long_df[
        (
            regime_long_df["regime"]
            ==
            regime
        )
        &
        (
            regime_long_df["model"]
            ==
            "FNO"
        )
    ].iloc[0]

    lno = regime_long_df[
        (
            regime_long_df["regime"]
            ==
            regime
        )
        &
        (
            regime_long_df["model"]
            ==
            "LNO"
        )
    ].iloc[0]

    structural_better = bool(
        lno["all_finite"]
        and
        lno["completed_steps"]
        ==
        100
        and
        lno[
            "worst_max_trace_error"
        ]
        <
        fno[
            "worst_max_trace_error"
        ]
        and
        lno[
            "worst_minimum_eigenvalue"
        ]
        >
        fno[
            "worst_minimum_eigenvalue"
        ]
        and
        lno[
            "worst_psd_violation_fraction"
        ]
        <=
        fno[
            "worst_psd_violation_fraction"
        ]
    )

    regime_advantage_rows.append(
        {
            "regime":
                regime,

            "FNO_completed_steps":
                int(
                    fno[
                        "completed_steps"
                    ]
                ),

            "LNO_completed_steps":
                int(
                    lno[
                        "completed_steps"
                    ]
                ),

            "FNO_worst_trace_error":
                float(
                    fno[
                        "worst_max_trace_error"
                    ]
                ),

            "LNO_worst_trace_error":
                float(
                    lno[
                        "worst_max_trace_error"
                    ]
                ),

            "FNO_worst_minimum_eigenvalue":
                float(
                    fno[
                        "worst_minimum_eigenvalue"
                    ]
                ),

            "LNO_worst_minimum_eigenvalue":
                float(
                    lno[
                        "worst_minimum_eigenvalue"
                    ]
                ),

            "FNO_worst_psd_violation_fraction":
                float(
                    fno[
                        "worst_psd_violation_fraction"
                    ]
                ),

            "LNO_worst_psd_violation_fraction":
                float(
                    lno[
                        "worst_psd_violation_fraction"
                    ]
                ),

            "LNO_structurally_better":
                structural_better,
        }
    )


regime_advantage_df = pd.DataFrame(
    regime_advantage_rows
)


# ============================================================
# 12. FINAL EVIDENCE TABLE
# ============================================================

final_evidence_rows = [
    {
        "evidence_category":
            "One-step R accuracy",

        "FNO_value":
            float(
                accuracy_fno[
                    "R_Relative_L2"
                ]
            ),

        "LNO_value":
            float(
                accuracy_lno[
                    "R_Relative_L2"
                ]
            ),

        "interpretation":
            "Lower Relative-L2 is better.",
    },

    {
        "evidence_category":
            "One-step amplitude accuracy",

        "FNO_value":
            float(
                accuracy_fno[
                    "Amplitude_Relative_L2"
                ]
            ),

        "LNO_value":
            float(
                accuracy_lno[
                    "Amplitude_Relative_L2"
                ]
            ),

        "interpretation":
            "Lower Relative-L2 is better.",
    },

    {
        "evidence_category":
            "One-step structural trace error",

        "FNO_value":
            1.149819e-2,

        "LNO_value":
            2.838e-8,

        "interpretation":
            "Lower trace error is better.",
    },

    {
        "evidence_category":
            "One-step minimum eigenvalue",

        "FNO_value":
            -7.177095e-1,

        "LNO_value":
            -3.285e-7,

        "interpretation":
            "Higher minimum eigenvalue is better.",
    },

    {
        "evidence_category":
            "One-step meaningful PSD violation fraction",

        "FNO_value":
            0.340796,

        "LNO_value":
            0.183762,

        "interpretation":
            "Lower meaningful violation fraction is better.",
    },

    {
        "evidence_category":
            "100-step rollout completion",

        "FNO_value":
            int(
                long_horizon_df.loc[
                    long_horizon_df["model"] == "FNO",
                    "rollout_steps"
                ].max()
            ),

        "LNO_value":
            int(
                long_horizon_df.loc[
                    long_horizon_df["model"] == "LNO",
                    "rollout_steps"
                ].max()
            ),

        "interpretation":
            "Both completed the requested rollout.",
    },

    {
        "evidence_category":
            "100-step worst trace error",

        "FNO_value":
            float(
                long_horizon_df.loc[
                    long_horizon_df["model"] == "FNO",
                    "worst_max_trace_error"
                ].iloc[0]
            ),

        "LNO_value":
            float(
                long_horizon_df.loc[
                    long_horizon_df["model"] == "LNO",
                    "worst_max_trace_error"
                ].iloc[0]
            ),

        "interpretation":
            "Lower is better.",
    },

    {
        "evidence_category":
            "100-step worst minimum eigenvalue",

        "FNO_value":
            float(
                long_horizon_df.loc[
                    long_horizon_df["model"] == "FNO",
                    "worst_minimum_eigenvalue"
                ].iloc[0]
            ),

        "LNO_value":
            float(
                long_horizon_df.loc[
                    long_horizon_df["model"] == "LNO",
                    "worst_minimum_eigenvalue"
                ].iloc[0]
            ),

        "interpretation":
            "Higher is better.",
    },

    {
        "evidence_category":
            "100-step worst PSD violation",

        "FNO_value":
            float(
                long_horizon_df.loc[
                    long_horizon_df["model"] == "FNO",
                    "worst_negative_eigenvalue_fraction"
                ].iloc[0]
            ),

        "LNO_value":
            float(
                long_horizon_df.loc[
                    long_horizon_df["model"] == "LNO",
                    "worst_negative_eigenvalue_fraction"
                ].iloc[0]
            ),

        "interpretation":
            "Lower is better.",
    },
]

final_evidence_df = pd.DataFrame(
    final_evidence_rows
)

FINAL_EVIDENCE_CSV = (
    OUT_DIR
    /
    "final_evidence_summary.csv"
)

final_evidence_df.to_csv(
    FINAL_EVIDENCE_CSV,
    index=False
)

ACCURACY_CSV = (
    OUT_DIR
    /
    "final_accuracy_comparison.csv"
)

accuracy_df.to_csv(
    ACCURACY_CSV,
    index=False
)

ACCURACY_CHANGE_CSV = (
    OUT_DIR
    /
    "accuracy_relative_change.csv"
)

accuracy_change.to_csv(
    ACCURACY_CHANGE_CSV,
    index=False
)

REGIME_ADVANTAGE_CSV = (
    OUT_DIR
    /
    "regime_structural_advantage.csv"
)

regime_advantage_df.to_csv(
    REGIME_ADVANTAGE_CSV,
    index=False
)


# ============================================================
# 13. KEY COUNTS
# ============================================================

regime_advantage_count = int(
    regime_advantage_df[
        "LNO_structurally_better"
    ].sum()
)

regime_total = int(
    len(regime_advantage_df)
)


# ============================================================
# 14. JSON SUMMARY
# ============================================================

SUMMARY_JSON = (
    OUT_DIR
    /
    "final_comparative_evidence.json"
)

json_summary = {
    "step": 25,

    "title":
        "Final Comparative Evidence Consolidation",

    "diagnostic_type":
        (
            "Consolidation of independently computed "
            "accuracy, structural, and long-horizon "
            "evidence from Steps 19-24."
        ),

    "ground_truth_accuracy_claim":
        False,

    "no_retraining":
        True,

    "no_checkpoint_modification":
        True,

    "new_model_inference":
        False,

    "one_step_accuracy":
        accuracy_df.to_dict(
            orient="records"
        ),

    "one_step_structural":
        structural_df.to_dict(
            orient="records"
        ),

    "global_long_horizon":
        long_horizon_df.to_dict(
            orient="records"
        ),

    "regime_advantage_count":
        regime_advantage_count,

    "regime_total":
        regime_total,

    "all_regimes_structurally_better":
        bool(
            regime_advantage_count
            ==
            regime_total
        ),

    "outputs": {
        "accuracy":
            str(ACCURACY_CSV),

        "accuracy_relative_change":
            str(ACCURACY_CHANGE_CSV),

        "structural":
            str(STRUCTURAL_CSV),

        "long_horizon":
            str(LONG_HORIZON_CSV),

        "regime_stability":
            str(REGIME_CSV),

        "regime_advantage":
            str(REGIME_ADVANTAGE_CSV),

        "final_evidence":
            str(FINAL_EVIDENCE_CSV),

        "summary":
            str(SUMMARY_JSON),
    },
}

with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_summary,
        f,
        indent=2,
        allow_nan=True
    )


# ============================================================
# 15. FINAL DISPLAY
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "STEP 25 COMPLETE — FINAL COMPARATIVE EVIDENCE"
)

print(
    "=" * 110
)

print(
    "\n[+] One-step accuracy:"
)

display(
    accuracy_df.round(8)
)

print(
    "\n[+] One-step structural evidence:"
)

display(
    structural_df.round(8)
)

print(
    "\n[+] Global long-horizon evidence:"
)

display(
    long_horizon_df.round(8)
)

print(
    "\n[+] Regime structural advantage:"
)

display(
    regime_advantage_df
)

print(
    "\n[+] LNO structurally better in:",
    f"{regime_advantage_count}/{regime_total}",
    "regimes"
)

print(
    "\nResults:",
    OUT_DIR
)

print(
    "\n[INFO] No model retraining performed."
)

print(
    "[INFO] No checkpoint modification performed."
)

print(
    "[INFO] No new model inference performed."
)

print(
    "[INFO] Step 25 consolidates previously computed evidence."
)

print(
    "[INFO] This is not a new ground-truth accuracy benchmark."
)

print(
    "=" * 110
)
