# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 59
# Step            : STEP_19
# Step Heading    : # STEP 19 — FINAL FNO vs LNO COMPARISON
# Step Cell No.   : 2
# ============================================================

# ============================================================
# STEP 19 — FINAL FNO vs LNO COMPARISON
# ============================================================
#
# FINAL APPLES-TO-APPLES COMPARISON
#
# FNO:
#   Step 17 — Final FNO baseline
#
# LNO:
#   Step 18 — Final Repaired LNO
#
# SAME:
#   Dataset V3 Expanded
#   9,900 transitions
#   6,930 train
#   1,485 validation
#   1,485 test
#   Same test indices
#   Same target
#   Same amplitude normalization
#
# IMPORTANT:
#   No retraining.
#   Only evaluation / comparison.
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 2. ROOT
# ============================================================

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

print("=" * 110)
print("STEP 19 — FINAL FNO vs LNO COMPARISON")
print("=" * 110)


# ============================================================
# 3. DIRECTORIES
# ============================================================

FNO_DIR = (
    ROOT
    / "results"
    / "step17_final_fno"
)

LNO_DIR = (
    ROOT
    / "results"
    / "step18_final_lno"
)

STEP16_DIR = (
    ROOT
    / "results"
    / "step16_final_splits"
)

OUT_DIR = (
    ROOT
    / "results"
    / "step19_final_comparison"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. INPUT FILES
# ============================================================

FNO_SUMMARY_PATH = (
    FNO_DIR
    / "fno_final_summary.json"
)

LNO_SUMMARY_PATH = (
    LNO_DIR
    / "lno_final_summary.json"
)

FNO_PRED_PATH = (
    FNO_DIR
    / "fno_final_test_predictions.npz"
)

LNO_PRED_PATH = (
    LNO_DIR
    / "lno_final_test_predictions.npz"
)

FNO_REGIME_PATH = (
    FNO_DIR
    / "fno_final_regime_metrics.csv"
)

LNO_REGIME_PATH = (
    LNO_DIR
    / "lno_final_regime_metrics.csv"
)

SPLIT_PATH = (
    STEP16_DIR
    / "final_trajectory_level_split.npz"
)

TRANSITION_MANIFEST_PATH = (
    STEP16_DIR
    / "final_transition_manifest.csv"
)

TRAJECTORY_MANIFEST_PATH = (
    STEP16_DIR
    / "final_trajectory_manifest.csv"
)


# ============================================================
# 5. REQUIRED FILE CHECK
# ============================================================

required_files = {

    "FNO summary":
        FNO_SUMMARY_PATH,

    "LNO summary":
        LNO_SUMMARY_PATH,

    "FNO test predictions":
        FNO_PRED_PATH,

    "LNO test predictions":
        LNO_PRED_PATH,

    "FNO regime metrics":
        FNO_REGIME_PATH,

    "LNO regime metrics":
        LNO_REGIME_PATH,

    "Step16 final split":
        SPLIT_PATH,

    "Step16 transition manifest":
        TRANSITION_MANIFEST_PATH,

    "Step16 trajectory manifest":
        TRAJECTORY_MANIFEST_PATH,
}

print(
    "\n" + "=" * 110
)

print(
    "A. REQUIRED ARTIFACT CHECK"
)

print(
    "=" * 110
)

missing = []

for name, path in required_files.items():

    exists = path.is_file()

    print(
        f"{name:30s}: {exists}"
    )

    if not exists:

        missing.append(
            str(path)
        )

if missing:

    raise FileNotFoundError(
        "\nMissing required Step-19 artifacts:\n"
        +
        "\n".join(
            missing
        )
    )

print(
    "[PASS] All required Step-19 artifacts found."
)


# ============================================================
# 6. LOAD JSON SUMMARIES
# ============================================================

with open(
    FNO_SUMMARY_PATH,
    "r"
) as f:

    fno_summary = json.load(
        f
    )

with open(
    LNO_SUMMARY_PATH,
    "r"
) as f:

    lno_summary = json.load(
        f
    )

print(
    "\n[PASS] FNO and LNO summaries loaded."
)


# ============================================================
# 7. LOAD TEST PREDICTIONS
# ============================================================

fno_pred = np.load(
    FNO_PRED_PATH
)

lno_pred = np.load(
    LNO_PRED_PATH
)

print(
    "[PASS] Test prediction files loaded."
)


# ============================================================
# 8. REQUIRED PREDICTION KEYS
# ============================================================

prediction_keys = [

    "R_prediction",
    "R_target",

    "amplitude_prediction",
    "amplitude_target",

    "environment",
    "test_indices",
]

for key in prediction_keys:

    if key not in fno_pred:

        raise KeyError(
            f"FNO prediction file missing key: {key}"
        )

    if key not in lno_pred:

        raise KeyError(
            f"LNO prediction file missing key: {key}"
        )

print(
    "[PASS] Required prediction keys present."
)


# ============================================================
# 9. EXTRACT ARRAYS
# ============================================================

fno_test_indices = (
    fno_pred[
        "test_indices"
    ]
    .astype(
        np.int64
    )
)

lno_test_indices = (
    lno_pred[
        "test_indices"
    ]
    .astype(
        np.int64
    )
)

fno_R_pred = (
    fno_pred[
        "R_prediction"
    ]
    .astype(
        np.float64
    )
)

lno_R_pred = (
    lno_pred[
        "R_prediction"
    ]
    .astype(
        np.float64
    )
)

fno_R_target = (
    fno_pred[
        "R_target"
    ]
    .astype(
        np.float64
    )
)

lno_R_target = (
    lno_pred[
        "R_target"
    ]
    .astype(
        np.float64
    )
)

fno_A_pred = (
    fno_pred[
        "amplitude_prediction"
    ]
    .astype(
        np.float64
    )
)

lno_A_pred = (
    lno_pred[
        "amplitude_prediction"
    ]
    .astype(
        np.float64
    )
)

fno_A_target = (
    fno_pred[
        "amplitude_target"
    ]
    .astype(
        np.float64
    )
)

lno_A_target = (
    lno_pred[
        "amplitude_target"
    ]
    .astype(
        np.float64
    )
)

fno_environment = (
    fno_pred[
        "environment"
    ]
    .astype(
        np.float64
    )
)

lno_environment = (
    lno_pred[
        "environment"
    ]
    .astype(
        np.float64
    )
)


# ============================================================
# 10. SAME TEST SET CHECK
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "B. TEST-SET IDENTITY CHECK"
)

print(
    "=" * 110
)

assert np.array_equal(
    fno_test_indices,
    lno_test_indices
)

print(
    "[PASS] FNO and LNO use identical test indices."
)

print(
    "Test transitions:",
    len(
        fno_test_indices
    )
)

assert len(
    fno_test_indices
) == 1485


# ============================================================
# 11. SAME TARGET CHECK
# ============================================================

assert (
    fno_R_target.shape
    ==
    lno_R_target.shape
)

assert (
    fno_A_target.shape
    ==
    lno_A_target.shape
)

R_target_difference = float(
    np.max(
        np.abs(
            fno_R_target
            -
            lno_R_target
        )
    )
)

A_target_difference = float(
    np.max(
        np.abs(
            fno_A_target
            -
            lno_A_target
        )
    )
)

assert np.allclose(
    fno_R_target,
    lno_R_target,
    atol=1e-7,
    rtol=1e-6
)

assert np.allclose(
    fno_A_target,
    lno_A_target,
    atol=1e-5,
    rtol=1e-6
)

print(
    "Max R target difference:",
    f"{R_target_difference:.12e}"
)

print(
    "Max amplitude target difference:",
    f"{A_target_difference:.12e}"
)

print(
    "[PASS] FNO and LNO have identical targets."
)


# ============================================================
# 12. SAME ENVIRONMENT CHECK
# ============================================================

assert (
    fno_environment.shape
    ==
    lno_environment.shape
)

environment_difference = float(
    np.max(
        np.abs(
            fno_environment
            -
            lno_environment
        )
    )
)

assert np.allclose(
    fno_environment,
    lno_environment,
    atol=1e-7,
    rtol=1e-6
)

print(
    "Max environment difference:",
    f"{environment_difference:.12e}"
)

print(
    "[PASS] Environment conditioning is identical."
)


# ============================================================
# 13. SHAPE CHECK
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "C. PREDICTION SHAPES"
)

print(
    "=" * 110
)

print(
    "FNO R:",
    fno_R_pred.shape
)

print(
    "LNO R:",
    lno_R_pred.shape
)

print(
    "FNO amplitude:",
    fno_A_pred.shape
)

print(
    "LNO amplitude:",
    lno_A_pred.shape
)

assert (
    fno_R_pred.shape
    ==
    lno_R_pred.shape
    ==
    fno_R_target.shape
)

assert (
    fno_A_pred.shape
    ==
    lno_A_pred.shape
    ==
    fno_A_target.shape
)

print(
    "[PASS] FNO/LNO prediction dimensions match."
)


# ============================================================
# 14. FINITE CHECK
# ============================================================

all_prediction_arrays = {

    "FNO R prediction":
        fno_R_pred,

    "LNO R prediction":
        lno_R_pred,

    "FNO R target":
        fno_R_target,

    "LNO R target":
        lno_R_target,

    "FNO amplitude prediction":
        fno_A_pred,

    "LNO amplitude prediction":
        lno_A_pred,

    "FNO amplitude target":
        fno_A_target,

    "LNO amplitude target":
        lno_A_target,
}

print(
    "\n" + "=" * 110
)

print(
    "D. FINITE-VALUE CHECK"
)

print(
    "=" * 110
)

for name, arr in all_prediction_arrays.items():

    finite = bool(
        np.isfinite(
            arr
        ).all()
    )

    print(
        f"{name:32s}:",
        finite
    )

    assert finite

print(
    "[PASS] All prediction and target arrays are finite."
)


# ============================================================
# 15. METRIC FUNCTIONS
# ============================================================

def calculate_metrics(
    prediction,
    target
):

    error = (
        prediction
        -
        target
    )

    mse = float(
        np.mean(
            error ** 2
        )
    )

    rmse = float(
        np.sqrt(
            mse
        )
    )

    mae = float(
        np.mean(
            np.abs(
                error
            )
        )
    )

    numerator = np.linalg.norm(
        error
    )

    denominator = (
        np.linalg.norm(
            target
        )
        +
        1e-12
    )

    relative_l2 = float(
        numerator
        /
        denominator
    )

    return {

        "MSE":
            mse,

        "RMSE":
            rmse,

        "MAE":
            mae,

        "Relative-L2":
            relative_l2,
    }


def percentage_change(
    new_value,
    old_value
):

    return float(
        (
            new_value
            -
            old_value
        )
        /
        (
            abs(
                old_value
            )
            +
            1e-12
        )
        *
        100.0
    )


# ============================================================
# 16. CALCULATE OVERALL METRICS
# ============================================================

fno_R_metrics = calculate_metrics(
    fno_R_pred,
    fno_R_target
)

lno_R_metrics = calculate_metrics(
    lno_R_pred,
    lno_R_target
)

fno_A_metrics = calculate_metrics(
    fno_A_pred,
    fno_A_target
)

lno_A_metrics = calculate_metrics(
    lno_A_pred,
    lno_A_target
)


# ============================================================
# 17. OVERALL COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame(
    [

        {

            "model":
                "FNO",

            "R_MSE":
                fno_R_metrics[
                    "MSE"
                ],

            "R_RMSE":
                fno_R_metrics[
                    "RMSE"
                ],

            "R_MAE":
                fno_R_metrics[
                    "MAE"
                ],

            "R_Relative_L2":
                fno_R_metrics[
                    "Relative-L2"
                ],

            "Amplitude_MSE":
                fno_A_metrics[
                    "MSE"
                ],

            "Amplitude_RMSE":
                fno_A_metrics[
                    "RMSE"
                ],

            "Amplitude_MAE":
                fno_A_metrics[
                    "MAE"
                ],

            "Amplitude_Relative_L2":
                fno_A_metrics[
                    "Relative-L2"
                ],

            "Mean_trace_error":
                float(
                    fno_summary[
                        "structural_diagnostics"
                    ][
                        "mean_trace_error"
                    ]
                ),

            "Max_trace_error":
                float(
                    fno_summary[
                        "structural_diagnostics"
                    ][
                        "max_trace_error"
                    ]
                ),

            "Minimum_eigenvalue":
                float(
                    fno_summary[
                        "structural_diagnostics"
                    ][
                        "minimum_eigenvalue"
                    ]
                ),

            "Negative_eigenvalue_fraction":
                float(
                    fno_summary[
                        "structural_diagnostics"
                    ][
                        "negative_eigenvalue_fraction"
                    ]
                ),
        },


        {

            "model":
                "LNO",

            "R_MSE":
                lno_R_metrics[
                    "MSE"
                ],

            "R_RMSE":
                lno_R_metrics[
                    "RMSE"
                ],

            "R_MAE":
                lno_R_metrics[
                    "MAE"
                ],

            "R_Relative_L2":
                lno_R_metrics[
                    "Relative-L2"
                ],

            "Amplitude_MSE":
                lno_A_metrics[
                    "MSE"
                ],

            "Amplitude_RMSE":
                lno_A_metrics[
                    "RMSE"
                ],

            "Amplitude_MAE":
                lno_A_metrics[
                    "MAE"
                ],

            "Amplitude_Relative_L2":
                lno_A_metrics[
                    "Relative-L2"
                ],

            "Mean_trace_error":
                float(
                    lno_summary[
                        "structural_diagnostics"
                    ][
                        "mean_trace_error"
                    ]
                ),

            "Max_trace_error":
                float(
                    lno_summary[
                        "structural_diagnostics"
                    ][
                        "max_trace_error"
                    ]
                ),

            "Minimum_eigenvalue":
                float(
                    lno_summary[
                        "structural_diagnostics"
                    ][
                        "minimum_eigenvalue"
                    ]
                ),

            "Negative_eigenvalue_fraction":
                float(
                    lno_summary[
                        "structural_diagnostics"
                    ][
                        "negative_eigenvalue_fraction"
                    ]
                ),
        },

    ]
)


print(
    "\n" + "=" * 110
)

print(
    "E. OVERALL FNO vs LNO"
)

print(
    "=" * 110
)

display(
    comparison_df.round(
        8
    )
)


# ============================================================
# 18. LNO RELATIVE CHANGE VS FNO
# ============================================================

relative_change_rows = []

comparison_metrics = [

    (
        "R_RMSE",
        "R_RMSE"
    ),

    (
        "R_Relative_L2",
        "R_Relative_L2"
    ),

    (
        "Amplitude_RMSE",
        "Amplitude_RMSE"
    ),

    (
        "Amplitude_Relative_L2",
        "Amplitude_Relative_L2"
    ),

    (
        "Mean_trace_error",
        "Mean_trace_error"
    ),

    (
        "Max_trace_error",
        "Max_trace_error"
    ),
]

for (
    metric_name,
    column_name
) in comparison_metrics:

    fno_value = float(
        comparison_df.loc[
            comparison_df[
                "model"
            ] == "FNO",
            column_name
        ].iloc[0]
    )

    lno_value = float(
        comparison_df.loc[
            comparison_df[
                "model"
            ] == "LNO",
            column_name
        ].iloc[0]
    )

    relative_change_rows.append({

        "metric":
            metric_name,

        "FNO":
            fno_value,

        "LNO":
            lno_value,

        "LNO_percent_change_vs_FNO":
            percentage_change(
                lno_value,
                fno_value
            ),
    })


relative_change_df = pd.DataFrame(
    relative_change_rows
)

print(
    "\n" + "=" * 110
)

print(
    "F. LNO RELATIVE CHANGE VS FNO"
)

print(
    "=" * 110
)

display(
    relative_change_df.round(
        8
    )
)


# ============================================================
# 19. REGIME-WISE COMPARISON — CORRECTED
# ============================================================

fno_regime = pd.read_csv(
    FNO_REGIME_PATH
)

lno_regime = pd.read_csv(
    LNO_REGIME_PATH
)

print(
    "\n" + "=" * 110
)

print(
    "G. REGIME-WISE COMPARISON"
)

print(
    "=" * 110
)

# ------------------------------------------------------------
# Basic checks
# ------------------------------------------------------------

assert "regime" in fno_regime.columns
assert "regime" in lno_regime.columns

assert set(
    fno_regime[
        "regime"
    ]
) == set(
    lno_regime[
        "regime"
    ]
)


# ------------------------------------------------------------
# Merge
# ------------------------------------------------------------

merged_regime = fno_regime.merge(
    lno_regime,
    on="regime",
    suffixes=(
        "_FNO",
        "_LNO"
    )
)


# ------------------------------------------------------------
# Transition counts
#
# Both files may contain "transitions".
# After merge they become:
#
#   transitions_FNO
#   transitions_LNO
#
# We verify they agree and create one clean column.
# ------------------------------------------------------------

if (
    "transitions_FNO" in merged_regime.columns
    and
    "transitions_LNO" in merged_regime.columns
):

    transition_difference = (
        merged_regime[
            "transitions_FNO"
        ].to_numpy()
        -
        merged_regime[
            "transitions_LNO"
        ].to_numpy()
    )

    assert np.all(
        transition_difference == 0
    )

    merged_regime[
        "transitions"
    ] = (
        merged_regime[
            "transitions_FNO"
        ]
        .astype(
            int
        )
    )

elif "transitions" in merged_regime.columns:

    merged_regime[
        "transitions"
    ] = (
        merged_regime[
            "transitions"
        ]
        .astype(
            int
        )
    )

else:

    raise KeyError(
        "No transition-count column found in merged regime table."
    )


# ------------------------------------------------------------
# R Relative-L2 change
#
# Positive = LNO worse
# Negative = LNO better
# ------------------------------------------------------------

merged_regime[
    "R_relative_L2_change_percent"
] = (

    (
        merged_regime[
            "R_relative_L2_LNO"
        ]
        -
        merged_regime[
            "R_relative_L2_FNO"
        ]
    )
    /
    (
        np.abs(
            merged_regime[
                "R_relative_L2_FNO"
            ]
        )
        +
        1e-12
    )
    *
    100.0
)


# ------------------------------------------------------------
# Amplitude Relative-L2 change
# ------------------------------------------------------------

merged_regime[
    "Amplitude_relative_L2_change_percent"
] = (

    (
        merged_regime[
            "amplitude_relative_L2_LNO"
        ]
        -
        merged_regime[
            "amplitude_relative_L2_FNO"
        ]
    )
    /
    (
        np.abs(
            merged_regime[
                "amplitude_relative_L2_FNO"
            ]
        )
        +
        1e-12
    )
    *
    100.0
)


# ------------------------------------------------------------
# R winner
# ------------------------------------------------------------

merged_regime[
    "R_winner"
] = np.where(

    merged_regime[
        "R_relative_L2_LNO"
    ]
    <
    merged_regime[
        "R_relative_L2_FNO"
    ],

    "LNO",

    "FNO"
)


# ------------------------------------------------------------
# Amplitude winner
# ------------------------------------------------------------

merged_regime[
    "Amplitude_winner"
] = np.where(

    merged_regime[
        "amplitude_relative_L2_LNO"
    ]
    <
    merged_regime[
        "amplitude_relative_L2_FNO"
    ],

    "LNO",

    "FNO"
)


# ------------------------------------------------------------
# Final clean regime table
# ------------------------------------------------------------

regime_output = merged_regime[
    [

        "regime",

        "transitions",

        "R_relative_L2_FNO",
        "R_relative_L2_LNO",
        "R_relative_L2_change_percent",

        "amplitude_relative_L2_FNO",
        "amplitude_relative_L2_LNO",
        "Amplitude_relative_L2_change_percent",

        "R_winner",
        "Amplitude_winner",
    ]
].copy()


print(
    "\nRegime comparison:"
)

display(
    regime_output.round(
        6
    )
)


# ============================================================
# 20. REGIME WIN COUNTS
# ============================================================

R_LNO_wins = int(
    np.sum(
        regime_output[
            "R_winner"
        ]
        ==
        "LNO"
    )
)

R_FNO_wins = int(
    np.sum(
        regime_output[
            "R_winner"
        ]
        ==
        "FNO"
    )
)

A_LNO_wins = int(
    np.sum(
        regime_output[
            "Amplitude_winner"
        ]
        ==
        "LNO"
    )
)

A_FNO_wins = int(
    np.sum(
        regime_output[
            "Amplitude_winner"
        ]
        ==
        "FNO"
    )
)

print(
    "\nR regime wins:"
)

print(
    "  LNO:",
    R_LNO_wins
)

print(
    "  FNO:",
    R_FNO_wins
)

print(
    "\nAmplitude regime wins:"
)

print(
    "  LNO:",
    A_LNO_wins
)

print(
    "  FNO:",
    A_FNO_wins
)


# ============================================================
# 21. PAIRED SAMPLE-WISE ANALYSIS
# ============================================================

fno_R_error = (
    fno_R_pred
    -
    fno_R_target
)

lno_R_error = (
    lno_R_pred
    -
    lno_R_target
)

fno_A_error = (
    fno_A_pred
    -
    fno_A_target
)

lno_A_error = (
    lno_A_pred
    -
    lno_A_target
)


# ------------------------------------------------------------
# R per-transition RMSE
# ------------------------------------------------------------

R_axes = tuple(
    range(
        1,
        fno_R_error.ndim
    )
)

fno_R_transition_rmse = np.sqrt(
    np.mean(
        fno_R_error ** 2,
        axis=R_axes
    )
)

lno_R_transition_rmse = np.sqrt(
    np.mean(
        lno_R_error ** 2,
        axis=R_axes
    )
)


# ------------------------------------------------------------
# Amplitude per-transition RMSE
# ------------------------------------------------------------

fno_A_transition_rmse = np.sqrt(
    np.mean(
        fno_A_error ** 2,
        axis=1
    )
)

lno_A_transition_rmse = np.sqrt(
    np.mean(
        lno_A_error ** 2,
        axis=1
    )
)


# ------------------------------------------------------------
# Paired table
# ------------------------------------------------------------

paired_df = pd.DataFrame({

    "test_index":
        fno_test_indices,

    "FNO_R_RMSE":
        fno_R_transition_rmse,

    "LNO_R_RMSE":
        lno_R_transition_rmse,

    "LNO_minus_FNO_R":
        (
            lno_R_transition_rmse
            -
            fno_R_transition_rmse
        ),

    "FNO_amplitude_RMSE":
        fno_A_transition_rmse,

    "LNO_amplitude_RMSE":
        lno_A_transition_rmse,

    "LNO_minus_FNO_amplitude":
        (
            lno_A_transition_rmse
            -
            fno_A_transition_rmse
        ),
})


# ------------------------------------------------------------
# Sample win counts
# ------------------------------------------------------------

R_LNO_sample_wins = int(
    np.sum(
        paired_df[
            "LNO_minus_FNO_R"
        ]
        < 0
    )
)

R_FNO_sample_wins = int(
    np.sum(
        paired_df[
            "LNO_minus_FNO_R"
        ]
        > 0
    )
)

A_LNO_sample_wins = int(
    np.sum(
        paired_df[
            "LNO_minus_FNO_amplitude"
        ]
        < 0
    )
)

A_FNO_sample_wins = int(
    np.sum(
        paired_df[
            "LNO_minus_FNO_amplitude"
        ]
        > 0
    )
)

print(
    "\n" + "=" * 110
)

print(
    "H. PAIRED TEST-TRANSITION ANALYSIS"
)

print(
    "=" * 110
)

print(
    "R:"
)

print(
    "  LNO better:",
    R_LNO_sample_wins,
    "/",
    len(
        paired_df
    )
)

print(
    "  FNO better:",
    R_FNO_sample_wins,
    "/",
    len(
        paired_df
    )
)

print(
    "\nAmplitude:"
)

print(
    "  LNO better:",
    A_LNO_sample_wins,
    "/",
    len(
        paired_df
    )
)

print(
    "  FNO better:",
    A_FNO_sample_wins,
    "/",
    len(
        paired_df
    )
)


# ============================================================
# 22. STRUCTURAL COMPARISON
# ============================================================

fno_trace = float(
    comparison_df.loc[
        comparison_df[
            "model"
        ] == "FNO",
        "Mean_trace_error"
    ].iloc[0]
)

lno_trace = float(
    comparison_df.loc[
        comparison_df[
            "model"
        ] == "LNO",
        "Mean_trace_error"
    ].iloc[0]
)

fno_eig = float(
    comparison_df.loc[
        comparison_df[
            "model"
        ] == "FNO",
        "Minimum_eigenvalue"
    ].iloc[0]
)

lno_eig = float(
    comparison_df.loc[
        comparison_df[
            "model"
        ] == "LNO",
        "Minimum_eigenvalue"
    ].iloc[0]
)

fno_negative_fraction = float(
    comparison_df.loc[
        comparison_df[
            "model"
        ] == "FNO",
        "Negative_eigenvalue_fraction"
    ].iloc[0]
)

lno_negative_fraction = float(
    comparison_df.loc[
        comparison_df[
            "model"
        ] == "LNO",
        "Negative_eigenvalue_fraction"
    ].iloc[0]
)

print(
    "\n" + "=" * 110
)

print(
    "I. STRUCTURAL COMPARISON"
)

print(
    "=" * 110
)

print(
    "FNO mean trace error:",
    f"{fno_trace:.12e}"
)

print(
    "LNO mean trace error:",
    f"{lno_trace:.12e}"
)

print(
    "FNO minimum eigenvalue:",
    f"{fno_eig:.12e}"
)

print(
    "LNO minimum eigenvalue:",
    f"{lno_eig:.12e}"
)

print(
    "FNO negative eigenvalue fraction:",
    f"{fno_negative_fraction:.6f}"
)

print(
    "LNO negative eigenvalue fraction:",
    f"{lno_negative_fraction:.6f}"
)


# ============================================================
# 23. ROLLOUT COMPARISON
# ============================================================

fno_rollout = fno_summary.get(
    "rollout",
    {}
)

lno_rollout = lno_summary.get(
    "rollout",
    {}
)

print(
    "\n" + "=" * 110
)

print(
    "J. ROLLOUT COMPARISON"
)

print(
    "=" * 110
)

print(
    "FNO rollout:"
)

print(
    "  Finite:",
    fno_rollout.get(
        "finite",
        "N/A"
    )
)

print(
    "  Stable:",
    fno_rollout.get(
        "stable",
        "N/A"
    )
)

print(
    "  Minimum eigenvalue:",
    fno_rollout.get(
        "minimum_eigenvalue",
        "N/A"
    )
)

print(
    "  Maximum trace error:",
    fno_rollout.get(
        "maximum_trace_error",
        "N/A"
    )
)

print(
    "\nLNO rollout:"
)

print(
    "  Finite:",
    lno_rollout.get(
        "finite",
        "N/A"
    )
)

print(
    "  Stable:",
    lno_rollout.get(
        "stable",
        "N/A"
    )
)

print(
    "  Minimum eigenvalue:",
    lno_rollout.get(
        "minimum_eigenvalue",
        "N/A"
    )
)

print(
    "  Maximum trace error:",
    lno_rollout.get(
        "maximum_trace_error",
        "N/A"
    )
)


# ============================================================
# 24. LNO DYNAMICAL COMPONENTS
# ============================================================

lno_components = lno_summary.get(
    "dynamical_components",
    {}
)

print(
    "\n" + "=" * 110
)

print(
    "K. LNO DYNAMICAL COMPONENTS"
)

print(
    "=" * 110
)

for key in [

    "target_delta_R_norm",

    "lindblad_delta_R_norm",

    "neural_delta_R_norm",

    "combined_delta_R_norm",

    "kappa_lindblad",

    "kappa_neural",

]:

    print(
        f"{key:30s}:",
        lno_components.get(
            key,
            "N/A"
        )
    )


# ============================================================
# 25. ENVIRONMENT RESPONSE COMPARISON
# ============================================================

FNO_ENV_PATH = (
    FNO_DIR
    / "fno_final_environment_response.csv"
)

LNO_ENV_PATH = (
    LNO_DIR
    / "lno_final_environment_response.csv"
)

environment_comparison_df = None

if (
    FNO_ENV_PATH.is_file()
    and
    LNO_ENV_PATH.is_file()
):

    fno_env_df = pd.read_csv(
        FNO_ENV_PATH
    )

    lno_env_df = pd.read_csv(
        LNO_ENV_PATH
    )

    assert np.allclose(
        fno_env_df[
            "gamma"
        ].to_numpy(),
        lno_env_df[
            "gamma"
        ].to_numpy()
    )

    environment_comparison_df = pd.DataFrame({

        "gamma":
            fno_env_df[
                "gamma"
            ].to_numpy(),

        "FNO_R_change":
            fno_env_df[
                "R_change"
            ].to_numpy(),

        "LNO_R_change":
            lno_env_df[
                "R_change"
            ].to_numpy(),

        "FNO_amplitude_change":
            fno_env_df[
                "amplitude_change"
            ].to_numpy(),

        "LNO_amplitude_change":
            lno_env_df[
                "amplitude_change"
            ].to_numpy(),

    })

    print(
        "\n" + "=" * 110
    )

    print(
        "L. ENVIRONMENT RESPONSE COMPARISON"
    )

    print(
        "=" * 110
    )

    display(
        environment_comparison_df.round(
            8
        )
    )

else:

    print(
        "\n[INFO] Environment-response files not available; skipping comparison."
    )


# ============================================================
# 26. SAVE CSV OUTPUTS
# ============================================================

COMPARISON_PATH = (
    OUT_DIR
    / "final_comparison.csv"
)

RELATIVE_CHANGE_PATH = (
    OUT_DIR
    / "final_relative_change.csv"
)

REGIME_PATH = (
    OUT_DIR
    / "final_regime_comparison.csv"
)

PAIRED_PATH = (
    OUT_DIR
    / "paired_test_comparison.csv"
)

comparison_df.to_csv(
    COMPARISON_PATH,
    index=False
)

relative_change_df.to_csv(
    RELATIVE_CHANGE_PATH,
    index=False
)

regime_output.to_csv(
    REGIME_PATH,
    index=False
)

paired_df.to_csv(
    PAIRED_PATH,
    index=False
)

if environment_comparison_df is not None:

    environment_comparison_df.to_csv(
        OUT_DIR
        / "environment_response_comparison.csv",
        index=False
    )


# ============================================================
# 27. PLOT 1 — OVERALL R RELATIVE-L2
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    [0, 1],
    [
        fno_R_metrics[
            "Relative-L2"
        ],
        lno_R_metrics[
            "Relative-L2"
        ],
    ]
)

plt.xticks(
    [0, 1],
    [
        "FNO",
        "LNO"
    ]
)

plt.ylabel(
    "R Relative-L2"
)

plt.title(
    "FNO vs LNO — Information-State Error"
)

plt.tight_layout()

plt.savefig(
    OUT_DIR
    / "01_overall_R_relative_L2.png",
    dpi=200
)

plt.show()


# ============================================================
# 28. PLOT 2 — REGIME R RELATIVE-L2
# ============================================================

plt.figure(
    figsize=(10, 6)
)

regime_names = regime_output[
    "regime"
].tolist()

regime_x = np.arange(
    len(
        regime_names
    )
)

width = 0.38

plt.bar(
    regime_x - width / 2,
    regime_output[
        "R_relative_L2_FNO"
    ].to_numpy(),
    width=width,
    label="FNO"
)

plt.bar(
    regime_x + width / 2,
    regime_output[
        "R_relative_L2_LNO"
    ].to_numpy(),
    width=width,
    label="LNO"
)

plt.xticks(
    regime_x,
    regime_names,
    rotation=20
)

plt.ylabel(
    "R Relative-L2"
)

plt.title(
    "R Error by Environment Regime"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT_DIR
    / "02_regime_R_relative_L2.png",
    dpi=200
)

plt.show()


# ============================================================
# 29. PLOT 3 — OVERALL AMPLITUDE RELATIVE-L2
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    [0, 1],
    [
        fno_A_metrics[
            "Relative-L2"
        ],
        lno_A_metrics[
            "Relative-L2"
        ],
    ]
)

plt.xticks(
    [0, 1],
    [
        "FNO",
        "LNO"
    ]
)

plt.ylabel(
    "Amplitude Relative-L2"
)

plt.title(
    "FNO vs LNO — Information-State Amplitude"
)

plt.tight_layout()

plt.savefig(
    OUT_DIR
    / "03_overall_amplitude_relative_L2.png",
    dpi=200
)

plt.show()


# ============================================================
# 30. PLOT 4 — TRACE PRESERVATION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    [0, 1],
    [
        fno_trace,
        lno_trace,
    ]
)

plt.xticks(
    [0, 1],
    [
        "FNO",
        "LNO"
    ]
)

plt.ylabel(
    "Mean |trace - 1|"
)

plt.title(
    "Information-State Trace Preservation"
)

plt.tight_layout()

plt.savefig(
    OUT_DIR
    / "04_trace_constraint.png",
    dpi=200
)

plt.show()


# ============================================================
# 31. PLOT 5 — MINIMUM EIGENVALUE
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    [0, 1],
    [
        fno_eig,
        lno_eig,
    ]
)

plt.xticks(
    [0, 1],
    [
        "FNO",
        "LNO"
    ]
)

plt.ylabel(
    "Minimum eigenvalue"
)

plt.title(
    "Minimum Information-State Eigenvalue"
)

plt.tight_layout()

plt.savefig(
    OUT_DIR
    / "05_minimum_eigenvalue.png",
    dpi=200
)

plt.show()


# ============================================================
# 32. PLOT 6 — NEGATIVE EIGENVALUE FRACTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    [0, 1],
    [
        fno_negative_fraction,
        lno_negative_fraction,
    ]
)

plt.xticks(
    [0, 1],
    [
        "FNO",
        "LNO"
    ]
)

plt.ylabel(
    "Fraction of negative eigenvalues"
)

plt.title(
    "Structural Constraint Violation"
)

plt.tight_layout()

plt.savefig(
    OUT_DIR
    / "06_negative_eigenvalue_fraction.png",
    dpi=200
)

plt.show()


# ============================================================
# 33. PLOT 7 — REGIME AMPLITUDE RELATIVE-L2
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    regime_x - width / 2,
    regime_output[
        "amplitude_relative_L2_FNO"
    ].to_numpy(),
    width=width,
    label="FNO"
)

plt.bar(
    regime_x + width / 2,
    regime_output[
        "amplitude_relative_L2_LNO"
    ].to_numpy(),
    width=width,
    label="LNO"
)

plt.xticks(
    regime_x,
    regime_names,
    rotation=20
)

plt.ylabel(
    "Amplitude Relative-L2"
)

plt.title(
    "Amplitude Error by Environment Regime"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT_DIR
    / "07_regime_amplitude_relative_L2.png",
    dpi=200
)

plt.show()


# ============================================================
# 34. ENVIRONMENT RESPONSE PLOT
# ============================================================

if environment_comparison_df is not None:

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        environment_comparison_df[
            "gamma"
        ].to_numpy(),

        environment_comparison_df[
            "FNO_R_change"
        ].to_numpy(),

        marker="o",
        label="FNO"
    )

    plt.plot(
        environment_comparison_df[
            "gamma"
        ].to_numpy(),

        environment_comparison_df[
            "LNO_R_change"
        ].to_numpy(),

        marker="o",
        label="LNO"
    )

    plt.xlabel(
        "Gamma"
    )

    plt.ylabel(
        "R response magnitude"
    )

    plt.title(
        "Environment Coupling Response"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT_DIR
        / "08_environment_response.png",
        dpi=200
    )

    plt.show()


# ============================================================
# 35. OVERALL WINNERS
# ============================================================

R_overall_winner = (

    "LNO"

    if (
        lno_R_metrics[
            "Relative-L2"
        ]
        <
        fno_R_metrics[
            "Relative-L2"
        ]
    )

    else

    "FNO"
)


A_overall_winner = (

    "LNO"

    if (
        lno_A_metrics[
            "Relative-L2"
        ]
        <
        fno_A_metrics[
            "Relative-L2"
        ]
    )

    else

    "FNO"
)


trace_winner = (

    "LNO"

    if (
        lno_trace
        <
        fno_trace
    )

    else

    "FNO"
)


# ============================================================
# 36. FINAL JSON SUMMARY
# ============================================================

final_summary = {

    "step":
        19,

    "title":
        "Final FNO vs Final Repaired LNO Comparison",

    "fair_comparison":
        {

            "same_test_indices":
                True,

            "same_targets":
                True,

            "same_environment":
                True,

            "test_transitions":
                int(
                    len(
                        fno_test_indices
                    )
                ),
        },


    "overall":
        {

            "FNO":
                {

                    "R":
                        fno_R_metrics,

                    "amplitude":
                        fno_A_metrics,
                },

            "LNO":
                {

                    "R":
                        lno_R_metrics,

                    "amplitude":
                        lno_A_metrics,
                },
        },


    "relative_change":
        {

            "R_RMSE":
                percentage_change(
                    lno_R_metrics[
                        "RMSE"
                    ],
                    fno_R_metrics[
                        "RMSE"
                    ]
                ),

            "R_Relative_L2":
                percentage_change(
                    lno_R_metrics[
                        "Relative-L2"
                    ],
                    fno_R_metrics[
                        "Relative-L2"
                    ]
                ),

            "Amplitude_RMSE":
                percentage_change(
                    lno_A_metrics[
                        "RMSE"
                    ],
                    fno_A_metrics[
                        "RMSE"
                    ]
                ),

            "Amplitude_Relative_L2":
                percentage_change(
                    lno_A_metrics[
                        "Relative-L2"
                    ],
                    fno_A_metrics[
                        "Relative-L2"
                    ]
                ),
        },


    "structural":
        {

            "FNO_mean_trace_error":
                fno_trace,

            "LNO_mean_trace_error":
                lno_trace,

            "FNO_minimum_eigenvalue":
                fno_eig,

            "LNO_minimum_eigenvalue":
                lno_eig,

            "FNO_negative_eigenvalue_fraction":
                fno_negative_fraction,

            "LNO_negative_eigenvalue_fraction":
                lno_negative_fraction,
        },


    "rollout":
        {

            "FNO":
                fno_rollout,

            "LNO":
                lno_rollout,
        },


    "regime_wins":
        {

            "R_LNO":
                R_LNO_wins,

            "R_FNO":
                R_FNO_wins,

            "Amplitude_LNO":
                A_LNO_wins,

            "Amplitude_FNO":
                A_FNO_wins,
        },


    "paired_sample_wins":
        {

            "R_LNO":
                R_LNO_sample_wins,

            "R_FNO":
                R_FNO_sample_wins,

            "Amplitude_LNO":
                A_LNO_sample_wins,

            "Amplitude_FNO":
                A_FNO_sample_wins,
        },


    "overall_winners":
        {

            "R_accuracy":
                R_overall_winner,

            "Amplitude_accuracy":
                A_overall_winner,

            "Trace_preservation":
                trace_winner,
        },


    "LNO_dynamical_components":
        lno_components,


    "outputs":
        {

            "overall":
                str(
                    COMPARISON_PATH
                ),

            "relative_change":
                str(
                    RELATIVE_CHANGE_PATH
                ),

            "regime":
                str(
                    REGIME_PATH
                ),

            "paired":
                str(
                    PAIRED_PATH
                ),
        },


    "plots":
        [

            "01_overall_R_relative_L2.png",

            "02_regime_R_relative_L2.png",

            "03_overall_amplitude_relative_L2.png",

            "04_trace_constraint.png",

            "05_minimum_eigenvalue.png",

            "06_negative_eigenvalue_fraction.png",

            "07_regime_amplitude_relative_L2.png",

            "08_environment_response.png",
        ],
}


FINAL_JSON_PATH = (
    OUT_DIR
    / "final_comparison.json"
)

with open(
    FINAL_JSON_PATH,
    "w"
) as f:

    json.dump(
        final_summary,
        f,
        indent=2,
        allow_nan=False
    )


# ============================================================
# 37. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "STEP 19 COMPLETE — FINAL FNO vs LNO COMPARISON"
)

print(
    "=" * 110
)

print(
    "\nOVERALL ACCURACY"
)

print(
    "FNO R Relative-L2:",
    f"{fno_R_metrics['Relative-L2']:.8f}"
)

print(
    "LNO R Relative-L2:",
    f"{lno_R_metrics['Relative-L2']:.8f}"
)

print(
    "FNO amplitude Relative-L2:",
    f"{fno_A_metrics['Relative-L2']:.8f}"
)

print(
    "LNO amplitude Relative-L2:",
    f"{lno_A_metrics['Relative-L2']:.8f}"
)

print(
    "\nSTRUCTURE"
)

print(
    "FNO mean trace error:",
    f"{fno_trace:.8e}"
)

print(
    "LNO mean trace error:",
    f"{lno_trace:.8e}"
)

print(
    "FNO minimum eigenvalue:",
    f"{fno_eig:.8e}"
)

print(
    "LNO minimum eigenvalue:",
    f"{lno_eig:.8e}"
)

print(
    "\nREGIME WINS"
)

print(
    "R — LNO:",
    R_LNO_wins,
    "| FNO:",
    R_FNO_wins
)

print(
    "Amplitude — LNO:",
    A_LNO_wins,
    "| FNO:",
    A_FNO_wins
)

print(
    "\nPAIRED TEST-TRANSITION WINS"
)

print(
    "R — LNO:",
    R_LNO_sample_wins,
    "| FNO:",
    R_FNO_sample_wins
)

print(
    "Amplitude — LNO:",
    A_LNO_sample_wins,
    "| FNO:",
    A_FNO_sample_wins
)

print(
    "\nOVERALL WINNERS"
)

print(
    "R accuracy:",
    R_overall_winner
)

print(
    "Amplitude accuracy:",
    A_overall_winner
)

print(
    "Trace preservation:",
    trace_winner
)

print(
    "\nFINAL FILES"
)

print(
    "Overall comparison:",
    COMPARISON_PATH
)

print(
    "Relative change:",
    RELATIVE_CHANGE_PATH
)

print(
    "Regime comparison:",
    REGIME_PATH
)

print(
    "Paired comparison:",
    PAIRED_PATH
)

print(
    "Final JSON:",
    FINAL_JSON_PATH
)

print(
    "\n[PASS] STEP 19 COMPLETE."
)

print(
    "[INFO] No model was retrained."
)

print(
    "[INFO] FNO and LNO were compared on the exact same unseen test set."
)

print("=" * 110)
