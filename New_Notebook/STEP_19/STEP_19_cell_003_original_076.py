# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 76
# Step            : STEP_19
# Step Heading    : # STEP 19 — FINAL FNO vs FINAL REPAIRED LNO COMPARISON
# Step Cell No.   : 3
# ============================================================

# ============================================================
# STEP 19 — FINAL FNO vs FINAL REPAIRED LNO COMPARISON
# AUTO-DETECT ARTIFACT VERSION
# ============================================================
#
# Purpose:
#   Fresh apples-to-apples comparison using the CURRENT final
#   FNO and LNO prediction artifacts.
#
# Important:
#   - No retraining
#   - No checkpoint modification
#   - No new model inference
#   - Automatically searches /content for uploaded artifacts
#   - Same test transitions / same targets / same environment
# ============================================================

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. ROOT
# ============================================================

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

print("=" * 110)
print("STEP 19 — FINAL FNO vs FINAL REPAIRED LNO COMPARISON")
print("=" * 110)


# ============================================================
# 2. AUTO-DETECT FILE
# ============================================================

def locate_file_auto(
    filename,
    preferred_paths=None
):

    preferred_paths = preferred_paths or []

    # --------------------------------------------------------
    # First try preferred paths
    # --------------------------------------------------------

    for p in preferred_paths:

        p = Path(p)

        if p.is_file():
            return p

    # --------------------------------------------------------
    # Search repository recursively
    # --------------------------------------------------------

    repo_matches = list(
        ROOT.rglob(filename)
    )

    if repo_matches:

        repo_matches.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        return repo_matches[0]

    # --------------------------------------------------------
    # Search all of /content
    # --------------------------------------------------------

    content_matches = []

    for p in Path(
        "/content"
    ).rglob(filename):

        try:

            if p.is_file():
                content_matches.append(p)

        except Exception:
            pass

    if content_matches:

        # Newest uploaded/generated copy first
        content_matches.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        return content_matches[0]

    raise FileNotFoundError(
        f"Could not locate required file anywhere under /content:\n"
        f"{filename}"
    )


# ============================================================
# 3. DIRECTORIES
# ============================================================

STEP17_DIR = (
    ROOT
    / "results"
    / "step17_final_fno"
)

STEP18_DIR = (
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
# 4. AUTO-DETECT INPUT ARTIFACTS
# ============================================================

FNO_PRED_PATH = locate_file_auto(
    "fno_final_test_predictions.npz",
    preferred_paths=[
        STEP17_DIR
        / "fno_final_test_predictions.npz"
    ]
)

LNO_PRED_PATH = locate_file_auto(
    "lno_final_test_predictions.npz",
    preferred_paths=[
        STEP18_DIR
        / "lno_final_test_predictions.npz"
    ]
)

FNO_SUMMARY_PATH = locate_file_auto(
    "fno_final_summary.json",
    preferred_paths=[
        STEP17_DIR
        / "fno_final_summary.json"
    ]
)

LNO_SUMMARY_PATH = locate_file_auto(
    "lno_final_summary.json",
    preferred_paths=[
        STEP18_DIR
        / "lno_final_summary.json"
    ]
)

SPLIT_PATH = locate_file_auto(
    "final_trajectory_level_split.npz",
    preferred_paths=[
        STEP16_DIR
        / "final_trajectory_level_split.npz"
    ]
)


# Optional regime metric files
FNO_REGIME_PATH = locate_file_auto(
    "fno_final_regime_metrics.csv",
    preferred_paths=[
        STEP17_DIR
        / "fno_final_regime_metrics.csv"
    ]
)

LNO_REGIME_PATH = locate_file_auto(
    "lno_final_regime_metrics.csv",
    preferred_paths=[
        STEP18_DIR
        / "lno_final_regime_metrics.csv"
    ]
)


# ============================================================
# 5. PRINT LOCATIONS
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "A. AUTO-DETECTED ARTIFACT LOCATIONS"
)

print(
    "=" * 110
)

print(
    "FNO prediction:",
    FNO_PRED_PATH
)

print(
    "LNO prediction:",
    LNO_PRED_PATH
)

print(
    "FNO summary:",
    FNO_SUMMARY_PATH
)

print(
    "LNO summary:",
    LNO_SUMMARY_PATH
)

print(
    "Final split:",
    SPLIT_PATH
)

print(
    "FNO regime metrics:",
    FNO_REGIME_PATH
)

print(
    "LNO regime metrics:",
    LNO_REGIME_PATH
)


# ============================================================
# 6. LOAD PREDICTIONS
# ============================================================

fno_pred = np.load(
    FNO_PRED_PATH
)

lno_pred = np.load(
    LNO_PRED_PATH
)

required_keys = [
    "R_prediction",
    "R_target",
    "amplitude_prediction",
    "amplitude_target",
    "environment",
    "test_indices",
]

for key in required_keys:

    assert key in fno_pred, (
        f"FNO prediction file missing key: {key}"
    )

    assert key in lno_pred, (
        f"LNO prediction file missing key: {key}"
    )

print(
    "\n[PASS] Test prediction files loaded."
)

print(
    "[PASS] Required prediction keys present."
)


# ============================================================
# 7. EXTRACT ARRAYS
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
# 8. SAME TEST SET
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

assert len(
    fno_test_indices
) == 1485

print(
    "[PASS] FNO and LNO use identical test indices."
)

print(
    "Test transitions:",
    len(
        fno_test_indices
    )
)


# ============================================================
# 9. SAME TARGET CHECK
# ============================================================

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
# 10. SAME ENVIRONMENT CHECK
# ============================================================

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
# 11. SHAPE CHECK
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
# 12. FINITE CHECK
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "D. FINITE-VALUE CHECK"
)

print(
    "=" * 110
)

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
# 13. METRIC FUNCTION
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

    relative_l2 = float(
        np.linalg.norm(
            error
        )
        /
        (
            np.linalg.norm(
                target
            )
            +
            1e-12
        )
    )

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Relative-L2": relative_l2,
    }


# ============================================================
# 14. OVERALL ACCURACY METRICS
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
# 15. STRUCTURAL METRICS
# ============================================================

def calculate_structural_metrics(
    R
):

    trace = np.trace(
        R,
        axis1=-2,
        axis2=-1
    )

    trace_error = np.abs(
        trace
        -
        1.0
    )

    R_sym = (
        0.5
        *
        (
            R
            +
            np.swapaxes(
                R,
                -1,
                -2
            )
        )
    )

    eig = np.linalg.eigvalsh(
        R_sym
    )

    return {

        "mean_trace_error":
            float(
                np.mean(
                    trace_error
                )
            ),

        "max_trace_error":
            float(
                np.max(
                    trace_error
                )
            ),

        "minimum_eigenvalue":
            float(
                np.min(
                    eig
                )
            ),

        "negative_eigenvalue_fraction":
            float(
                np.mean(
                    eig
                    <
                    -1e-6
                )
            ),

        "mean_symmetry_error":
            float(
                np.mean(
                    np.abs(
                        R
                        -
                        np.swapaxes(
                            R,
                            -1,
                            -2
                        )
                    )
                ),
            ),
    }


fno_struct = calculate_structural_metrics(
    fno_R_pred
)

lno_struct = calculate_structural_metrics(
    lno_R_pred
)


# ============================================================
# 16. OVERALL COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame(
    [
        {
            "model":
                "FNO",

            "R_MSE":
                fno_R_metrics["MSE"],

            "R_RMSE":
                fno_R_metrics["RMSE"],

            "R_MAE":
                fno_R_metrics["MAE"],

            "R_Relative_L2":
                fno_R_metrics["Relative-L2"],

            "Amplitude_MSE":
                fno_A_metrics["MSE"],

            "Amplitude_RMSE":
                fno_A_metrics["RMSE"],

            "Amplitude_MAE":
                fno_A_metrics["MAE"],

            "Amplitude_Relative_L2":
                fno_A_metrics["Relative-L2"],

            "Mean_trace_error":
                fno_struct["mean_trace_error"],

            "Max_trace_error":
                fno_struct["max_trace_error"],

            "Minimum_eigenvalue":
                fno_struct["minimum_eigenvalue"],

            "Negative_eigenvalue_fraction":
                fno_struct[
                    "negative_eigenvalue_fraction"
                ],
        },

        {
            "model":
                "LNO",

            "R_MSE":
                lno_R_metrics["MSE"],

            "R_RMSE":
                lno_R_metrics["RMSE"],

            "R_MAE":
                lno_R_metrics["MAE"],

            "R_Relative_L2":
                lno_R_metrics["Relative-L2"],

            "Amplitude_MSE":
                lno_A_metrics["MSE"],

            "Amplitude_RMSE":
                lno_A_metrics["RMSE"],

            "Amplitude_MAE":
                lno_A_metrics["MAE"],

            "Amplitude_Relative_L2":
                lno_A_metrics["Relative-L2"],

            "Mean_trace_error":
                lno_struct["mean_trace_error"],

            "Max_trace_error":
                lno_struct["max_trace_error"],

            "Minimum_eigenvalue":
                lno_struct["minimum_eigenvalue"],

            "Negative_eigenvalue_fraction":
                lno_struct[
                    "negative_eigenvalue_fraction"
                ],
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
# 17. RELATIVE CHANGE
# ============================================================

def percentage_change(
    lno_value,
    fno_value
):

    return float(
        (
            (
                lno_value
                -
                fno_value
            )
            /
            (
                abs(
                    fno_value
                )
                +
                1e-12
            )
        )
        *
        100.0
    )


relative_change_rows = []

for metric in [

    "R_RMSE",
    "R_Relative_L2",
    "Amplitude_RMSE",
    "Amplitude_Relative_L2",
    "Mean_trace_error",
    "Max_trace_error",

]:

    fno_value = float(
        comparison_df.loc[
            comparison_df[
                "model"
            ]
            ==
            "FNO",
            metric
        ].iloc[0]
    )

    lno_value = float(
        comparison_df.loc[
            comparison_df[
                "model"
            ]
            ==
            "LNO",
            metric
        ].iloc[0]
    )

    relative_change_rows.append(
        {
            "metric":
                metric,

            "FNO":
                fno_value,

            "LNO":
                lno_value,

            "LNO_percent_change_vs_FNO":
                percentage_change(
                    lno_value,
                    fno_value
                ),
        }
    )

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
# 18. REGIME DEFINITIONS
# ============================================================

regimes = {

    "low_noise":
        (0.01, 0.02),

    "stochastic":
        (0.05, 0.45),

    "heavy_dissipation":
        (0.35, 0.15),

    "collapse":
        (0.50, 0.65),

    "metastable":
        (0.15, 0.30),

}


# ============================================================
# 19. REGIME-WISE ACCURACY
# ============================================================

gamma_test = (
    fno_environment[
        :,
        0,
        0
    ]
)

sigma_test = (
    fno_environment[
        :,
        0,
        1
    ]
)

regime_rows = []

for regime_name, (
    gamma_value,
    sigma_value
) in regimes.items():

    mask = (
        np.isclose(
            gamma_test,
            gamma_value,
            atol=1e-5
        )
        &
        np.isclose(
            sigma_test,
            sigma_value,
            atol=1e-5
        )
    )

    assert int(
        mask.sum()
    ) == 297, (
        f"Unexpected transition count "
        f"for {regime_name}: "
        f"{int(mask.sum())}"
    )

    fno_r = calculate_metrics(
        fno_R_pred[mask],
        fno_R_target[mask]
    )

    lno_r = calculate_metrics(
        lno_R_pred[mask],
        lno_R_target[mask]
    )

    fno_a = calculate_metrics(
        fno_A_pred[mask],
        fno_A_target[mask]
    )

    lno_a = calculate_metrics(
        lno_A_pred[mask],
        lno_A_target[mask]
    )

    regime_rows.append(
        {
            "regime":
                regime_name,

            "transitions":
                int(
                    mask.sum()
                ),

            "R_relative_L2_FNO":
                fno_r[
                    "Relative-L2"
                ],

            "R_relative_L2_LNO":
                lno_r[
                    "Relative-L2"
                ],

            "R_relative_L2_change_percent":
                percentage_change(
                    lno_r[
                        "Relative-L2"
                    ],
                    fno_r[
                        "Relative-L2"
                    ]
                ),

            "amplitude_relative_L2_FNO":
                fno_a[
                    "Relative-L2"
                ],

            "amplitude_relative_L2_LNO":
                lno_a[
                    "Relative-L2"
                ],

            "Amplitude_relative_L2_change_percent":
                percentage_change(
                    lno_a[
                        "Relative-L2"
                    ],
                    fno_a[
                        "Relative-L2"
                    ]
                ),

            "R_winner":
                (
                    "LNO"
                    if
                    lno_r[
                        "Relative-L2"
                    ]
                    <
                    fno_r[
                        "Relative-L2"
                    ]
                    else
                    "FNO"
                ),

            "Amplitude_winner":
                (
                    "LNO"
                    if
                    lno_a[
                        "Relative-L2"
                    ]
                    <
                    fno_a[
                        "Relative-L2"
                    ]
                    else
                    "FNO"
                ),
        }
    )

regime_output = pd.DataFrame(
    regime_rows
)

print(
    "\n" + "=" * 110
)

print(
    "G. REGIME-WISE ACCURACY COMPARISON"
)

print(
    "=" * 110
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
# 21. PAIRED TEST-TRANSITION ANALYSIS
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


paired_df = pd.DataFrame(
    {

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
    }
)


R_LNO_sample_wins = int(
    np.sum(
        paired_df[
            "LNO_minus_FNO_R"
        ]
        <
        0
    )
)

R_FNO_sample_wins = int(
    np.sum(
        paired_df[
            "LNO_minus_FNO_R"
        ]
        >
        0
    )
)

A_LNO_sample_wins = int(
    np.sum(
        paired_df[
            "LNO_minus_FNO_amplitude"
        ]
        <
        0
    )
)

A_FNO_sample_wins = int(
    np.sum(
        paired_df[
            "LNO_minus_FNO_amplitude"
        ]
        >
        0
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
# 22. OVERALL WINNERS
# ============================================================

R_overall_winner = (
    "LNO"
    if
    lno_R_metrics[
        "Relative-L2"
    ]
    <
    fno_R_metrics[
        "Relative-L2"
    ]
    else
    "FNO"
)

A_overall_winner = (
    "LNO"
    if
    lno_A_metrics[
        "Relative-L2"
    ]
    <
    fno_A_metrics[
        "Relative-L2"
    ]
    else
    "FNO"
)

trace_winner = (
    "LNO"
    if
    lno_struct[
        "mean_trace_error"
    ]
    <
    fno_struct[
        "mean_trace_error"
    ]
    else
    "FNO"
)


# ============================================================
# 23. SAVE CSV OUTPUTS
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


# ============================================================
# 24. PLOTS
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
    /
    "01_overall_R_relative_L2.png",
    dpi=200
)

plt.close()


plt.figure(
    figsize=(10, 6)
)

regime_x = np.arange(
    len(
        regime_output
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
    regime_output[
        "regime"
    ].tolist(),
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
    /
    "02_regime_R_relative_L2.png",
    dpi=200
)

plt.close()


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
    /
    "03_overall_amplitude_relative_L2.png",
    dpi=200
)

plt.close()


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    [0, 1],
    [
        fno_struct[
            "mean_trace_error"
        ],
        lno_struct[
            "mean_trace_error"
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
    "Mean |trace - 1|"
)

plt.title(
    "Information-State Trace Preservation"
)

plt.tight_layout()

plt.savefig(
    OUT_DIR
    /
    "04_trace_constraint.png",
    dpi=200
)

plt.close()


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    [0, 1],
    [
        fno_struct[
            "minimum_eigenvalue"
        ],
        lno_struct[
            "minimum_eigenvalue"
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
    "Minimum eigenvalue"
)

plt.title(
    "Minimum Information-State Eigenvalue"
)

plt.tight_layout()

plt.savefig(
    OUT_DIR
    /
    "05_minimum_eigenvalue.png",
    dpi=200
)

plt.close()


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    [0, 1],
    [
        fno_struct[
            "negative_eigenvalue_fraction"
        ],
        lno_struct[
            "negative_eigenvalue_fraction"
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
    "Fraction of meaningful negative eigenvalues"
)

plt.title(
    "Structural Constraint Violation"
)

plt.tight_layout()

plt.savefig(
    OUT_DIR
    /
    "06_negative_eigenvalue_fraction.png",
    dpi=200
)

plt.close()


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
    regime_output[
        "regime"
    ].tolist(),
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
    /
    "07_regime_amplitude_relative_L2.png",
    dpi=200
)

plt.close()


# ============================================================
# 25. FINAL JSON
# ============================================================

FINAL_JSON_PATH = (
    OUT_DIR
    /
    "final_comparison.json"
)

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

                    "structural":
                        fno_struct,
                },

            "LNO":
                {

                    "R":
                        lno_R_metrics,

                    "amplitude":
                        lno_A_metrics,

                    "structural":
                        lno_struct,
                },
        },

    "relative_change":
        relative_change_df.to_dict(
            orient="records"
        ),

    "regime_comparison":
        regime_output.to_dict(
            orient="records"
        ),

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

    "overall_winners":
        {

            "R_accuracy":
                R_overall_winner,

            "Amplitude_accuracy":
                A_overall_winner,

            "Trace_preservation":
                trace_winner,
        },

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

            "json":
                str(
                    FINAL_JSON_PATH
                ),
        },
}


with open(
    FINAL_JSON_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_summary,
        f,
        indent=2,
        allow_nan=False
    )


# ============================================================
# 26. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "STEP 19 COMPLETE — FRESH FINAL FNO vs LNO COMPARISON"
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
    f"{fno_struct['mean_trace_error']:.12e}"
)

print(
    "LNO mean trace error:",
    f"{lno_struct['mean_trace_error']:.12e}"
)

print(
    "FNO minimum eigenvalue:",
    f"{fno_struct['minimum_eigenvalue']:.12e}"
)

print(
    "LNO minimum eigenvalue:",
    f"{lno_struct['minimum_eigenvalue']:.12e}"
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
    "[INFO] No checkpoint was modified."
)

print(
    "[INFO] No new model inference was performed."
)

print(
    "[INFO] Current final prediction artifacts were "
    "auto-detected under /content."
)

print("=" * 110)
