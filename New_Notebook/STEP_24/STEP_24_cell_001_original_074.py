# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 74
# Step            : STEP_24
# Step Heading    : # STEP 24 — ERROR & STABILITY VISUALIZATION
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 24 — ERROR & STABILITY VISUALIZATION
# ============================================================
#
# Purpose:
#   Convert Step-23 regime-wise rollout diagnostics into
#   consolidated comparative figures and summary tables.
#
# Important:
#   - No retraining
#   - No checkpoint modification
#   - No new model inference
#   - Uses Step-23 saved rollout data
#   - Not a ground-truth accuracy benchmark
# ============================================================

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. ROOT / OUTPUT PATHS
# ============================================================

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

STEP23_DIR = (
    ROOT
    / "results"
    / "step23_regime_analysis"
)

DETAIL_CSV = (
    STEP23_DIR
    / "regime_rollout_details.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "step24_stability_visualization"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

assert DETAIL_CSV.is_file(), (
    f"Missing Step-23 rollout details:\n{DETAIL_CSV}"
)


print("=" * 110)
print("STEP 24 — ERROR & STABILITY VISUALIZATION")
print("=" * 110)

print(
    "[+] Source:",
    DETAIL_CSV
)

print(
    "[+] Output:",
    OUT_DIR
)


# ============================================================
# 2. LOAD STEP-23 DATA
# ============================================================

detail_df = pd.read_csv(
    DETAIL_CSV
)

required_columns = [
    "regime",
    "model",
    "step",
    "R_norm",
    "amplitude_norm",
    "mean_trace_error",
    "max_trace_error",
    "minimum_eigenvalue",
    "negative_eigenvalue_fraction",
    "finite",
]

missing = [
    c
    for c in required_columns
    if c not in detail_df.columns
]

assert not missing, (
    "Missing required Step-23 columns: "
    + str(missing)
)


print(
    "\n[+] Rows loaded:",
    len(detail_df)
)

print(
    "[+] Models:",
    sorted(
        detail_df["model"].unique()
    )
)

print(
    "[+] Regimes:",
    list(
        detail_df["regime"].drop_duplicates()
    )
)


# ============================================================
# 3. BASIC VALIDITY CHECK
# ============================================================

assert set(
    detail_df["model"].unique()
) == {
    "FNO",
    "LNO"
}

expected_regimes = [
    "low_noise",
    "stochastic",
    "heavy_dissipation",
    "collapse",
    "metastable",
]

for regime in expected_regimes:

    for model in [
        "FNO",
        "LNO"
    ]:

        subset = detail_df[
            (
                detail_df["regime"]
                ==
                regime
            )
            &
            (
                detail_df["model"]
                ==
                model
            )
        ]

        assert len(subset) > 0, (
            f"No data for {regime} / {model}"
        )

        assert subset["step"].max() == 100, (
            f"Incomplete rollout for {regime} / {model}"
        )

        assert bool(
            subset["finite"].all()
        ), (
            f"Non-finite rollout detected for "
            f"{regime} / {model}"
        )

print(
    "[PASS] All five regimes have complete finite 100-step rollouts."
)


# ============================================================
# 4. CONSOLIDATED SUMMARY
# ============================================================

summary_rows = []

for regime in expected_regimes:

    for model in [
        "FNO",
        "LNO"
    ]:

        df = detail_df[
            (
                detail_df["regime"]
                ==
                regime
            )
            &
            (
                detail_df["model"]
                ==
                model
            )
        ]

        summary_rows.append(
            {
                "regime": regime,
                "model": model,
                "completed_steps": int(
                    df["step"].max()
                ),
                "worst_minimum_eigenvalue": float(
                    df[
                        "minimum_eigenvalue"
                    ].min()
                ),
                "worst_max_trace_error": float(
                    df[
                        "max_trace_error"
                    ].max()
                ),
                "worst_psd_violation_fraction": float(
                    df[
                        "negative_eigenvalue_fraction"
                    ].max()
                ),
                "max_R_norm": float(
                    df[
                        "R_norm"
                    ].max()
                ),
                "min_R_norm": float(
                    df[
                        "R_norm"
                    ].min()
                ),
                "max_amplitude_norm": float(
                    df[
                        "amplitude_norm"
                    ].max()
                ),
                "min_amplitude_norm": float(
                    df[
                        "amplitude_norm"
                    ].min()
                ),
                "all_finite": bool(
                    df[
                        "finite"
                    ].all()
                ),
            }
        )


summary_df = pd.DataFrame(
    summary_rows
)


SUMMARY_CSV = (
    OUT_DIR
    /
    "stability_visualization_summary.csv"
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)


# ============================================================
# 5. FIGURE 1 — R NORM
# ============================================================

fig, axes = plt.subplots(
    5,
    1,
    figsize=(9, 18),
    sharex=True
)

for ax, regime in zip(
    axes,
    expected_regimes
):

    for model in [
        "FNO",
        "LNO"
    ]:

        df = detail_df[
            (
                detail_df["regime"]
                ==
                regime
            )
            &
            (
                detail_df["model"]
                ==
                model
            )
        ]

        ax.plot(
            df["step"],
            df["R_norm"],
            linewidth=2,
            label=model
        )

    ax.set_ylabel(
        "R norm"
    )

    ax.set_title(
        regime
    )

    ax.grid(
        True,
        alpha=0.25
    )

axes[-1].set_xlabel(
    "Rollout step"
)

axes[0].legend(
    frameon=False
)

fig.suptitle(
    "Step 24 — R-Norm Across Environmental Regimes",
    fontsize=14
)

plt.tight_layout()

RNORM_PLOT = (
    OUT_DIR
    /
    "stability_R_norm_all_regimes.png"
)

plt.savefig(
    RNORM_PLOT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. FIGURE 2 — TRACE ERROR
# ============================================================

fig, axes = plt.subplots(
    5,
    1,
    figsize=(9, 18),
    sharex=True
)

for ax, regime in zip(
    axes,
    expected_regimes
):

    for model in [
        "FNO",
        "LNO"
    ]:

        df = detail_df[
            (
                detail_df["regime"]
                ==
                regime
            )
            &
            (
                detail_df["model"]
                ==
                model
            )
        ]

        ax.plot(
            df["step"],
            df["max_trace_error"],
            linewidth=2,
            label=model
        )

    ax.set_ylabel(
        "Max trace error"
    )

    ax.set_title(
        regime
    )

    ax.grid(
        True,
        alpha=0.25
    )

axes[-1].set_xlabel(
    "Rollout step"
)

axes[0].legend(
    frameon=False
)

fig.suptitle(
    "Step 24 — Trace Preservation Across Environmental Regimes",
    fontsize=14
)

plt.tight_layout()

TRACE_PLOT = (
    OUT_DIR
    /
    "stability_trace_error_all_regimes.png"
)

plt.savefig(
    TRACE_PLOT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. FIGURE 3 — MINIMUM EIGENVALUE
# ============================================================

fig, axes = plt.subplots(
    5,
    1,
    figsize=(9, 18),
    sharex=True
)

for ax, regime in zip(
    axes,
    expected_regimes
):

    for model in [
        "FNO",
        "LNO"
    ]:

        df = detail_df[
            (
                detail_df["regime"]
                ==
                regime
            )
            &
            (
                detail_df["model"]
                ==
                model
            )
        ]

        ax.plot(
            df["step"],
            df["minimum_eigenvalue"],
            linewidth=2,
            label=model
        )

    ax.axhline(
        0.0,
        linewidth=1
    )

    ax.set_ylabel(
        "Min eigenvalue"
    )

    ax.set_title(
        regime
    )

    ax.grid(
        True,
        alpha=0.25
    )

axes[-1].set_xlabel(
    "Rollout step"
)

axes[0].legend(
    frameon=False
)

fig.suptitle(
    "Step 24 — PSD Stability Across Environmental Regimes",
    fontsize=14
)

plt.tight_layout()

EIG_PLOT = (
    OUT_DIR
    /
    "stability_minimum_eigenvalue_all_regimes.png"
)

plt.savefig(
    EIG_PLOT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. FIGURE 4 — AMPLITUDE NORM
# ============================================================

fig, axes = plt.subplots(
    5,
    1,
    figsize=(9, 18),
    sharex=True
)

for ax, regime in zip(
    axes,
    expected_regimes
):

    for model in [
        "FNO",
        "LNO"
    ]:

        df = detail_df[
            (
                detail_df["regime"]
                ==
                regime
            )
            &
            (
                detail_df["model"]
                ==
                model
            )
        ]

        ax.plot(
            df["step"],
            df["amplitude_norm"],
            linewidth=2,
            label=model
        )

    ax.set_ylabel(
        "Amplitude norm"
    )

    ax.set_title(
        regime
    )

    ax.grid(
        True,
        alpha=0.25
    )

axes[-1].set_xlabel(
    "Rollout step"
)

axes[0].legend(
    frameon=False
)

fig.suptitle(
    "Step 24 — Amplitude Stability Across Environmental Regimes",
    fontsize=14
)

plt.tight_layout()

AMP_PLOT = (
    OUT_DIR
    /
    "stability_amplitude_norm_all_regimes.png"
)

plt.savefig(
    AMP_PLOT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. FIGURE 5 — WORST-CASE STRUCTURAL COMPARISON
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 14)
)

x = np.arange(
    len(expected_regimes)
)

width = 0.35


# ------------------------------------------------------------
# Trace error
# ------------------------------------------------------------

fno_trace = []
lno_trace = []

for regime in expected_regimes:

    fno_trace.append(
        summary_df[
            (
                summary_df["regime"]
                ==
                regime
            )
            &
            (
                summary_df["model"]
                ==
                "FNO"
            )
        ][
            "worst_max_trace_error"
        ].iloc[0]
    )

    lno_trace.append(
        summary_df[
            (
                summary_df["regime"]
                ==
                regime
            )
            &
            (
                summary_df["model"]
                ==
                "LNO"
            )
        ][
            "worst_max_trace_error"
        ].iloc[0]
    )

axes[0].bar(
    x - width / 2,
    fno_trace,
    width,
    label="FNO"
)

axes[0].bar(
    x + width / 2,
    lno_trace,
    width,
    label="LNO"
)

axes[0].set_yscale(
    "log"
)

axes[0].set_ylabel(
    "Worst max trace error"
)

axes[0].set_title(
    "Worst Trace Error"
)

axes[0].set_xticks(
    x,
    expected_regimes,
    rotation=20
)

axes[0].legend(
    frameon=False
)

axes[0].grid(
    True,
    axis="y",
    alpha=0.25
)


# ------------------------------------------------------------
# PSD violation
# ------------------------------------------------------------

fno_psd = []
lno_psd = []

for regime in expected_regimes:

    fno_psd.append(
        summary_df[
            (
                summary_df["regime"]
                ==
                regime
            )
            &
            (
                summary_df["model"]
                ==
                "FNO"
            )
        ][
            "worst_psd_violation_fraction"
        ].iloc[0]
    )

    lno_psd.append(
        summary_df[
            (
                summary_df["regime"]
                ==
                regime
            )
            &
            (
                summary_df["model"]
                ==
                "LNO"
            )
        ][
            "worst_psd_violation_fraction"
        ].iloc[0]
    )

axes[1].bar(
    x - width / 2,
    fno_psd,
    width,
    label="FNO"
)

axes[1].bar(
    x + width / 2,
    lno_psd,
    width,
    label="LNO"
)

axes[1].set_ylabel(
    "Worst PSD-violation fraction"
)

axes[1].set_title(
    "Worst Meaningful PSD Violation"
)

axes[1].set_xticks(
    x,
    expected_regimes,
    rotation=20
)

axes[1].grid(
    True,
    axis="y",
    alpha=0.25
)


# ------------------------------------------------------------
# Minimum eigenvalue
# ------------------------------------------------------------

fno_eig = []
lno_eig = []

for regime in expected_regimes:

    fno_eig.append(
        summary_df[
            (
                summary_df["regime"]
                ==
                regime
            )
            &
            (
                summary_df["model"]
                ==
                "FNO"
            )
        ][
            "worst_minimum_eigenvalue"
        ].iloc[0]
    )

    lno_eig.append(
        summary_df[
            (
                summary_df["regime"]
                ==
                regime
            )
            &
            (
                summary_df["model"]
                ==
                "LNO"
            )
        ][
            "worst_minimum_eigenvalue"
        ].iloc[0]
    )

axes[2].bar(
    x - width / 2,
    fno_eig,
    width,
    label="FNO"
)

axes[2].bar(
    x + width / 2,
    lno_eig,
    width,
    label="LNO"
)

axes[2].set_ylabel(
    "Worst minimum eigenvalue"
)

axes[2].set_title(
    "Worst Minimum Eigenvalue"
)

axes[2].set_xticks(
    x,
    expected_regimes,
    rotation=20
)

axes[2].grid(
    True,
    axis="y",
    alpha=0.25
)

plt.tight_layout()

WORST_PLOT = (
    OUT_DIR
    /
    "stability_worst_case_comparison.png"
)

plt.savefig(
    WORST_PLOT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. JSON SUMMARY
# ============================================================

SUMMARY_JSON = (
    OUT_DIR
    /
    "stability_visualization_summary.json"
)

json_summary = {
    "step": 24,
    "title":
        "Error and Stability Visualization",

    "diagnostic_type":
        (
            "Consolidated visualization of Step-23 "
            "regime-wise long-horizon stability diagnostics."
        ),

    "ground_truth_accuracy_claim":
        False,

    "source_step":
        23,

    "regimes":
        expected_regimes,

    "models":
        [
            "FNO",
            "LNO"
        ],

    "rollout_steps":
        100,

    "summary":
        summary_df.to_dict(
            orient="records"
        ),

    "outputs": {
        "summary_csv":
            str(SUMMARY_CSV),

        "R_norm_plot":
            str(RNORM_PLOT),

        "trace_error_plot":
            str(TRACE_PLOT),

        "minimum_eigenvalue_plot":
            str(EIG_PLOT),

        "amplitude_norm_plot":
            str(AMP_PLOT),

        "worst_case_comparison":
            str(WORST_PLOT),

        "summary_json":
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
# 11. FINAL DISPLAY
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "STEP 24 COMPLETE — ERROR & STABILITY VISUALIZATION"
)

print(
    "=" * 110
)

display(
    summary_df.round(8)
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
    "[INFO] This is a visualization/diagnostic stage."
)

print(
    "[INFO] It is not a new ground-truth accuracy benchmark."
)

print(
    "=" * 110
)
