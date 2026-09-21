# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 61
# Step            : STEP_18
# Step Heading    : # Uses ONLY already-generated Step-18 artifacts.
# Step Cell No.   : 5
# ============================================================

# ============================================================
# STEP 18B — PUBLICATION-QUALITY SQUARE FIGURE PACK
# ============================================================
#
# Uses ONLY already-generated Step-18 artifacts.
# NO MODEL RETRAINING.
#
# Output:
#   results/step18_final_lno/figures_square/
#
# Each figure:
#   - square 1:1 layout
#   - 300 DPI PNG
#   - SVG
#   - PDF
#
# ============================================================

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# 1. ROOT
# ============================================================

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

print("=" * 110)
print("STEP 18B — PUBLICATION-QUALITY SQUARE FIGURE PACK")
print("=" * 110)


# ============================================================
# 2. PATHS
# ============================================================

LNO_DIR = (
    ROOT
    / "results"
    / "step18_final_lno"
)

FIG_DIR = (
    LNO_DIR
    / "figures_square"
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


HISTORY_PATH = (
    LNO_DIR
    / "lno_final_training_history.csv"
)

REGIME_PATH = (
    LNO_DIR
    / "lno_final_regime_metrics.csv"
)

ENV_PATH = (
    LNO_DIR
    / "lno_final_environment_response.csv"
)

ROLLOUT_PATH = (
    LNO_DIR
    / "lno_final_rollout_stability.csv"
)

PRED_PATH = (
    LNO_DIR
    / "lno_final_test_predictions.npz"
)

SUMMARY_PATH = (
    LNO_DIR
    / "lno_final_summary.json"
)


# ============================================================
# 3. VERIFY ARTIFACTS
# ============================================================

required = {

    "Training history":
        HISTORY_PATH,

    "Regime metrics":
        REGIME_PATH,

    "Environment response":
        ENV_PATH,

    "Rollout stability":
        ROLLOUT_PATH,

    "Test predictions":
        PRED_PATH,

    "LNO summary":
        SUMMARY_PATH,

}

for name, path in required.items():

    exists = path.is_file()

    print(
        f"{name:25s}: {exists}"
    )

    if not exists:

        raise FileNotFoundError(
            f"Missing required Step-18 artifact:\n{path}"
        )

print(
    "\n[PASS] All Step-18 plotting artifacts found."
)


# ============================================================
# 4. LOAD DATA
# ============================================================

history = pd.read_csv(
    HISTORY_PATH
)

regime_df = pd.read_csv(
    REGIME_PATH
)

environment_df = pd.read_csv(
    ENV_PATH
)

rollout_df = pd.read_csv(
    ROLLOUT_PATH
)

pred = np.load(
    PRED_PATH
)

with open(
    SUMMARY_PATH,
    "r"
) as f:

    summary = json.load(
        f
    )

print(
    "[PASS] All plotting data loaded."
)


# ============================================================
# 5. TYPOGRAPHY / FIGURE SETTINGS
# ============================================================

plt.rcParams.update({

    "font.family":
        "serif",

    "font.serif":
        [
            "DejaVu Serif"
        ],

    "font.size":
        12,

    "axes.titlesize":
        16,

    "axes.labelsize":
        13,

    "xtick.labelsize":
        11,

    "ytick.labelsize":
        11,

    "legend.fontsize":
        10,

    "figure.dpi":
        150,

    "savefig.dpi":
        300,

    "axes.linewidth":
        1.0,

})


# ============================================================
# 6. HELPER
# ============================================================

pdf_path = (
    FIG_DIR
    / "step18_square_figure_pack.pdf"
)

pdf = PdfPages(
    pdf_path
)


def save_square(
    fig,
    name
):

    png_path = (
        FIG_DIR
        / f"{name}.png"
    )

    svg_path = (
        FIG_DIR
        / f"{name}.svg"
    )

    single_pdf_path = (
        FIG_DIR
        / f"{name}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        svg_path,
        bbox_inches="tight"
    )

    fig.savefig(
        single_pdf_path,
        bbox_inches="tight"
    )

    pdf.savefig(
        fig,
        bbox_inches="tight"
    )

    plt.show()

    plt.close(
        fig
    )

    print(
        f"[+] {name}"
    )


def scientific(
    value,
    digits=2
):

    return (
        f"{value:.{digits}e}"
    )


# ============================================================
# FIGURE 01
# TRAINING TOTAL LOSS
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.plot(
    history[
        "epoch"
    ],
    history[
        "train_loss"
    ],
    marker="o",
    linewidth=2,
    markersize=4,
    label="Train"
)

ax.plot(
    history[
        "epoch"
    ],
    history[
        "val_loss"
    ],
    marker="s",
    linewidth=2,
    markersize=4,
    label="Validation"
)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "Total Loss"
)

ax.set_title(
    "LNO Training Convergence"
)

ax.grid(
    True,
    alpha=0.20
)

ax.legend(
    frameon=False
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

save_square(
    fig,
    "01_training_convergence"
)


# ============================================================
# FIGURE 02
# R vs AMPLITUDE LOSS
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.plot(
    history[
        "epoch"
    ],
    history[
        "train_R_loss"
    ],
    marker="o",
    linewidth=2,
    markersize=4,
    label="Train R loss"
)

ax.plot(
    history[
        "epoch"
    ],
    history[
        "val_R_loss"
    ],
    marker="s",
    linewidth=2,
    markersize=4,
    label="Validation R loss"
)

ax.plot(
    history[
        "epoch"
    ],
    history[
        "train_amplitude_loss"
    ],
    marker="^",
    linewidth=2,
    markersize=4,
    label="Train amplitude loss"
)

ax.plot(
    history[
        "epoch"
    ],
    history[
        "val_amplitude_loss"
    ],
    marker="D",
    linewidth=2,
    markersize=4,
    label="Validation amplitude loss"
)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "MSE Loss"
)

ax.set_title(
    "Decomposed Learning Dynamics"
)

ax.grid(
    True,
    alpha=0.20
)

ax.legend(
    frameon=False,
    loc="best"
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

save_square(
    fig,
    "02_decomposed_learning"
)


# ============================================================
# FIGURE 03
# LEARNED DYNAMICAL COUPLINGS
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.plot(
    history[
        "epoch"
    ],
    history[
        "kappa_lindblad"
    ],
    marker="o",
    linewidth=2,
    markersize=4,
    label=r"$\kappa_L$ — Lindblad"
)

ax.plot(
    history[
        "epoch"
    ],
    history[
        "kappa_neural"
    ],
    marker="s",
    linewidth=2,
    markersize=4,
    label=r"$\kappa_N$ — Neural"
)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "Learned Coupling"
)

ax.set_title(
    "Learned Dynamical Couplings"
)

ax.grid(
    True,
    alpha=0.20
)

ax.legend(
    frameon=False
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

save_square(
    fig,
    "03_learned_couplings"
)


# ============================================================
# FIGURE 04
# ENVIRONMENT RESPONSE — R
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.plot(
    environment_df[
        "gamma"
    ],
    environment_df[
        "R_change"
    ],
    marker="o",
    linewidth=2.5,
    markersize=6
)

ax.set_xlabel(
    r"Environment Dissipation $\gamma$"
)

ax.set_ylabel(
    r"$\|R_{\mathrm{out}}-R_{\mathrm{in}}\|$"
)

ax.set_title(
    "Environment-Driven Information-State Response"
)

ax.grid(
    True,
    alpha=0.20
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

save_square(
    fig,
    "04_environment_R_response"
)


# ============================================================
# FIGURE 05
# ENVIRONMENT RESPONSE — AMPLITUDE
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.plot(
    environment_df[
        "gamma"
    ],
    environment_df[
        "amplitude_change"
    ],
    marker="o",
    linewidth=2.5,
    markersize=6
)

ax.set_xlabel(
    r"Environment Dissipation $\gamma$"
)

ax.set_ylabel(
    r"Amplitude response magnitude"
)

ax.set_title(
    "Environment-Driven Amplitude Response"
)

ax.grid(
    True,
    alpha=0.20
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

save_square(
    fig,
    "05_environment_amplitude_response"
)


# ============================================================
# FIGURE 06
# REGIME-WISE R RELATIVE L2
# ============================================================

regime_order = [
    "low_noise",
    "stochastic",
    "metastable",
    "heavy_dissipation",
    "collapse",
]

regime_plot = regime_df.copy()

regime_plot["order"] = regime_plot[
    "regime"
].apply(
    lambda x:
        regime_order.index(x)
        if x in regime_order
        else 999
)

regime_plot = regime_plot.sort_values(
    "order"
)

fig, ax = plt.subplots(
    figsize=(7, 7)
)

x = np.arange(
    len(
        regime_plot
    )
)

ax.bar(
    x,
    regime_plot[
        "R_relative_L2"
    ].to_numpy()
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    regime_plot[
        "regime"
    ],
    rotation=25,
    ha="right"
)

ax.set_ylabel(
    "R Relative-L2"
)

ax.set_title(
    "LNO Information-State Error by Regime"
)

ax.grid(
    True,
    axis="y",
    alpha=0.20
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

save_square(
    fig,
    "06_regime_R_relative_L2"
)


# ============================================================
# FIGURE 07
# REGIME-WISE AMPLITUDE
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.bar(
    x,
    regime_plot[
        "amplitude_relative_L2"
    ].to_numpy()
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    regime_plot[
        "regime"
    ],
    rotation=25,
    ha="right"
)

ax.set_ylabel(
    "Amplitude Relative-L2"
)

ax.set_title(
    "LNO Amplitude Error by Regime"
)

ax.grid(
    True,
    axis="y",
    alpha=0.20
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

save_square(
    fig,
    "07_regime_amplitude_relative_L2"
)


# ============================================================
# FIGURE 08
# 40-STEP ROLLOUT NORM
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.plot(
    rollout_df[
        "step"
    ],
    rollout_df[
        "R_norm"
    ],
    linewidth=2.5,
    marker="o",
    markersize=3
)

ax.set_xlabel(
    "Rollout step"
)

ax.set_ylabel(
    r"$\|R\|_F$"
)

ax.set_title(
    "Long-Horizon Information-State Rollout"
)

ax.grid(
    True,
    alpha=0.20
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

save_square(
    fig,
    "08_rollout_R_norm"
)


# ============================================================
# FIGURE 09
# ROLLOUT MINIMUM EIGENVALUE
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.plot(
    rollout_df[
        "step"
    ],
    rollout_df[
        "min_eigenvalue"
    ],
    linewidth=2.5,
    marker="o",
    markersize=3
)

ax.axhline(
    0.0,
    linestyle="--",
    linewidth=1.5
)

ax.set_xlabel(
    "Rollout step"
)

ax.set_ylabel(
    "Minimum eigenvalue"
)

ax.set_title(
    "Lindblad Rollout Spectral Stability"
)

ax.grid(
    True,
    alpha=0.20
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

save_square(
    fig,
    "09_rollout_minimum_eigenvalue"
)


# ============================================================
# FIGURE 10
# TRACE PRESERVATION DURING ROLLOUT
# ============================================================

trace_error = np.abs(
    rollout_df[
        "trace_mean"
    ].to_numpy()
    -
    1.0
)

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.semilogy(
    rollout_df[
        "step"
    ],
    trace_error + 1e-16,
    linewidth=2.5,
    marker="o",
    markersize=3
)

ax.set_xlabel(
    "Rollout step"
)

ax.set_ylabel(
    r"$|\mathrm{Tr}(R)-1|$"
)

ax.set_title(
    "Trace Preservation Across Rollout"
)

ax.grid(
    True,
    alpha=0.20
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

save_square(
    fig,
    "10_rollout_trace_error"
)


# ============================================================
# FIGURE 11
# TEST-SET R ERROR DISTRIBUTION
# ============================================================

R_prediction = pred[
    "R_prediction"
].astype(
    np.float64
)

R_target = pred[
    "R_target"
].astype(
    np.float64
)

R_sample_rmse = np.sqrt(
    np.mean(
        (
            R_prediction
            -
            R_target
        )
        ** 2,
        axis=(
            1,
            2,
            3
        )
    )
)

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.hist(
    R_sample_rmse,
    bins=35,
    edgecolor="black",
    linewidth=0.6
)

ax.axvline(
    np.median(
        R_sample_rmse
    ),
    linestyle="--",
    linewidth=2,
    label="Median"
)

ax.set_xlabel(
    "Per-transition R RMSE"
)

ax.set_ylabel(
    "Number of test transitions"
)

ax.set_title(
    "Test-Set Information-State Error Distribution"
)

ax.grid(
    True,
    axis="y",
    alpha=0.20
)

ax.legend(
    frameon=False
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

save_square(
    fig,
    "11_test_R_error_distribution"
)


# ============================================================
# FIGURE 12
# TEST-SET AMPLITUDE ERROR DISTRIBUTION
# ============================================================

A_prediction = pred[
    "amplitude_prediction"
].astype(
    np.float64
)

A_target = pred[
    "amplitude_target"
].astype(
    np.float64
)

A_sample_rmse = np.sqrt(
    np.mean(
        (
            A_prediction
            -
            A_target
        )
        ** 2,
        axis=1
    )
)

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.hist(
    A_sample_rmse,
    bins=35,
    edgecolor="black",
    linewidth=0.6
)

ax.axvline(
    np.median(
        A_sample_rmse
    ),
    linestyle="--",
    linewidth=2,
    label="Median"
)

ax.set_xlabel(
    "Per-transition amplitude RMSE"
)

ax.set_ylabel(
    "Number of test transitions"
)

ax.set_title(
    "Test-Set Amplitude Error Distribution"
)

ax.grid(
    True,
    axis="y",
    alpha=0.20
)

ax.legend(
    frameon=False
)

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

save_square(
    fig,
    "12_test_amplitude_error_distribution"
)


# ============================================================
# CLOSE PDF
# ============================================================

pdf.close()


# ============================================================
# 7. WRITE FIGURE MANIFEST
# ============================================================

figure_manifest = {

    "figure_pack":
        "Step 18 Publication-Quality Square Figure Pack",

    "format":
        "1:1 square",

    "dpi":
        300,

    "figures":
        [

            "01_training_convergence",
            "02_decomposed_learning",
            "03_learned_couplings",
            "04_environment_R_response",
            "05_environment_amplitude_response",
            "06_regime_R_relative_L2",
            "07_regime_amplitude_relative_L2",
            "08_rollout_R_norm",
            "09_rollout_minimum_eigenvalue",
            "10_rollout_trace_error",
            "11_test_R_error_distribution",
            "12_test_amplitude_error_distribution",

        ],

    "combined_pdf":
        str(
            pdf_path
        ),
}

manifest_path = (
    FIG_DIR
    / "figure_manifest.json"
)

with open(
    manifest_path,
    "w"
) as f:

    json.dump(
        figure_manifest,
        f,
        indent=2
    )


# ============================================================
# 8. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "STEP 18B COMPLETE — SQUARE FIGURE PACK"
)

print(
    "=" * 110
)

print(
    "Figures:",
    12
)

print(
    "Directory:",
    FIG_DIR
)

print(
    "Combined PDF:",
    pdf_path
)

print(
    "Manifest:",
    manifest_path
)

print(
    "\n[PASS] 12 publication-quality square figures generated."
)

print(
    "[PASS] PNG + SVG + individual PDF versions saved."
)

print(
    "[PASS] Combined square PDF figure pack saved."
)

print(
    "[INFO] No model training or data generation was performed."
)

print("=" * 110)
