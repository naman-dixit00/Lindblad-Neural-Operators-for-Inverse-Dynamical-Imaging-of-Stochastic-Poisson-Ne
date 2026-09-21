# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 62
# Step            : STEP_18
# Step Heading    : #   Step 18 Final Repaired LNO artifacts
# Step Cell No.   : 6
# ============================================================

# ============================================================
# STEP 18B — PUBLICATION-QUALITY SQUARE FIGURE PACK
# FINAL POLISHED + SCIENTIFICALLY CORRECTED
# ============================================================
#
# Source:
#   Step 18 Final Repaired LNO artifacts
#
# NO RETRAINING
# NO DATA REGENERATION
#
# Output:
#   results/step18_final_lno/figures_square/
#
# Formats:
#   PNG  -> 600 DPI
#   SVG  -> vector
#   PDF  -> vector
#   Combined PDF figure pack
#
# Figure geometry:
#   3.5 x 3.5 inch
#   1:1 square
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

from matplotlib.ticker import ScalarFormatter
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# 2. ROOT
# ============================================================

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

print("=" * 110)
print(
    "STEP 18B — PUBLICATION-QUALITY SQUARE FIGURE PACK"
)
print(
    "FINAL POLISHED — LNO DYNAMICS"
)
print("=" * 110)


# ============================================================
# 3. PATHS
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
# 4. VERIFY ARTIFACTS
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "A. STEP-18 ARTIFACT VERIFICATION"
)

print(
    "=" * 110
)

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

missing = []

for name, path in required.items():

    exists = path.is_file()

    print(
        f"{name:28s}: {exists}"
    )

    if not exists:

        missing.append(
            str(path)
        )

if missing:

    raise FileNotFoundError(
        "\nMissing required Step-18 artifacts:\n"
        +
        "\n".join(
            missing
        )
    )

print(
    "\n[PASS] All Step-18 plotting artifacts found."
)


# ============================================================
# 5. LOAD DATA
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
# 6. COLUMN VALIDATION
# ============================================================

expected_history_columns = {

    "epoch",

    "train_loss",
    "train_R_loss",
    "train_amplitude_loss",

    "val_loss",
    "val_R_loss",
    "val_amplitude_loss",

    "kappa_lindblad",
    "kappa_neural",
}

expected_regime_columns = {

    "regime",

    "R_relative_L2",

    "amplitude_relative_L2",
}

expected_environment_columns = {

    "gamma",

    "R_change",

    "amplitude_change",
}

expected_rollout_columns = {

    "step",

    "R_norm",

    "trace_mean",

    "min_eigenvalue",

}

missing_history_columns = (
    expected_history_columns
    -
    set(
        history.columns
    )
)

missing_regime_columns = (
    expected_regime_columns
    -
    set(
        regime_df.columns
    )
)

missing_environment_columns = (
    expected_environment_columns
    -
    set(
        environment_df.columns
    )
)

missing_rollout_columns = (
    expected_rollout_columns
    -
    set(
        rollout_df.columns
    )
)

if missing_history_columns:

    raise KeyError(
        "Missing history columns: "
        +
        str(
            sorted(
                missing_history_columns
            )
        )
    )

if missing_regime_columns:

    raise KeyError(
        "Missing regime columns: "
        +
        str(
            sorted(
                missing_regime_columns
            )
        )
    )

if missing_environment_columns:

    raise KeyError(
        "Missing environment columns: "
        +
        str(
            sorted(
                missing_environment_columns
            )
        )
    )

if missing_rollout_columns:

    raise KeyError(
        "Missing rollout columns: "
        +
        str(
            sorted(
                missing_rollout_columns
            )
        )
    )

print(
    "[PASS] All expected plotting columns verified."
)


# ============================================================
# 7. BASIC DATA SANITY
# ============================================================

for name, frame in {

    "history":
        history,

    "regime":
        regime_df,

    "environment":
        environment_df,

    "rollout":
        rollout_df,

}.items():

    numeric_values = frame.select_dtypes(
        include=[
            np.number
        ]
    ).to_numpy()

    assert np.isfinite(
        numeric_values
    ).all()

    print(
        f"[PASS] {name.capitalize()} data finite."
    )


# ============================================================
# 8. TEST PREDICTION VALIDATION
# ============================================================

prediction_keys = [

    "R_prediction",

    "R_target",

    "amplitude_prediction",

    "amplitude_target",

]

for key in prediction_keys:

    if key not in pred:

        raise KeyError(
            f"Missing prediction key: {key}"
        )

    assert np.isfinite(
        pred[key]
    ).all()

print(
    "[PASS] Prediction arrays finite."
)


# ============================================================
# 9. PUBLICATION-QUALITY SCIENTIFIC PALETTE
# ============================================================

SCI_COLORS = {

    "primary":
        "#003366",

    "secondary":
        "#006699",

    "accent1":
        "#990000",

    "accent2":
        "#E67300",

    "accent3":
        "#006633",

    "dark":
        "#2B2B2B",

    "gray":
        "#7A7A7A",

    "light_gray":
        "#B0B0B0",

}


# ============================================================
# 10. TYPOGRAPHY / MATPLOTLIB SETTINGS
# ============================================================

plt.rcParams.update({

    "font.family":
        "serif",

    "font.serif":
        [
            "Times New Roman",
            "DejaVu Serif",
        ],

    "mathtext.fontset":
        "stix",

    "font.size":
        9,

    "axes.titlesize":
        10,

    "axes.labelsize":
        9,

    "xtick.labelsize":
        8,

    "ytick.labelsize":
        8,

    "legend.fontsize":
        7.5,

    "figure.dpi":
        200,

    "savefig.dpi":
        600,

    "axes.linewidth":
        0.8,

    "xtick.direction":
        "in",

    "ytick.direction":
        "in",

    "xtick.major.size":
        3.5,

    "ytick.major.size":
        3.5,

    "xtick.minor.size":
        2,

    "ytick.minor.size":
        2,

    "xtick.top":
        True,

    "ytick.right":
        True,

    "grid.linestyle":
        ":",

    "grid.alpha":
        0.6,

    "grid.color":
        SCI_COLORS[
            "light_gray"
        ],

})


# ============================================================
# 11. FIGURE SIZE
# ============================================================

FIGSIZE = (
    3.5,
    3.5
)


# ============================================================
# 12. PDF PACK
# ============================================================

pdf_path = (
    FIG_DIR
    / "step18_square_figure_pack.pdf"
)

pdf = PdfPages(
    pdf_path
)


# ============================================================
# 13. HELPER FUNCTIONS
# ============================================================

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
        dpi=600,
        facecolor="white",
        edgecolor="none",
        bbox_inches="tight"
    )

    fig.savefig(
        svg_path,
        facecolor="white",
        edgecolor="none",
        bbox_inches="tight"
    )

    fig.savefig(
        single_pdf_path,
        facecolor="white",
        edgecolor="none",
        bbox_inches="tight"
    )

    pdf.savefig(
        fig,
        facecolor="white",
        edgecolor="none",
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
# 14. BEST EPOCH / BEST COUPLINGS
# ============================================================

best_epoch = int(
    summary.get(
        "best_epoch",
        history[
            "epoch"
        ].iloc[-1]
    )
)

best_epoch_rows = history[
    history[
        "epoch"
    ]
    ==
    best_epoch
]

if best_epoch_rows.empty:

    raise ValueError(
        f"Best epoch {best_epoch} not found "
        "in training history."
    )

best_row = best_epoch_rows.iloc[
    0
]

best_kappa_L = float(
    best_row[
        "kappa_lindblad"
    ]
)

best_kappa_N = float(
    best_row[
        "kappa_neural"
    ]
)

print(
    "\n[+] Best epoch:",
    best_epoch
)

print(
    "[+] Best epoch kappa_L:",
    f"{best_kappa_L:.8f}"
)

print(
    "[+] Best epoch kappa_N:",
    f"{best_kappa_N:.8f}"
)


# ============================================================
# FIGURE 01
# TRAINING TOTAL LOSS
# ============================================================

fig, ax = plt.subplots(
    figsize=FIGSIZE,
    layout="constrained"
)

ax.plot(

    history[
        "epoch"
    ],

    history[
        "train_loss"
    ],

    color=SCI_COLORS[
        "primary"
    ],

    marker="o",

    linewidth=1.2,

    markersize=4,

    markeredgecolor="white",

    markeredgewidth=0.5,

    label="Train",

)

ax.plot(

    history[
        "epoch"
    ],

    history[
        "val_loss"
    ],

    color=SCI_COLORS[
        "accent1"
    ],

    marker="s",

    linewidth=1.2,

    markersize=4,

    markeredgecolor="white",

    markeredgewidth=0.5,

    label="Validation",

)

ax.axvline(

    best_epoch,

    color=SCI_COLORS[
        "gray"
    ],

    linestyle=":",

    linewidth=1.0,

)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "Total Loss (MSE)"
)

ax.set_title(
    "LNO Training Convergence"
)

ax.grid(
    True
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
    "01_training_convergence"
)


# ============================================================
# FIGURE 02
# DECOMPOSED LEARNING DYNAMICS
# ============================================================

fig, ax = plt.subplots(
    figsize=FIGSIZE,
    layout="constrained"
)

ax.plot(

    history[
        "epoch"
    ],

    history[
        "train_R_loss"
    ],

    color=SCI_COLORS[
        "primary"
    ],

    marker="o",

    linewidth=1.2,

    markersize=3.5,

    markeredgecolor="white",

    markeredgewidth=0.4,

    label="Train $R$"
)

ax.plot(

    history[
        "epoch"
    ],

    history[
        "val_R_loss"
    ],

    color=SCI_COLORS[
        "secondary"
    ],

    marker="s",

    linewidth=1.2,

    markersize=3.5,

    linestyle="--",

    markeredgecolor="white",

    markeredgewidth=0.4,

    label="Val $R$"
)

ax.plot(

    history[
        "epoch"
    ],

    history[
        "train_amplitude_loss"
    ],

    color=SCI_COLORS[
        "accent1"
    ],

    marker="^",

    linewidth=1.2,

    markersize=3.5,

    markeredgecolor="white",

    markeredgewidth=0.4,

    label="Train amplitude"
)

ax.plot(

    history[
        "epoch"
    ],

    history[
        "val_amplitude_loss"
    ],

    color=SCI_COLORS[
        "accent2"
    ],

    marker="D",

    linewidth=1.2,

    markersize=3.5,

    linestyle="--",

    markeredgecolor="white",

    markeredgewidth=0.4,

    label="Val amplitude"
)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "Component MSE Loss"
)

ax.set_title(
    "Decomposed Learning Dynamics"
)

ax.grid(
    True
)

ax.legend(
    frameon=False,
    ncol=2,
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
    figsize=FIGSIZE,
    layout="constrained"
)

ax.plot(

    history[
        "epoch"
    ],

    history[
        "kappa_lindblad"
    ],

    color=SCI_COLORS[
        "primary"
    ],

    marker="o",

    linewidth=1.5,

    markersize=4,

    markeredgecolor="white",

    markeredgewidth=0.5,

    label=r"$\kappa_L$ (Lindblad)"

)

ax.plot(

    history[
        "epoch"
    ],

    history[
        "kappa_neural"
    ],

    color=SCI_COLORS[
        "accent2"
    ],

    marker="s",

    linewidth=1.5,

    markersize=4,

    markeredgecolor="white",

    markeredgewidth=0.5,

    label=r"$\kappa_N$ (Neural)"

)

ax.axvline(

    best_epoch,

    color=SCI_COLORS[
        "gray"
    ],

    linestyle=":",

    linewidth=1.0,

    alpha=0.9,

)

ax.annotate(

    (
        f"Best epoch = {best_epoch}\n"
        rf"$\kappa_L={best_kappa_L:.3f}$" "\n"
        rf"$\kappa_N={best_kappa_N:.3f}$"
    ),

    xy=(

        best_epoch,

        best_kappa_L

    ),

    xytext=(

        best_epoch + 1.0,

        best_kappa_L - 0.20

    ),

    arrowprops={

        "arrowstyle":
            "->",

        "linewidth":
            0.8,

        "color":
            SCI_COLORS[
                "dark"
            ],
    },

    fontsize=7.2,

    bbox={

        "boxstyle":
            "round,pad=0.3",

        "facecolor":
            "white",

        "edgecolor":
            "#CCCCCC",

        "alpha":
            0.95,
    },

)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "Learned Coupling Magnitude"
)

ax.set_title(
    "Learned Dynamical Couplings"
)

ax.grid(
    True
)

ax.legend(
    frameon=False,
    loc="upper left"
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
# CONTROLLED GAMMA SWEEP — R
# ============================================================

fig, ax = plt.subplots(
    figsize=FIGSIZE,
    layout="constrained"
)

ax.plot(

    environment_df[
        "gamma"
    ],

    environment_df[
        "R_change"
    ],

    color=SCI_COLORS[
        "primary"
    ],

    marker="o",

    linewidth=1.4,

    markersize=4.5,

    markeredgecolor="white",

    markeredgewidth=0.5,
)

ax.set_xlabel(
    r"Dissipation parameter $\gamma$"
)

ax.set_ylabel(
    r"$\|R_{\mathrm{out}}-R_{\mathrm{in}}\|_F$"
)

ax.set_title(
    r"Controlled $\gamma$-Sweep of Info-State Response"
)

ax.grid(
    True
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
# CONTROLLED GAMMA SWEEP — AMPLITUDE
# ============================================================

fig, ax = plt.subplots(
    figsize=FIGSIZE,
    layout="constrained"
)

ax.plot(

    environment_df[
        "gamma"
    ],

    environment_df[
        "amplitude_change"
    ],

    color=SCI_COLORS[
        "accent1"
    ],

    marker="o",

    linewidth=1.4,

    markersize=4.5,

    markeredgecolor="white",

    markeredgewidth=0.5,
)

ax.set_xlabel(
    r"Dissipation parameter $\gamma$"
)

ax.set_ylabel(
    "Amplitude Response Magnitude"
)

ax.set_title(
    r"Controlled $\gamma$-Sweep of Amplitude Response"
)

ax.grid(
    True
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

regime_plot[
    "order"
] = regime_plot[
    "regime"
].apply(

    lambda x:

        regime_order.index(
            x
        )

        if x in regime_order

        else 999

)

regime_plot = regime_plot.sort_values(
    "order"
)

x = np.arange(
    len(
        regime_plot
    )
)

regime_labels = [

    str(
        value
    )
    .replace(
        "_",
        " "
    )
    .title()

    for value in regime_plot[
        "regime"
    ]

]

fig, ax = plt.subplots(
    figsize=FIGSIZE,
    layout="constrained"
)

bars = ax.bar(

    x,

    regime_plot[
        "R_relative_L2"
    ].to_numpy(),

    color=SCI_COLORS[
        "secondary"
    ],

    edgecolor=SCI_COLORS[
        "dark"
    ],

    linewidth=0.8,

    width=0.60,

    alpha=0.9,
)

ax.bar_label(

    bars,

    fmt="%.3f",

    padding=2,

    fontsize=6.5,
)

ax.set_xticks(
    x
)

ax.set_xticklabels(

    regime_labels,

    rotation=30,

    ha="right"
)

ax.set_ylabel(
    r"Relative $L_2$ Error ($R$)"
)

ax.set_title(
    "Info-State Error Across Regimes"
)

ax.grid(
    True,
    axis="y"
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
# REGIME-WISE AMPLITUDE RELATIVE L2
# ============================================================

fig, ax = plt.subplots(
    figsize=FIGSIZE,
    layout="constrained"
)

bars = ax.bar(

    x,

    regime_plot[
        "amplitude_relative_L2"
    ].to_numpy(),

    color=SCI_COLORS[
        "accent2"
    ],

    edgecolor=SCI_COLORS[
        "dark"
    ],

    linewidth=0.8,

    width=0.60,

    alpha=0.9,
)

ax.bar_label(

    bars,

    fmt="%.3f",

    padding=2,

    fontsize=6.5,
)

ax.set_xticks(
    x
)

ax.set_xticklabels(

    regime_labels,

    rotation=30,

    ha="right"
)

ax.set_ylabel(
    r"Relative $L_2$ Error (Amplitude)"
)

ax.set_title(
    "Amplitude Error Across Regimes"
)

ax.grid(
    True,
    axis="y"
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
# REPRESENTATIVE 40-STEP ROLLOUT
# ============================================================

fig, ax = plt.subplots(
    figsize=FIGSIZE,
    layout="constrained"
)

ax.plot(

    rollout_df[
        "step"
    ],

    rollout_df[
        "R_norm"
    ],

    color=SCI_COLORS[
        "primary"
    ],

    marker="o",

    linewidth=1.3,

    markersize=3.2,

    markeredgecolor="white",

    markeredgewidth=0.4,
)

ax.set_xlabel(
    "Rollout Step"
)

ax.set_ylabel(
    r"Frobenius Norm $\|R\|_F$"
)

ax.set_title(
    "Representative 40-Step Info-State Rollout"
)

ax.grid(
    True
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
# REPRESENTATIVE ROLLOUT MINIMUM EIGENVALUE
# ============================================================

fig, ax = plt.subplots(
    figsize=FIGSIZE,
    layout="constrained"
)

ax.plot(

    rollout_df[
        "step"
    ],

    rollout_df[
        "min_eigenvalue"
    ],

    color=SCI_COLORS[
        "accent3"
    ],

    marker="o",

    linewidth=1.3,

    markersize=3.2,

    markeredgecolor="white",

    markeredgewidth=0.4,
)

ax.axhline(

    0.0,

    color=SCI_COLORS[
        "dark"
    ],

    linestyle="-",

    linewidth=0.7,

    alpha=0.7,

)

ax.axhline(

    -1e-6,

    color=SCI_COLORS[
        "accent1"
    ],

    linestyle="--",

    linewidth=0.9,

    alpha=0.85,

)

formatter = ScalarFormatter(
    useMathText=True
)

formatter.set_powerlimits(
    (-3, 3)
)

ax.yaxis.set_major_formatter(
    formatter
)

ax.text(

    0.98,

    0.05,

    r"Tolerance: $\lambda_{\min}>-10^{-6}$",

    transform=ax.transAxes,

    ha="right",

    va="bottom",

    fontsize=6.6,

    color=SCI_COLORS[
        "dark"
    ],
)

ax.set_xlabel(
    "Rollout Step"
)

ax.set_ylabel(
    r"Minimum Eigenvalue $\lambda_{\min}$"
)

ax.set_title(
    "Minimum Eigenvalue — Representative Rollout"
)

ax.grid(
    True
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
# TRACE PRESERVATION
# ============================================================

trace_error = np.abs(

    rollout_df[
        "trace_mean"
    ].to_numpy()

    -

    1.0

)

fig, ax = plt.subplots(
    figsize=FIGSIZE,
    layout="constrained"
)

ax.semilogy(

    rollout_df[
        "step"
    ],

    trace_error
    +
    1e-16,

    color=SCI_COLORS[
        "primary"
    ],

    marker="o",

    linewidth=1.3,

    markersize=3.2,

    markeredgecolor="white",

    markeredgewidth=0.4,
)

ax.axhline(

    1e-6,

    color=SCI_COLORS[
        "accent1"
    ],

    linestyle="--",

    linewidth=0.8,

    alpha=0.8,

)

ax.text(

    0.98,

    0.05,

    r"Values below $10^{-16}$ shown at plotting floor",

    transform=ax.transAxes,

    ha="right",

    va="bottom",

    fontsize=6.3,

    color=SCI_COLORS[
        "gray"
    ],
)

ax.set_xlabel(
    "Rollout Step"
)

ax.set_ylabel(
    r"$|\mathrm{Tr}(R)-1|$"
)

ax.set_title(
    "Trace Preservation — Representative Rollout"
)

ax.grid(
    True,
    which="both"
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

assert (
    R_prediction.shape
    ==
    R_target.shape
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

median_R = float(
    np.median(
        R_sample_rmse
    )
)

fig, ax = plt.subplots(
    figsize=FIGSIZE,
    layout="constrained"
)

ax.hist(

    R_sample_rmse,

    bins=35,

    color=SCI_COLORS[
        "secondary"
    ],

    edgecolor="white",

    linewidth=0.5,

    alpha=0.9,

)

ax.axvline(

    median_R,

    color=SCI_COLORS[
        "dark"
    ],

    linestyle="--",

    linewidth=1.2,

    label=(
        "Median: "
        +
        scientific(
            median_R
        )
    ),

)

ax.set_xlabel(
    r"Per-Transition $R$ RMSE"
)

ax.set_ylabel(
    "Transition Count"
)

ax.set_title(
    "Test-Set Info-State Error Distribution"
)

ax.grid(
    True,
    axis="y"
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

assert (
    A_prediction.shape
    ==
    A_target.shape
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

median_A = float(
    np.median(
        A_sample_rmse
    )
)

fig, ax = plt.subplots(
    figsize=FIGSIZE,
    layout="constrained"
)

ax.hist(

    A_sample_rmse,

    bins=35,

    color=SCI_COLORS[
        "accent2"
    ],

    edgecolor="white",

    linewidth=0.5,

    alpha=0.9,

)

ax.axvline(

    median_A,

    color=SCI_COLORS[
        "dark"
    ],

    linestyle="--",

    linewidth=1.2,

    label=(
        "Median: "
        +
        scientific(
            median_A
        )
    ),

)

ax.set_xlabel(
    "Per-Transition Amplitude RMSE"
)

ax.set_ylabel(
    "Transition Count"
)

ax.set_title(
    "Test-Set Amplitude Error Distribution"
)

ax.grid(
    True,
    axis="y"
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
# 15. CLOSE COMBINED PDF
# ============================================================

pdf.close()

print(
    "\n[PASS] Combined PDF closed successfully."
)


# ============================================================
# 16. FIGURE MANIFEST
# ============================================================

figure_manifest = {

    "figure_pack":
        (
            "Step 18 Publication-Quality "
            "Square Figure Pack — LNO Dynamics"
        ),

    "source":
        "Step 18 Final Repaired LNO artifacts",

    "training_performed":
        False,

    "geometry":
        {

            "width_inch":
                3.5,

            "height_inch":
                3.5,

            "aspect_ratio":
                "1:1",

        },

    "raster_dpi":
        600,

    "best_epoch":
        best_epoch,

    "best_epoch_kappa_lindblad":
        best_kappa_L,

    "best_epoch_kappa_neural":
        best_kappa_N,

    "figures":
        [

            {

                "id":
                    1,

                "name":
                    "01_training_convergence",

                "description":
                    "Train and validation total loss",

            },

            {

                "id":
                    2,

                "name":
                    "02_decomposed_learning",

                "description":
                    "R and amplitude learning components",

            },

            {

                "id":
                    3,

                "name":
                    "03_learned_couplings",

                "description":
                    "Learned Lindblad and neural couplings",

            },

            {

                "id":
                    4,

                "name":
                    "04_environment_R_response",

                "description":
                    "Controlled gamma sweep of R response",

            },

            {

                "id":
                    5,

                "name":
                    "05_environment_amplitude_response",

                "description":
                    "Controlled gamma sweep of amplitude response",

            },

            {

                "id":
                    6,

                "name":
                    "06_regime_R_relative_L2",

                "description":
                    "R relative-L2 error by environment regime",

            },

            {

                "id":
                    7,

                "name":
                    "07_regime_amplitude_relative_L2",

                "description":
                    "Amplitude relative-L2 error by regime",

            },

            {

                "id":
                    8,

                "name":
                    "08_rollout_R_norm",

                "description":
                    "Representative 40-step R rollout",

            },

            {

                "id":
                    9,

                "name":
                    "09_rollout_minimum_eigenvalue",

                "description":
                    "Representative rollout minimum eigenvalue",

            },

            {

                "id":
                    10,

                "name":
                    "10_rollout_trace_error",

                "description":
                    "Representative rollout trace preservation",

            },

            {

                "id":
                    11,

                "name":
                    "11_test_R_error_distribution",

                "description":
                    "Distribution of per-transition R RMSE",

            },

            {

                "id":
                    12,

                "name":
                    "12_test_amplitude_error_distribution",

                "description":
                    "Distribution of per-transition amplitude RMSE",

            },

        ],

    "formats":
        [

            "PNG",

            "SVG",

            "PDF",

            "Combined PDF",

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
# 17. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "STEP 18B COMPLETE — FINAL SQUARE FIGURE PACK"
)

print(
    "=" * 110
)

print(
    "Figures generated:",
    12
)

print(
    "Output directory:",
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
    "\nFormats:"
)

print(
    "  PNG  — 600 DPI"
)

print(
    "  SVG  — vector"
)

print(
    "  PDF  — vector"
)

print(
    "  Combined PDF — all 12 figures"
)

print(
    "\n[PASS] Publication-quality square figure pack generated."
)

print(
    "[PASS] No model retraining performed."
)

print(
    "[PASS] No dataset modification performed."
)

print(
    "[INFO] Representative-rollout wording used for single-rollout diagnostics."
)

print(
    "[INFO] Numerical plotting floor explicitly documented."
)

print("=" * 110)
