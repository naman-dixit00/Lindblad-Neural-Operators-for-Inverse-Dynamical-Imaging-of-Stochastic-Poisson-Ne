# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 89
# Step            : STEP_30
# Step Heading    : # STEP 30 — FINAL IEEE / JOURNAL-QUALITY VISUALIZATION PACKAGE
# Step Cell No.   : 1
# ============================================================

# ======================================================================================
# STEP 30 — FINAL IEEE / JOURNAL-QUALITY VISUALIZATION PACKAGE
# ======================================================================================
#
# FULL PUBLICATION PACKAGE
#
# Generates:
#
# 2D PUBLICATION FIGURES
#   01. Overall FNO vs Original LNO vs C-LNO accuracy
#   02. Regime-wise R accuracy
#   03. Structural preservation
#   04. C-LNO 100-step trace stability
#   05. C-LNO 100-step PSD stability
#   06. C-LNO R-norm stability
#   07. Regime-wise structural stability
#   08. Final paper summary
#
# 3D FIGURES
#   09. C-LNO long-horizon R-norm surface
#   10. 3D R-matrix structure
#
# ARCHITECTURE
#   11. Publication-quality LNO architecture schematic
#
# GIFS
#   12. LNO architecture working animation
#   13. 3D C-LNO R-norm evolution animation
#
# FORMATS
#   PNG 600 DPI
#   SVG vector
#   PDF vector
#
# ASPECT RATIOS
#   Square
#   IEEE two-column
#   Wide
#
# IMPORTANT
#   - NO training
#   - NO model inference
#   - NO checkpoint modification
#   - Uses ONLY validated Step 27 / 28 / 29 evidence
#   - Staged execution
#   - Aggressive figure cleanup / garbage collection
#   - Memory-safe GIF generation
#   - No Unicode subscript glyph dependency
# ======================================================================================


# ======================================================================================
# 0. IMPORTS
# ======================================================================================

import os
import gc
import json
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from PIL import Image


# ======================================================================================
# 1. ROOT AUTO-DETECTION
# ======================================================================================

CONTENT_ROOT = Path("/content")

repo_candidates = [
    p
    for p in CONTENT_ROOT.iterdir()
    if (
        p.is_dir()
        and (p / "results").is_dir()
        and (p / "physics").is_dir()
        and (p / "training").is_dir()
    )
]

assert repo_candidates, "Repository root not found."

ROOT = repo_candidates[0]

os.chdir(ROOT)

print("=" * 110)
print("STEP 30 — IEEE / JOURNAL-QUALITY VISUALIZATION PACKAGE")
print("=" * 110)
print("[+] Root:", ROOT)


# ======================================================================================
# 2. SOURCE EVIDENCE DIRECTORIES
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

STEP29_DIR = (
    ROOT
    / "results"
    / "step29_final_c_lno_evaluation"
)

assert STEP27_DIR.is_dir(), STEP27_DIR
assert STEP28_DIR.is_dir(), STEP28_DIR
assert STEP29_DIR.is_dir(), STEP29_DIR


# ======================================================================================
# 3. REQUIRED SOURCE FILES
# ======================================================================================

STEP27_OVERALL = (
    STEP27_DIR
    / "three_model_comparison.csv"
)

STEP27_REGIME = (
    STEP27_DIR
    / "regime_comparison.csv"
)

STEP28_ROLLOUT = (
    STEP28_DIR
    / "c_lno_rollout_details.csv"
)

STEP28_REGIME = (
    STEP28_DIR
    / "c_lno_regime_summary.csv"
)

STEP28_GLOBAL = (
    STEP28_DIR
    / "c_lno_global_summary.csv"
)

STEP29_JSON = (
    STEP29_DIR
    / "step29_final_summary.json"
)

for path in [
    STEP27_OVERALL,
    STEP27_REGIME,
    STEP28_ROLLOUT,
    STEP28_REGIME,
    STEP28_GLOBAL,
    STEP29_JSON,
]:
    assert path.is_file(), f"Missing evidence file: {path}"


# ======================================================================================
# 4. LOAD VALIDATED EVIDENCE
# ======================================================================================

overall_df = pd.read_csv(STEP27_OVERALL)
regime_df = pd.read_csv(STEP27_REGIME)
rollout_df = pd.read_csv(STEP28_ROLLOUT)
stability_df = pd.read_csv(STEP28_REGIME)
global_df = pd.read_csv(STEP28_GLOBAL)

with open(
    STEP29_JSON,
    "r",
    encoding="utf-8"
) as f:
    final_json = json.load(f)

print("[PASS] Step 27/28/29 evidence loaded.")


# ======================================================================================
# 5. VALIDATE SOURCE SHAPES / CONTENT
# ======================================================================================

assert set(
    overall_df["model"]
) == {
    "FNO",
    "Final_LNO",
    "Step26_C_LNO"
}

assert len(regime_df) == 5
assert len(stability_df) == 5
assert len(rollout_df) == 2500

print("[PASS] Evidence integrity checks passed.")


# ======================================================================================
# 6. PUBLICATION STYLE
# ======================================================================================

mpl.rcParams.update({

    "font.family": "serif",

    "font.serif": [
        "Times New Roman",
        "Times",
        "STIXGeneral",
        "DejaVu Serif"
    ],

    "mathtext.fontset": "stix",

    "axes.titlesize": 12,
    "axes.labelsize": 10,

    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,

    "legend.fontsize": 8.5,

    "axes.linewidth": 0.75,

    "lines.linewidth": 1.7,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.dpi": 600,

    "figure.dpi": 120,

    "axes.titlepad": 7,
})

print("[PASS] Publication style configured.")


# ======================================================================================
# 7. COLOR SYSTEM
# ======================================================================================

COLORS = {

    "FNO":
        "#2457C5",

    "Final_LNO":
        "#00897B",

    "C_LNO":
        "#C6285C",

    "low_noise":
        "#1565C0",

    "stochastic":
        "#7E57C2",

    "heavy_dissipation":
        "#EF6C00",

    "collapse":
        "#C6285C",

    "metastable":
        "#00897B",

    "dark":
        "#20242A",

    "gray":
        "#6B7280",

    "light_grid":
        "#D7DBE0"
}


# ======================================================================================
# 8. OUTPUT DIRECTORIES
# ======================================================================================

OUT = (
    ROOT
    / "results"
    / "step30_ieee_publication_package"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

PNG_DIR = OUT / "PNG"
SVG_DIR = OUT / "SVG"
PDF_DIR = OUT / "PDF"

SQUARE_DIR = OUT / "SQUARE"
TWO_COLUMN_DIR = OUT / "TWO_COLUMN"
WIDE_DIR = OUT / "WIDE"

THREED_DIR = OUT / "3D"
GIF_DIR = OUT / "GIF"
ARCH_DIR = OUT / "ARCHITECTURE"

EVIDENCE_DIR = OUT / "validated_evidence"

TEMP_DIR = OUT / "_temp_gif_frames"

for directory in [
    PNG_DIR,
    SVG_DIR,
    PDF_DIR,
    SQUARE_DIR,
    TWO_COLUMN_DIR,
    WIDE_DIR,
    THREED_DIR,
    GIF_DIR,
    ARCH_DIR,
    EVIDENCE_DIR,
    TEMP_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

print("[PASS] Output directories ready.")


# ======================================================================================
# 9. HELPERS
# ======================================================================================

def cleanup():
    """
    Aggressive matplotlib / Python memory cleanup.
    """
    plt.close("all")
    gc.collect()


def style_axes(ax):
    """
    Clean publication axes.
    """
    ax.grid(
        True,
        alpha=0.20,
        linewidth=0.55
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.75)


def save_layout_variants(
    plot_function,
    basename
):
    """
    Generate square / two-column / wide versions.
    """

    variants = {

        "square":
            (
                SQUARE_DIR,
                (3.45, 3.45)
            ),

        "two_column":
            (
                TWO_COLUMN_DIR,
                (7.10, 3.55)
            ),

        "wide":
            (
                WIDE_DIR,
                (9.20, 4.20)
            ),
    }

    for variant_name, (
        target_dir,
        figsize
    ) in variants.items():

        fig = plot_function(figsize)

        fig.savefig(
            target_dir
            /
            f"{basename}_{variant_name}.png",
            dpi=600,
            bbox_inches="tight",
            facecolor="white"
        )

        fig.savefig(
            target_dir
            /
            f"{basename}_{variant_name}.pdf",
            bbox_inches="tight",
            facecolor="white"
        )

        fig.savefig(
            target_dir
            /
            f"{basename}_{variant_name}.svg",
            bbox_inches="tight",
            facecolor="white"
        )

        plt.close(fig)
        cleanup()


# ======================================================================================
# 10. MODEL DEFINITIONS
# ======================================================================================

MODEL_ORDER = [
    "FNO",
    "Final_LNO",
    "Step26_C_LNO"
]

MODEL_LABELS = {

    "FNO":
        "FNO",

    "Final_LNO":
        "Original LNO",

    "Step26_C_LNO":
        "C-LNO"
}


# ======================================================================================
# 11. FIGURE 01 — OVERALL ACCURACY
# ======================================================================================

def fig01_overall(figsize):

    fig, ax = plt.subplots(
        figsize=figsize
    )

    x = np.arange(
        len(MODEL_ORDER)
    )

    width = 0.34

    r_values = [
        float(
            overall_df.loc[
                overall_df["model"] == model,
                "R_Relative_L2"
            ].iloc[0]
        )
        for model in MODEL_ORDER
    ]

    a_values = [
        float(
            overall_df.loc[
                overall_df["model"] == model,
                "Amplitude_Relative_L2"
            ].iloc[0]
        )
        for model in MODEL_ORDER
    ]

    ax.bar(
        x - width / 2,
        r_values,
        width,
        color=COLORS["FNO"],
        label="R Relative-L2",
        edgecolor="white",
        linewidth=0.7
    )

    ax.bar(
        x + width / 2,
        a_values,
        width,
        color=COLORS["C_LNO"],
        label="Amplitude Relative-L2",
        edgecolor="white",
        linewidth=0.7
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            MODEL_LABELS[m]
            for m in MODEL_ORDER
        ]
    )

    ax.set_ylabel(
        "Relative-L2 error"
    )

    ax.set_title(
        "Overall One-Step Reconstruction Accuracy",
        fontweight="bold"
    )

    ax.legend(
        frameon=False
    )

    style_axes(ax)

    for i, v in enumerate(
        r_values
    ):

        ax.text(
            i - width / 2,
            v + 0.006,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=7.5
        )

    for i, v in enumerate(
        a_values
    ):

        ax.text(
            i + width / 2,
            v + 0.006,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=7.5
        )

    fig.subplots_adjust(
        left=0.16,
        right=0.97,
        bottom=0.18,
        top=0.88
    )

    return fig


print("\n[1/8] Generating overall accuracy figures...")

save_layout_variants(
    fig01_overall,
    "01_overall_accuracy"
)

print("[PASS] Figure 01 complete.")


# ======================================================================================
# 12. FIGURE 02 — REGIME-WISE R ACCURACY
# ======================================================================================

def fig02_regime_r(figsize):

    fig, ax = plt.subplots(
        figsize=figsize
    )

    x = np.arange(
        len(regime_df)
    )

    ax.plot(
        x,
        regime_df[
            "R_Relative_L2_FNO"
        ].to_numpy(),
        marker="o",
        markersize=4,
        linewidth=1.9,
        color=COLORS["FNO"],
        label="FNO"
    )

    ax.plot(
        x,
        regime_df[
            "R_Relative_L2_Final_LNO"
        ].to_numpy(),
        marker="o",
        markersize=4,
        linewidth=1.9,
        color=COLORS["Final_LNO"],
        label="Original LNO"
    )

    ax.plot(
        x,
        regime_df[
            "R_Relative_L2_C_LNO"
        ].to_numpy(),
        marker="o",
        markersize=4.5,
        linewidth=2.2,
        color=COLORS["C_LNO"],
        label="C-LNO"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        regime_df["regime"],
        rotation=22,
        ha="right"
    )

    ax.set_ylabel(
        "R Relative-L2"
    )

    ax.set_title(
        "Regime-Wise R Reconstruction Accuracy",
        fontweight="bold"
    )

    ax.legend(
        frameon=False
    )

    style_axes(ax)

    fig.subplots_adjust(
        left=0.15,
        right=0.98,
        bottom=0.25,
        top=0.88
    )

    return fig


print("\n[2/8] Generating regime-wise R figures...")

save_layout_variants(
    fig02_regime_r,
    "02_regime_wise_R_accuracy"
)

print("[PASS] Figure 02 complete.")


# ======================================================================================
# 13. FIGURE 03 — STRUCTURAL PRESERVATION
# ======================================================================================

def fig03_structure(figsize):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize
    )

    labels = [
        "FNO",
        "Original LNO",
        "C-LNO"
    ]

    model_colors = [
        COLORS["FNO"],
        COLORS["Final_LNO"],
        COLORS["C_LNO"]
    ]

    trace_values = [
        float(
            overall_df.loc[
                overall_df["model"] == model,
                "Mean_trace_error"
            ].iloc[0]
        )
        for model in MODEL_ORDER
    ]

    eig_values = [
        float(
            overall_df.loc[
                overall_df["model"] == model,
                "Minimum_eigenvalue"
            ].iloc[0]
        )
        for model in MODEL_ORDER
    ]

    # Avoid zero on log scale
    trace_values = np.maximum(
        np.asarray(trace_values, dtype=float),
        1e-12
    )

    axes[0].bar(
        labels,
        trace_values,
        color=model_colors,
        edgecolor="white",
        linewidth=0.7
    )

    axes[0].set_yscale(
        "log"
    )

    axes[0].set_ylabel(
        "Mean trace error"
    )

    axes[0].set_title(
        "Trace Preservation"
    )

    axes[1].bar(
        labels,
        eig_values,
        color=model_colors,
        edgecolor="white",
        linewidth=0.7
    )

    axes[1].axhline(
        0.0,
        color=COLORS["dark"],
        linewidth=1.0,
        linestyle="--"
    )

    axes[1].set_ylabel(
        "Minimum eigenvalue"
    )

    axes[1].set_title(
        "PSD Diagnostic"
    )

    for ax in axes:

        style_axes(ax)

        ax.tick_params(
            axis="x",
            rotation=18
        )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.20,
        top=0.86,
        wspace=0.30
    )

    return fig


print("\n[3/8] Generating structural figures...")

save_layout_variants(
    fig03_structure,
    "03_structural_preservation"
)

print("[PASS] Figure 03 complete.")


# ======================================================================================
# 14. FIGURE 04 — 100-STEP TRACE STABILITY
# ======================================================================================

def fig04_trace_stability(figsize):

    fig, ax = plt.subplots(
        figsize=figsize
    )

    for regime in stability_df["regime"]:

        df = rollout_df[
            rollout_df["regime"] == regime
        ]

        curve = (
            df.groupby("step")[
                "max_trace_error"
            ]
            .mean()
        )

        curve_values = np.maximum(
            curve.to_numpy(dtype=float),
            1e-14
        )

        ax.plot(
            curve.index,
            curve_values,
            linewidth=1.65,
            color=COLORS.get(
                regime,
                COLORS["gray"]
            ),
            label=regime
        )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "Rollout step"
    )

    ax.set_ylabel(
        "Maximum trace error"
    )

    ax.set_title(
        "C-LNO Long-Horizon Trace Preservation",
        fontweight="bold"
    )

    ax.legend(
        frameon=False,
        fontsize=7.5
    )

    style_axes(ax)

    fig.subplots_adjust(
        left=0.15,
        right=0.98,
        bottom=0.16,
        top=0.88
    )

    return fig


print("\n[4/8] Generating trace-stability figures...")

save_layout_variants(
    fig04_trace_stability,
    "04_C_LNO_100step_trace_stability"
)

print("[PASS] Figure 04 complete.")


# ======================================================================================
# 15. FIGURE 05 — 100-STEP PSD STABILITY
# ======================================================================================

def fig05_psd_stability(figsize):

    fig, ax = plt.subplots(
        figsize=figsize
    )

    for regime in stability_df["regime"]:

        df = rollout_df[
            rollout_df["regime"] == regime
        ]

        curve = (
            df.groupby("step")[
                "minimum_eigenvalue"
            ]
            .min()
        )

        ax.plot(
            curve.index,
            curve.values,
            linewidth=1.65,
            color=COLORS.get(
                regime,
                COLORS["gray"]
            ),
            label=regime
        )

    ax.axhline(
        0.0,
        color=COLORS["dark"],
        linewidth=1.0,
        linestyle="--"
    )

    ax.set_xlabel(
        "Rollout step"
    )

    ax.set_ylabel(
        "Minimum eigenvalue"
    )

    ax.set_title(
        "C-LNO Long-Horizon PSD Stability",
        fontweight="bold"
    )

    ax.legend(
        frameon=False,
        fontsize=7.5
    )

    style_axes(ax)

    fig.subplots_adjust(
        left=0.15,
        right=0.98,
        bottom=0.16,
        top=0.88
    )

    return fig


print("\n[5/8] Generating PSD-stability figures...")

save_layout_variants(
    fig05_psd_stability,
    "05_C_LNO_100step_PSD_stability"
)

print("[PASS] Figure 05 complete.")


# ======================================================================================
# 16. FIGURE 06 — R NORM STABILITY
# ======================================================================================

def fig06_rnorm(figsize):

    fig, ax = plt.subplots(
        figsize=figsize
    )

    for regime in stability_df["regime"]:

        df = rollout_df[
            rollout_df["regime"] == regime
        ]

        curve = (
            df.groupby("step")[
                "R_norm"
            ]
            .mean()
        )

        ax.plot(
            curve.index,
            curve.values,
            linewidth=1.6,
            color=COLORS.get(
                regime,
                COLORS["gray"]
            ),
            label=regime
        )

    ax.set_xlabel(
        "Rollout step"
    )

    ax.set_ylabel(
        "R norm"
    )

    ax.set_title(
        "C-LNO Long-Horizon R-Norm Stability",
        fontweight="bold"
    )

    ax.legend(
        frameon=False,
        fontsize=7.5
    )

    style_axes(ax)

    fig.subplots_adjust(
        left=0.14,
        right=0.98,
        bottom=0.16,
        top=0.88
    )

    return fig


print("\n[6/8] Generating R-norm figures...")

save_layout_variants(
    fig06_rnorm,
    "06_C_LNO_R_norm_stability"
)

print("[PASS] Figure 06 complete.")


# ======================================================================================
# 17. FIGURE 07 — REGIME STRUCTURAL STABILITY
# ======================================================================================

def fig07_regime_structure(figsize):

    fig, ax = plt.subplots(
        figsize=figsize
    )

    x = np.arange(
        len(stability_df)
    )

    trace = stability_df[
        "worst_max_trace_error"
    ].to_numpy(dtype=float)

    eig_abs = np.abs(
        stability_df[
            "worst_minimum_eigenvalue"
        ].to_numpy(dtype=float)
    )

    trace = np.maximum(
        trace,
        1e-14
    )

    eig_abs = np.maximum(
        eig_abs,
        1e-14
    )

    width = 0.34

    ax.bar(
        x - width / 2,
        trace,
        width,
        color=COLORS["FNO"],
        label="Worst trace error"
    )

    ax.bar(
        x + width / 2,
        eig_abs,
        width,
        color=COLORS["Final_LNO"],
        label="Absolute worst minimum eigenvalue"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        stability_df["regime"],
        rotation=22,
        ha="right"
    )

    ax.set_ylabel(
        "Magnitude"
    )

    ax.set_title(
        "C-LNO Structural Stability Across Regimes",
        fontweight="bold"
    )

    ax.legend(
        frameon=False,
        fontsize=8
    )

    style_axes(ax)

    fig.subplots_adjust(
        left=0.14,
        right=0.98,
        bottom=0.24,
        top=0.88
    )

    return fig


print("\n[7/8] Generating regime-stability figures...")

save_layout_variants(
    fig07_regime_structure,
    "07_regime_structural_stability"
)

print("[PASS] Figure 07 complete.")


# ======================================================================================
# 18. FIGURE 08 — FINAL PAPER SUMMARY
# ======================================================================================

def fig08_final_summary(figsize):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize
    )

    x = np.arange(
        len(MODEL_ORDER)
    )

    model_colors = [
        COLORS["FNO"],
        COLORS["Final_LNO"],
        COLORS["C_LNO"]
    ]

    # ------------------------------------------------------------------
    # A — R accuracy
    # ------------------------------------------------------------------

    r_values = [
        float(
            overall_df.loc[
                overall_df["model"] == m,
                "R_Relative_L2"
            ].iloc[0]
        )
        for m in MODEL_ORDER
    ]

    axes[0, 0].bar(
        x,
        r_values,
        color=model_colors,
        edgecolor="white"
    )

    axes[0, 0].set_xticks(
        x
    )

    axes[0, 0].set_xticklabels(
        [
            "FNO",
            "LNO",
            "C-LNO"
        ]
    )

    axes[0, 0].set_ylabel(
        "R Relative-L2"
    )

    axes[0, 0].set_title(
        "(a) One-Step R Accuracy"
    )

    style_axes(
        axes[0, 0]
    )

    # ------------------------------------------------------------------
    # B — amplitude
    # ------------------------------------------------------------------

    a_values = [
        float(
            overall_df.loc[
                overall_df["model"] == m,
                "Amplitude_Relative_L2"
            ].iloc[0]
        )
        for m in MODEL_ORDER
    ]

    axes[0, 1].bar(
        x,
        a_values,
        color=model_colors,
        edgecolor="white"
    )

    axes[0, 1].set_xticks(
        x
    )

    axes[0, 1].set_xticklabels(
        [
            "FNO",
            "LNO",
            "C-LNO"
        ]
    )

    axes[0, 1].set_ylabel(
        "Amplitude Relative-L2"
    )

    axes[0, 1].set_title(
        "(b) One-Step Amplitude Accuracy"
    )

    style_axes(
        axes[0, 1]
    )

    # ------------------------------------------------------------------
    # C — R across regimes
    # ------------------------------------------------------------------

    axes[1, 0].plot(
        regime_df["regime"],
        regime_df["R_Relative_L2_FNO"],
        marker="o",
        linewidth=1.9,
        color=COLORS["FNO"],
        label="FNO"
    )

    axes[1, 0].plot(
        regime_df["regime"],
        regime_df["R_Relative_L2_C_LNO"],
        marker="o",
        linewidth=2.2,
        color=COLORS["C_LNO"],
        label="C-LNO"
    )

    axes[1, 0].tick_params(
        axis="x",
        rotation=22
    )

    axes[1, 0].set_ylabel(
        "R Relative-L2"
    )

    axes[1, 0].set_title(
        "(c) R Accuracy Across Regimes"
    )

    axes[1, 0].legend(
        frameon=False
    )

    style_axes(
        axes[1, 0]
    )

    # ------------------------------------------------------------------
    # D — C-LNO structural stability
    # ------------------------------------------------------------------

    stability_curve = (
        rollout_df.groupby("step")[
            "max_trace_error"
        ]
        .mean()
    )

    stability_values = np.maximum(
        stability_curve.to_numpy(dtype=float),
        1e-14
    )

    axes[1, 1].plot(
        stability_curve.index,
        stability_values,
        linewidth=2.2,
        color=COLORS["C_LNO"]
    )

    axes[1, 1].set_yscale(
        "log"
    )

    axes[1, 1].set_xlabel(
        "Rollout step"
    )

    axes[1, 1].set_ylabel(
        "Mean maximum trace error"
    )

    axes[1, 1].set_title(
        "(d) 100-Step Structural Stability"
    )

    style_axes(
        axes[1, 1]
    )

    fig.suptitle(
        "Lindblad Neural Operators — Final Comparative Evidence",
        fontsize=14,
        fontweight="bold"
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.10,
        top=0.90,
        wspace=0.28,
        hspace=0.32
    )

    return fig


print("\n[8/8] Generating final summary figures...")

save_layout_variants(
    fig08_final_summary,
    "08_final_paper_summary"
)

print("[PASS] Figure 08 complete.")


# ======================================================================================
# 19. LOAD DATASET ONLY FOR 3D R-MATRIX
# ======================================================================================

print("\n" + "=" * 110)
print("STAGE 2 — 3D VISUALIZATION")
print("=" * 110)

DATASET_PATH = (
    CONTENT_ROOT
    / "environment_conditioned_lno_dataset_v3_expanded.npz"
)

if not DATASET_PATH.is_file():

    DATASET_PATH = (
        ROOT
        / "results"
        / "dataset_v3_expanded"
        / "environment_conditioned_lno_dataset_v3_expanded.npz"
    )

assert DATASET_PATH.is_file(), DATASET_PATH


# ======================================================================================
# 20. 3D FIGURE 09 — LONG-HORIZON R-NORM SURFACE
# ======================================================================================

regime_order = stability_df[
    "regime"
].tolist()

pivot = (
    rollout_df
    .groupby(
        [
            "regime",
            "step"
        ]
    )[
        "R_norm"
    ]
    .mean()
    .reset_index()
)

steps = sorted(
    rollout_df[
        "step"
    ].unique()
)

Z = np.zeros(
    (
        len(regime_order),
        len(steps)
    ),
    dtype=np.float64
)

for i, regime in enumerate(
    regime_order
):

    temp = (
        pivot[
            pivot["regime"] == regime
        ]
        .set_index("step")
    )

    Z[i, :] = [
        float(
            temp.loc[
                step,
                "R_norm"
            ]
        )
        for step in steps
    ]

X_grid, Y_grid = np.meshgrid(
    steps,
    np.arange(
        len(regime_order)
    )
)

fig = plt.figure(
    figsize=(9.2, 6.8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

surface = ax.plot_surface(
    X_grid,
    Y_grid,
    Z,
    cmap="viridis",
    linewidth=0,
    antialiased=True,
    alpha=0.95
)

ax.set_xlabel(
    "Rollout step",
    labelpad=8
)

ax.set_ylabel(
    "Environment regime",
    labelpad=8
)

ax.set_zlabel(
    "Mean R norm",
    labelpad=8
)

ax.set_yticks(
    np.arange(
        len(regime_order)
    )
)

ax.set_yticklabels(
    regime_order
)

ax.set_title(
    "C-LNO 100-Step R-Norm Stability Across Regimes",
    pad=16,
    fontweight="bold"
)

ax.view_init(
    elev=28,
    azim=-122
)

fig.colorbar(
    surface,
    ax=ax,
    shrink=0.62,
    pad=0.10,
    label="Mean R norm"
)

fig.subplots_adjust(
    left=0.02,
    right=0.88,
    bottom=0.03,
    top=0.92
)

for extension, kwargs in [

    (
        "png",
        {
            "dpi": 600,
            "bbox_inches": "tight",
            "facecolor": "white"
        }
    ),

    (
        "pdf",
        {
            "bbox_inches": "tight",
            "facecolor": "white"
        }
    ),

    (
        "svg",
        {
            "bbox_inches": "tight",
            "facecolor": "white"
        }
    )
]:

    output_path = (
        THREED_DIR
        /
        f"09_3D_C_LNO_R_norm_stability.{extension}"
    )

    fig.savefig(
        output_path,
        **kwargs
    )

plt.close(fig)
cleanup()

print("[PASS] 3D Figure 09 complete.")


# ======================================================================================
# 21. 3D FIGURE 10 — R MATRIX STRUCTURE
# ======================================================================================

# Use context manager so the NPZ handle is released deterministically.
with np.load(
    DATASET_PATH,
    allow_pickle=False
) as dataset:

    X_R = dataset[
        "X_R"
    ].astype(
        np.float32
    )

    print(
        "[PASS] Dataset loaded for 3D R-matrix visualization."
    )

    R_sample = X_R[0]


R_mean = R_sample.mean(
    axis=0
)

ix = np.arange(
    R_mean.shape[0]
)

iy = np.arange(
    R_mean.shape[1]
)

XX, YY = np.meshgrid(
    ix,
    iy
)

fig = plt.figure(
    figsize=(8.2, 7.0)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

surface = ax.plot_surface(
    XX,
    YY,
    R_mean,
    cmap="coolwarm",
    edgecolor="white",
    linewidth=0.28,
    antialiased=True
)

ax.set_xlabel(
    "Information-state index i",
    labelpad=8
)

ax.set_ylabel(
    "Information-state index j",
    labelpad=8
)

ax.set_zlabel(
    r"$R_{ij}$",
    labelpad=8
)

ax.set_title(
    "Normalized Internal-Information Structure",
    pad=18,
    fontweight="bold"
)

ax.view_init(
    elev=32,
    azim=-128
)

fig.colorbar(
    surface,
    ax=ax,
    shrink=0.66,
    pad=0.10,
    label="R value"
)

fig.subplots_adjust(
    left=0.02,
    right=0.88,
    bottom=0.03,
    top=0.92
)

for extension, kwargs in [

    (
        "png",
        {
            "dpi": 600,
            "bbox_inches": "tight",
            "facecolor": "white"
        }
    ),

    (
        "pdf",
        {
            "bbox_inches": "tight",
            "facecolor": "white"
        }
    ),

    (
        "svg",
        {
            "bbox_inches": "tight",
            "facecolor": "white"
        }
    )
]:

    output_path = (
        THREED_DIR
        /
        f"10_3D_R_matrix_structure.{extension}"
    )

    fig.savefig(
        output_path,
        **kwargs
    )

plt.close(fig)
cleanup()

print("[PASS] 3D Figure 10 complete.")


# ======================================================================================
# 22. FREE DATASET MEMORY
# ======================================================================================

del X_R

cleanup()

print("[PASS] Large dataset array released from memory.")


# ======================================================================================
# 23. PUBLICATION-QUALITY LNO ARCHITECTURE SCHEMATIC
# ======================================================================================

print("\n" + "=" * 110)
print("STAGE 3 — LNO ARCHITECTURE")
print("=" * 110)


fig, ax = plt.subplots(
    figsize=(15, 8)
)

ax.set_xlim(
    0,
    15
)

ax.set_ylim(
    0,
    8
)

ax.axis(
    "off"
)


def add_box(
    x,
    y,
    width,
    height,
    title,
    subtitle,
    facecolor,
    title_size=10.5,
    subtitle_size=8.2
):

    patch = FancyBboxPatch(
        (
            x,
            y
        ),
        width,
        height,
        boxstyle=(
            "round,pad=0.06,"
            "rounding_size=0.10"
        ),
        linewidth=1.05,
        edgecolor=COLORS["dark"],
        facecolor=facecolor
    )

    ax.add_patch(
        patch
    )

    ax.text(
        x + width / 2,
        y + height * 0.63,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold"
    )

    ax.text(
        x + width / 2,
        y + height * 0.31,
        subtitle,
        ha="center",
        va="center",
        fontsize=subtitle_size,
        color=COLORS["gray"]
    )


def add_arrow(
    start,
    end,
    color=None,
    linewidth=1.45,
    rad=0.0
):

    if color is None:
        color = COLORS["dark"]

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=linewidth,
        color=color,
        connectionstyle=(
            f"arc3,rad={rad}"
        )
    )

    ax.add_patch(
        arrow
    )


# ------------------------------------------------------------------
# Input
# ------------------------------------------------------------------

add_box(
    0.45,
    5.8,
    2.05,
    1.05,
    "Input state",
    "R(t), A(t), gamma, sigma",
    "#EAF2FF"
)

# ------------------------------------------------------------------
# Projection
# ------------------------------------------------------------------

add_box(
    3.05,
    5.8,
    2.05,
    1.05,
    "Input projection",
    "1 x 1 convolution",
    "#EEF9F6"
)

# ------------------------------------------------------------------
# Fourier operator
# ------------------------------------------------------------------

add_box(
    5.65,
    5.8,
    2.25,
    1.05,
    "Fourier operator",
    "Spectral + pointwise blocks x4",
    "#F1ECFF"
)

# ------------------------------------------------------------------
# Neural R generator
# ------------------------------------------------------------------

add_box(
    8.55,
    5.8,
    2.15,
    1.05,
    "Neural R generator",
    "G_N",
    "#FFF0F3"
)

# ------------------------------------------------------------------
# Lindblad branch
# ------------------------------------------------------------------

add_box(
    8.55,
    2.95,
    2.15,
    1.05,
    "Lindblad generator",
    "G_L(R)",
    "#FFF3E5"
)

# ------------------------------------------------------------------
# Coupling
# ------------------------------------------------------------------

add_box(
    11.25,
    4.35,
    2.10,
    1.05,
    "Positive coupling",
    "kappa_L, kappa_N",
    "#EAF8F5"
)

# ------------------------------------------------------------------
# Physical-state projection
# ------------------------------------------------------------------

add_box(
    11.25,
    1.60,
    2.10,
    1.05,
    "Physical-state projection",
    "PSD + trace normalization",
    "#F1F3F5"
)

# ------------------------------------------------------------------
# R output
# ------------------------------------------------------------------

add_box(
    13.75,
    0.70,
    1.05,
    0.90,
    "R(t+1)",
    "valid state",
    "#FDECEF",
    title_size=9.5,
    subtitle_size=7.3
)

# ------------------------------------------------------------------
# Amplitude output
# ------------------------------------------------------------------

add_box(
    13.75,
    6.15,
    1.05,
    0.90,
    "A(t+1)",
    "direct head",
    "#EDF1FF",
    title_size=9.5,
    subtitle_size=7.3
)


# ------------------------------------------------------------------
# Arrows
# ------------------------------------------------------------------

add_arrow(
    (2.50, 6.32),
    (3.05, 6.32)
)

add_arrow(
    (5.10, 6.32),
    (5.65, 6.32)
)

add_arrow(
    (7.90, 6.32),
    (8.55, 6.32),
    color=COLORS["C_LNO"],
    linewidth=1.8
)

# Fourier features -> Lindblad input
add_arrow(
    (6.78, 5.80),
    (9.15, 4.00),
    color=COLORS["Final_LNO"],
    linewidth=1.55
)

# Neural R -> coupling
add_arrow(
    (10.70, 6.18),
    (11.25, 4.95),
    color=COLORS["C_LNO"],
    linewidth=1.75
)

# Lindblad -> coupling
add_arrow(
    (10.70, 3.47),
    (11.25, 4.75),
    color=COLORS["Final_LNO"],
    linewidth=1.75
)

# Coupling -> projection
add_arrow(
    (12.30, 4.35),
    (12.30, 2.65),
    linewidth=1.55
)

# Projection -> R output
add_arrow(
    (13.35, 1.95),
    (13.75, 1.20),
    linewidth=1.55
)

# Shared feature -> amplitude head
add_arrow(
    (10.70, 6.32),
    (13.75, 6.55),
    color=COLORS["FNO"],
    linewidth=1.45
)


# ------------------------------------------------------------------
# Dynamic equation annotation
# ------------------------------------------------------------------

ax.text(
    9.65,
    7.18,
    r"$R(t+\Delta t) = "
    r"\mathcal{P}_{\mathrm{PSD},\mathrm{tr}=1}"
    r"\left[R(t) + "
    r"\kappa_L \gamma G_L(R) + "
    r"\kappa_N G_N\right]$",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold"
)

ax.text(
    7.48,
    4.72,
    "parallel dynamical generators",
    ha="center",
    va="center",
    fontsize=8.5,
    color=COLORS["gray"],
    style="italic"
)


# ------------------------------------------------------------------
# Main title
# ------------------------------------------------------------------

ax.text(
    7.50,
    7.72,
    "Repaired Lindblad Neural Operator — Computational Architecture",
    ha="center",
    va="center",
    fontsize=17,
    fontweight="bold"
)

ax.text(
    7.50,
    7.42,
    "Fourier operator + dissipative Lindblad generator + trainable positive coupling + physical-state projection",
    ha="center",
    va="center",
    fontsize=9.0,
    color=COLORS["gray"]
)


ARCH_PNG = (
    ARCH_DIR
    / "11_LNO_architecture_publication.png"
)

ARCH_PDF = (
    ARCH_DIR
    / "11_LNO_architecture_publication.pdf"
)

ARCH_SVG = (
    ARCH_DIR
    / "11_LNO_architecture_publication.svg"
)


fig.savefig(
    ARCH_PNG,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    ARCH_PDF,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    ARCH_SVG,
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)
cleanup()

print("[PASS] LNO architecture schematic complete.")


# ======================================================================================
# 24. ARCHITECTURE GIF
# ======================================================================================

print("\n" + "=" * 110)
print("STAGE 4 — ARCHITECTURE GIF")
print("=" * 110)


ARCH_TEMP = (
    TEMP_DIR
    / "architecture_frames"
)

if ARCH_TEMP.exists():
    shutil.rmtree(
        ARCH_TEMP
    )

ARCH_TEMP.mkdir(
    parents=True,
    exist_ok=True
)

architecture_frames = []


# Full workflow coordinates
workflow_points = np.array(
    [
        [1.48, 6.32],
        [4.08, 6.32],
        [6.78, 6.32],
        [9.62, 6.32],
        [12.30, 4.85],
        [12.30, 2.10],
        [14.28, 1.15],
    ],
    dtype=np.float64
)


def interpolate_points(
    points,
    per_segment=10
):

    pieces = []

    for i in range(
        len(points) - 1
    ):

        a = points[i]
        b = points[i + 1]

        for t in np.linspace(
            0,
            1,
            per_segment,
            endpoint=False
        ):

            pieces.append(
                (1 - t) * a
                +
                t * b
            )

    pieces.append(
        points[-1]
    )

    return np.asarray(
        pieces
    )


architecture_path = interpolate_points(
    workflow_points,
    per_segment=10
)

# Exactly 90 frames.
frame_indices = np.linspace(
    0,
    len(architecture_path) - 1,
    90,
    dtype=np.int64
)


for frame_number, index in enumerate(
    frame_indices
):

    fig, ax = plt.subplots(
        figsize=(11, 5.8)
    )

    ax.set_xlim(
        0,
        15
    )

    ax.set_ylim(
        0,
        8
    )

    ax.axis(
        "off"
    )

    simple_nodes = [

        (0.55, 5.50, 1.75, 0.95, "Input"),

        (2.80, 5.50, 1.85, 0.95, "Fourier"),

        (5.35, 6.00, 1.75, 0.90, "Neural R"),

        (5.35, 2.90, 1.75, 0.90, "Lindblad"),

        (8.05, 4.55, 1.85, 0.95, "Coupling"),

        (10.65, 4.55, 1.95, 0.95, "Projection"),

        (13.20, 5.75, 1.25, 0.85, "A(t+1)"),

        (13.20, 2.55, 1.25, 0.85, "R(t+1)")
    ]

    for x0, y0, w, h, label in simple_nodes:

        face = "#F7F9FB"

        if label == "Neural R":
            face = "#FFF0F3"

        elif label == "Lindblad":
            face = "#FFF3E5"

        elif label == "Coupling":
            face = "#EAF8F5"

        elif label == "Projection":
            face = "#F1F3F5"

        elif label == "R(t+1)":
            face = "#FDECEF"

        elif label == "A(t+1)":
            face = "#EDF1FF"

        patch = FancyBboxPatch(
            (
                x0,
                y0
            ),
            w,
            h,
            boxstyle=(
                "round,pad=0.05,"
                "rounding_size=0.09"
            ),
            linewidth=1.0,
            edgecolor=COLORS["dark"],
            facecolor=face
        )

        ax.add_patch(
            patch
        )

        ax.text(
            x0 + w / 2,
            y0 + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold"
        )

    static_arrows = [

        ((2.30, 5.98), (2.80, 5.98)),

        ((4.65, 5.98), (5.35, 6.42)),

        ((4.65, 5.98), (5.35, 3.35)),

        ((7.10, 6.42), (8.05, 5.02)),

        ((7.10, 3.35), (8.05, 4.98)),

        ((9.90, 5.02), (10.65, 5.02)),

        ((12.60, 5.02), (13.20, 6.12)),

        ((12.60, 5.02), (13.20, 2.98)),
    ]

    for start, end in static_arrows:

        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.25,
                color=COLORS["dark"]
            )
        )

    particle_position = architecture_path[
        index
    ]

    ax.scatter(
        [
            particle_position[0]
        ],
        [
            particle_position[1]
        ],
        s=95,
        color=COLORS["C_LNO"],
        zorder=20,
        edgecolors="white",
        linewidths=1.2
    )

    ax.text(
        7.5,
        7.55,
        "LNO Computational Workflow",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold"
    )

    ax.text(
        7.5,
        7.18,
        "Conceptual data-flow animation — not an activation recording",
        ha="center",
        va="center",
        fontsize=8.5,
        color=COLORS["gray"]
    )

    frame_path = (
        ARCH_TEMP
        /
        f"frame_{frame_number:03d}.png"
    )

    fig.savefig(
        frame_path,
        dpi=110,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    architecture_frames.append(
        frame_path
    )

    cleanup()


# ------------------------------------------------------------------
# ACTUAL GENERATED GIF PATH
# ------------------------------------------------------------------

ARCH_GIF = (
    GIF_DIR
    / "12_LNO_architecture_working.gif"
)


assert architecture_frames, "Architecture GIF frames were not generated."


with Image.open(
    architecture_frames[0]
) as first_image:

    first = first_image.convert(
        "P",
        palette=Image.ADAPTIVE
    )

    frames = []

    for frame_path in architecture_frames[1:]:

        with Image.open(
            frame_path
        ) as frame_image:

            frames.append(
                frame_image.convert(
                    "P",
                    palette=Image.ADAPTIVE
                )
            )

    first.save(
        ARCH_GIF,
        save_all=True,
        append_images=frames,
        duration=int(
            1000 / 14
        ),
        loop=0,
        optimize=False
    )

    for frame in frames:
        frame.close()


print(
    "[PASS] Architecture working GIF generated:",
    ARCH_GIF
)

shutil.rmtree(
    ARCH_TEMP,
    ignore_errors=True
)

cleanup()


# ======================================================================================
# 25. 3D R-NORM GIF
# ======================================================================================

print("\n" + "=" * 110)
print("STAGE 5 — 3D R-NORM EVOLUTION GIF")
print("=" * 110)


GIF3D_TEMP = (
    TEMP_DIR
    / "rnorm_3d_frames"
)

if GIF3D_TEMP.exists():
    shutil.rmtree(
        GIF3D_TEMP
    )

GIF3D_TEMP.mkdir(
    parents=True,
    exist_ok=True
)


# Exactly 100 frames.
three_d_frame_paths = []


for frame_number, step in enumerate(
    steps
):

    current_values = Z[
        :,
        frame_number
    ]

    fig = plt.figure(
        figsize=(8.2, 6.4)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.bar(
        np.arange(
            len(regime_order)
        ),
        current_values,
        zs=0,
        zdir="y",
        color=[
            COLORS.get(
                regime,
                COLORS["gray"]
            )
            for regime in regime_order
        ],
        alpha=0.90,
        edgecolor="white",
        linewidth=0.4
    )

    ax.set_xlim(
        -0.5,
        len(regime_order) - 0.5
    )

    zmax = max(
        float(
            Z.max()
        ),
        1.0
    )

    ax.set_zlim(
        0,
        zmax * 1.08
    )

    ax.set_xticks(
        np.arange(
            len(regime_order)
        )
    )

    ax.set_xticklabels(
        regime_order,
        rotation=18,
        ha="right"
    )

    ax.set_yticks([])

    ax.set_xlabel(
        "Environment regime",
        labelpad=7
    )

    ax.set_zlabel(
        "R norm",
        labelpad=7
    )

    ax.set_title(
        f"C-LNO R-Norm Evolution — Rollout Step {step}",
        fontweight="bold",
        pad=15
    )

    ax.view_init(
        elev=27,
        azim=-120
    )

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.03,
        top=0.90
    )

    frame_path = (
        GIF3D_TEMP
        /
        f"frame_{frame_number:03d}.png"
    )

    fig.savefig(
        frame_path,
        dpi=100,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    three_d_frame_paths.append(
        frame_path
    )

    cleanup()


# ------------------------------------------------------------------
# ACTUAL GENERATED GIF PATH
# ------------------------------------------------------------------

R_NORM_GIF = (
    GIF_DIR
    / "13_C_LNO_3D_R_norm_evolution.gif"
)


assert three_d_frame_paths, (
    "3D R-norm GIF frames were not generated."
)


with Image.open(
    three_d_frame_paths[0]
) as first_image:

    first = first_image.convert(
        "P",
        palette=Image.ADAPTIVE
    )

    frames = []

    for frame_path in three_d_frame_paths[1:]:

        with Image.open(
            frame_path
        ) as frame_image:

            frames.append(
                frame_image.convert(
                    "P",
                    palette=Image.ADAPTIVE
                )
            )

    first.save(
        R_NORM_GIF,
        save_all=True,
        append_images=frames,
        duration=int(
            1000 / 14
        ),
        loop=0,
        optimize=False
    )

    for frame in frames:
        frame.close()


print(
    "[PASS] 3D R-norm evolution GIF generated:",
    R_NORM_GIF
)

shutil.rmtree(
    GIF3D_TEMP,
    ignore_errors=True
)

cleanup()


# ======================================================================================
# 26. COPY VALIDATED EVIDENCE INTO PACKAGE
# ======================================================================================

print("\n" + "=" * 110)
print("STAGE 6 — COPY VALIDATED EVIDENCE")
print("=" * 110)


for source in [
    STEP27_OVERALL,
    STEP27_REGIME,
    STEP28_ROLLOUT,
    STEP28_REGIME,
    STEP28_GLOBAL,
    STEP29_JSON,
]:

    shutil.copy2(
        source,
        EVIDENCE_DIR
        /
        source.name
    )


print(
    "[PASS] Validated Step 27/28/29 evidence copied."
)


# ======================================================================================
# 27. FIGURE MANIFEST
# ======================================================================================

figure_manifest = {

    "step":
        30,

    "title":
        "IEEE / Journal Quality Publication Visualization Package",

    "training_performed":
        False,

    "new_model_inference":
        False,

    "checkpoint_modified":
        False,

    "evidence_sources":
        [
            str(STEP27_DIR),
            str(STEP28_DIR),
            str(STEP29_DIR)
        ],

    "core_figures":
        [
            "01_overall_accuracy",
            "02_regime_wise_R_accuracy",
            "03_structural_preservation",
            "04_C_LNO_100step_trace_stability",
            "05_C_LNO_100step_PSD_stability",
            "06_C_LNO_R_norm_stability",
            "07_regime_structural_stability",
            "08_final_paper_summary"
        ],

    "three_d_figures":
        [
            "09_3D_C_LNO_R_norm_stability",
            "10_3D_R_matrix_structure"
        ],

    "architecture":
        [
            "11_LNO_architecture_publication"
        ],

    "gifs":
        [
            "12_LNO_architecture_working.gif",
            "13_C_LNO_3D_R_norm_evolution.gif"
        ],

    "aspect_ratios":
        [
            "square",
            "two_column",
            "wide"
        ],

    "formats":
        [
            "PNG_600DPI",
            "SVG",
            "PDF"
        ]
}

MANIFEST_PATH = (
    OUT
    /
    "STEP30_FIGURE_MANIFEST.json"
)

with open(
    MANIFEST_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        figure_manifest,
        f,
        indent=2
    )

print(
    "[PASS] Figure manifest written."
)


# ======================================================================================
# 28. README FOR PUBLICATION PACKAGE
# ======================================================================================

README_TEXT = """
STEP 30 — IEEE / JOURNAL-QUALITY VISUALIZATION PACKAGE
========================================================

This package contains publication-oriented figures generated exclusively
from previously validated Step 27, Step 28, and Step 29 evidence.

NO:
- model retraining
- checkpoint modification
- new model inference
- new ground-truth benchmark

2D figures:
01 overall accuracy
02 regime-wise R accuracy
03 structural preservation
04 C-LNO long-horizon trace stability
05 C-LNO long-horizon PSD stability
06 C-LNO R-norm stability
07 regime-wise structural stability
08 final paper summary

3D figures:
09 C-LNO 100-step R-norm surface
10 normalized internal-information structure R_ij

Architecture:
11 repaired LNO computational architecture

Animations:
12 conceptual LNO computational workflow
13 C-LNO 3D R-norm evolution

Each core figure is exported in square, IEEE two-column, and wide layouts.
PNG figures use 600 DPI. Vector SVG/PDF versions are also provided.

The GIFs are visualization aids. The architecture GIF is conceptual and
does not represent recorded neural activations.
"""


README_PATH = (
    OUT
    /
    "README.txt"
)

with open(
    README_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        README_TEXT
    )

print(
    "[PASS] README written."
)


# ======================================================================================
# 29. CREATE ZIP
# ======================================================================================

print("\n" + "=" * 110)
print("STAGE 7 — ZIP PACKAGE")
print("=" * 110)


ZIP_PATH = (
    ROOT
    / "results"
    / "STEP30_IEEE_PUBLICATION_PACKAGE.zip"
)

if ZIP_PATH.exists():
    ZIP_PATH.unlink()


with zipfile.ZipFile(
    ZIP_PATH,
    "w",
    zipfile.ZIP_DEFLATED
) as z:

    for root, dirs, files in os.walk(
        OUT
    ):

        for file in files:

            full_path = (
                Path(root)
                /
                file
            )

            # Never include temporary frame directory.
            if TEMP_DIR in full_path.parents:
                continue

            arcname = (
                full_path
                .relative_to(
                    OUT.parent
                )
            )

            z.write(
                full_path,
                arcname
            )


print(
    "[PASS] ZIP package created."
)


# ======================================================================================
# 30. FINAL FILE COUNT / VALIDATION
# ======================================================================================

cleanup()

all_output_files = []

for root, dirs, files in os.walk(
    OUT
):

    for file in files:

        full_path = Path(root) / file

        if TEMP_DIR in full_path.parents:
            continue

        all_output_files.append(
            full_path
        )


png_count = sum(
    p.suffix.lower() == ".png"
    for p in all_output_files
)

svg_count = sum(
    p.suffix.lower() == ".svg"
    for p in all_output_files
)

pdf_count = sum(
    p.suffix.lower() == ".pdf"
    for p in all_output_files
)

gif_count = sum(
    p.suffix.lower() == ".gif"
    for p in all_output_files
)


# ======================================================================================
# 31. REQUIRED FILES
# ======================================================================================
#
# IMPORTANT FIX:
# The previous code referenced GIF_PATH, which was never defined.
# The actual variables are ARCH_GIF and R_NORM_GIF.
# ======================================================================================

required_final_outputs = [

    ARCH_PNG,
    ARCH_PDF,
    ARCH_SVG,

    # FIXED:
    ARCH_GIF,
    R_NORM_GIF,

    THREED_DIR
    /
    "09_3D_C_LNO_R_norm_stability.png",

    THREED_DIR
    /
    "09_3D_C_LNO_R_norm_stability.pdf",

    THREED_DIR
    /
    "09_3D_C_LNO_R_norm_stability.svg",

    THREED_DIR
    /
    "10_3D_R_matrix_structure.png",

    THREED_DIR
    /
    "10_3D_R_matrix_structure.pdf",

    THREED_DIR
    /
    "10_3D_R_matrix_structure.svg",

    MANIFEST_PATH,

    README_PATH
]


for required in required_final_outputs:

    assert required.is_file(), (
        f"Required Step 30 output missing: {required}"
    )


# ======================================================================================
# 32. CORE ASPECT-RATIO VALIDATION
# ======================================================================================

square_pngs = list(
    SQUARE_DIR.glob("*.png")
)

two_column_pngs = list(
    TWO_COLUMN_DIR.glob("*.png")
)

wide_pngs = list(
    WIDE_DIR.glob("*.png")
)

assert len(
    square_pngs
) >= 8, (
    f"Expected >=8 square PNGs, found {len(square_pngs)}"
)

assert len(
    two_column_pngs
) >= 8, (
    f"Expected >=8 two-column PNGs, found {len(two_column_pngs)}"
)

assert len(
    wide_pngs
) >= 8, (
    f"Expected >=8 wide PNGs, found {len(wide_pngs)}"
)


# ======================================================================================
# 33. GIF VALIDATION
# ======================================================================================

assert ARCH_GIF.is_file(), (
    f"Architecture GIF missing: {ARCH_GIF}"
)

assert R_NORM_GIF.is_file(), (
    f"3D R-norm GIF missing: {R_NORM_GIF}"
)

with Image.open(
    ARCH_GIF
) as img:

    architecture_gif_frames = getattr(
        img,
        "n_frames",
        1
    )

with Image.open(
    R_NORM_GIF
) as img:

    rnorm_gif_frames = getattr(
        img,
        "n_frames",
        1
    )

assert architecture_gif_frames >= 2
assert rnorm_gif_frames >= 2


# ======================================================================================
# 34. FINAL STATUS
# ======================================================================================

print("\n" + "=" * 110)
print("STEP 30 COMPLETE — IEEE / JOURNAL PUBLICATION PACKAGE")
print("=" * 110)

print(
    "\n[PASS] Core 2D figures:"
    f" {len(square_pngs)} square /"
    f" {len(two_column_pngs)} two-column /"
    f" {len(wide_pngs)} wide"
)

print(
    "[PASS] PNG / SVG / PDF publication exports generated."
)

print(
    "[PASS] 3D C-LNO R-norm surface generated."
)

print(
    "[PASS] 3D R-matrix structure generated."
)

print(
    "[PASS] High-quality LNO architecture schematic generated."
)

print(
    "[PASS] LNO architecture working GIF generated."
)

print(
    f"[PASS] Architecture GIF frames: {architecture_gif_frames}"
)

print(
    "[PASS] 3D C-LNO R-norm evolution GIF generated."
)

print(
    f"[PASS] 3D R-norm GIF frames: {rnorm_gif_frames}"
)

print(
    "[PASS] Validated Step 27/28/29 evidence copied."
)

print(
    f"[PASS] PNG files: {png_count}"
)

print(
    f"[PASS] SVG files: {svg_count}"
)

print(
    f"[PASS] PDF files: {pdf_count}"
)

print(
    f"[PASS] GIF files: {gif_count}"
)

print(
    "\nOutput directory:"
)

print(
    OUT
)

print(
    "\nZIP package:"
)

print(
    ZIP_PATH
)

print(
    "\nZIP size:"
)

print(
    f"{ZIP_PATH.stat().st_size / (1024**2):.2f} MB"
)

print(
    "\n[INFO] No model training performed."
)

print(
    "[INFO] No model inference performed."
)

print(
    "[INFO] No checkpoint modified."
)

print(
    "[INFO] All visualizations are derived from validated evidence."
)

print(
    "\n[PASS] STEP 30 COMPLETE."
)

print("=" * 110)
