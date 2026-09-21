# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 23
# Step            : STEP_22
# Step Heading    : # STEP 22 — FINAL PAPER FIGURES + IEEE-READY TABLES
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 22 — FINAL PAPER FIGURES + IEEE-READY TABLES
#
# Produces:
#   * High-resolution PNG
#   * Vector PDF
#   * Vector SVG
#   * IEEE-ready LaTeX tables
#   * CSV summaries
#   * Figure/table manifest
#
# Main paper story:
#   FNO vs LNO
#
# NO TRAINING
# NO MODEL MODIFICATION
# ============================================================

import os
import json
import shutil
import math
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

print("=" * 100)
print("STEP 22 — FINAL PAPER FIGURES + IEEE TABLES")
print("=" * 100)

# ============================================================
# 2. OUTPUT DIRECTORIES
# ============================================================

PAPER_DIR = (
    ROOT
    / "results"
    / "paper_figures_tables"
)

FIG_DIR = (
    PAPER_DIR
    / "figures"
)

TABLE_DIR = (
    PAPER_DIR
    / "tables"
)

DATA_DIR = (
    PAPER_DIR
    / "data"
)

for directory in [
    PAPER_DIR,
    FIG_DIR,
    TABLE_DIR,
    DATA_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

print(
    "[+] Output:",
    PAPER_DIR
)

# ============================================================
# 3. IEEE FIGURE CONFIGURATION
# ============================================================

# IEEE single-column width ≈ 3.5 in
# IEEE double-column width ≈ 7.16 in

SINGLE_COL = (
    3.5,
    2.45
)

DOUBLE_COL = (
    7.16,
    3.9
)

DPI = 600

plt.rcParams.update({

    "font.family":
        "serif",

    "font.size":
        8,

    "axes.labelsize":
        8,

    "axes.titlesize":
        8.5,

    "xtick.labelsize":
        7,

    "ytick.labelsize":
        7,

    "legend.fontsize":
        7,

    "axes.linewidth":
        0.8,

    "xtick.major.width":
        0.7,

    "ytick.major.width":
        0.7,

    "savefig.dpi":
        DPI,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42,

})

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def save_figure(
    fig,
    filename
):

    base = (
        FIG_DIR
        / filename
    )

    fig.savefig(
        str(base) + ".png",
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.03
    )

    fig.savefig(
        str(base) + ".pdf",
        bbox_inches="tight",
        pad_inches=0.03
    )

    fig.savefig(
        str(base) + ".svg",
        bbox_inches="tight",
        pad_inches=0.03
    )

    plt.close(fig)

    print(
        "[FIGURE]",
        filename
    )


def read_csv(
    path,
    label
):

    path = Path(path)

    if not path.is_file():

        print(
            f"[MISSING] {label}: {path}"
        )

        return None

    df = pd.read_csv(
        path
    )

    print(
        f"[LOADED] {label}: {path}"
    )

    return df


def read_json(
    path,
    label
):

    path = Path(path)

    if not path.is_file():

        print(
            f"[MISSING] {label}: {path}"
        )

        return None

    with open(
        path,
        "r"
    ) as f:

        obj = json.load(
            f
        )

    print(
        f"[LOADED] {label}: {path}"
    )

    return obj


def latex_escape(
    text
):

    text = str(text)

    replacements = {

        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "$": r"\$",
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    return text


# ============================================================
# 5. LOAD VERIFIED RESULTS
# ============================================================

comparison_csv = (
    ROOT
    / "results"
    / "final_comparison"
    / "fno_vs_lno_final.csv"
)

comparison_json = (
    ROOT
    / "results"
    / "final_comparison"
    / "fno_vs_lno_final.json"
)

ood_csv = (
    ROOT
    / "results"
    / "final_ood"
    / "final_fno_vs_lno_ood_sigma_sweep.csv"
)

ood_json = (
    ROOT
    / "results"
    / "final_ood"
    / "final_fno_vs_lno_ood_sigma_sweep.json"
)

rollout_csv = (
    ROOT
    / "results"
    / "final_long_horizon"
    / "long_horizon_rollout.csv"
)

rollout_json = (
    ROOT
    / "results"
    / "final_long_horizon"
    / "long_horizon_summary.json"
)

spectral_timestep_csv = (
    ROOT
    / "results"
    / "final_spectral_stability"
    / "spectral_stability_vs_timestep.csv"
)

spectral_mode_csv = (
    ROOT
    / "results"
    / "final_spectral_stability"
    / "spectral_energy_vs_mode.csv"
)

spectral_json = (
    ROOT
    / "results"
    / "final_spectral_stability"
    / "spectral_stability_summary.json"
)

comparison_df = read_csv(
    comparison_csv,
    "FNO vs LNO comparison"
)

comparison_meta = read_json(
    comparison_json,
    "FNO vs LNO JSON"
)

ood_df = read_csv(
    ood_csv,
    "OOD sigma sweep"
)

ood_meta = read_json(
    ood_json,
    "OOD JSON"
)

rollout_df = read_csv(
    rollout_csv,
    "Long-horizon rollout"
)

rollout_meta = read_json(
    rollout_json,
    "Long-horizon JSON"
)

spectral_df = read_csv(
    spectral_timestep_csv,
    "Spectral timestep analysis"
)

spectral_mode_df = read_csv(
    spectral_mode_csv,
    "Spectral mode analysis"
)

spectral_meta = read_json(
    spectral_json,
    "Spectral JSON"
)

# ============================================================
# 6. CORE TABLE I — CLEAN BENCHMARK
# ============================================================

if comparison_df is not None:

    core_metrics = [
        "RMSE",
        "Relative_L2",
        "MAE",
    ]

    clean_rows = (
        comparison_df[
            comparison_df["Metric"].isin(
                core_metrics
            )
        ]
        .copy()
    )

    # Preserve logical order
    clean_rows["Metric"] = pd.Categorical(
        clean_rows["Metric"],
        categories=core_metrics,
        ordered=True
    )

    clean_rows = clean_rows.sort_values(
        "Metric"
    )

    clean_rows.to_csv(
        DATA_DIR
        / "table_clean_benchmark.csv",
        index=False
    )

    # --------------------------------------------------------
    # LaTeX
    # --------------------------------------------------------

    latex = []

    latex.append(
        r"\begin{table}[t]"
    )

    latex.append(
        r"\caption{In-distribution reconstruction performance "
        r"on the 3,000-sample test set. Lower is better.}"
    )

    latex.append(
        r"\label{tab:clean_performance}"
    )

    latex.append(
        r"\centering"
    )

    latex.append(
        r"\begin{tabular}{lccc}"
    )

    latex.append(
        r"\hline"
    )

    latex.append(
        r"Metric & FNO & LNO & LNO Improvement (\%) \\"
    )

    latex.append(
        r"\hline"
    )

    for _, row in clean_rows.iterrows():

        latex.append(
            f"{latex_escape(row['Metric'])} & "
            f"{row['FNO']:.4e} & "
            f"{row['LNO']:.4e} & "
            f"{row['LNO_improvement_percent']:+.2f} \\\\"
        )

    latex.append(
        r"\hline"
    )

    latex.append(
        r"\end{tabular}"
    )

    latex.append(
        r"\end{table}"
    )

    with open(
        TABLE_DIR
        / "table_clean_benchmark.tex",
        "w"
    ) as f:

        f.write(
            "\n".join(
                latex
            )
        )

    # --------------------------------------------------------
    # Figure 1 — Clean benchmark
    # --------------------------------------------------------

    plot_df = clean_rows.copy()

    metrics = list(
        plot_df["Metric"]
        .astype(str)
    )

    fno_values = (
        plot_df["FNO"]
        .astype(float)
        .values
    )

    lno_values = (
        plot_df["LNO"]
        .astype(float)
        .values
    )

    x = np.arange(
        len(metrics)
    )

    width = 0.34

    fig, ax = plt.subplots(
        figsize=SINGLE_COL
    )

    ax.bar(
        x - width / 2,
        fno_values,
        width,
        label="FNO"
    )

    ax.bar(
        x + width / 2,
        lno_values,
        width,
        label="LNO"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        metrics,
        rotation=25,
        ha="right"
    )

    ax.set_ylabel(
        "Error (log scale)"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        axis="y",
        alpha=0.20,
        linewidth=0.5
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig1_clean_benchmark"
    )


# ============================================================
# 7. CORE TABLE II — OOD
# ============================================================

if ood_df is not None:

    ood_export = ood_df[
        [
            "sigma",
            "FNO_RMSE",
            "LNO_RMSE",
            "LNO_improvement_percent",
            "Winner",
        ]
    ].copy()

    ood_export.to_csv(
        DATA_DIR
        / "table_ood_sigma.csv",
        index=False
    )

    latex = []

    latex.append(
        r"\begin{table}[t]"
    )

    latex.append(
        r"\caption{Out-of-distribution robustness under "
        r"systematic stochastic perturbation.}"
    )

    latex.append(
        r"\label{tab:ood}"
    )

    latex.append(
        r"\centering"
    )

    latex.append(
        r"\begin{tabular}{ccccc}"
    )

    latex.append(
        r"\hline"
    )

    latex.append(
        r"$\sigma$ & FNO RMSE & LNO RMSE & "
        r"LNO Improvement (\%) & Winner \\"
    )

    latex.append(
        r"\hline"
    )

    for _, row in ood_export.iterrows():

        latex.append(
            f"{row['sigma']:.2f} & "
            f"{row['FNO_RMSE']:.4e} & "
            f"{row['LNO_RMSE']:.4e} & "
            f"{row['LNO_improvement_percent']:+.2f} & "
            f"{latex_escape(row['Winner'])} \\\\"
        )

    latex.append(
        r"\hline"
    )

    latex.append(
        r"\end{tabular}"
    )

    latex.append(
        r"\end{table}"
    )

    with open(
        TABLE_DIR
        / "table_ood_sigma.tex",
        "w"
    ) as f:

        f.write(
            "\n".join(
                latex
            )
        )

    # --------------------------------------------------------
    # Figure 2 — OOD robustness
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=DOUBLE_COL
    )

    ax.plot(
        ood_df["sigma"],
        ood_df["FNO_RMSE"],
        marker="o",
        markersize=3.5,
        linewidth=1.3,
        label="FNO"
    )

    ax.plot(
        ood_df["sigma"],
        ood_df["LNO_RMSE"],
        marker="s",
        markersize=3.5,
        linewidth=1.3,
        label="LNO"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        r"Perturbation strength $\sigma$"
    )

    ax.set_ylabel(
        "RMSE (log scale)"
    )

    ax.legend(
        frameon=False,
        ncol=2
    )

    ax.grid(
        which="both",
        alpha=0.20,
        linewidth=0.5
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig2_ood_robustness"
    )

    # --------------------------------------------------------
    # Figure 2b — Relative improvement
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=SINGLE_COL
    )

    ax.plot(
        ood_df["sigma"],
        ood_df[
            "LNO_improvement_percent"
        ],
        marker="o",
        markersize=3.5,
        linewidth=1.3
    )

    ax.axhline(
        0.0,
        linewidth=0.8,
        linestyle="--"
    )

    ax.set_xlabel(
        r"Perturbation strength $\sigma$"
    )

    ax.set_ylabel(
        "LNO improvement over FNO (%)"
    )

    ax.grid(
        alpha=0.20,
        linewidth=0.5
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig2b_ood_lno_advantage"
    )


# ============================================================
# 8. CORE TABLE III — LONG HORIZON
# ============================================================

if rollout_meta is not None:

    FNO = rollout_meta["FNO"]
    LNO = rollout_meta["LNO"]

    long_rows = [

        {
            "Metric":
                "Initial energy",

            "FNO":
                FNO["initial_energy"],

            "LNO":
                LNO["initial_energy"],
        },

        {
            "Metric":
                "Final energy",

            "FNO":
                FNO["final_energy"],

            "LNO":
                LNO["final_energy"],
        },

        {
            "Metric":
                "Absolute energy drift",

            "FNO":
                FNO["absolute_drift"],

            "LNO":
                LNO["absolute_drift"],
        },

        {
            "Metric":
                "Relative energy drift",

            "FNO":
                FNO["relative_drift"],

            "LNO":
                LNO["relative_drift"],
        },

        {
            "Metric":
                "Maximum |state|",

            "FNO":
                FNO["max_abs_state"],

            "LNO":
                LNO["max_abs_state"],
        },
    ]

    long_df = pd.DataFrame(
        long_rows
    )

    long_df.to_csv(
        DATA_DIR
        / "table_long_horizon.csv",
        index=False
    )

    latex = []

    latex.append(
        r"\begin{table}[t]"
    )

    latex.append(
        r"\caption{Long-horizon autoregressive stability over "
        r"40 rollout steps.}"
    )

    latex.append(
        r"\label{tab:long_horizon}"
    )

    latex.append(
        r"\centering"
    )

    latex.append(
        r"\begin{tabular}{lcc}"
    )

    latex.append(
        r"\hline"
    )

    latex.append(
        r"Metric & FNO & LNO \\"
    )

    latex.append(
        r"\hline"
    )

    for _, row in long_df.iterrows():

        latex.append(
            f"{latex_escape(row['Metric'])} & "
            f"{row['FNO']:.4e} & "
            f"{row['LNO']:.4e} \\\\"
        )

    latex.append(
        r"\hline"
    )

    latex.append(
        r"\end{tabular}"
    )

    latex.append(
        r"\end{table}"
    )

    with open(
        TABLE_DIR
        / "table_long_horizon.tex",
        "w"
    ) as f:

        f.write(
            "\n".join(
                latex
            )
        )

    # --------------------------------------------------------
    # Figure 3 — Energy evolution
    # --------------------------------------------------------

    if rollout_df is not None:

        fig, ax = plt.subplots(
            figsize=DOUBLE_COL
        )

        ax.plot(
            rollout_df["step"],
            rollout_df[
                "FNO_mean_energy"
            ],
            linewidth=1.3,
            label="FNO"
        )

        ax.plot(
            rollout_df["step"],
            rollout_df[
                "LNO_mean_energy"
            ],
            linewidth=1.3,
            label="LNO"
        )

        ax.set_xlabel(
            "Autoregressive step"
        )

        ax.set_ylabel(
            "Mean state energy"
        )

        ax.legend(
            frameon=False,
            ncol=2
        )

        ax.grid(
            alpha=0.20,
            linewidth=0.5
        )

        fig.tight_layout()

        save_figure(
            fig,
            "fig3_long_horizon_energy"
        )

        # ----------------------------------------------------
        # Figure 3b — absolute drift
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=SINGLE_COL
        )

        ax.plot(
            rollout_df["step"],
            rollout_df[
                "FNO_abs_energy_drift"
            ],
            linewidth=1.3,
            label="FNO"
        )

        ax.plot(
            rollout_df["step"],
            rollout_df[
                "LNO_abs_energy_drift"
            ],
            linewidth=1.3,
            label="LNO"
        )

        ax.set_xlabel(
            "Autoregressive step"
        )

        ax.set_ylabel(
            "Absolute energy drift"
        )

        ax.legend(
            frameon=False
        )

        ax.grid(
            alpha=0.20,
            linewidth=0.5
        )

        fig.tight_layout()

        save_figure(
            fig,
            "fig3b_energy_drift"
        )


# ============================================================
# 9. SPECTRAL STABILITY TABLE
# ============================================================

if spectral_meta is not None:

    spectral_rows = [

        {
            "Metric":
                "Max total spectral growth",

            "FNO":
                spectral_meta[
                    "FNO"
                ][
                    "max_total_spectral_growth"
                ],

            "LNO":
                spectral_meta[
                    "LNO"
                ][
                    "max_total_spectral_growth"
                ],
        },

        {
            "Metric":
                "Final total spectral growth",

            "FNO":
                spectral_meta[
                    "FNO"
                ][
                    "final_total_spectral_growth"
                ],

            "LNO":
                spectral_meta[
                    "LNO"
                ][
                    "final_total_spectral_growth"
                ],
        },

        {
            "Metric":
                "Max high-frequency growth",

            "FNO":
                spectral_meta[
                    "FNO"
                ][
                    "max_high_frequency_growth"
                ],

            "LNO":
                spectral_meta[
                    "LNO"
                ][
                    "max_high_frequency_growth"
                ],
        },

        {
            "Metric":
                "Final high-frequency fraction",

            "FNO":
                spectral_meta[
                    "FNO"
                ][
                    "final_high_frequency_fraction"
                ],

            "LNO":
                spectral_meta[
                    "LNO"
                ][
                    "final_high_frequency_fraction"
                ],
        },
    ]

    spectral_summary_df = pd.DataFrame(
        spectral_rows
    )

    spectral_summary_df.to_csv(
        DATA_DIR
        / "table_spectral_stability.csv",
        index=False
    )

    latex = []

    latex.append(
        r"\begin{table}[t]"
    )

    latex.append(
        r"\caption{Spectral stability statistics over the "
        r"40-step autoregressive rollout.}"
    )

    latex.append(
        r"\label{tab:spectral_stability}"
    )

    latex.append(
        r"\centering"
    )

    latex.append(
        r"\begin{tabular}{lcc}"
    )

    latex.append(
        r"\hline"
    )

    latex.append(
        r"Metric & FNO & LNO \\"
    )

    latex.append(
        r"\hline"
    )

    for _, row in spectral_summary_df.iterrows():

        latex.append(
            f"{latex_escape(row['Metric'])} & "
            f"{row['FNO']:.4e} & "
            f"{row['LNO']:.4e} \\\\"
        )

    latex.append(
        r"\hline"
    )

    latex.append(
        r"\end{tabular}"
    )

    latex.append(
        r"\end{table}"
    )

    with open(
        TABLE_DIR
        / "table_spectral_stability.tex",
        "w"
    ) as f:

        f.write(
            "\n".join(
                latex
            )
        )


# ============================================================
# 10. FIGURE 4 — HIGH-FREQUENCY FRACTION
# ============================================================

if spectral_df is not None:

    fig, ax = plt.subplots(
        figsize=DOUBLE_COL
    )

    ax.plot(
        spectral_df["timestep"],
        spectral_df[
            "FNO_high_frequency_fraction"
        ],
        linewidth=1.3,
        label="FNO"
    )

    ax.plot(
        spectral_df["timestep"],
        spectral_df[
            "LNO_high_frequency_fraction"
        ],
        linewidth=1.3,
        label="LNO"
    )

    ax.set_xlabel(
        "Autoregressive timestep"
    )

    ax.set_ylabel(
        "High-frequency spectral fraction"
    )

    ax.legend(
        frameon=False,
        ncol=2
    )

    ax.grid(
        alpha=0.20,
        linewidth=0.5
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig4_high_frequency_stability"
    )

    # --------------------------------------------------------
    # Figure 4b — total spectral growth
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=SINGLE_COL
    )

    ax.plot(
        spectral_df["timestep"],
        spectral_df[
            "FNO_total_spectral_growth"
        ],
        linewidth=1.3,
        label="FNO"
    )

    ax.plot(
        spectral_df["timestep"],
        spectral_df[
            "LNO_total_spectral_growth"
        ],
        linewidth=1.3,
        label="LNO"
    )

    ax.axhline(
        1.0,
        linewidth=0.7,
        linestyle="--"
    )

    ax.set_xlabel(
        "Autoregressive timestep"
    )

    ax.set_ylabel(
        "Normalized spectral energy"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        alpha=0.20,
        linewidth=0.5
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig4b_total_spectral_growth"
    )


# ============================================================
# 11. FIGURE 5 — SPECTRAL ENERGY DISTRIBUTION
# ============================================================

if spectral_mode_df is not None:

    fig, ax = plt.subplots(
        figsize=DOUBLE_COL
    )

    ax.plot(
        spectral_mode_df[
            "fourier_mode"
        ],
        spectral_mode_df[
            "FNO_mean_spectral_energy"
        ],
        linewidth=1.2,
        label="FNO"
    )

    ax.plot(
        spectral_mode_df[
            "fourier_mode"
        ],
        spectral_mode_df[
            "LNO_mean_spectral_energy"
        ],
        linewidth=1.2,
        label="LNO"
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "Fourier mode"
    )

    ax.set_ylabel(
        "Mean spectral energy"
    )

    ax.legend(
        frameon=False,
        ncol=2
    )

    ax.grid(
        which="both",
        alpha=0.20,
        linewidth=0.5
    )

    fig.tight_layout()

    save_figure(
        fig,
        "fig5_spectral_energy_by_mode"
    )


# ============================================================
# 12. PAPER RESULTS SUMMARY
# ============================================================

paper_summary = {

    "core_claim_axis":
        "FNO vs LNO",

    "figures": [

        "fig1_clean_benchmark",
        "fig2_ood_robustness",
        "fig2b_ood_lno_advantage",
        "fig3_long_horizon_energy",
        "fig3b_energy_drift",
        "fig4_high_frequency_stability",
        "fig4b_total_spectral_growth",
        "fig5_spectral_energy_by_mode",
    ],

    "tables": [

        "table_clean_benchmark",
        "table_ood_sigma",
        "table_long_horizon",
        "table_spectral_stability",
    ],

    "notes": [

        "Clean benchmark uses RMSE, Relative_L2 and MAE.",

        "OOD results use the systematic sigma sweep.",

        "Long-horizon evaluation is stability/energy analysis, "
        "not future-ground-truth multi-step RMSE.",

        "Spectral figures use normalized Fourier-energy diagnostics.",

        "Secondary metrics should be cross-checked against "
        "their original evaluator before inclusion."
    ],
}

SUMMARY_PATH = (
    PAPER_DIR
    / "paper_figures_tables_summary.json"
)

with open(
    SUMMARY_PATH,
    "w"
) as f:

    json.dump(
        paper_summary,
        f,
        indent=2
    )

# ============================================================
# 13. MANIFEST
# ============================================================

all_files = []

for root, dirs, files in os.walk(
    PAPER_DIR
):

    for filename in files:

        path = (
            Path(root)
            / filename
        )

        all_files.append({

            "path":
                str(
                    path.relative_to(
                        PAPER_DIR
                    )
                ),

            "size_bytes":
                path.stat().st_size,

        })

manifest_path = (
    PAPER_DIR
    / "manifest.csv"
)

pd.DataFrame(
    all_files
).to_csv(
    manifest_path,
    index=False
)

# ============================================================
# 14. FINAL REPORT
# ============================================================

print("\n" + "=" * 100)
print("STEP 22 — FINAL PAPER FIGURES + TABLES COMPLETE")
print("=" * 100)

print(
    "[+] Figures directory:",
    FIG_DIR
)

print(
    "[+] Tables directory:",
    TABLE_DIR
)

print(
    "[+] Data directory:",
    DATA_DIR
)

print(
    "[+] Summary:",
    SUMMARY_PATH
)

print(
    "[+] Manifest:",
    manifest_path
)

print(
    "\n[+] Generated files:",
    len(all_files)
)

print("=" * 100)
