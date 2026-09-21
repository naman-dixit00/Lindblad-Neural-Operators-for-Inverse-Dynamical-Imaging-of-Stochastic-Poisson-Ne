# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 68
# Step            : STEP_21
# Step Heading    : # STEP 21 — COLLAPSE STRESS TEST
# Step Cell No.   : 4
# ============================================================

# ============================================================
# STEP 21 — COLLAPSE STRESS TEST
# FINAL DIAGNOSTIC VERSION
# ============================================================
#
# Goal:
#   Stress-test FNO vs final repaired LNO specifically on
#   collapse-regime states.
#
# Diagnostics:
#   1. Prediction response under progressively harder collapse
#   2. Trace preservation
#   3. Minimum eigenvalue
#   4. PSD violation
#   5. Rollout stability
#   6. State/amplitude response
#   7. Lindblad vs neural dynamical contribution
#
# IMPORTANT:
#   This is a STRESS / RESPONSE diagnostic.
#   It is NOT claimed as a new ground-truth accuracy benchmark
#   unless independently simulated collapse trajectories exist.
#
# No model retraining is performed.
# No checkpoint is modified.
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import matplotlib.pyplot as plt


# ============================================================
# 2. ROOT
# ============================================================

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 110)
print("STEP 21 — COLLAPSE STRESS TEST")
print("=" * 110)

print(
    "[+] Device:",
    device
)


# ============================================================
# 3. FLEXIBLE FILE LOCATOR
# ============================================================
#
# Because two large NPZ files may have been manually uploaded
# outside the repository folder, search /content recursively.
#
# ============================================================

def locate_file(
    filename,
    preferred_paths=None,
    required=True
):

    if preferred_paths is None:
        preferred_paths = []

    # --------------------------------------------------------
    # 1. Preferred exact locations
    # --------------------------------------------------------

    for path in preferred_paths:

        path = Path(path)

        if path.is_file():

            return path


    # --------------------------------------------------------
    # 2. Search repository
    # --------------------------------------------------------

    repo_matches = list(
        ROOT.rglob(
            filename
        )
    )

    if repo_matches:

        return repo_matches[0]


    # --------------------------------------------------------
    # 3. Search /content
    # --------------------------------------------------------

    content_root = Path(
        "/content"
    )

    matches = []

    for path in content_root.rglob(
        filename
    ):

        if path.is_file():

            matches.append(
                path
            )

    if matches:

        # Prefer newest file if several uploads exist.
        matches.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        return matches[0]


    if required:

        raise FileNotFoundError(
            f"Could not locate required file:\n{filename}"
        )

    return None


# ============================================================
# 4. LOCATE CORE ARTIFACTS
# ============================================================

DATASET_PATH = locate_file(
    "environment_conditioned_lno_dataset_v3_expanded.npz",
    preferred_paths=[
        ROOT
        / "results"
        / "dataset_v3_expanded"
        / "environment_conditioned_lno_dataset_v3_expanded.npz"
    ]
)

SPLIT_PATH = locate_file(
    "final_trajectory_level_split.npz",
    preferred_paths=[
        ROOT
        / "results"
        / "step16_final_splits"
        / "final_trajectory_level_split.npz"
    ]
)

FNO_CHECKPOINT = locate_file(
    "fno_final_best.pt",
    preferred_paths=[
        ROOT
        / "results"
        / "step17_final_fno"
        / "fno_final_best.pt"
    ]
)

LNO_CHECKPOINT = locate_file(
    "lno_final_best.pt",
    preferred_paths=[
        ROOT
        / "results"
        / "step18_final_lno"
        / "lno_final_best.pt"
    ]
)

print(
    "\n" + "=" * 110
)

print(
    "A. ARTIFACT LOCATIONS"
)

print(
    "=" * 110
)

print(
    "Dataset:",
    DATASET_PATH
)

print(
    "Split:",
    SPLIT_PATH
)

print(
    "FNO checkpoint:",
    FNO_CHECKPOINT
)

print(
    "LNO checkpoint:",
    LNO_CHECKPOINT
)


# ============================================================
# 5. CHECK ARCHITECTURES
# ============================================================

if "FinalFNO" not in globals():

    raise RuntimeError(
        "FinalFNO class is not defined in this runtime."
        "\nPlease rerun the Step-17 model-definition/recovery cell."
    )

if "FinalRepairedLNO" not in globals():

    raise RuntimeError(
        "FinalRepairedLNO class is not defined in this runtime."
        "\nPlease rerun the Step-18 model-definition cell."
    )

print(
    "\n[PASS] FNO and repaired-LNO classes available."
)


# ============================================================
# 6. OUTPUT DIRECTORY
# ============================================================

OUT_DIR = (
    ROOT
    / "results"
    / "step21_collapse_stress"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULTS_PATH = (
    OUT_DIR
    / "collapse_stress_results.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "collapse_stress_summary.json"
)

ROLLOUT_PATH = (
    OUT_DIR
    / "collapse_stress_rollout.csv"
)

REGIME_PATH = (
    OUT_DIR
    / "collapse_stress_regime_summary.csv"
)

PLOT_PATH = (
    OUT_DIR
    / "collapse_stress_response.png"
)


# ============================================================
# 7. LOAD DATASET
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "B. LOADING COLLAPSE DATA"
)

print(
    "=" * 110
)

data = np.load(
    DATASET_PATH
)

X_R_all = data[
    "X_R"
].astype(
    np.float32
)

X_A_all = data[
    "X_amplitude"
].astype(
    np.float32
)

ENV_all = data[
    "environment"
].astype(
    np.float32
)

print(
    "X_R:",
    X_R_all.shape
)

print(
    "X_amplitude:",
    X_A_all.shape
)

print(
    "Environment:",
    ENV_all.shape
)


# ============================================================
# 8. LOAD TEST SPLIT
# ============================================================

split = np.load(
    SPLIT_PATH
)

test_indices = (
    split[
        "test_indices"
    ]
    .astype(
        np.int64
    )
)

print(
    "Test transitions:",
    len(
        test_indices
    )
)

assert len(
    test_indices
) == 1485


# ============================================================
# 9. IDENTIFY COLLAPSE TRANSITIONS
# ============================================================

test_environment = ENV_all[test_indices]

# Collapse regime:
#   gamma = 0.50
#   sigma = 0.65
#
# Environment is stored as (transitions, Nx, 2),
# so reduce the spatial dimension to obtain one boolean
# decision per transition.

collapse_gamma_mask = np.all(
    np.isclose(
        test_environment[..., 0],
        0.50,
        atol=1e-5
    ),
    axis=1
)

collapse_sigma_mask = np.all(
    np.isclose(
        test_environment[..., 1],
        0.65,
        atol=1e-5
    ),
    axis=1
)

collapse_mask = (
    collapse_gamma_mask
    &
    collapse_sigma_mask
)

collapse_indices = test_indices[collapse_mask]

print(
    "\n[+] Collapse test transitions:",
    len(collapse_indices)
)

assert len(collapse_indices) > 0

print(
    "[PASS] Collapse test states identified."
)

print(
    "\n[+] Collapse test transitions:",
    len(
        collapse_indices
    )
)

assert len(
    collapse_indices
) > 0

print(
    "[PASS] Collapse test states identified."
)


# ============================================================
# 10. EXTRACT COLLAPSE TEST SET
# ============================================================

X_R_collapse = X_R_all[
    collapse_indices
]

X_A_collapse = X_A_all[
    collapse_indices
]

ENV_collapse = ENV_all[
    collapse_indices
]


# ============================================================
# 11. FNO NORMALIZATION
# ============================================================

# ------------------------------------------------------------
# Try Step-17 summary first.
# ------------------------------------------------------------

FNO_SUMMARY = locate_file(
    "fno_final_summary.json",
    preferred_paths=[
        ROOT
        / "results"
        / "step17_final_fno"
        / "fno_final_summary.json"
    ]
)

with open(
    FNO_SUMMARY,
    "r"
) as f:

    fno_summary = json.load(
        f
    )


amp_norm = (
    fno_summary[
        "amplitude_normalization"
    ]
)

amp_mean = float(
    amp_norm[
        "mean"
    ]
)

amp_std = float(
    amp_norm[
        "std"
    ]
)

X_A_collapse_norm = (
    X_A_collapse
    -
    amp_mean
) / amp_std

print(
    "\n[+] FNO-consistent amplitude normalization:"
)

print(
    "Mean:",
    amp_mean
)

print(
    "Std:",
    amp_std
)


# ============================================================
# 12. TORCH INPUTS
# ============================================================

X_R_t = torch.from_numpy(
    X_R_collapse
).to(
    device
)

X_A_t = torch.from_numpy(
    X_A_collapse_norm.astype(
        np.float32
    )
).to(
    device
)

ENV_t = torch.from_numpy(
    ENV_collapse
).to(
    device
)

gamma_base = ENV_t[
    ...,
    0
]

sigma_base = ENV_t[
    ...,
    1
]


# ============================================================
# 13. LOAD FINAL FNO
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "C. LOADING FINAL MODELS"
)

print(
    "=" * 110
)

fno_model = FinalFNO(
    dim=6,
    width=64,
    modes=16,
    depth=4
).to(
    device
)

fno_ckpt = torch.load(
    FNO_CHECKPOINT,
    map_location=device
)

fno_model.load_state_dict(
    fno_ckpt[
        "model_state_dict"
    ]
)

fno_model.eval()

print(
    "[PASS] Final FNO loaded."
)


# ============================================================
# 14. LOAD FINAL REPAIRED LNO
# ============================================================

lno_model = FinalRepairedLNO(
    dim=6,
    width=64,
    modes=16,
    depth=4,
    lindblad_channels=4
).to(
    device
)

lno_ckpt = torch.load(
    LNO_CHECKPOINT,
    map_location=device
)

lno_model.load_state_dict(
    lno_ckpt[
        "model_state_dict"
    ]
)

lno_model.eval()

print(
    "[PASS] Final repaired LNO loaded."
)


# ============================================================
# 15. STRESS LEVELS
# ============================================================
#
# Base collapse:
#   gamma = 0.50
#   sigma = 0.65
#
# Stress increases both dissipation and stochastic forcing.
#
# IMPORTANT:
# These are counterfactual environment probes applied to
# the same collapse input states.
# They are NOT new ground-truth simulations.
#
# ============================================================

STRESS_LEVELS = [

    {
        "level":
            0,

        "gamma":
            0.50,

        "sigma":
            0.65,

        "label":
            "baseline_collapse",
    },

    {
        "level":
            1,

        "gamma":
            0.60,

        "sigma":
            0.75,

        "label":
            "collapse_stress_1",
    },

    {
        "level":
            2,

        "gamma":
            0.70,

        "sigma":
            0.85,

        "label":
            "collapse_stress_2",
    },

    {
        "level":
            3,

        "gamma":
            0.80,

        "sigma":
            1.00,

        "label":
            "collapse_stress_3",
    },

    {
        "level":
            4,

        "gamma":
            0.90,

        "sigma":
            1.15,

        "label":
            "collapse_stress_4",
    },

    {
        "level":
            5,

        "gamma":
            1.00,

        "sigma":
            1.30,

        "label":
            "collapse_stress_5",
    },

]


# ============================================================
# 16. STRUCTURAL METRIC FUNCTION
# ============================================================

def get_structural_metrics(
    R
):

    trace = torch.diagonal(
        R,
        dim1=-2,
        dim2=-1
    ).sum(
        dim=-1
    )

    trace_error = torch.abs(
        trace
        -
        1.0
    )

    eig = torch.linalg.eigvalsh(
        R
    )

    return {

        "R_norm":
            float(
                torch.linalg.vector_norm(
                    R
                ).item()
            ),

        "mean_trace_error":
            float(
                trace_error.mean().item()
            ),

        "max_trace_error":
            float(
                trace_error.max().item()
            ),

        "minimum_eigenvalue":
            float(
                eig.min().item()
            ),

        "negative_eigenvalue_fraction":
            float(
                (
                    eig
                    <
                    -1e-6
                )
                .float()
                .mean()
                .item()
            ),

        "finite":
            bool(
                torch.isfinite(
                    R
                ).all().item()
            ),

    }


# ============================================================
# 17. RUN STRESS TEST
# ============================================================

rows = []

rollout_rows = []

baseline_fno_R = None
baseline_lno_R = None


print(
    "\n" + "=" * 110
)

print(
    "D. PROGRESSIVE COLLAPSE STRESS"
)

print(
    "=" * 110
)


with torch.no_grad():

    for stress in STRESS_LEVELS:

        level = stress[
            "level"
        ]

        gamma_value = stress[
            "gamma"
        ]

        sigma_value = stress[
            "sigma"
        ]

        label = stress[
            "label"
        ]

        print(
            "\n" + "-" * 90
        )

        print(
            f"Stress level {level}: "
            f"{label}"
        )

        gamma = torch.full(
            (
                len(
                    collapse_indices
                ),
                X_R_collapse.shape[1]
            ),
            gamma_value,
            dtype=torch.float32,
            device=device
        )

        sigma = torch.full(
            (
                len(
                    collapse_indices
                ),
                X_R_collapse.shape[1]
            ),
            sigma_value,
            dtype=torch.float32,
            device=device
        )


        # ----------------------------------------------------
        # FNO
        # ----------------------------------------------------

        fno_result = fno_model(
            X_R_t,
            X_A_t,
            gamma,
            sigma
        )

        fno_R = fno_result[
            "R_next"
        ]

        fno_A = fno_result[
            "amplitude_next"
        ]


        # ----------------------------------------------------
        # LNO
        # ----------------------------------------------------

        lno_result = lno_model(
            X_R_t,
            X_A_t,
            gamma,
            sigma
        )

        lno_R = lno_result[
            "R_next"
        ]

        lno_A = lno_result[
            "amplitude_next"
        ]


        # ----------------------------------------------------
        # Baselines
        # ----------------------------------------------------

        if level == 0:

            baseline_fno_R = (
                fno_R.clone()
            )

            baseline_lno_R = (
                lno_R.clone()
            )


        # ----------------------------------------------------
        # Structural metrics
        # ----------------------------------------------------

        fno_struct = get_structural_metrics(
            fno_R
        )

        lno_struct = get_structural_metrics(
            lno_R
        )


        # ----------------------------------------------------
        # Response from baseline collapse
        # ----------------------------------------------------

        fno_response = float(
            torch.linalg.vector_norm(
                fno_R
                -
                baseline_fno_R
            ).item()
        )

        lno_response = float(
            torch.linalg.vector_norm(
                lno_R
                -
                baseline_lno_R
            ).item()
        )


        # ----------------------------------------------------
        # Amplitude norms
        # ----------------------------------------------------

        fno_A_norm = float(
            torch.linalg.vector_norm(
                fno_A
            ).item()
        )

        lno_A_norm = float(
            torch.linalg.vector_norm(
                lno_A
            ).item()
        )


        # ----------------------------------------------------
        # LNO dynamical decomposition
        # ----------------------------------------------------

        lindblad_norm = np.nan
        neural_norm = np.nan

        if (
            isinstance(
                lno_result,
                dict
            )
        ):

            if (
                "delta_R_lindblad"
                in lno_result
            ):

                lindblad_norm = float(
                    torch.linalg.vector_norm(
                        lno_result[
                            "delta_R_lindblad"
                        ]
                    ).item()
                )

            if (
                "delta_R_neural"
                in lno_result
            ):

                neural_norm = float(
                    torch.linalg.vector_norm(
                        lno_result[
                            "delta_R_neural"
                        ]
                    ).item()
                )


        # ----------------------------------------------------
        # Append FNO
        # ----------------------------------------------------

        rows.append({

            "stress_level":
                level,

            "label":
                label,

            "model":
                "FNO",

            "gamma":
                gamma_value,

            "sigma":
                sigma_value,

            "R_norm":
                fno_struct[
                    "R_norm"
                ],

            "amplitude_norm":
                fno_A_norm,

            "mean_trace_error":
                fno_struct[
                    "mean_trace_error"
                ],

            "max_trace_error":
                fno_struct[
                    "max_trace_error"
                ],

            "minimum_eigenvalue":
                fno_struct[
                    "minimum_eigenvalue"
                ],

            "negative_eigenvalue_fraction":
                fno_struct[
                    "negative_eigenvalue_fraction"
                ],

            "finite":
                fno_struct[
                    "finite"
                ],

            "response_from_baseline":
                fno_response,

            "Lindblad_delta_norm":
                np.nan,

            "Neural_delta_norm":
                np.nan,

        })


        # ----------------------------------------------------
        # Append LNO
        # ----------------------------------------------------

        rows.append({

            "stress_level":
                level,

            "label":
                label,

            "model":
                "LNO",

            "gamma":
                gamma_value,

            "sigma":
                sigma_value,

            "R_norm":
                lno_struct[
                    "R_norm"
                ],

            "amplitude_norm":
                lno_A_norm,

            "mean_trace_error":
                lno_struct[
                    "mean_trace_error"
                ],

            "max_trace_error":
                lno_struct[
                    "max_trace_error"
                ],

            "minimum_eigenvalue":
                lno_struct[
                    "minimum_eigenvalue"
                ],

            "negative_eigenvalue_fraction":
                lno_struct[
                    "negative_eigenvalue_fraction"
                ],

            "finite":
                lno_struct[
                    "finite"
                ],

            "response_from_baseline":
                lno_response,

            "Lindblad_delta_norm":
                lindblad_norm,

            "Neural_delta_norm":
                neural_norm,

        })


        print(
            f"  gamma={gamma_value:.2f} "
            f"sigma={sigma_value:.2f}"
        )

        print(
            "  FNO min eigenvalue:",
            f"{fno_struct['minimum_eigenvalue']:.6e}"
        )

        print(
            "  LNO min eigenvalue:",
            f"{lno_struct['minimum_eigenvalue']:.6e}"
        )

        print(
            "  FNO trace error:",
            f"{fno_struct['max_trace_error']:.6e}"
        )

        print(
            "  LNO trace error:",
            f"{lno_struct['max_trace_error']:.6e}"
        )

        print(
            "  FNO finite:",
            fno_struct["finite"]
        )

        print(
            "  LNO finite:",
            lno_struct["finite"]
        )


# ============================================================
# 18. RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    rows
)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

print(
    "\n[PASS] Stress results saved."
)


# ============================================================
# 19. FINAL OOD/COLLAPSE RISK SUMMARY
# ============================================================

fno_rows = results_df[
    results_df[
        "model"
    ]
    ==
    "FNO"
]

lno_rows = results_df[
    results_df[
        "model"
    ]
    ==
    "LNO"
]


# Worst structural behavior
# More negative eigenvalue = worse.
# Larger trace error = worse.

fno_worst_eig = float(
    fno_rows[
        "minimum_eigenvalue"
    ].min()
)

lno_worst_eig = float(
    lno_rows[
        "minimum_eigenvalue"
    ].min()
)

fno_worst_trace = float(
    fno_rows[
        "max_trace_error"
    ].max()
)

lno_worst_trace = float(
    lno_rows[
        "max_trace_error"
    ].max()
)

fno_worst_psd_fraction = float(
    fno_rows[
        "negative_eigenvalue_fraction"
    ].max()
)

lno_worst_psd_fraction = float(
    lno_rows[
        "negative_eigenvalue_fraction"
    ].max()
)

fno_all_finite = bool(
    fno_rows[
        "finite"
    ].all()
)

lno_all_finite = bool(
    lno_rows[
        "finite"
    ].all()
)


# ============================================================
# 20. STRUCTURAL ADVANTAGE FLAG
# ============================================================

lno_structurally_better = bool(

    (
        lno_worst_trace
        <
        fno_worst_trace
    )

    and

    (
        lno_worst_eig
        >
        fno_worst_eig
    )

    and

    (
        lno_worst_psd_fraction
        <=
        fno_worst_psd_fraction
    )

)


# ============================================================
# 21. COLLAPSE RESPONSE PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.plot(

    fno_rows[
        "stress_level"
    ],

    fno_rows[
        "response_from_baseline"
    ],

    marker="o",

    linewidth=2,

    label="FNO"

)

ax.plot(

    lno_rows[
        "stress_level"
    ],

    lno_rows[
        "response_from_baseline"
    ],

    marker="s",

    linewidth=2,

    label="LNO"

)

ax.set_xlabel(
    "Collapse stress level"
)

ax.set_ylabel(
    "R response from baseline"
)

ax.set_title(
    "Progressive Collapse-Stress Response"
)

ax.grid(
    True,
    alpha=0.25
)

ax.legend(
    frameon=False
)

plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# 22. REPRESENTATIVE LNO ROLLOUT
# ============================================================
#
# Use one collapse state and progressively apply the hardest
# collapse environment to test repeated forward application.
#
# ============================================================

REP_INDEX = 0

R0 = X_R_t[
    REP_INDEX:
    REP_INDEX + 1
]

A0 = X_A_t[
    REP_INDEX:
    REP_INDEX + 1
]

gamma_rollout = 1.00
sigma_rollout = 1.30

R_roll = R0.clone()

A_roll = A0.clone()

rollout_steps = 40

with torch.no_grad():

    for step in range(
        rollout_steps
    ):

        B = R_roll.shape[0]
        NX = R_roll.shape[1]

        gamma_roll = torch.full(
            (
                B,
                NX
            ),
            gamma_rollout,
            dtype=torch.float32,
            device=device
        )

        sigma_roll = torch.full(
            (
                B,
                NX
            ),
            sigma_rollout,
            dtype=torch.float32,
            device=device
        )

        result = lno_model(
            R_roll,
            A_roll,
            gamma_roll,
            sigma_roll
        )

        R_roll = result[
            "R_next"
        ]

        A_roll = result[
            "amplitude_next"
        ]

        eig = torch.linalg.eigvalsh(
            R_roll
        )

        trace = torch.diagonal(
            R_roll,
            dim1=-2,
            dim2=-1
        ).sum(
            dim=-1
        )

        rollout_rows.append({

            "step":
                step,

            "R_norm":
                float(
                    torch.linalg.vector_norm(
                        R_roll
                    ).item()
                ),

            "amplitude_norm":
                float(
                    torch.linalg.vector_norm(
                        A_roll
                    ).item()
                ),

            "trace_error":
                float(
                    torch.abs(
                        trace
                        -
                        1.0
                    ).max().item()
                ),

            "minimum_eigenvalue":
                float(
                    eig.min().item()
                ),

            "finite":
                bool(
                    torch.isfinite(
                        R_roll
                    ).all().item()
                ),
        })


rollout_df = pd.DataFrame(
    rollout_rows
)

rollout_df.to_csv(
    ROLLOUT_PATH,
    index=False
)


rollout_stable = bool(
    rollout_df[
        "finite"
    ].all()
)

rollout_min_eig = float(
    rollout_df[
        "minimum_eigenvalue"
    ].min()
)

rollout_max_trace = float(
    rollout_df[
        "trace_error"
    ].max()
)


# ============================================================
# 23. REGIME SUMMARY
# ============================================================

regime_summary_rows = [

    {

        "regime":
            "collapse",

        "transitions":
            int(
                len(
                    collapse_indices
                )
            ),

        "FNO_worst_minimum_eigenvalue":
            fno_worst_eig,

        "LNO_worst_minimum_eigenvalue":
            lno_worst_eig,

        "FNO_worst_trace_error":
            fno_worst_trace,

        "LNO_worst_trace_error":
            lno_worst_trace,

        "FNO_worst_negative_eigenvalue_fraction":
            fno_worst_psd_fraction,

        "LNO_worst_negative_eigenvalue_fraction":
            lno_worst_psd_fraction,

        "FNO_all_finite":
            fno_all_finite,

        "LNO_all_finite":
            lno_all_finite,

        "LNO_structurally_better":
            lno_structurally_better,

    }
]

regime_summary_df = pd.DataFrame(
    regime_summary_rows
)

regime_summary_df.to_csv(
    REGIME_PATH,
    index=False
)


# ============================================================
# 24. FINAL JSON SUMMARY
# ============================================================

summary = {

    "step":
        21,

    "title":
        "Collapse Stress Test",

    "test_regime":
        "collapse",

    "collapse_test_transitions":
        int(
            len(
                collapse_indices
            )
        ),

    "stress_levels":
        STRESS_LEVELS,

    "ground_truth_accuracy_claim":
        False,

    "diagnostic_type":
        (
            "Counterfactual environment-response and "
            "structural-stability stress test."
        ),

    "FNO":
        {

            "worst_minimum_eigenvalue":
                fno_worst_eig,

            "worst_max_trace_error":
                fno_worst_trace,

            "worst_negative_eigenvalue_fraction":
                fno_worst_psd_fraction,

            "all_finite":
                fno_all_finite,

        },

    "LNO":
        {

            "worst_minimum_eigenvalue":
                lno_worst_eig,

            "worst_max_trace_error":
                lno_worst_trace,

            "worst_negative_eigenvalue_fraction":
                lno_worst_psd_fraction,

            "all_finite":
                lno_all_finite,

            "representative_40_step_rollout_finite":
                rollout_stable,

            "representative_40_step_rollout_minimum_eigenvalue":
                rollout_min_eig,

            "representative_40_step_rollout_max_trace_error":
                rollout_max_trace,

        },

    "structural_comparison":
        {

            "LNO_better":
                lno_structurally_better,

        },

    "outputs":
        {

            "results":
                str(
                    RESULTS_PATH
                ),

            "rollout":
                str(
                    ROLLOUT_PATH
                ),

            "regime_summary":
                str(
                    REGIME_PATH
                ),

            "plot":
                str(
                    PLOT_PATH
                ),

            "summary":
                str(
                    SUMMARY_PATH
                ),

        },
}


with open(
    SUMMARY_PATH,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        allow_nan=False
    )


# ============================================================
# 25. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "STEP 21 COMPLETE — COLLAPSE STRESS TEST"
)

print(
    "=" * 110
)

print(
    "\nCOLLAPSE TEST SIZE:",
    len(
        collapse_indices
    )
)

print(
    "\nFNO:"
)

print(
    "  Worst minimum eigenvalue:",
    f"{fno_worst_eig:.8e}"
)

print(
    "  Worst trace error:",
    f"{fno_worst_trace:.8e}"
)

print(
    "  Worst negative-eigenvalue fraction:",
    f"{fno_worst_psd_fraction:.8f}"
)

print(
    "  All finite:",
    fno_all_finite
)

print(
    "\nLNO:"
)

print(
    "  Worst minimum eigenvalue:",
    f"{lno_worst_eig:.8e}"
)

print(
    "  Worst trace error:",
    f"{lno_worst_trace:.8e}"
)

print(
    "  Worst negative-eigenvalue fraction:",
    f"{lno_worst_psd_fraction:.8f}"
)

print(
    "  All finite:",
    lno_all_finite
)

print(
    "\nLNO 40-step rollout:"
)

print(
    "  Finite:",
    rollout_stable
)

print(
    "  Minimum eigenvalue:",
    f"{rollout_min_eig:.8e}"
)

print(
    "  Maximum trace error:",
    f"{rollout_max_trace:.8e}"
)

print(
    "\nSTRUCTURAL ADVANTAGE FLAG:",
    lno_structurally_better
)

print(
    "\nResults:",
    RESULTS_PATH
)

print(
    "Rollout:",
    ROLLOUT_PATH
)

print(
    "Regime summary:",
    REGIME_PATH
)

print(
    "Plot:",
    PLOT_PATH
)

print(
    "Summary:",
    SUMMARY_PATH
)

print(
    "\n[PASS] STEP 21 COMPLETE."
)

print(
    "[INFO] No model retraining performed."
)

print(
    "[INFO] Collapse stress uses counterfactual environment probes."
)

print(
    "[INFO] True stress-regime accuracy requires independently simulated"
)

print(
    "       ground-truth collapse trajectories at the stressed parameters."
)

print("=" * 110)
