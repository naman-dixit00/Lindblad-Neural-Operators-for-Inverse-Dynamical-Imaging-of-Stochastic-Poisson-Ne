# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 64
# Step            : STEP_20
# Step Heading    : # STEP 20 — OOD SIGMA SWEEP
# Step Cell No.   : 2
# ============================================================

# ============================================================
# STEP 20 — OOD SIGMA SWEEP
# ============================================================
#
# Purpose:
#   Stress-test final FNO and final LNO beyond the
#   training noise_sigma range.
#
# Training sigma max:
#   0.65
#
# OOD sigma:
#   0.75, 0.85, 1.00, 1.25
#
# IMPORTANT:
#   This is an OOD RESPONSE / STABILITY diagnostic.
#   It is NOT OOD prediction accuracy because no new
#   ground-truth trajectories are generated in this cell.
#
# ============================================================

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# ============================================================
# 1. ROOT
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
print("STEP 20 — OOD SIGMA SWEEP")
print("=" * 110)

print("[+] Device:", device)


# ============================================================
# 2. PATHS
# ============================================================

DATASET_PATH = (
    ROOT
    / "results"
    / "dataset_v3_expanded"
    / "environment_conditioned_lno_dataset_v3_expanded.npz"
)

SPLIT_PATH = (
    ROOT
    / "results"
    / "step16_final_splits"
    / "final_trajectory_level_split.npz"
)

FNO_CHECKPOINT = (
    ROOT
    / "results"
    / "step17_final_fno"
    / "fno_final_best.pt"
)

LNO_CHECKPOINT = (
    ROOT
    / "results"
    / "step18_final_lno"
    / "lno_final_best.pt"
)

FNO_SUMMARY = (
    ROOT
    / "results"
    / "step17_final_fno"
    / "fno_final_summary.json"
)

OUT_DIR = (
    ROOT
    / "results"
    / "step20_ood_sigma"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_PATH = (
    OUT_DIR
    / "ood_sigma_results.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "ood_sigma_summary.json"
)

PLOT_PATH = (
    OUT_DIR
    / "ood_sigma_stability.png"
)


# ============================================================
# 3. REQUIRED FILE CHECK
# ============================================================

required = {

    "Dataset":
        DATASET_PATH,

    "Split":
        SPLIT_PATH,

    "FNO checkpoint":
        FNO_CHECKPOINT,

    "LNO checkpoint":
        LNO_CHECKPOINT,

    "FNO summary":
        FNO_SUMMARY,
}

for name, path in required.items():

    exists = path.is_file()

    print(
        f"{name:22s}: {exists}"
    )

    if not exists:

        raise FileNotFoundError(
            f"Missing {name}:\n{path}"
        )

print(
    "[PASS] Required Step-20 artifacts found."
)


# ============================================================
# 4. ARCHITECTURE AVAILABILITY
# ============================================================

if "FinalFNO" not in globals():

    raise RuntimeError(
        "FinalFNO class is not currently defined in this Colab "
        "runtime. Re-run the Step-17 FNO recovery/model-definition "
        "cell before Step 20."
    )

if "FinalRepairedLNO" not in globals():

    raise RuntimeError(
        "FinalRepairedLNO class is not currently defined in this "
        "Colab runtime. Re-run the Step-18 model-definition cell "
        "before Step 20."
    )

print(
    "[PASS] FNO and LNO architecture classes available."
)


# ============================================================
# 5. LOAD DATASET
# ============================================================

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
    "\n[+] Dataset:",
    X_R_all.shape
)

print(
    "[+] Amplitude:",
    X_A_all.shape
)

print(
    "[+] Environment:",
    ENV_all.shape
)


# ============================================================
# 6. LOAD TEST SPLIT
# ============================================================

split = np.load(
    SPLIT_PATH
)

test_indices = split[
    "test_indices"
].astype(
    np.int64
)

assert len(
    test_indices
) == 1485

X_R_test = X_R_all[
    test_indices
]

X_A_test = X_A_all[
    test_indices
]

ENV_test = ENV_all[
    test_indices
]

print(
    "\n[+] Test transitions:",
    len(test_indices)
)


# ============================================================
# 7. FNO-CONSISTENT AMPLITUDE NORMALIZATION
# ============================================================

with open(
    FNO_SUMMARY,
    "r"
) as f:

    fno_summary = json.load(
        f
    )

norm = fno_summary[
    "amplitude_normalization"
]

amp_mean = float(
    norm["mean"]
)

amp_std = float(
    norm["std"]
)

X_A_test_norm = (
    X_A_test
    -
    amp_mean
) / amp_std

print(
    "[+] Amplitude mean:",
    amp_mean
)

print(
    "[+] Amplitude std:",
    amp_std
)

print(
    "[PASS] FNO-consistent normalization applied."
)


# ============================================================
# 8. CREATE FINAL FNO
# ============================================================

fno_model = FinalFNO(
    dim=6,
    width=64,
    modes=16,
    depth=4
).to(
    device
)

fno_checkpoint = torch.load(
    FNO_CHECKPOINT,
    map_location=device
)

fno_model.load_state_dict(
    fno_checkpoint[
        "model_state_dict"
    ]
)

fno_model.eval()

print(
    "[PASS] Final FNO checkpoint loaded."
)


# ============================================================
# 9. CREATE FINAL LNO
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

lno_checkpoint = torch.load(
    LNO_CHECKPOINT,
    map_location=device
)

lno_model.load_state_dict(
    lno_checkpoint[
        "model_state_dict"
    ]
)

lno_model.eval()

print(
    "[PASS] Final LNO checkpoint loaded."
)


# ============================================================
# 10. TORCH TEST INPUTS
# ============================================================

X_R_t = torch.from_numpy(
    X_R_test
).to(
    device
)

X_A_t = torch.from_numpy(
    X_A_test_norm.astype(
        np.float32
    )
).to(
    device
)

ENV_t = torch.from_numpy(
    ENV_test
).to(
    device
)

gamma_t = ENV_t[
    ...,
    0
]

sigma_original_t = ENV_t[
    ...,
    1
]


# ============================================================
# 11. OOD SIGMA VALUES
# ============================================================

TRAINING_SIGMA_MAX = 0.65

SIGMA_SWEEP = [

    0.65,   # boundary / reference
    0.75,   # OOD
    0.85,   # OOD
    1.00,   # OOD
    1.25,   # OOD

]

print(
    "\n[+] Training sigma max:",
    TRAINING_SIGMA_MAX
)

print(
    "[+] Sigma sweep:",
    SIGMA_SWEEP
)


# ============================================================
# 12. BATCHED INFERENCE
# ============================================================

BATCH_SIZE = 32

rows = []

baseline_outputs = {

    "FNO_R":
        None,

    "LNO_R":
        None,

}


def structural_metrics(
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

    finite = bool(
        torch.isfinite(
            R
        ).all().item()
    )

    mean_trace_error = float(
        trace_error.mean().item()
    )

    max_trace_error = float(
        trace_error.max().item()
    )

    min_eigenvalue = float(
        eig.min().item()
    )

    negative_fraction = float(
        (
            eig < -1e-6
        )
        .float()
        .mean()
        .item()
    )

    norm = float(
        torch.linalg.vector_norm(
            R
        ).item()
    )

    return {

        "R_norm":
            norm,

        "mean_trace_error":
            mean_trace_error,

        "max_trace_error":
            max_trace_error,

        "minimum_eigenvalue":
            min_eigenvalue,

        "negative_eigenvalue_fraction":
            negative_fraction,

        "finite":
            finite,

    }


# ============================================================
# 13. SWEEP
# ============================================================

for sigma_value in SIGMA_SWEEP:

    print(
        "\n" + "-" * 90
    )

    print(
        f"[+] Evaluating sigma = {sigma_value:.2f}"
    )

    fno_R_parts = []

    fno_A_parts = []

    lno_R_parts = []

    lno_A_parts = []

    lno_L_parts = []

    lno_N_parts = []


    with torch.no_grad():

        for start in range(
            0,
            len(test_indices),
            BATCH_SIZE
        ):

            end = min(
                start + BATCH_SIZE,
                len(test_indices)
            )

            R_batch = X_R_t[
                start:end
            ]

            A_batch = X_A_t[
                start:end
            ]

            gamma_batch = gamma_t[
                start:end
            ]

            sigma_batch = torch.full(
                (
                    end - start,
                    X_R_test.shape[1]
                ),
                float(
                    sigma_value
                ),
                dtype=torch.float32,
                device=device
            )


            # --------------------------------------------
            # FNO
            # --------------------------------------------

            fno_R_out, fno_A_out = fno_model(
                R_batch,
                A_batch,
                gamma_batch,
                sigma_batch
            )

            fno_R_parts.append(
                fno_R_out.cpu()
            )

            fno_A_parts.append(
                fno_A_out.cpu()
            )


            # --------------------------------------------
            # LNO
            # --------------------------------------------

            lno_out = lno_model(
                R_batch,
                A_batch,
                gamma_batch,
                sigma_batch
            )

            lno_R_parts.append(
                lno_out[
                    "R_next"
                ].cpu()
            )

            lno_A_parts.append(
                lno_out[
                    "amplitude_next"
                ].cpu()
            )

            lno_L_parts.append(
                lno_out[
                    "delta_R_lindblad"
                ].cpu()
            )

            lno_N_parts.append(
                lno_out[
                    "delta_R_neural"
                ].cpu()
            )


    fno_R = torch.cat(
        fno_R_parts,
        dim=0
    )

    fno_A = torch.cat(
        fno_A_parts,
        dim=0
    )

    lno_R = torch.cat(
        lno_R_parts,
        dim=0
    )

    lno_A = torch.cat(
        lno_A_parts,
        dim=0
    )

    lno_L = torch.cat(
        lno_L_parts,
        dim=0
    )

    lno_N = torch.cat(
        lno_N_parts,
        dim=0
    )


    # ========================================================
    # BASELINE STORAGE
    # ========================================================

    if abs(
        sigma_value
        -
        TRAINING_SIGMA_MAX
    ) < 1e-8:

        baseline_outputs[
            "FNO_R"
        ] = fno_R.clone()

        baseline_outputs[
            "LNO_R"
        ] = lno_R.clone()


    # ========================================================
    # STRUCTURE
    # ========================================================

    fno_struct = structural_metrics(
        fno_R
    )

    lno_struct = structural_metrics(
        lno_R
    )


    # ========================================================
    # AMPLITUDE / RESPONSE
    # ========================================================

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

    lno_L_norm = float(
        torch.linalg.vector_norm(
            lno_L
        ).item()
    )

    lno_N_norm = float(
        torch.linalg.vector_norm(
            lno_N
        ).item()
    )


    # ========================================================
    # RESPONSE FROM SIGMA=0.65 BASELINE
    # ========================================================

    if baseline_outputs[
        "FNO_R"
    ] is not None:

        fno_sigma_response = float(
            torch.linalg.vector_norm(
                fno_R
                -
                baseline_outputs[
                    "FNO_R"
                ]
            ).item()
        )

        lno_sigma_response = float(
            torch.linalg.vector_norm(
                lno_R
                -
                baseline_outputs[
                    "LNO_R"
                ]
            ).item()
        )

    else:

        fno_sigma_response = 0.0
        lno_sigma_response = 0.0


    # ========================================================
    # APPEND ROWS
    # ========================================================

    rows.append({

        "model":
            "FNO",

        "sigma":
            float(
                sigma_value
            ),

        "ood":
            bool(
                sigma_value
                >
                TRAINING_SIGMA_MAX
            ),

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

        "sigma_response_from_0p65":
            fno_sigma_response,

        "Lindblad_delta_norm":
            np.nan,

        "Neural_delta_norm":
            np.nan,

    })


    rows.append({

        "model":
            "LNO",

        "sigma":
            float(
                sigma_value
            ),

        "ood":
            bool(
                sigma_value
                >
                TRAINING_SIGMA_MAX
            ),

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

        "sigma_response_from_0p65":
            lno_sigma_response,

        "Lindblad_delta_norm":
            lno_L_norm,

        "Neural_delta_norm":
            lno_N_norm,

    })


    # --------------------------------------------------------
    # Incremental save
    # --------------------------------------------------------

    pd.DataFrame(
        rows
    ).to_csv(
        RESULTS_PATH,
        index=False
    )

    print(
        f"[PASS] sigma={sigma_value:.2f} complete."
    )


# ============================================================
# 14. RESULT TABLE
# ============================================================

results_df = pd.DataFrame(
    rows
)

print(
    "\n" + "=" * 110
)

print(
    "OOD SIGMA SWEEP RESULTS"
)

print(
    "=" * 110
)

display(
    results_df.round(
        8
    )
)


# ============================================================
# 15. MONOTONIC RESPONSE CHECK
# ============================================================

fno_ood = results_df[
    results_df[
        "model"
    ] == "FNO"
].sort_values(
    "sigma"
)

lno_ood = results_df[
    results_df[
        "model"
    ] == "LNO"
].sort_values(
    "sigma"
)

fno_response = (
    fno_ood[
        "sigma_response_from_0p65"
    ]
    .to_numpy()
)

lno_response = (
    lno_ood[
        "sigma_response_from_0p65"
    ]
    .to_numpy()
)

fno_monotonic = bool(
    np.all(
        np.diff(
            fno_response
        ) >= -1e-10
    )
)

lno_monotonic = bool(
    np.all(
        np.diff(
            lno_response
        ) >= -1e-10
    )
)

print(
    "\nFNO sigma-response monotonic:",
    fno_monotonic
)

print(
    "LNO sigma-response monotonic:",
    lno_monotonic
)


# ============================================================
# 16. OOD STRUCTURAL SUMMARY
# ============================================================

fno_ood_rows = fno_ood[
    fno_ood[
        "ood"
    ]
]

lno_ood_rows = lno_ood[
    lno_ood[
        "ood"
    ]
]

fno_max_trace_ood = float(
    fno_ood_rows[
        "max_trace_error"
    ].max()
)

lno_max_trace_ood = float(
    lno_ood_rows[
        "max_trace_error"
    ].max()
)

fno_min_eig_ood = float(
    fno_ood_rows[
        "minimum_eigenvalue"
    ].min()
)

lno_min_eig_ood = float(
    lno_ood_rows[
        "minimum_eigenvalue"
    ].min()
)

fno_max_negative_fraction = float(
    fno_ood_rows[
        "negative_eigenvalue_fraction"
    ].max()
)

lno_max_negative_fraction = float(
    lno_ood_rows[
        "negative_eigenvalue_fraction"
    ].max()
)

fno_all_finite = bool(
    fno_ood_rows[
        "finite"
    ].all()
)

lno_all_finite = bool(
    lno_ood_rows[
        "finite"
    ].all()
)


# ============================================================
# 17. PLOT
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.plot(

    fno_ood[
        "sigma"
    ],

    fno_ood[
        "sigma_response_from_0p65"
    ],

    marker="o",

    linewidth=2,

    label="FNO",

)

plt.plot(

    lno_ood[
        "sigma"
    ],

    lno_ood[
        "sigma_response_from_0p65"
    ],

    marker="s",

    linewidth=2,

    label="LNO",

)

plt.axvline(

    TRAINING_SIGMA_MAX,

    linestyle="--",

    linewidth=1.2,

    label="Training boundary",

)

plt.xlabel(
    r"Noise level $\sigma$"
)

plt.ylabel(
    "Response from σ = 0.65"
)

plt.title(
    "OOD Noise-Stress Response"
)

plt.grid(
    True,
    alpha=0.25
)

plt.legend(
    frameon=False
)

plt.tight_layout()

plt.savefig(
    PLOT_PATH,
    dpi=300
)

plt.show()


# ============================================================
# 18. FINAL JSON
# ============================================================

summary = {

    "step":
        20,

    "title":
        "OOD Sigma Sweep",

    "training_sigma_max":
        TRAINING_SIGMA_MAX,

    "sigma_values":
        SIGMA_SWEEP,

    "true_ood_accuracy":
        False,

    "description":
        (
            "Model-only OOD response and structural-stability "
            "diagnostic. True OOD accuracy requires new "
            "ground-truth trajectories at unseen sigma."
        ),

    "FNO":
        {

            "ood_max_trace_error":
                fno_max_trace_ood,

            "ood_minimum_eigenvalue":
                fno_min_eig_ood,

            "ood_max_negative_eigenvalue_fraction":
                fno_max_negative_fraction,

            "all_finite":
                fno_all_finite,

            "response_monotonic":
                fno_monotonic,

        },

    "LNO":
        {

            "ood_max_trace_error":
                lno_max_trace_ood,

            "ood_minimum_eigenvalue":
                lno_min_eig_ood,

            "ood_max_negative_eigenvalue_fraction":
                lno_max_negative_fraction,

            "all_finite":
                lno_all_finite,

            "response_monotonic":
                lno_monotonic,

        },

    "outputs":
        {

            "csv":
                str(
                    RESULTS_PATH
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
# 19. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "STEP 20 COMPLETE — OOD SIGMA SWEEP"
)

print(
    "=" * 110
)

print(
    "FNO OOD minimum eigenvalue:",
    f"{fno_min_eig_ood:.8e}"
)

print(
    "LNO OOD minimum eigenvalue:",
    f"{lno_min_eig_ood:.8e}"
)

print(
    "FNO OOD max trace error:",
    f"{fno_max_trace_ood:.8e}"
)

print(
    "LNO OOD max trace error:",
    f"{lno_max_trace_ood:.8e}"
)

print(
    "FNO OOD finite:",
    fno_all_finite
)

print(
    "LNO OOD finite:",
    lno_all_finite
)

print(
    "FNO response monotonic:",
    fno_monotonic
)

print(
    "LNO response monotonic:",
    lno_monotonic
)

print(
    "\nResults:",
    RESULTS_PATH
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
    "\n[PASS] STEP 20 COMPLETE."
)

print(
    "[INFO] This is an OOD stress diagnostic, not OOD accuracy."
)

print("=" * 110)
