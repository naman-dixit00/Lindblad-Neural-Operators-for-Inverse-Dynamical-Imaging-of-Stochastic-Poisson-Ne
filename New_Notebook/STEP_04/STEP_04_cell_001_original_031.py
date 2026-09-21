# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 31
# Step            : STEP_04
# Step Heading    : # STEP 4 — ENVIRONMENT-CONDITIONED DATASET V2
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 4 — ENVIRONMENT-CONDITIONED DATASET V2
#
# INPUT:
#   R(t)
#   gamma
#   noise_sigma
#
# TARGET:
#   R(t+1)
#
# ALSO PRESERVE:
#   physical state
#   noise
#   dissipation
#   instability
#   entropy
#   environment-induced response
#
# NO TRAINING
# NO MODEL MODIFICATION
# NO OLD DATASET OVERWRITE
# ============================================================

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# 1. ROOT
# ============================================================

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

print("=" * 100)
print("STEP 4 — ENVIRONMENT-CONDITIONED DATASET V2")
print("=" * 100)

# ============================================================
# 2. SOURCE / OUTPUT
# ============================================================

SOURCE_DIR = (
    ROOT
    / "results"
    / "step3_paired_trajectories"
)

OUT_DIR = (
    ROOT
    / "results"
    / "dataset_v2"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

assert SOURCE_DIR.is_dir(), (
    f"Missing Step 3 directory:\n{SOURCE_DIR}"
)

print(
    "[+] Source:",
    SOURCE_DIR
)

print(
    "[+] Output:",
    OUT_DIR
)

# ============================================================
# 3. REGIME ORDER
# ============================================================

REGIMES = {
    "low_noise": {
        "noise_sigma": 0.02,
        "gamma": 0.01,
    },

    "stochastic": {
        "noise_sigma": 0.45,
        "gamma": 0.05,
    },

    "heavy_dissipation": {
        "noise_sigma": 0.15,
        "gamma": 0.35,
    },

    "collapse": {
        "noise_sigma": 0.65,
        "gamma": 0.50,
    },

    "metastable": {
        "noise_sigma": 0.30,
        "gamma": 0.15,
    },
}

# ============================================================
# 4. CONTAINERS
# ============================================================

X_information = []
Y_information = []

X_state = []
Y_state = []

environment_features = []

environment_response = []

metadata_rows = []

# ============================================================
# 5. PROCESS EACH REGIME
# ============================================================

for regime_name, regime in REGIMES.items():

    print(
        "\n" + "=" * 100
    )

    print(
        f"PROCESSING: {regime_name}"
    )

    print(
        "=" * 100
    )

    source_file = (
        SOURCE_DIR
        / f"{regime_name}_paired.npz"
    )

    assert source_file.is_file(), (
        f"Missing paired trajectory:\n{source_file}"
    )

    data = np.load(
        source_file
    )

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    R_ref = data[
        "R_reference"
    ]

    R_open = data[
        "R_open"
    ]

    ref_state = data[
        "reference_state"
    ]

    open_state = data[
        "open_state"
    ]

    delta_R_env = data[
        "delta_R_environment"
    ]

    delta_state_env = data[
        "delta_state_environment"
    ]

    ref_noise = data[
        "reference_noise"
    ]

    open_noise = data[
        "open_noise"
    ]

    ref_diss = data[
        "reference_dissipation"
    ]

    open_diss = data[
        "open_dissipation"
    ]

    ref_entropy = data[
        "reference_entropy"
    ]

    open_entropy = data[
        "open_entropy"
    ]

    ref_inst = data[
        "reference_instability"
    ]

    open_inst = data[
        "open_instability"
    ]

    # --------------------------------------------------------
    # Basic shape checks
    # --------------------------------------------------------

    assert R_ref.ndim == 4
    assert R_open.ndim == 4

    T, NX, D1, D2 = (
        R_ref.shape
    )

    assert (
        R_open.shape
        ==
        R_ref.shape
    )

    assert D1 == D2

    assert (
        ref_state.shape[0]
        ==
        T
    )

    assert (
        open_state.shape
        ==
        ref_state.shape
    )

    print(
        "[+] R shape:",
        R_ref.shape
    )

    print(
        "[+] State shape:",
        ref_state.shape
    )

    # ========================================================
    # 6. CONSTRUCT ONE-STEP LEARNING PAIRS
    #
    # We use the OPEN trajectory as the actual dynamical
    # learning trajectory.
    #
    # Therefore:
    #
    #   X_t = R_open(t)
    #   Y_t = R_open(t+1)
    #
    # Environment metadata is attached to every transition.
    # ========================================================

    if T < 2:

        raise RuntimeError(
            f"Regime {regime_name} has "
            f"only {T} timestep(s)."
        )

    X_R = R_open[
        :-1
    ]

    Y_R = R_open[
        1:
    ]

    X_S = open_state[
        :-1
    ]

    Y_S = open_state[
        1:
    ]

    # ========================================================
    # 7. ENVIRONMENT CONDITION VECTOR
    #
    # sigma and gamma are kept explicit.
    # ========================================================

    env = np.zeros(
        (
            T - 1,
            NX,
            2
        ),
        dtype=np.float32
    )

    env[
        :,
        :,
        0
    ] = regime[
        "gamma"
    ]

    env[
        :,
        :,
        1
    ] = regime[
        "noise_sigma"
    ]

    # ========================================================
    # 8. ENVIRONMENT-INDUCED RESPONSE
    #
    # For the same underlying stochastic realization:
    #
    #   ΔR_env(t)
    #     =
    #   R_open(t) - R_reference(t)
    #
    # This is attached to each t.
    # ========================================================

    env_response = (
        delta_R_env[
            :-1
        ]
    )

    # ========================================================
    # 9. APPEND
    # ========================================================

    X_information.append(
        X_R.astype(
            np.float32
        )
    )

    Y_information.append(
        Y_R.astype(
            np.float32
        )
    )

    X_state.append(
        X_S.astype(
            np.float32
        )
    )

    Y_state.append(
        Y_S.astype(
            np.float32
        )
    )

    environment_features.append(
        env
    )

    environment_response.append(
        env_response.astype(
            np.float32
        )
    )

    # ========================================================
    # 10. METADATA
    # ========================================================

    for t in range(
        T - 1
    ):

        metadata_rows.append({

            "regime":
                regime_name,

            "timestep":
                t,

            "gamma":
                regime[
                    "gamma"
                ],

            "noise_sigma":
                regime[
                    "noise_sigma"
                ],

            "trajectory_length":
                T,

            "spatial_points":
                NX,

            "reference_gamma":
                0.0,

            "environment_conditioned":
                True,
        })

    print(
        "[+] Transitions generated:",
        T - 1
    )

# ============================================================
# 11. CONCATENATE ALL REGIMES
# ============================================================

X_information = np.concatenate(
    X_information,
    axis=0
)

Y_information = np.concatenate(
    Y_information,
    axis=0
)

X_state = np.concatenate(
    X_state,
    axis=0
)

Y_state = np.concatenate(
    Y_state,
    axis=0
)

environment_features = np.concatenate(
    environment_features,
    axis=0
)

environment_response = np.concatenate(
    environment_response,
    axis=0
)

metadata_df = pd.DataFrame(
    metadata_rows
)

# ============================================================
# 12. SHAPE VERIFICATION
# ============================================================

N_TRANSITIONS = (
    X_information.shape[0]
)

print(
    "\n" + "=" * 100
)

print(
    "DATASET V2 SHAPES"
)

print(
    "=" * 100
)

print(
    "X_information :",
    X_information.shape
)

print(
    "Y_information :",
    Y_information.shape
)

print(
    "X_state       :",
    X_state.shape
)

print(
    "Y_state       :",
    Y_state.shape
)

print(
    "Environment   :",
    environment_features.shape
)

print(
    "ΔR_env        :",
    environment_response.shape
)

print(
    "Metadata      :",
    metadata_df.shape
)

# ============================================================
# 13. FINITE CHECK
# ============================================================

arrays_to_check = {

    "X_information":
        X_information,

    "Y_information":
        Y_information,

    "X_state":
        X_state,

    "Y_state":
        Y_state,

    "environment_features":
        environment_features,

    "environment_response":
        environment_response,
}

print(
    "\n" + "=" * 100
)

print(
    "FINITE CHECK"
)

print(
    "=" * 100
)

for name, arr in arrays_to_check.items():

    finite = bool(
        np.isfinite(
            arr
        ).all()
    )

    print(
        f"{name:25s}:",
        finite
    )

    assert finite

# ============================================================
# 14. INFORMATION-STATE TRACE CHECK
# ============================================================

X_trace = np.trace(
    X_information,
    axis1=-2,
    axis2=-1
)

Y_trace = np.trace(
    Y_information,
    axis1=-2,
    axis2=-1
)

X_trace_error = float(
    np.max(
        np.abs(
            X_trace
            -
            1.0
        )
    )
)

Y_trace_error = float(
    np.max(
        np.abs(
            Y_trace
            -
            1.0
        )
    )
)

print(
    "\n" + "=" * 100
)

print(
    "INFORMATION-STATE NORMALIZATION"
)

print(
    "=" * 100
)

print(
    "X trace max error:",
    f"{X_trace_error:.12e}"
)

print(
    "Y trace max error:",
    f"{Y_trace_error:.12e}"
)

assert X_trace_error < 1e-6
assert Y_trace_error < 1e-6

print(
    "[PASS] X and Y information states are normalized."
)

# ============================================================
# 15. ENVIRONMENT RESPONSE STATISTICS
# ============================================================

env_response_norm = np.linalg.norm(
    environment_response.reshape(
        N_TRANSITIONS,
        NX,
        -1
    ),
    axis=-1
)

mean_env_response = float(
    np.mean(
        env_response_norm
    )
)

median_env_response = float(
    np.median(
        env_response_norm
    )
)

p95_env_response = float(
    np.percentile(
        env_response_norm,
        95
    )
)

max_env_response = float(
    np.max(
        env_response_norm
    )
)

print(
    "\n" + "=" * 100
)

print(
    "ENVIRONMENT-INDUCED INFORMATION RESPONSE"
)

print(
    "=" * 100
)

print(
    "Mean ||ΔR_env||:",
    f"{mean_env_response:.12e}"
)

print(
    "Median:",
    f"{median_env_response:.12e}"
)

print(
    "95th percentile:",
    f"{p95_env_response:.12e}"
)

print(
    "Maximum:",
    f"{max_env_response:.12e}"
)

# ============================================================
# 16. SAVE NUMPY DATASET
# ============================================================

NPZ_PATH = (
    OUT_DIR
    / "environment_conditioned_lno_dataset_v2.npz"
)

np.savez_compressed(

    NPZ_PATH,

    X_information=
        X_information,

    Y_information=
        Y_information,

    X_state=
        X_state,

    Y_state=
        Y_state,

    environment=
        environment_features,

    environment_response=
        environment_response,

)

print(
    "\n[+] Dataset saved:",
    NPZ_PATH
)

# ============================================================
# 17. SAVE METADATA
# ============================================================

METADATA_CSV = (
    OUT_DIR
    / "dataset_metadata.csv"
)

metadata_df.to_csv(
    METADATA_CSV,
    index=False
)

# ============================================================
# 18. SAVE SUMMARY JSON
# ============================================================

SUMMARY = {

    "dataset_name":
        "Environment-Conditioned LNO Dataset v2",

    "transitions":
        int(N_TRANSITIONS),

    "trajectory_timesteps":
        100,

    "spatial_points":
        128,

    "information_matrix_dimension":
        6,

    "input":
        "R_open(t)",

    "target":
        "R_open(t+1)",

    "environment_variables": [
        "gamma",
        "noise_sigma",
    ],

    "environmental_response":
        "R_open(t) - R_reference(t)",

    "regimes":
        REGIMES,

    "information_state_definition":
        "R = z z^T / trace(z z^T)",

    "important_note":
        "R is currently a mathematical internal-information "
        "representation. Its suitability as the physical "
        "state representation for the Lindblad-constrained "
        "kernel will be evaluated in Step 5.",

    "mean_environment_information_response":
        mean_env_response,

    "median_environment_information_response":
        median_env_response,

    "p95_environment_information_response":
        p95_env_response,

    "max_environment_information_response":
        max_env_response,
}

SUMMARY_JSON = (
    OUT_DIR
    / "dataset_v2_summary.json"
)

with open(
    SUMMARY_JSON,
    "w"
) as f:

    json.dump(
        SUMMARY,
        f,
        indent=2
    )

# ============================================================
# 19. MANIFEST
# ============================================================

MANIFEST = {

    "dataset":
        str(
            NPZ_PATH
        ),

    "metadata":
        str(
            METADATA_CSV
        ),

    "summary":
        str(
            SUMMARY_JSON
        ),

    "X_information_shape":
        list(
            X_information.shape
        ),

    "Y_information_shape":
        list(
            Y_information.shape
        ),

    "X_state_shape":
        list(
            X_state.shape
        ),

    "Y_state_shape":
        list(
            Y_state.shape
        ),

    "environment_shape":
        list(
            environment_features.shape
        ),

    "environment_response_shape":
        list(
            environment_response.shape
        ),

    "regime_count":
        len(
            REGIMES
        ),

    "total_transitions":
        int(
            N_TRANSITIONS
        ),
}

MANIFEST_JSON = (
    OUT_DIR
    / "dataset_v2_manifest.json"
)

with open(
    MANIFEST_JSON,
    "w"
) as f:

    json.dump(
        MANIFEST,
        f,
        indent=2
    )

# ============================================================
# 20. FINAL
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "STEP 4 — DATASET V2 COMPLETE"
)

print(
    "=" * 100
)

print(
    "[+] Total transitions:",
    N_TRANSITIONS
)

print(
    "[+] Input:",
    "R_open(t)"
)

print(
    "[+] Target:",
    "R_open(t+1)"
)

print(
    "[+] Environment:",
    "(gamma, noise_sigma)"
)

print(
    "[+] Environment response:",
    "ΔR_env(t)"
)

print(
    "[+] Dataset:",
    NPZ_PATH
)

print(
    "[+] Metadata:",
    METADATA_CSV
)

print(
    "[+] Summary:",
    SUMMARY_JSON
)

print(
    "[+] Manifest:",
    MANIFEST_JSON
)

print("=" * 100)
