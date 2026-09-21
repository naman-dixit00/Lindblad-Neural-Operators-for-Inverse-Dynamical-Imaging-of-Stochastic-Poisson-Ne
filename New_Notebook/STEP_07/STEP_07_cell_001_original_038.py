# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 38
# Step            : STEP_07
# Step Heading    : # STEP 7 — DATASET V3
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 7 — DATASET V3
#          FULL-INFORMATION AMPLITUDE + R REPRESENTATION
#
# Purpose:
#   Replace the temporary prototype amplitude |state| with
#   the TRUE amplitude of the complete internal-information
#   vector:
#
#       z = [state, phi, flux, noise, dissipation, instability]
#
#       R = z z^T / Tr(z z^T)
#       a = ||z||
#
# Current dataset:
#   5 regimes × 99 transitions = 495 transitions
#
# IMPORTANT:
#   This is still the TESTING dataset.
#   We do NOT expand the sample count yet.
#
# NO TRAINING
# NO OLD DATASET OVERWRITE
# NO OLD CHECKPOINT MODIFICATION
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
print("STEP 7 — DATASET V3 / FULL-INFORMATION AMPLITUDE")
print("=" * 100)

# ============================================================
# 2. SOURCE
# ============================================================

SOURCE_DIR = (
    ROOT
    / "results"
    / "step3_paired_trajectories"
)

assert SOURCE_DIR.is_dir(), (
    f"Missing Step 3 directory:\n{SOURCE_DIR}"
)

# ============================================================
# 3. OUTPUT
# ============================================================

OUT_DIR = (
    ROOT
    / "results"
    / "dataset_v3"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
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
# 4. REGIMES
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
# 5. FEATURE ORDER
# ============================================================

FEATURE_NAMES = [
    "state",
    "phi",
    "flux",
    "noise",
    "dissipation",
    "instability",
]

# ============================================================
# 6. CONTAINERS
# ============================================================

X_R_list = []
Y_R_list = []

X_A_list = []
Y_A_list = []

X_Z_list = []
Y_Z_list = []

X_STATE_list = []
Y_STATE_list = []

ENV_list = []

DELTA_R_ENV_list = []
DELTA_A_ENV_list = []

DELTA_STATE_ENV_list = []

metadata_rows = []

# ============================================================
# 7. PROCESS REGIMES
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

    paired_path = (
        SOURCE_DIR
        / f"{regime_name}_paired.npz"
    )

    assert paired_path.is_file(), (
        f"Missing paired trajectory:\n{paired_path}"
    )

    data = np.load(
        paired_path
    )

    # ========================================================
    # 8. LOAD OPEN TRAJECTORY
    # ========================================================

    open_state = data[
        "open_state"
    ].astype(
        np.float64
    )

    open_phi = data[
        "open_phi"
    ].astype(
        np.float64
    )

    open_flux = data[
        "open_flux"
    ].astype(
        np.float64
    )

    open_noise = data[
        "open_noise"
    ].astype(
        np.float64
    )

    open_diss = data[
        "open_dissipation"
    ].astype(
        np.float64
    )

    open_inst = data[
        "open_instability"
    ].astype(
        np.float64
    )

    # ========================================================
    # 9. LOAD REFERENCE TRAJECTORY
    # ========================================================

    ref_state = data[
        "reference_state"
    ].astype(
        np.float64
    )

    ref_phi = data[
        "reference_phi"
    ].astype(
        np.float64
    )

    ref_flux = data[
        "reference_flux"
    ].astype(
        np.float64
    )

    ref_noise = data[
        "reference_noise"
    ].astype(
        np.float64
    )

    ref_diss = data[
        "reference_dissipation"
    ].astype(
        np.float64
    )

    ref_inst = data[
        "reference_instability"
    ].astype(
        np.float64
    )

    # ========================================================
    # 10. SHAPE CHECK
    # ========================================================

    T, NX = open_state.shape

    assert (
        ref_state.shape
        ==
        open_state.shape
    )

    for arr_name, arr in {

        "open_phi":
            open_phi,

        "open_flux":
            open_flux,

        "open_noise":
            open_noise,

        "open_diss":
            open_diss,

        "open_inst":
            open_inst,

        "ref_phi":
            ref_phi,

        "ref_flux":
            ref_flux,

        "ref_noise":
            ref_noise,

        "ref_diss":
            ref_diss,

        "ref_inst":
            ref_inst,

    }.items():

        assert arr.shape == (
            T,
            NX
        ), (
            f"{arr_name} shape mismatch: "
            f"{arr.shape}"
        )

    print(
        "[+] Trajectory:",
        T,
        "×",
        NX
    )

    # ========================================================
    # 11. BUILD COMPLETE OPEN INFORMATION VECTOR
    #
    # z_open:
    #   [T, NX, 6]
    # ========================================================

    z_open = np.stack(
        [
            open_state,
            open_phi,
            open_flux,
            open_noise,
            open_diss,
            open_inst,
        ],
        axis=-1
    )

    # ========================================================
    # 12. BUILD COMPLETE REFERENCE VECTOR
    # ========================================================

    z_ref = np.stack(
        [
            ref_state,
            ref_phi,
            ref_flux,
            ref_noise,
            ref_diss,
            ref_inst,
        ],
        axis=-1
    )

    assert z_open.shape == (
        T,
        NX,
        6
    )

    assert z_ref.shape == (
        T,
        NX,
        6
    )

    # ========================================================
    # 13. JOINT NORMALIZATION
    #
    # Open and reference use the same statistics.
    # ========================================================

    combined_z = np.concatenate(
        [
            z_open,
            z_ref,
        ],
        axis=0
    )

    feature_mean = np.mean(
        combined_z,
        axis=(0, 1)
    )

    feature_std = np.std(
        combined_z,
        axis=(0, 1)
    )

    feature_std = np.where(
        feature_std < 1e-12,
        1.0,
        feature_std
    )

    z_open_n = (
        z_open
        -
        feature_mean[
            None,
            None,
            :
        ]
    ) / feature_std[
        None,
        None,
        :
    ]

    z_ref_n = (
        z_ref
        -
        feature_mean[
            None,
            None,
            :
        ]
    ) / feature_std[
        None,
        None,
        :
    ]

    # ========================================================
    # 14. TRUE FULL-INFORMATION AMPLITUDE
    #
    # This is the important correction:
    #
    #   a(x,t) = ||z(x,t)||
    #
    # using ALL SIX information channels.
    # ========================================================

    A_open = np.linalg.norm(
        z_open_n,
        axis=-1
    )

    A_ref = np.linalg.norm(
        z_ref_n,
        axis=-1
    )

    # ========================================================
    # 15. CONSTRUCT R
    # ========================================================

    def build_R(
        z
    ):

        outer = (
            z[
                ...,
                :,
                None
            ]
            *
            z[
                ...,
                None,
                :
            ]
        )

        trace = np.trace(
            outer,
            axis1=-2,
            axis2=-1
        )

        R = (
            outer
            /
            (
                np.abs(
                    trace
                )
                + 1e-12
            )[
                ...,
                None,
                None
            ]
        )

        return R

    R_open = build_R(
        z_open_n
    )

    R_ref = build_R(
        z_ref_n
    )

    # ========================================================
    # 16. VERIFY R
    # ========================================================

    trace_open = np.trace(
        R_open,
        axis1=-2,
        axis2=-1
    )

    trace_ref = np.trace(
        R_ref,
        axis1=-2,
        axis2=-1
    )

    open_trace_error = float(
        np.max(
            np.abs(
                trace_open
                -
                1.0
            )
        )
    )

    ref_trace_error = float(
        np.max(
            np.abs(
                trace_ref
                -
                1.0
            )
        )
    )

    open_symmetry_error = float(
        np.max(
            np.abs(
                R_open
                -
                np.swapaxes(
                    R_open,
                    -1,
                    -2
                )
            )
        )
    )

    ref_symmetry_error = float(
        np.max(
            np.abs(
                R_ref
                -
                np.swapaxes(
                    R_ref,
                    -1,
                    -2
                )
            )
        )
    )

    assert (
        open_trace_error
        <
        1e-6
    )

    assert (
        ref_trace_error
        <
        1e-6
    )

    assert (
        open_symmetry_error
        <
        1e-6
    )

    assert (
        ref_symmetry_error
        <
        1e-6
    )

    # ========================================================
    # 17. ONE-STEP PAIRS
    #
    # Input = t
    # Target = t+1
    #
    # This is a numerical representation of trajectory
    # evolution, not the scientific definition of the goal.
    # ========================================================

    X_R = R_open[
        :-1
    ]

    Y_R = R_open[
        1:
    ]

    X_A = A_open[
        :-1
    ]

    Y_A = A_open[
        1:
    ]

    X_Z = z_open_n[
        :-1
    ]

    Y_Z = z_open_n[
        1:
    ]

    X_S = open_state[
        :-1
    ]

    Y_S = open_state[
        1:
    ]

    # Environment controls
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
    ] = float(
        regime["gamma"]
    )

    env[
        :,
        :,
        1
    ] = float(
        regime["noise_sigma"]
    )

    # ========================================================
    # 18. ENVIRONMENT RESPONSE
    # ========================================================

    delta_R_env = (
        R_open
        -
        R_ref
    )

    delta_A_env = (
        A_open
        -
        A_ref
    )

    delta_state_env = (
        open_state
        -
        ref_state
    )

    X_delta_R_env = (
        delta_R_env[
            :-1
        ]
    )

    X_delta_A_env = (
        delta_A_env[
            :-1
        ]
    )

    X_delta_state_env = (
        delta_state_env[
            :-1
        ]
    )

    # ========================================================
    # 19. APPEND
    # ========================================================

    X_R_list.append(
        X_R.astype(
            np.float32
        )
    )

    Y_R_list.append(
        Y_R.astype(
            np.float32
        )
    )

    X_A_list.append(
        X_A.astype(
            np.float32
        )
    )

    Y_A_list.append(
        Y_A.astype(
            np.float32
        )
    )

    X_Z_list.append(
        X_Z.astype(
            np.float32
        )
    )

    Y_Z_list.append(
        Y_Z.astype(
            np.float32
        )
    )

    X_STATE_list.append(
        X_S.astype(
            np.float32
        )
    )

    Y_STATE_list.append(
        Y_S.astype(
            np.float32
        )
    )

    ENV_list.append(
        env
    )

    DELTA_R_ENV_list.append(
        X_delta_R_env.astype(
            np.float32
        )
    )

    DELTA_A_ENV_list.append(
        X_delta_A_env.astype(
            np.float32
        )
    )

    DELTA_STATE_ENV_list.append(
        X_delta_state_env.astype(
            np.float32
        )
    )

    # ========================================================
    # 20. METADATA
    # ========================================================

    for t in range(
        T - 1
    ):

        metadata_rows.append({

            "regime":
                regime_name,

            "timestep":
                int(t),

            "gamma":
                float(
                    regime["gamma"]
                ),

            "noise_sigma":
                float(
                    regime["noise_sigma"]
                ),

            "trajectory_length":
                int(T),

            "spatial_points":
                int(NX),

            "information_dimension":
                6,

            "reference_gamma":
                0.0,

            "representation":
                "R + full_information_amplitude",

            "environment_conditioned":
                True,
        })

    print(
        "[+] Transitions:",
        T - 1
    )

# ============================================================
# 21. CONCATENATE
# ============================================================

X_R = np.concatenate(
    X_R_list,
    axis=0
)

Y_R = np.concatenate(
    Y_R_list,
    axis=0
)

X_A = np.concatenate(
    X_A_list,
    axis=0
)

Y_A = np.concatenate(
    Y_A_list,
    axis=0
)

X_Z = np.concatenate(
    X_Z_list,
    axis=0
)

Y_Z = np.concatenate(
    Y_Z_list,
    axis=0
)

X_STATE = np.concatenate(
    X_STATE_list,
    axis=0
)

Y_STATE = np.concatenate(
    Y_STATE_list,
    axis=0
)

ENV = np.concatenate(
    ENV_list,
    axis=0
)

DELTA_R_ENV = np.concatenate(
    DELTA_R_ENV_list,
    axis=0
)

DELTA_A_ENV = np.concatenate(
    DELTA_A_ENV_list,
    axis=0
)

DELTA_STATE_ENV = np.concatenate(
    DELTA_STATE_ENV_list,
    axis=0
)

metadata_df = pd.DataFrame(
    metadata_rows
)

# ============================================================
# 22. DATASET SHAPES
# ============================================================

N_TRANSITIONS = (
    X_R.shape[0]
)

print(
    "\n" + "=" * 100
)

print(
    "DATASET V3 SHAPES"
)

print(
    "=" * 100
)

print(
    "X_R              :",
    X_R.shape
)

print(
    "Y_R              :",
    Y_R.shape
)

print(
    "X_amplitude      :",
    X_A.shape
)

print(
    "Y_amplitude      :",
    Y_A.shape
)

print(
    "X_information_z  :",
    X_Z.shape
)

print(
    "Y_information_z  :",
    Y_Z.shape
)

print(
    "X_state          :",
    X_STATE.shape
)

print(
    "Y_state          :",
    Y_STATE.shape
)

print(
    "Environment      :",
    ENV.shape
)

print(
    "ΔR_env           :",
    DELTA_R_ENV.shape
)

print(
    "Δamplitude_env   :",
    DELTA_A_ENV.shape
)

print(
    "Δstate_env       :",
    DELTA_STATE_ENV.shape
)

print(
    "Metadata          :",
    metadata_df.shape
)

# ============================================================
# 23. FINITE CHECK
# ============================================================

all_arrays = {

    "X_R":
        X_R,

    "Y_R":
        Y_R,

    "X_amplitude":
        X_A,

    "Y_amplitude":
        Y_A,

    "X_information_z":
        X_Z,

    "Y_information_z":
        Y_Z,

    "X_state":
        X_STATE,

    "Y_state":
        Y_STATE,

    "Environment":
        ENV,

    "delta_R_env":
        DELTA_R_ENV,

    "delta_amplitude_env":
        DELTA_A_ENV,

    "delta_state_env":
        DELTA_STATE_ENV,
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

for name, arr in all_arrays.items():

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
# 24. NORMALIZATION CHECK
# ============================================================

X_trace = np.trace(
    X_R,
    axis1=-2,
    axis2=-1
)

Y_trace = np.trace(
    Y_R,
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
    "R NORMALIZATION"
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

# ============================================================
# 25. FULL-INFORMATION AMPLITUDE STATISTICS
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "FULL-INFORMATION AMPLITUDE"
)

print(
    "=" * 100
)

print(
    "X amplitude min:",
    f"{X_A.min():.12e}"
)

print(
    "X amplitude max:",
    f"{X_A.max():.12e}"
)

print(
    "X amplitude mean:",
    f"{X_A.mean():.12e}"
)

print(
    "Y amplitude mean:",
    f"{Y_A.mean():.12e}"
)

# ============================================================
# 26. ENVIRONMENT RESPONSE STATISTICS
# ============================================================

delta_R_norm = np.linalg.norm(
    DELTA_R_ENV.reshape(
        N_TRANSITIONS,
        NX,
        -1
    ),
    axis=-1
)

delta_A_abs = np.abs(
    DELTA_A_ENV
)

print(
    "\n" + "=" * 100
)

print(
    "ENVIRONMENT-INDUCED RESPONSE"
)

print(
    "=" * 100
)

print(
    "Mean ||ΔR_env||:",
    f"{delta_R_norm.mean():.12e}"
)

print(
    "95th percentile ||ΔR_env||:",
    f"{np.percentile(delta_R_norm,95):.12e}"
)

print(
    "Maximum ||ΔR_env||:",
    f"{delta_R_norm.max():.12e}"
)

print(
    "Mean |Δamplitude_env|:",
    f"{delta_A_abs.mean():.12e}"
)

print(
    "95th percentile |Δamplitude_env|:",
    f"{np.percentile(delta_A_abs,95):.12e}"
)

print(
    "Maximum |Δamplitude_env|:",
    f"{delta_A_abs.max():.12e}"
)

# ============================================================
# 27. REGIME-WISE SUMMARY
# ============================================================

regime_summary = []

for regime_name, regime in REGIMES.items():

    mask = (
        metadata_df[
            "regime"
        ].to_numpy()
        ==
        regime_name
    )

    regime_R = delta_R_norm[
        mask
    ]

    regime_A = delta_A_abs[
        mask
    ]

    regime_summary.append({

        "regime":
            regime_name,

        "gamma":
            float(
                regime["gamma"]
            ),

        "noise_sigma":
            float(
                regime["noise_sigma"]
            ),

        "transitions":
            int(
                np.sum(mask)
            ),

        "mean_delta_R":
            float(
                regime_R.mean()
            ),

        "p95_delta_R":
            float(
                np.percentile(
                    regime_R,
                    95
                )
            ),

        "max_delta_R":
            float(
                regime_R.max()
            ),

        "mean_abs_delta_amplitude":
            float(
                regime_A.mean()
            ),

        "max_abs_delta_amplitude":
            float(
                regime_A.max()
            ),
    })

regime_summary_df = pd.DataFrame(
    regime_summary
)

print(
    "\n" + "=" * 100
)

print(
    "REGIME-WISE ENVIRONMENT RESPONSE"
)

print(
    "=" * 100
)

display(
    regime_summary_df.round(8)
)

# ============================================================
# 28. SAVE MAIN DATASET
# ============================================================

DATASET_PATH = (
    OUT_DIR
    / "environment_conditioned_lno_dataset_v3.npz"
)

np.savez_compressed(

    DATASET_PATH,

    X_R=
        X_R,

    Y_R=
        Y_R,

    X_amplitude=
        X_A,

    Y_amplitude=
        Y_A,

    X_information_z=
        X_Z,

    Y_information_z=
        Y_Z,

    X_state=
        X_STATE,

    Y_state=
        Y_STATE,

    environment=
        ENV,

    delta_R_environment=
        DELTA_R_ENV,

    delta_amplitude_environment=
        DELTA_A_ENV,

    delta_state_environment=
        DELTA_STATE_ENV,
)

print(
    "\n[+] Dataset saved:",
    DATASET_PATH
)

# ============================================================
# 29. SAVE METADATA
# ============================================================

METADATA_PATH = (
    OUT_DIR
    / "dataset_v3_metadata.csv"
)

metadata_df.to_csv(
    METADATA_PATH,
    index=False
)

# ============================================================
# 30. SAVE REGIME SUMMARY
# ============================================================

REGIME_SUMMARY_PATH = (
    OUT_DIR
    / "regime_environment_summary.csv"
)

regime_summary_df.to_csv(
    REGIME_SUMMARY_PATH,
    index=False
)

# ============================================================
# 31. SAVE FEATURE NORMALIZATION
# ============================================================

NORMALIZATION = {

    name: {

        "mean":
            float(
                feature_mean[i]
            ),

        "std":
            float(
                feature_std[i]
            ),
    }

    for i, name in enumerate(
        FEATURE_NAMES
    )
}

with open(
    OUT_DIR
    / "feature_normalization.json",
    "w"
) as f:

    json.dump(
        NORMALIZATION,
        f,
        indent=2
    )

# ============================================================
# 32. DATASET SUMMARY
# ============================================================

summary = {

    "dataset_name":
        "Environment-Conditioned LNO Dataset V3",

    "testing_dataset":
        True,

    "final_training_dataset":
        False,

    "total_transitions":
        int(
            N_TRANSITIONS
        ),

    "timesteps_per_trajectory":
        100,

    "spatial_points":
        int(
            NX
        ),

    "information_dimension":
        6,

    "input_representation":
        "(R(t), amplitude(t), gamma, noise_sigma)",

    "target_representation":
        "(R(t+1), amplitude(t+1))",

    "R_definition":
        "R = z z^T / Tr(z z^T)",

    "amplitude_definition":
        "a = ||z||",

    "feature_names":
        FEATURE_NAMES,

    "environment_variables":
        [
            "gamma",
            "noise_sigma",
        ],

    "environment_response":
        "R_open - R_reference",

    "amplitude_environment_response":
        "a_open - a_reference",

    "mean_delta_R":
        float(
            delta_R_norm.mean()
        ),

    "p95_delta_R":
        float(
            np.percentile(
                delta_R_norm,
                95
            )
        ),

    "max_delta_R":
        float(
            delta_R_norm.max()
        ),

    "mean_abs_delta_amplitude":
        float(
            delta_A_abs.mean()
        ),

    "max_abs_delta_amplitude":
        float(
            delta_A_abs.max()
        ),

    "source":
        str(
            SOURCE_DIR
        ),

    "dataset_path":
        str(
            DATASET_PATH
        ),
}

SUMMARY_PATH = (
    OUT_DIR
    / "dataset_v3_summary.json"
)

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
# 33. FINAL
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "STEP 7 — DATASET V3 COMPLETE"
)

print(
    "=" * 100
)

print(
    "[+] Transitions:",
    N_TRANSITIONS
)

print(
    "[+] Full information amplitude explicitly stored: YES"
)

print(
    "[+] R stored: YES"
)

print(
    "[+] Environment (gamma, noise_sigma) stored: YES"
)

print(
    "[+] Environment-induced ΔR stored: YES"
)

print(
    "[+] Environment-induced Δamplitude stored: YES"
)

print(
    "[+] Original Dataset V2 untouched."
)

print(
    "[+] Current dataset remains TESTING-SCALE."
)

print(
    "[+] Final large-scale dataset expansion is still pending."
)

print(
    "\nDataset:",
    DATASET_PATH
)

print(
    "Metadata:",
    METADATA_PATH
)

print(
    "Summary:",
    SUMMARY_PATH
)

print("=" * 100)
