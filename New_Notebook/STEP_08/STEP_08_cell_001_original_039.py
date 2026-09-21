# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 39
# Step            : STEP_08
# Step Heading    : # STEP 8 — DATASET V3 TRAINING-READINESS / TARGET AUDIT
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 8 — DATASET V3 TRAINING-READINESS / TARGET AUDIT
#
# PURPOSE:
#   Verify that Dataset V3 is mathematically and statistically
#   suitable for controlled LNO/FNO training.
#
# THIS CELL DOES NOT TRAIN ANY MODEL.
#
# CHECKS:
#   1. Tensor alignment
#   2. Input/target consistency
#   3. Target non-triviality
#   4. Environment diversity
#   5. Collapse-regime representation
#   6. R structural validity
#   7. Amplitude validity
#   8. X/Y leakage checks
#   9. Environment-response signal
#  10. Normalization readiness
#
# NO DATASET OVERWRITE
# NO CHECKPOINT MODIFICATION
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
print("STEP 8 — DATASET V3 TRAINING-READINESS / TARGET AUDIT")
print("=" * 100)

# ============================================================
# 2. LOAD DATASET V3
# ============================================================

DATASET_PATH = (
    ROOT
    / "results"
    / "dataset_v3"
    / "environment_conditioned_lno_dataset_v3.npz"
)

METADATA_PATH = (
    ROOT
    / "results"
    / "dataset_v3"
    / "dataset_v3_metadata.csv"
)

assert DATASET_PATH.is_file(), (
    f"Missing Dataset V3:\n{DATASET_PATH}"
)

assert METADATA_PATH.is_file(), (
    f"Missing metadata:\n{METADATA_PATH}"
)

data = np.load(
    DATASET_PATH
)

metadata = pd.read_csv(
    METADATA_PATH
)

print(
    "[+] Dataset:",
    DATASET_PATH
)

print(
    "[+] Metadata:",
    METADATA_PATH
)

# ============================================================
# 3. REQUIRED ARRAYS
# ============================================================

X_R = data[
    "X_R"
].astype(
    np.float64
)

Y_R = data[
    "Y_R"
].astype(
    np.float64
)

X_A = data[
    "X_amplitude"
].astype(
    np.float64
)

Y_A = data[
    "Y_amplitude"
].astype(
    np.float64
)

X_Z = data[
    "X_information_z"
].astype(
    np.float64
)

Y_Z = data[
    "Y_information_z"
].astype(
    np.float64
)

X_STATE = data[
    "X_state"
].astype(
    np.float64
)

Y_STATE = data[
    "Y_state"
].astype(
    np.float64
)

ENV = data[
    "environment"
].astype(
    np.float64
)

DELTA_R_ENV = data[
    "delta_R_environment"
].astype(
    np.float64
)

DELTA_A_ENV = data[
    "delta_amplitude_environment"
].astype(
    np.float64
)

DELTA_STATE_ENV = data[
    "delta_state_environment"
].astype(
    np.float64
)

# ============================================================
# 4. GLOBAL SHAPES
# ============================================================

N, NX, D, D2 = X_R.shape

print(
    "\n" + "=" * 100
)

print(
    "A. SHAPE AUDIT"
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

assert (
    X_R.shape
    ==
    Y_R.shape
)

assert (
    X_A.shape
    ==
    Y_A.shape
)

assert (
    X_Z.shape
    ==
    Y_Z.shape
)

assert (
    X_STATE.shape
    ==
    Y_STATE.shape
)

assert (
    ENV.shape
    ==
    (
        N,
        NX,
        2
    )
)

assert (
    DELTA_R_ENV.shape
    ==
    (
        N,
        NX,
        6,
        6
    )
)

assert (
    DELTA_A_ENV.shape
    ==
    (
        N,
        NX
    )
)

assert (
    DELTA_STATE_ENV.shape
    ==
    (
        N,
        NX
    )
)

assert D == 6
assert D2 == 6

print(
    "[PASS] All tensor shapes aligned."
)

# ============================================================
# 5. FINITE AUDIT
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "B. FINITE-VALUE AUDIT"
)

print(
    "=" * 100
)

arrays = {

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

    "delta_R_environment":
        DELTA_R_ENV,

    "delta_amplitude_environment":
        DELTA_A_ENV,

    "delta_state_environment":
        DELTA_STATE_ENV,
}

finite_results = {}

for name, arr in arrays.items():

    finite = bool(
        np.isfinite(
            arr
        ).all()
    )

    finite_results[
        name
    ] = finite

    print(
        f"{name:30s}:",
        finite
    )

    assert finite

print(
    "[PASS] Every V3 tensor is finite."
)

# ============================================================
# 6. METADATA ALIGNMENT
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "C. METADATA ALIGNMENT"
)

print(
    "=" * 100
)

print(
    "Metadata rows:",
    len(metadata)
)

print(
    "Dataset transitions:",
    N
)

assert len(metadata) == N

required_metadata_columns = [
    "regime",
    "timestep",
    "gamma",
    "noise_sigma",
    "trajectory_length",
    "spatial_points",
    "information_dimension",
]

for column in required_metadata_columns:

    assert column in metadata.columns, (
        f"Missing metadata column: {column}"
    )

print(
    "[PASS] Metadata matches dataset transitions."
)

# ============================================================
# 7. ENVIRONMENT DIVERSITY
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "D. ENVIRONMENT DIVERSITY"
)

print(
    "=" * 100
)

gamma_values = np.unique(
    ENV[
        ...,
        0
    ]
)

sigma_values = np.unique(
    ENV[
        ...,
        1
    ]
)

print(
    "Unique gamma values:",
    gamma_values
)

print(
    "Unique noise_sigma values:",
    sigma_values
)

assert len(
    gamma_values
) >= 2

assert len(
    sigma_values
) >= 2

print(
    "[PASS] Environment variables have variation."
)

# ============================================================
# 8. REGIME COUNTS
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "E. REGIME DISTRIBUTION"
)

print(
    "=" * 100
)

regime_counts = (
    metadata[
        "regime"
    ]
    .value_counts()
    .sort_index()
)

display(
    regime_counts.to_frame(
        "transitions"
    )
)

expected_regimes = {
    "low_noise",
    "stochastic",
    "heavy_dissipation",
    "collapse",
    "metastable",
}

assert (
    set(
        regime_counts.index
    )
    ==
    expected_regimes
)

print(
    "[PASS] All five regimes represented."
)

# ============================================================
# 9. COLLAPSE REGIME AUDIT
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "F. COLLAPSE-REGIME AUDIT"
)

print(
    "=" * 100
)

collapse_mask = (
    metadata[
        "regime"
    ].to_numpy()
    ==
    "collapse"
)

collapse_count = int(
    collapse_mask.sum()
)

assert collapse_count > 0

collapse_R = (
    X_R[
        collapse_mask
    ]
)

collapse_A = (
    X_A[
        collapse_mask
    ]
)

collapse_delta_R = (
    np.linalg.norm(
        DELTA_R_ENV[
            collapse_mask
        ].reshape(
            collapse_count,
            NX,
            -1
        ),
        axis=-1
    )
)

collapse_delta_A = np.abs(
    DELTA_A_ENV[
        collapse_mask
    ]
)

print(
    "Collapse transitions:",
    collapse_count
)

print(
    "Collapse mean amplitude:",
    f"{collapse_A.mean():.12e}"
)

print(
    "Collapse mean ||ΔR_env||:",
    f"{collapse_delta_R.mean():.12e}"
)

print(
    "Collapse mean |Δa_env|:",
    f"{collapse_delta_A.mean():.12e}"
)

assert (
    collapse_delta_R.mean()
    >
    0.0
)

assert (
    collapse_delta_A.mean()
    >
    0.0
)

print(
    "[PASS] Collapse regime contains measurable environment response."
)

# ============================================================
# 10. R TRACE CHECK
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "G. R NORMALIZATION / STRUCTURE AUDIT"
)

print(
    "=" * 100
)

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
            X_trace - 1.0
        )
    )
)

Y_trace_error = float(
    np.max(
        np.abs(
            Y_trace - 1.0
        )
    )
)

print(
    "X trace error:",
    f"{X_trace_error:.12e}"
)

print(
    "Y trace error:",
    f"{Y_trace_error:.12e}"
)

assert X_trace_error < 1e-6
assert Y_trace_error < 1e-6

# ============================================================
# 11. R SYMMETRY
# ============================================================

X_symmetry_error = float(
    np.max(
        np.abs(
            X_R
            -
            np.swapaxes(
                X_R,
                -1,
                -2
            )
        )
    )
)

Y_symmetry_error = float(
    np.max(
        np.abs(
            Y_R
            -
            np.swapaxes(
                Y_R,
                -1,
                -2
            )
        )
    )
)

print(
    "X symmetry error:",
    f"{X_symmetry_error:.12e}"
)

print(
    "Y symmetry error:",
    f"{Y_symmetry_error:.12e}"
)

assert X_symmetry_error < 1e-6
assert Y_symmetry_error < 1e-6

# ============================================================
# 12. PSD CHECK
# ============================================================

subset = X_R[
    :min(
        N,
        64
    ),
    ::max(
        1,
        NX // 16
    )
]

eigvals = np.linalg.eigvalsh(
    subset
)

minimum_eigenvalue = float(
    eigvals.min()
)

print(
    "Minimum X_R eigenvalue:",
    f"{minimum_eigenvalue:.12e}"
)

assert minimum_eigenvalue >= -1e-6

print(
    "[PASS] R normalization / symmetry / PSD checks."
)

# ============================================================
# 13. TARGET NON-TRIVIALITY
#
# Check whether Y is simply identical to X.
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "H. TARGET NON-TRIVIALITY"
)

print(
    "=" * 100
)

R_transition = (
    Y_R
    -
    X_R
)

A_transition = (
    Y_A
    -
    X_A
)

R_transition_norm = np.linalg.norm(
    R_transition.reshape(
        N,
        NX,
        -1
    ),
    axis=-1
)

A_transition_abs = np.abs(
    A_transition
)

mean_R_transition = float(
    R_transition_norm.mean()
)

median_R_transition = float(
    np.median(
        R_transition_norm
    )
)

mean_A_transition = float(
    A_transition_abs.mean()
)

print(
    "Mean ||Y_R - X_R||:",
    f"{mean_R_transition:.12e}"
)

print(
    "Median ||Y_R - X_R||:",
    f"{median_R_transition:.12e}"
)

print(
    "Mean |Y_a - X_a|:",
    f"{mean_A_transition:.12e}"
)

assert mean_R_transition > 0.0
assert mean_A_transition > 0.0

print(
    "[PASS] Target evolution is non-trivial."
)

# ============================================================
# 14. TRIVIAL COPY BASELINE
#
# If X -> Y is nearly identity, the learning task would be
# trivial. Measure that explicitly.
# ============================================================

copy_R_mse = float(
    np.mean(
        (
            X_R
            -
            Y_R
        ) ** 2
    )
)

copy_A_mse = float(
    np.mean(
        (
            X_A
            -
            Y_A
        ) ** 2
    )
)

print(
    "\nTrivial X→Y copy baseline:"
)

print(
    "R copy MSE:",
    f"{copy_R_mse:.12e}"
)

print(
    "Amplitude copy MSE:",
    f"{copy_A_mse:.12e}"
)

# ============================================================
# 15. ENVIRONMENT RESPONSE
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "I. ENVIRONMENT RESPONSE SIGNAL"
)

print(
    "=" * 100
)

delta_R_norm = np.linalg.norm(
    DELTA_R_ENV.reshape(
        N,
        NX,
        -1
    ),
    axis=-1
)

delta_A_abs = np.abs(
    DELTA_A_ENV
)

env_response_mean_R = float(
    delta_R_norm.mean()
)

env_response_mean_A = float(
    delta_A_abs.mean()
)

print(
    "Mean ||ΔR_env||:",
    f"{env_response_mean_R:.12e}"
)

print(
    "Mean |Δa_env|:",
    f"{env_response_mean_A:.12e}"
)

assert (
    env_response_mean_R
    >
    0.0
)

assert (
    env_response_mean_A
    >
    0.0
)

# ============================================================
# 16. REGIME-WISE TARGET CHANGE
# ============================================================

regime_target_rows = []

for regime_name in sorted(
    expected_regimes
):

    mask = (
        metadata[
            "regime"
        ].to_numpy()
        ==
        regime_name
    )

    r_change = float(
        R_transition_norm[
            mask
        ].mean()
    )

    a_change = float(
        A_transition_abs[
            mask
        ].mean()
    )

    env_r = float(
        delta_R_norm[
            mask
        ].mean()
    )

    env_a = float(
        delta_A_abs[
            mask
        ].mean()
    )

    regime_target_rows.append({

        "regime":
            regime_name,

        "gamma":
            float(
                metadata.loc[
                    mask,
                    "gamma"
                ].iloc[0]
            ),

        "noise_sigma":
            float(
                metadata.loc[
                    mask,
                    "noise_sigma"
                ].iloc[0]
            ),

        "transitions":
            int(
                mask.sum()
            ),

        "mean_R_target_change":
            r_change,

        "mean_amplitude_target_change":
            a_change,

        "mean_environment_R_response":
            env_r,

        "mean_environment_amplitude_response":
            env_a,
    })

regime_target_df = pd.DataFrame(
    regime_target_rows
)

print(
    "\nRegime-wise target dynamics:"
)

display(
    regime_target_df.round(8)
)

# ============================================================
# 17. TRAINING INPUT/OUTPUT LEAKAGE CHECK
#
# Environment response is derived from paired reference/open
# trajectories and should NOT be used as an accidental input
# unless explicitly included in a future training objective.
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "J. TARGET / INPUT LEAKAGE AUDIT"
)

print(
    "=" * 100
)

# Check that the actual target Y_R is not identical to
# delta_R_env or environment metadata.

delta_vs_target = np.mean(
    np.abs(
        Y_R
        -
        DELTA_R_ENV
    )
)

print(
    "Mean |Y_R - ΔR_env|:",
    f"{delta_vs_target:.12e}"
)

# We don't require it to be large as a mathematical criterion,
# but it confirms they are different objects.

# Check environment variables contain only controls.
env_gamma = ENV[
    ...,
    0
]

env_sigma = ENV[
    ...,
    1
]

print(
    "Gamma finite:",
    bool(
        np.isfinite(
            env_gamma
        ).all()
    )
)

print(
    "Sigma finite:",
    bool(
        np.isfinite(
            env_sigma
        ).all()
    )
)

# ============================================================
# 18. NORMALIZATION READINESS
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "K. NORMALIZATION READINESS"
)

print(
    "=" * 100
)

normalization_stats = {

    "R": {

        "mean":
            float(
                X_R.mean()
            ),

        "std":
            float(
                X_R.std()
            ),

        "min":
            float(
                X_R.min()
            ),

        "max":
            float(
                X_R.max()
            ),
    },

    "amplitude": {

        "mean":
            float(
                X_A.mean()
            ),

        "std":
            float(
                X_A.std()
            ),

        "min":
            float(
                X_A.min()
            ),

        "max":
            float(
                X_A.max()
            ),
    },

    "gamma": {

        "mean":
            float(
                env_gamma.mean()
            ),

        "std":
            float(
                env_gamma.std()
            ),

        "min":
            float(
                env_gamma.min()
            ),

        "max":
            float(
                env_gamma.max()
            ),
    },

    "noise_sigma": {

        "mean":
            float(
                env_sigma.mean()
            ),

        "std":
            float(
                env_sigma.std()
            ),

        "min":
            float(
                env_sigma.min()
            ),

        "max":
            float(
                env_sigma.max()
            ),
    },
}

print(
    json.dumps(
        normalization_stats,
        indent=2
    )
)

# ============================================================
# 19. FINAL READINESS SCORE
# ============================================================

shape_pass = True

finite_pass = all(
    finite_results.values()
)

metadata_pass = (
    len(metadata) == N
)

environment_diversity_pass = (
    len(gamma_values) >= 2
    and
    len(sigma_values) >= 2
)

collapse_pass = (
    collapse_count > 0
    and
    collapse_delta_R.mean() > 0.0
    and
    collapse_delta_A.mean() > 0.0
)

structure_pass = (
    X_trace_error < 1e-6
    and
    Y_trace_error < 1e-6
    and
    X_symmetry_error < 1e-6
    and
    Y_symmetry_error < 1e-6
    and
    minimum_eigenvalue >= -1e-6
)

target_pass = (
    mean_R_transition > 0.0
    and
    mean_A_transition > 0.0
)

environment_response_pass = (
    env_response_mean_R > 0.0
    and
    env_response_mean_A > 0.0
)

print(
    "\n" + "=" * 100
)

print(
    "STEP 8 — FINAL TRAINING READINESS"
)

print(
    "=" * 100
)

print(
    "Shapes:",
    shape_pass
)

print(
    "Finite:",
    finite_pass
)

print(
    "Metadata:",
    metadata_pass
)

print(
    "Environment diversity:",
    environment_diversity_pass
)

print(
    "Collapse regime:",
    collapse_pass
)

print(
    "R structure:",
    structure_pass
)

print(
    "Non-trivial target:",
    target_pass
)

print(
    "Environment response:",
    environment_response_pass
)

ready = all([
    shape_pass,
    finite_pass,
    metadata_pass,
    environment_diversity_pass,
    collapse_pass,
    structure_pass,
    target_pass,
    environment_response_pass,
])

print(
    "\nREADY FOR CONTROLLED TRAINING:",
    ready
)

# ============================================================
# 20. SAVE AUDIT ARTIFACTS
# ============================================================

OUT_DIR = (
    ROOT
    / "results"
    / "dataset_v3"
    / "training_readiness_audit"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

regime_target_df.to_csv(
    OUT_DIR
    / "regime_target_dynamics.csv",
    index=False
)

audit_summary = {

    "dataset":
        str(
            DATASET_PATH
        ),

    "transitions":
        int(N),

    "spatial_points":
        int(NX),

    "information_dimension":
        int(D),

    "shapes_pass":
        bool(shape_pass),

    "finite_pass":
        bool(finite_pass),

    "metadata_pass":
        bool(metadata_pass),

    "environment_diversity_pass":
        bool(environment_diversity_pass),

    "collapse_regime_pass":
        bool(collapse_pass),

    "structure_pass":
        bool(structure_pass),

    "target_nontrivial_pass":
        bool(target_pass),

    "environment_response_pass":
        bool(environment_response_pass),

    "ready_for_controlled_training":
        bool(ready),

    "mean_R_target_change":
        float(mean_R_transition),

    "mean_amplitude_target_change":
        float(mean_A_transition),

    "mean_environment_R_response":
        float(env_response_mean_R),

    "mean_environment_amplitude_response":
        float(env_response_mean_A),

    "R_copy_baseline_MSE":
        float(copy_R_mse),

    "amplitude_copy_baseline_MSE":
        float(copy_A_mse),

    "X_R_trace_error":
        float(X_trace_error),

    "Y_R_trace_error":
        float(Y_trace_error),

    "X_R_symmetry_error":
        float(X_symmetry_error),

    "Y_R_symmetry_error":
        float(Y_symmetry_error),

    "minimum_eigenvalue":
        float(minimum_eigenvalue),

    "normalization_stats":
        normalization_stats,
}

SUMMARY_PATH = (
    OUT_DIR
    / "training_readiness_summary.json"
)

with open(
    SUMMARY_PATH,
    "w"
) as f:

    json.dump(
        audit_summary,
        f,
        indent=2,
        allow_nan=False
    )

# ============================================================
# 21. FINAL
# ============================================================

print(
    "\n[+] Audit saved:",
    OUT_DIR
)

if ready:

    print(
        "\n[PASS] Dataset V3 is ready for controlled prototype training."
    )

    print(
        "[NEXT] Build train/validation/test splits and train "
        "FNO and LNO on the SAME task."
    )

else:

    print(
        "\n[REVIEW] Dataset V3 requires correction before training."
    )

print("=" * 100)
