# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 26
# Step            : STEP_01
# Step Heading    : # STEP 1 — INTERNAL INFORMATION STATE DEFINITION
# Step Cell No.   : 14
# ============================================================

# ============================================================
# STEP 1 — INTERNAL INFORMATION STATE DEFINITION
#
# Scientific definition:
#
#   Internal information state =
#   multivariate dynamical information carried by the
#   nanoscale ion-transport system and modified by the
#   external environment/noise/dissipation.
#
# We do NOT equate concentration alone with the full
# information state.
#
# We construct a matrix-valued Information State R(x,t)
# from the physically observed transport variables.
#
# NO TRAINING
# NO DATASET MODIFICATION
# ============================================================

import os
import json
import numpy as np
import pandas as pd

ROOT = (
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

TEST_DIR = os.path.join(
    ROOT,
    "dataset ",
    "test"
)

OUT_DIR = os.path.join(
    ROOT,
    "results",
    "physical_state_definition_v2"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

print("=" * 100)
print("STEP 1 — INTERNAL INFORMATION STATE DEFINITION")
print("=" * 100)

# ============================================================
# 1. LOAD PHYSICAL VARIABLES
# ============================================================

required_files = {

    "state":
        "state_t.npy",

    "phi":
        "phi.npy",

    "flux":
        "flux.npy",

    "noise":
        "noise.npy",

    "dissipation":
        "dissipation.npy",

    "gamma":
        "gamma.npy",

    "instability":
        "instability.npy",

    "entropy":
        "entropy.npy",

    "state_next":
        "state_t1.npy",
}

data = {}

for name, filename in required_files.items():

    path = os.path.join(
        TEST_DIR,
        filename
    )

    assert os.path.isfile(
        path
    ), (
        "Missing required file:\n"
        + path
    )

    data[name] = np.load(
        path
    ).astype(
        np.float64
    )

    print(
        f"[+] {name:15s}: "
        f"shape={data[name].shape}"
    )

# ============================================================
# 2. VERIFY SAMPLE / SPATIAL CONSISTENCY
# ============================================================

N = data["state"].shape[0]
NX = data["state"].shape[1]

assert N == 3000
assert NX == 128

for name in [
    "phi",
    "flux",
    "noise",
    "dissipation",
    "gamma",
    "instability",
]:
    assert data[name].shape == (
        N,
        NX
    ), (
        f"{name} shape mismatch: "
        f"{data[name].shape}"
    )

print(
    "\n[PASS] 3000 samples × 128 spatial points"
)

# ============================================================
# 3. DEFINE INTERNAL INFORMATION VECTOR
#
# Each spatial point contains a multivariate description:
#
#   z(x,t) =
#       [
#         transport state,
#         electrostatic / potential field,
#         flux,
#         stochastic forcing,
#         dissipation,
#         instability
#       ]
#
# gamma is kept as the EXTERNAL ENVIRONMENTAL CONTROL
# rather than embedding it directly inside z.
# ============================================================

feature_names = [
    "state",
    "phi",
    "flux",
    "noise",
    "dissipation",
    "instability",
]

features = np.stack(
    [
        data[name]
        for name in feature_names
    ],
    axis=1
)

# Shape:
#   [N, 6, NX]

print(
    "\n[+] Raw internal-information tensor:",
    features.shape
)

# ============================================================
# 4. ROBUST FEATURE NORMALIZATION
#
# We normalize each observable independently so that the
# information representation is not dominated by the raw
# numerical scale of dissipation/flux.
# ============================================================

normalized = np.zeros_like(
    features
)

normalization_stats = {}

for channel_idx, name in enumerate(
    feature_names
):

    x = features[
        :,
        channel_idx,
        :
    ]

    mean = np.mean(x)

    std = np.std(x)

    if std < 1e-12:

        std = 1.0

    normalized[
        :,
        channel_idx,
        :
    ] = (
        x - mean
    ) / std

    normalization_stats[
        name
    ] = {

        "mean":
            float(mean),

        "std":
            float(std),
    }

print(
    "[+] Feature-wise normalization complete."
)

# ============================================================
# 5. INFORMATION-STATE MATRIX
#
# For every sample/spatial point:
#
#   z ∈ R^6
#
# Define:
#
#   R = z z^T
#
# and normalize:
#
#   R <- R / trace(R)
#
# Therefore:
#
#   R >= 0
#   trace(R) = 1
#
# This gives a normalized matrix-valued representation
# suitable for a Lindblad-style matrix generator.
#
# This is an INFORMATION-STATE representation.
# It is NOT being identified as a microscopic density matrix.
# ============================================================

# z:
# [N, 6, NX]

z = normalized

# Move spatial coordinate before feature dimension:
#
# [N, NX, 6]

z_spatial = np.transpose(
    z,
    (0, 2, 1)
)

# Outer product:
#
# [N, NX, 6, 6]

information_matrix = (
    z_spatial[
        ...,
        :, None
    ]
    *
    z_spatial[
        ...,
        None, :
    ]
)

# Trace before normalization
trace = np.trace(
    information_matrix,
    axis1=-2,
    axis2=-1
)

# Positive normalization denominator
denominator = (
    np.abs(trace)
    + 1e-12
)

information_matrix = (
    information_matrix
    /
    denominator[
        ...,
        None,
        None
    ]
)

print(
    "\n[+] Information-state matrix shape:",
    information_matrix.shape
)

# ============================================================
# 6. MATRIX SANITY CHECK
# ============================================================

matrix_trace = np.trace(
    information_matrix,
    axis1=-2,
    axis2=-1
)

# Symmetry check
transpose_error = np.max(
    np.abs(
        information_matrix
        -
        np.swapaxes(
            information_matrix,
            -1,
            -2
        )
    )
)

# Eigenvalue check on a representative subset
subset = information_matrix[
    :100,
    ::8,
    :,
    :
]

eigenvalues = np.linalg.eigvalsh(
    subset
)

min_eigenvalue = (
    np.min(
        eigenvalues
    )
)

max_trace_error = np.max(
    np.abs(
        matrix_trace
        - 1.0
    )
)

print(
    "\n" + "=" * 100
)

print(
    "INFORMATION-STATE MATRIX SANITY"
)

print(
    "=" * 100
)

print(
    "Max trace error:",
    f"{max_trace_error:.12e}"
)

print(
    "Max symmetry error:",
    f"{transpose_error:.12e}"
)

print(
    "Minimum eigenvalue:",
    f"{min_eigenvalue:.12e}"
)

if (
    max_trace_error < 1e-6
    and
    transpose_error < 1e-6
    and
    min_eigenvalue >= -1e-8
):

    print(
        "[PASS] Information-state matrix is normalized, "
        "symmetric and positive-semidefinite up to "
        "floating-point tolerance."
    )

else:

    print(
        "[WARNING] Matrix sanity condition needs inspection."
    )

# ============================================================
# 7. ENVIRONMENTAL COUPLING EFFECT
#
# gamma is explicitly treated as external coupling.
#
# We measure whether the internal information representation
# varies systematically across the available gamma regimes.
# ============================================================

gamma_values = np.unique(
    np.round(
        data["gamma"].reshape(-1),
        6
    )
)

print(
    "\n" + "=" * 100
)

print(
    "ENVIRONMENTAL COUPLING RESPONSE"
)

print(
    "=" * 100
)

environment_rows = []

for gamma_value in gamma_values:

    mask = (
        np.isclose(
            data["gamma"][:, 0],
            gamma_value,
            atol=1e-6
        )
    )

    if not np.any(mask):

        continue

    R = information_matrix[
        mask
    ]

    # Mean deviation from identity/uniform baseline
    mean_R = np.mean(
        R,
        axis=(0, 1)
    )

    # Average off-diagonal magnitude
    offdiag = (
        mean_R
        - np.diag(
            np.diag(
                mean_R
            )
        )
    )

    offdiag_strength = np.mean(
        np.abs(
            offdiag
        )
    )

    # Information-state variability
    state_variability = np.mean(
        np.std(
            R.reshape(
                R.shape[0],
                R.shape[1],
                -1
            ),
            axis=0
        )
    )

    # Mean instability
    instability = np.mean(
        data[
            "instability"
        ][mask]
    )

    # Mean dissipation
    dissipation = np.mean(
        data[
            "dissipation"
        ][mask]
    )

    environment_rows.append({

        "gamma":
            float(gamma_value),

        "samples":
            int(np.sum(mask)),

        "information_offdiagonal_strength":
            float(offdiag_strength),

        "information_state_variability":
            float(state_variability),

        "mean_dissipation":
            float(dissipation),

        "mean_instability":
            float(instability),
    })

environment_df = pd.DataFrame(
    environment_rows
)

display(
    environment_df.round(8)
)

# ============================================================
# 8. STATE EVOLUTION IN INFORMATION SPACE
#
# Construct R(t+dt) from the next state while keeping the
# other observable fields associated with the current sample.
#
# This is a preliminary transition diagnostic.
# Full trajectory construction comes in Step 3.
# ============================================================

next_features = normalized.copy()

# Replace only the state observable with state_t1.
state_next = (
    data["state_next"]
)

state_mean = normalization_stats[
    "state"
]["mean"]

state_std = normalization_stats[
    "state"
]["std"]

next_features[
    :,
    0,
    :
] = (
    state_next
    - state_mean
) / state_std

next_z = np.transpose(
    next_features,
    (0, 2, 1)
)

next_R = (
    next_z[
        ...,
        :, None
    ]
    *
    next_z[
        ...,
        None, :
    ]
)

next_trace = np.trace(
    next_R,
    axis1=-2,
    axis2=-1
)

next_R = (
    next_R
    /
    (
        np.abs(
            next_trace
        )
        + 1e-12
    )[
        ...,
        None,
        None
    ]
)

R_change = (
    next_R
    -
    information_matrix
)

R_change_norm = np.linalg.norm(
    R_change.reshape(
        N,
        NX,
        -1
    ),
    axis=-1
)

print(
    "\n" + "=" * 100
)

print(
    "INFORMATION-STATE EVOLUTION"
)

print(
    "=" * 100
)

print(
    "Mean ||R(t+dt)-R(t)||:",
    f"{np.mean(R_change_norm):.12e}"
)

print(
    "Median:",
    f"{np.median(R_change_norm):.12e}"
)

print(
    "95th percentile:",
    f"{np.percentile(R_change_norm,95):.12e}"
)

print(
    "Maximum:",
    f"{np.max(R_change_norm):.12e}"
)

# ============================================================
# 9. ENVIRONMENT-INDUCED INFORMATION CHANGE PROXY
#
# For each sample:
#
#   ||ΔR||
#
# and relationship with gamma/dissipation/instability.
# ============================================================

sample_information_change = np.mean(
    R_change_norm,
    axis=1
)

gamma_sample = data[
    "gamma"
][:, 0]

mean_diss_sample = np.mean(
    data["dissipation"],
    axis=1
)

mean_inst_sample = np.mean(
    data["instability"],
    axis=1
)

def safe_corr(
    a,
    b
):

    a = np.asarray(a)
    b = np.asarray(b)

    if (
        np.std(a) < 1e-12
        or
        np.std(b) < 1e-12
    ):

        return 0.0

    return float(
        np.corrcoef(
            a,
            b
        )[0, 1]
    )

print(
    "\n" + "=" * 100
)

print(
    "ENVIRONMENT → INTERNAL INFORMATION RESPONSE"
)

print(
    "=" * 100
)

print(
    "corr(gamma, ||ΔR||)       :",
    f"{safe_corr(gamma_sample, sample_information_change):+.8f}"
)

print(
    "corr(dissipation, ||ΔR||):",
    f"{safe_corr(mean_diss_sample, sample_information_change):+.8f}"
)

print(
    "corr(instability, ||ΔR||):",
    f"{safe_corr(mean_inst_sample, sample_information_change):+.8f}"
)

# ============================================================
# 10. SAVE REPRESENTATION
# ============================================================

np.save(
    os.path.join(
        OUT_DIR,
        "internal_information_state.npy"
    ),
    information_matrix.astype(
        np.float32
    )
)

np.save(
    os.path.join(
        OUT_DIR,
        "internal_information_state_next.npy"
    ),
    next_R.astype(
        np.float32
    )
)

np.save(
    os.path.join(
        OUT_DIR,
        "internal_information_change_norm.npy"
    ),
    sample_information_change.astype(
        np.float32
    )
)

environment_df.to_csv(
    os.path.join(
        OUT_DIR,
        "environment_response.csv"
    ),
    index=False
)

# ============================================================
# 11. SPECIFICATION
# ============================================================

specification = {

    "physical_system":
        "stochastic non-equilibrium nanoscale ion transport",

    "internal_information_state":
        "multivariate dynamical information carried by "
        "the transport system and modified by environmental "
        "forcing, dissipation and instability",

    "feature_vector":
        feature_names,

    "environmental_control":
        "gamma",

    "information_state_matrix":
        "R = z z^T / trace(z z^T)",

    "matrix_dimension":
        6,

    "spatial_points":
        NX,

    "samples":
        N,

    "primary_transition":
        "R(t) -> R(t+dt)",

    "environmental_response":
        "||R(t+dt) - R(t)||",

    "important_note":
        "R is a mathematical information-state representation "
        "of the ion-transport observables. It is not identified "
        "with the microscopic physical concentration field or "
        "a quantum density matrix.",

    "intended_LNO_role":
        "Lindblad-constrained dynamical kernel acts on the "
        "defined matrix-valued internal-information state.",

    "target_for_next_steps":
        "learn the evolution of the internal information state "
        "under varying environmental coupling and stochastic forcing",
}

SPEC_PATH = os.path.join(
    OUT_DIR,
    "internal_information_state_specification.json"
)

with open(
    SPEC_PATH,
    "w"
) as f:

    json.dump(
        specification,
        f,
        indent=2
    )

# ============================================================
# 12. SUMMARY
# ============================================================

summary = {

    "samples":
        N,

    "spatial_points":
        NX,

    "feature_dimension":
        len(feature_names),

    "information_matrix_shape":
        list(
            information_matrix.shape
        ),

    "max_trace_error":
        float(max_trace_error),

    "max_symmetry_error":
        float(transpose_error),

    "minimum_eigenvalue":
        float(min_eigenvalue),

    "mean_information_change":
        float(
            np.mean(
                sample_information_change
            )
        ),

    "gamma_values":
        [
            float(x)
            for x in gamma_values
        ],
}

with open(
    os.path.join(
        OUT_DIR,
        "step1_summary.json"
    ),
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

print(
    "\n" + "=" * 100
)

print(
    "STEP 1 COMPLETE"
)

print(
    "[+] Internal information representation:",
    specification[
        "internal_information_state"
    ]
)

print(
    "[+] Information-state matrix:",
    information_matrix.shape
)

print(
    "[+] Environment variable:",
    "gamma"
)

print(
    "[+] Saved:",
    OUT_DIR
)

print("=" * 100)
