# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 30
# Step            : STEP_03
# Step Heading    : # STEP 3 — ACTUAL PAIRED OPEN / REFERENCE TRAJECTORIES
# Step Cell No.   : 2
# ============================================================

# ============================================================
# STEP 3 — ACTUAL PAIRED OPEN / REFERENCE TRAJECTORIES
#
# SAME:
#   initial state
#   noise_sigma
#   RNG seed
#
# DIFFERENT:
#   gamma
#
# Output:
#   R_reference(t)
#   R_open(t)
#   ΔR_env(t)
#
# NO TRAINING
# NO DATASET MODIFICATION
# ============================================================

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

# ============================================================
# 1. ROOT
# ============================================================

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

print("=" * 100)
print("STEP 3 — ACTUAL PAIRED OPEN / REFERENCE TRAJECTORIES")
print("=" * 100)

# ============================================================
# 2. CONFIG + SIMULATOR
# ============================================================

CONFIG_PATH = ROOT / "config.yaml"

assert CONFIG_PATH.is_file(), CONFIG_PATH

with open(
    CONFIG_PATH,
    "r"
) as f:

    config = yaml.safe_load(f)

from data.pnp_simulator import PhysicsOperatorSimulator
from data.regime_configs import REGIMES

reference_sim = PhysicsOperatorSimulator(
    config
)

open_sim = PhysicsOperatorSimulator(
    config
)

# ============================================================
# 3. OUTPUT
# ============================================================

OUT_DIR = (
    ROOT
    / "results"
    / "step3_paired_trajectories"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 4. INFORMATION-STATE BUILDER
#
# IMPORTANT:
# Open + reference are normalized jointly, so the difference
# between R_open and R_reference is meaningful.
# ============================================================

FEATURE_NAMES = [
    "state",
    "phi",
    "flux",
    "noise",
    "dissipation",
    "instability",
]


def make_information_state(
    state,
    phi,
    flux,
    noise,
    dissipation,
    instability,
    joint_mean,
    joint_std,
):

    raw = np.stack(
        [
            state,
            phi,
            flux,
            noise,
            dissipation,
            instability,
        ],
        axis=1
    ).astype(
        np.float64
    )

    z = (
        raw
        -
        joint_mean
    ) / joint_std

    z = np.transpose(
        z,
        (
            0,
            2,
            1
        )
    )

    # R = z z^T
    R = (
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
        R,
        axis1=-2,
        axis2=-1
    )

    R = (
        R
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

    return R.astype(
        np.float32
    )


# ============================================================
# 5. PROCESS ALL FIVE REGIMES
# ============================================================

results = []

for regime_name, regime in REGIMES.items():

    noise_sigma = float(
        regime["noise"]
    )

    open_gamma = float(
        regime["gamma"]
    )

    reference_gamma = 0.0

    print(
        "\n" + "=" * 100
    )

    print(
        f"REGIME: {regime_name}"
    )

    print(
        f"noise_sigma = {noise_sigma}"
    )

    print(
        f"reference_gamma = {reference_gamma}"
    )

    print(
        f"open_gamma = {open_gamma}"
    )

    print(
        "=" * 100
    )

    # ========================================================
    # A. SAME SEED → SAME STOCHASTIC REALIZATION
    # ========================================================

    np.random.seed(
        2026
    )

    reference = (
        reference_sim.compute_trajectory(
            noise_sigma=noise_sigma,
            gamma=reference_gamma
        )
    )

    np.random.seed(
        2026
    )

    opened = (
        open_sim.compute_trajectory(
            noise_sigma=noise_sigma,
            gamma=open_gamma
        )
    )

    (
        ref_state,
        ref_state_t1,
        ref_phi,
        ref_flux,
        ref_noise,
        ref_diss,
        ref_gamma,
        ref_entropy,
        ref_inst,
    ) = reference[:9]

    (
        open_state,
        open_state_t1,
        open_phi,
        open_flux,
        open_noise,
        open_diss,
        open_gamma_field,
        open_entropy,
        open_inst,
    ) = opened[:9]

    # ========================================================
    # B. BASIC CHECKS
    # ========================================================

    assert ref_state.shape == open_state.shape

    assert ref_noise.shape == open_noise.shape

    T, NX = ref_state.shape

    print(
        "[+] Trajectory:",
        T,
        "timesteps ×",
        NX,
        "spatial points"
    )

    # ========================================================
    # C. NOISE PAIRING CHECK
    # ========================================================

    noise_difference = (
        open_noise
        -
        ref_noise
    )

    noise_max_error = float(
        np.max(
            np.abs(
                noise_difference
            )
        )
    )

    noise_relative_error = float(
        np.linalg.norm(
            noise_difference
        )
        /
        (
            np.linalg.norm(
                ref_noise
            )
            + 1e-30
        )
    )

    print(
        "\n[NOISE PAIRING]"
    )

    print(
        "Max difference     :",
        f"{noise_max_error:.12e}"
    )

    print(
        "Relative difference:",
        f"{noise_relative_error:.12e}"
    )

    # ========================================================
    # D. STATE ENVIRONMENT EFFECT
    # ========================================================

    delta_state_env = (
        open_state
        -
        ref_state
    )

    state_env_norm = np.linalg.norm(
        delta_state_env,
        axis=-1
    )

    print(
        "\n[STATE ENVIRONMENT EFFECT]"
    )

    print(
        "Mean ||Δstate_env||:",
        f"{state_env_norm.mean():.12e}"
    )

    print(
        "Max  ||Δstate_env||:",
        f"{state_env_norm.max():.12e}"
    )

    # ========================================================
    # E. JOINT NORMALIZATION
    # ========================================================

    ref_features = np.stack(
        [
            ref_state,
            ref_phi,
            ref_flux,
            ref_noise,
            ref_diss,
            ref_inst,
        ],
        axis=1
    ).astype(
        np.float64
    )

    open_features = np.stack(
        [
            open_state,
            open_phi,
            open_flux,
            open_noise,
            open_diss,
            open_inst,
        ],
        axis=1
    ).astype(
        np.float64
    )

    combined = np.concatenate(
        [
            ref_features,
            open_features,
        ],
        axis=0
    )

    joint_mean = np.mean(
        combined,
        axis=(0, 2),
        keepdims=True
    )

    joint_std = np.std(
        combined,
        axis=(0, 2),
        keepdims=True
    )

    joint_std = np.where(
        joint_std < 1e-12,
        1.0,
        joint_std
    )

    # ========================================================
    # F. BUILD INFORMATION STATES
    # ========================================================

    R_ref = make_information_state(
        ref_state,
        ref_phi,
        ref_flux,
        ref_noise,
        ref_diss,
        ref_inst,
        joint_mean,
        joint_std,
    )

    R_open = make_information_state(
        open_state,
        open_phi,
        open_flux,
        open_noise,
        open_diss,
        open_inst,
        joint_mean,
        joint_std,
    )

    # ========================================================
    # G. ENVIRONMENT-INDUCED INFORMATION RESPONSE
    # ========================================================

    delta_R_env = (
        R_open
        -
        R_ref
    )

    delta_R_norm = np.linalg.norm(
        delta_R_env.reshape(
            T,
            NX,
            -1
        ),
        axis=-1
    )

    temporal_delta_R = np.mean(
        delta_R_norm,
        axis=1
    )

    print(
        "\n[INFORMATION RESPONSE]"
    )

    print(
        "Mean ||ΔR_env||:",
        f"{delta_R_norm.mean():.12e}"
    )

    print(
        "Median:",
        f"{np.median(delta_R_norm):.12e}"
    )

    print(
        "95th percentile:",
        f"{np.percentile(delta_R_norm, 95):.12e}"
    )

    print(
        "Maximum:",
        f"{delta_R_norm.max():.12e}"
    )

    # ========================================================
    # H. ENTROPY DIFFERENCE
    # ========================================================

    ref_entropy_flat = (
        ref_entropy.reshape(-1)
    )

    open_entropy_flat = (
        open_entropy.reshape(-1)
    )

    delta_entropy = (
        open_entropy_flat
        -
        ref_entropy_flat
    )

    # ========================================================
    # I. INSTABILITY DIFFERENCE
    # ========================================================

    ref_inst_mean = np.mean(
        ref_inst,
        axis=1
    )

    open_inst_mean = np.mean(
        open_inst,
        axis=1
    )

    delta_instability = (
        open_inst_mean
        -
        ref_inst_mean
    )

    # ========================================================
    # J. TEMPORAL CSV
    # ========================================================

    temporal_df = pd.DataFrame({

        "timestep":
            np.arange(T),

        "state_environment_effect":
            state_env_norm.mean(
                axis=0
            )
            if state_env_norm.ndim > 1
            else state_env_norm,

        "information_environment_effect":
            temporal_delta_R,

        "reference_entropy":
            ref_entropy_flat,

        "open_entropy":
            open_entropy_flat,

        "delta_entropy":
            delta_entropy,

        "reference_instability":
            ref_inst_mean,

        "open_instability":
            open_inst_mean,

        "delta_instability":
            delta_instability,
    })

    temporal_csv = (
        OUT_DIR
        / f"{regime_name}_temporal_response.csv"
    )

    temporal_df.to_csv(
        temporal_csv,
        index=False
    )

    # ========================================================
    # K. SAVE FULL PAIRED DATA
    # ========================================================

    paired_npz = (
        OUT_DIR
        / f"{regime_name}_paired.npz"
    )

    np.savez_compressed(

        paired_npz,

        reference_state=
            ref_state,

        open_state=
            open_state,

        delta_state_environment=
            delta_state_env.astype(
                np.float32
            ),

        reference_phi=
            ref_phi,

        open_phi=
            open_phi,

        reference_flux=
            ref_flux,

        open_flux=
            open_flux,

        reference_noise=
            ref_noise,

        open_noise=
            open_noise,

        reference_dissipation=
            ref_diss,

        open_dissipation=
            open_diss,

        reference_gamma=
            ref_gamma,

        open_gamma=
            open_gamma_field,

        reference_entropy=
            ref_entropy,

        open_entropy=
            open_entropy,

        reference_instability=
            ref_inst,

        open_instability=
            open_inst,

        R_reference=
            R_ref,

        R_open=
            R_open,

        delta_R_environment=
            delta_R_env,

        delta_R_norm=
            delta_R_norm.astype(
                np.float32
            ),

        temporal_delta_R=
            temporal_delta_R.astype(
                np.float32
            ),
    )

    # ========================================================
    # L. SUMMARY
    # ========================================================

    results.append({

        "regime":
            regime_name,

        "noise_sigma":
            noise_sigma,

        "reference_gamma":
            reference_gamma,

        "open_gamma":
            open_gamma,

        "timesteps":
            T,

        "spatial_points":
            NX,

        "noise_max_error":
            noise_max_error,

        "noise_relative_error":
            noise_relative_error,

        "mean_state_environment_effect":
            float(
                state_env_norm.mean()
            ),

        "mean_information_environment_effect":
            float(
                delta_R_norm.mean()
            ),

        "p95_information_environment_effect":
            float(
                np.percentile(
                    delta_R_norm,
                    95
                )
            ),

        "max_information_environment_effect":
            float(
                delta_R_norm.max()
            ),

        "mean_entropy_change":
            float(
                delta_entropy.mean()
            ),

        "mean_instability_change":
            float(
                delta_instability.mean()
            ),

        "paired_npz":
            str(paired_npz),

        "temporal_csv":
            str(temporal_csv),
    })

# ============================================================
# 22. SAVE GLOBAL SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    results
)

SUMMARY_CSV = (
    OUT_DIR
    / "paired_environment_summary.csv"
)

SUMMARY_JSON = (
    OUT_DIR
    / "paired_environment_summary.json"
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)

with open(
    SUMMARY_JSON,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=2
    )

# ============================================================
# 23. FINAL REPORT
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "STEP 3 — PAIRED ENVIRONMENT EXPERIMENT COMPLETE"
)

print(
    "=" * 100
)

display(
    summary_df.round(8)
)

print(
    "[+] Summary CSV:",
    SUMMARY_CSV
)

print(
    "[+] Summary JSON:",
    SUMMARY_JSON
)

print(
    "[+] Output directory:",
    OUT_DIR
)

print(
    "\nCore quantity:"
)

print(
    "ΔR_env(t) = R_open(t) - R_reference(t)"
)

print(
    "\nPairing requirement:"
)

print(
    "same initial state + same noise + different gamma"
)

print("=" * 100)
