# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 27
# Step            : STEP_02
# Step Heading    : # STEP 2 — SIMULATOR → TRAJECTORY / OBSERVABLE AUDIT
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 2 — SIMULATOR → TRAJECTORY / OBSERVABLE AUDIT
#
# PURPOSE:
#   1. Inspect exact simulator API
#   2. Generate one trajectory per regime
#   3. Verify timestep alignment
#   4. Build Information-State R(t)
#   5. Measure its temporal evolution
#
# NO TRAINING
# NO MODEL CHANGES
# NO EXISTING DATASET MODIFICATION
# ============================================================

import os
import json
import inspect
import numpy as np
import pandas as pd

ROOT = (
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

print("=" * 100)
print("STEP 2 — SIMULATOR / TRAJECTORY / OBSERVABLE AUDIT")
print("=" * 100)

# ============================================================
# 1. IMPORT EXACT PROJECT SIMULATOR
# ============================================================

from data.pnp_simulator import PhysicsOperatorSimulator
from data.regime_configs import REGIMES

print(
    "[+] PhysicsOperatorSimulator imported."
)

print(
    "[+] Constructor signature:"
)

print(
    inspect.signature(
        PhysicsOperatorSimulator
    )
)

print(
    "\n[+] compute_trajectory signature:"
)

print(
    inspect.signature(
        PhysicsOperatorSimulator.compute_trajectory
    )
)

print(
    "\n[+] Available regimes:"
)

for name, cfg in REGIMES.items():

    print(
        f"    {name}: "
        f"noise={cfg.get('noise')}, "
        f"gamma={cfg.get('gamma')}"
    )

# ============================================================
# 2. INSPECT INSTANCE CONSTRUCTION
#
# Try the project's expected constructor first.
# If it fails, print the exact reason rather than guessing.
# ============================================================

sim = None

try:

    sim = PhysicsOperatorSimulator()

    print(
        "\n[PASS] Simulator instantiated with default constructor."
    )

except Exception as exc:

    print(
        "\n[INFO] Default constructor failed:"
    )

    print(
        repr(exc)
    )

    print(
        "\n[+] Constructor source/signature indicates "
        "that configuration may be required."
    )

# ============================================================
# 3. IF DEFAULT CONSTRUCTOR FAILS, SHOW CONSTRUCTOR SOURCE
# ============================================================

if sim is None:

    try:

        print(
            "\n" + "=" * 100
        )

        print(
            "CONSTRUCTOR SOURCE"
        )

        print(
            "=" * 100
        )

        print(
            inspect.getsource(
                PhysicsOperatorSimulator.__init__
            )
        )

    except Exception as exc:

        print(
            "[INFO] Constructor source unavailable:",
            repr(exc)
        )

    raise RuntimeError(
        "Simulator requires explicit constructor arguments. "
        "The cell stopped before generating data so that "
        "we do not invent configuration."
    )

# ============================================================
# 4. RUN ONE TRAJECTORY PER AVAILABLE REGIME
# ============================================================

OUT_DIR = os.path.join(
    ROOT,
    "results",
    "step2_trajectory_audit"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

trajectory_manifest = []
trajectory_objects = {}

# deterministic seed
np.random.seed(2026)

for regime_name, parameters in REGIMES.items():

    noise_sigma = float(
        parameters["noise"]
    )

    gamma = float(
        parameters["gamma"]
    )

    print(
        "\n" + "=" * 100
    )

    print(
        f"REGIME: {regime_name}"
    )

    print(
        f"noise_sigma={noise_sigma}"
    )

    print(
        f"gamma={gamma}"
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # Reset deterministic RNG before each regime so the
    # starting random realization is reproducible.
    # --------------------------------------------------------

    np.random.seed(
        2026
    )

    try:

        result = sim.compute_trajectory(
            noise_sigma=noise_sigma,
            gamma=gamma
        )

    except Exception as exc:

        print(
            "[FAIL] compute_trajectory failed:"
        )

        print(
            repr(exc)
        )

        continue

    # --------------------------------------------------------
    # Verify return structure
    # --------------------------------------------------------

    if not isinstance(
        result,
        tuple
    ):

        print(
            "[WARNING] Return type:",
            type(result)
        )

        trajectory_objects[
            regime_name
        ] = result

        continue

    print(
        "[+] Number of returned objects:",
        len(result)
    )

    for idx, item in enumerate(
        result
    ):

        if isinstance(
            item,
            np.ndarray
        ):

            print(
                f"    [{idx}] "
                f"shape={item.shape}, "
                f"dtype={item.dtype}"
            )

        else:

            print(
                f"    [{idx}] "
                f"type={type(item)}"
            )

    # ========================================================
    # 5. CURRENT PROJECT RETURN ORDER
    #
    # From the simulator source audit:
    #
    # 0 states_t
    # 1 states_t1
    # 2 phis_t
    # 3 fluxes_t
    # 4 noises_t
    # 5 dissipations_t
    # 6 gammas_t
    # 7 entropies_t
    # 8 instabilities_t
    #
    # We verify before using.
    # ========================================================

    if len(result) < 9:

        print(
            "[WARNING] Fewer than 9 trajectory fields returned. "
            "Skipping structured analysis."
        )

        continue

    (
        states_t,
        states_t1,
        phis_t,
        fluxes_t,
        noises_t,
        dissipations_t,
        gammas_t,
        entropies_t,
        instabilities_t,
    ) = result[:9]

    # ========================================================
    # 6. SHAPE CHECKS
    # ========================================================

    arrays = {

        "state_t":
            states_t,

        "state_t1":
            states_t1,

        "phi":
            phis_t,

        "flux":
            fluxes_t,

        "noise":
            noises_t,

        "dissipation":
            dissipations_t,

        "gamma":
            gammas_t,

        "entropy":
            entropies_t,

        "instability":
            instabilities_t,
    }

    print(
        "\n[+] Structured trajectory fields:"
    )

    for name, arr in arrays.items():

        print(
            f"    {name:15s}: "
            f"shape={arr.shape}"
        )

    # --------------------------------------------------------
    # Number of trajectory timesteps
    # --------------------------------------------------------

    T = states_t.shape[0]

    NX = states_t.shape[-1]

    # All spatial trajectory fields must share
    # the same leading time dimension.

    for name in [
        "state_t",
        "state_t1",
        "phi",
        "flux",
        "noise",
        "dissipation",
        "gamma",
        "instability",
    ]:

        arr = arrays[name]

        assert arr.shape[0] == T, (
            f"{regime_name}: {name} has "
            f"{arr.shape[0]} timesteps; expected {T}"
        )

    # Entropy can be [T] or [T,1]
    assert entropies_t.shape[0] == T

    print(
        f"\n[PASS] Temporal alignment verified: "
        f"T={T}, NX={NX}"
    )

    # ========================================================
    # 7. FINITENESS
    # ========================================================

    all_finite = True

    for name, arr in arrays.items():

        if not np.isfinite(
            arr
        ).all():

            all_finite = False

            print(
                "[FAIL] Non-finite values:",
                name
            )

    print(
        "[+] All trajectory values finite:",
        all_finite
    )

    # ========================================================
    # 8. ACTUAL STATE TRANSITION
    # ========================================================

    delta_state = (
        states_t1
        -
        states_t
    )

    transition_l2 = np.linalg.norm(
        delta_state.reshape(
            T,
            -1
        ),
        axis=1
    )

    print(
        "\n[+] Mean transition ||Δstate||:",
        f"{transition_l2.mean():.12e}"
    )

    print(
        "[+] Max transition ||Δstate||:",
        f"{transition_l2.max():.12e}"
    )

    # ========================================================
    # 9. INTERNAL INFORMATION VECTOR
    #
    # Same representation established in Step 1:
    #
    # z =
    # [state, phi, flux, noise, dissipation, instability]
    #
    # gamma remains an external environmental control.
    # ========================================================

    feature_names = [
        "state",
        "phi",
        "flux",
        "noise",
        "dissipation",
        "instability",
    ]

    feature_stack = np.stack(
        [
            arrays[name]
            for name in feature_names
        ],
        axis=1
    )

    # Shape:
    # [T, 6, NX]

    # --------------------------------------------------------
    # Feature-wise normalization along entire trajectory
    # --------------------------------------------------------

    z = np.empty_like(
        feature_stack,
        dtype=np.float64
    )

    normalization = {}

    for channel_idx, name in enumerate(
        feature_names
    ):

        values = feature_stack[
            :,
            channel_idx,
            :
        ]

        mean = np.mean(
            values
        )

        std = np.std(
            values
        )

        if std < 1e-12:

            std = 1.0

        z[
            :,
            channel_idx,
            :
        ] = (
            values
            - mean
        ) / std

        normalization[
            name
        ] = {

            "mean":
                float(mean),

            "std":
                float(std),
        }

    # ========================================================
    # 10. INFORMATION-STATE MATRIX R(t)
    # ========================================================

    z_spatial = np.transpose(
        z,
        (
            0,
            2,
            1
        )
    )

    # [T, NX, 6, 6]
    R = (
        z_spatial[
            ...,
            :,
            None
        ]
        *
        z_spatial[
            ...,
            None,
            :
        ]
    )

    R_trace = np.trace(
        R,
        axis1=-2,
        axis2=-1
    )

    R = (
        R
        /
        (
            np.abs(
                R_trace
            )
            + 1e-12
        )[
            ...,
            None,
            None
        ]
    )

    print(
        "[+] Information-state R shape:",
        R.shape
    )

    # ========================================================
    # 11. INFORMATION EVOLUTION
    # ========================================================

    if T > 1:

        R_delta = (
            R[1:]
            -
            R[:-1]
        )

        R_delta_norm = np.linalg.norm(
            R_delta.reshape(
                T - 1,
                NX,
                -1
            ),
            axis=-1
        )

        mean_R_change = (
            R_delta_norm.mean()
        )

        max_R_change = (
            R_delta_norm.max()
        )

        print(
            "\n[+] Mean ||R(t+1)-R(t)||:",
            f"{mean_R_change:.12e}"
        )

        print(
            "[+] Max ||R(t+1)-R(t)||:",
            f"{max_R_change:.12e}"
        )

    else:

        mean_R_change = np.nan
        max_R_change = np.nan

    # ========================================================
    # 12. ENVIRONMENTAL COUPLING THROUGHOUT TRAJECTORY
    # ========================================================

    gamma_values = np.unique(
        np.round(
            gammas_t.reshape(-1),
            8
        )
    )

    print(
        "\n[+] Gamma values in trajectory:",
        gamma_values
    )

    # ========================================================
    # 13. SAVE TRAJECTORY DATA
    #
    # One NPZ per regime.
    # ========================================================

    npz_path = os.path.join(
        OUT_DIR,
        f"{regime_name}_trajectory.npz"
    )

    np.savez_compressed(
        npz_path,
        state_t=states_t,
        state_t1=states_t1,
        phi=phis_t,
        flux=fluxes_t,
        noise=noises_t,
        dissipation=dissipations_t,
        gamma=gammas_t,
        entropy=entropies_t,
        instability=instabilities_t,
        information_state_R=R.astype(
            np.float32
        ),
    )

    # ========================================================
    # 14. SUMMARY
    # ========================================================

    trajectory_manifest.append({

        "regime":
            regime_name,

        "noise_sigma":
            noise_sigma,

        "gamma":
            gamma,

        "timesteps":
            int(T),

        "spatial_points":
            int(NX),

        "all_finite":
            bool(all_finite),

        "mean_transition_l2":
            float(
                transition_l2.mean()
            ),

        "max_transition_l2":
            float(
                transition_l2.max()
            ),

        "mean_information_change":
            float(
                mean_R_change
            ),

        "max_information_change":
            float(
                max_R_change
            ),

        "trajectory_file":
            npz_path,
    })

    print(
        "[+] Saved:",
        npz_path
    )

# ============================================================
# 15. SAVE MANIFEST
# ============================================================

manifest_path = os.path.join(
    OUT_DIR,
    "trajectory_manifest.json"
)

with open(
    manifest_path,
    "w"
) as f:

    json.dump(
        trajectory_manifest,
        f,
        indent=2
    )

manifest_df = pd.DataFrame(
    trajectory_manifest
)

manifest_csv = os.path.join(
    OUT_DIR,
    "trajectory_manifest.csv"
)

manifest_df.to_csv(
    manifest_csv,
    index=False
)

print(
    "\n" + "=" * 100
)

print(
    "STEP 2 COMPLETE"
)

print(
    "[+] Manifest:",
    manifest_path
)

print(
    "[+] CSV:",
    manifest_csv
)

print(
    "[+] Output directory:",
    OUT_DIR
)

print("=" * 100)
