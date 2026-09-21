# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 25
# Step            : STEP_01
# Step Heading    : # STEP 1 — PHYSICAL STATE DEFINITION / DATA AUDIT
# Step Cell No.   : 13
# ============================================================

# ============================================================
# STEP 1 — PHYSICAL STATE DEFINITION / DATA AUDIT
#
# PURPOSE:
#   Precisely define the physical state that the LNO/FNO
#   will learn to evolve.
#
# NO TRAINING
# NO CHECKPOINT MODIFICATION
# NO DATA MODIFICATION
# ============================================================

import os
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

print("=" * 90)
print("STEP 1 — PHYSICAL STATE DEFINITION / DATA AUDIT")
print("=" * 90)

# ============================================================
# LOAD CURRENT PHYSICAL FIELDS
# ============================================================

files_to_load = {
    "state_t": "state_t.npy",
    "state_t1": "state_t1.npy",
    "phi": "phi.npy",
    "flux": "flux.npy",
    "noise": "noise.npy",
    "dissipation": "dissipation.npy",
    "gamma": "gamma.npy",
    "entropy": "entropy.npy",
    "instability": "instability.npy",
}

arrays = {}

for name, filename in files_to_load.items():

    path = os.path.join(
        TEST_DIR,
        filename
    )

    if os.path.isfile(path):

        arrays[name] = np.load(path)

        print(
            f"[+] {name:15s}: "
            f"shape={arrays[name].shape}, "
            f"dtype={arrays[name].dtype}"
        )

    else:

        print(
            f"[MISSING] {name:15s}: "
            f"{path}"
        )

# ============================================================
# BASIC STATISTICS
# ============================================================

print("\n" + "=" * 90)
print("PHYSICAL FIELD STATISTICS")
print("=" * 90)

rows = []

for name, arr in arrays.items():

    arr64 = arr.astype(
        np.float64,
        copy=False
    )

    rows.append({

        "variable":
            name,

        "shape":
            str(arr.shape),

        "min":
            float(np.nanmin(arr64)),

        "max":
            float(np.nanmax(arr64)),

        "mean":
            float(np.nanmean(arr64)),

        "std":
            float(np.nanstd(arr64)),

        "finite":
            bool(np.isfinite(arr64).all()),
    })

stats_df = pd.DataFrame(
    rows
)

display(
    stats_df
)

# ============================================================
# STATE TRANSITION
# ============================================================

if (
    "state_t" in arrays
    and
    "state_t1" in arrays
):

    state_t = arrays[
        "state_t"
    ].astype(
        np.float64
    )

    state_t1 = arrays[
        "state_t1"
    ].astype(
        np.float64
    )

    assert (
        state_t.shape
        ==
        state_t1.shape
    )

    delta_state = (
        state_t1
        - state_t
    )

    print(
        "\n" + "=" * 90
    )

    print(
        "STATE EVOLUTION"
    )

    print(
        "=" * 90
    )

    print(
        "Physical state candidate : c(x,t)"
    )

    print(
        "state_t shape            :",
        state_t.shape
    )

    print(
        "state_t1 shape           :",
        state_t1.shape
    )

    print(
        "delta_state shape        :",
        delta_state.shape
    )

    print(
        "||state_t|| mean         :",
        np.mean(
            np.linalg.norm(
                state_t.reshape(
                    len(state_t),
                    -1
                ),
                axis=1
            )
        )
    )

    print(
        "||delta_state|| mean     :",
        np.mean(
            np.linalg.norm(
                delta_state.reshape(
                    len(delta_state),
                    -1
                ),
                axis=1
            )
        )
    )

    relative_change = (
        np.linalg.norm(
            delta_state.reshape(
                len(delta_state),
                -1
            ),
            axis=1
        )
        /
        (
            np.linalg.norm(
                state_t.reshape(
                    len(state_t),
                    -1
                ),
                axis=1
            )
            + 1e-30
        )
    )

    print(
        "Relative state change mean:",
        float(
            relative_change.mean()
        )
    )

# ============================================================
# GAMMA / ENVIRONMENT
# ============================================================

if "gamma" in arrays:

    gamma = arrays[
        "gamma"
    ]

    print(
        "\n" + "=" * 90
    )

    print(
        "ENVIRONMENTAL COUPLING"
    )

    print(
        "=" * 90
    )

    print(
        "gamma shape:",
        gamma.shape
    )

    print(
        "gamma min:",
        float(
            np.min(gamma)
        )
    )

    print(
        "gamma max:",
        float(
            np.max(gamma)
        )
    )

    print(
        "gamma unique values:",
        np.unique(gamma)[:30]
    )

# ============================================================
# NOISE / DISSIPATION
# ============================================================

for name in [
    "noise",
    "dissipation",
    "flux",
]:

    if name not in arrays:
        continue

    arr = arrays[name]

    print(
        "\n" + "=" * 90
    )

    print(
        name.upper()
    )

    print(
        "=" * 90
    )

    print(
        "mean absolute:",
        float(
            np.mean(
                np.abs(arr)
            )
        )
    )

    print(
        "L2 mean:",
        float(
            np.mean(
                np.linalg.norm(
                    arr.reshape(
                        len(arr),
                        -1
                    ),
                    axis=1
                )
            )
        )
    )

# ============================================================
# SAVE STATE SPECIFICATION
# ============================================================

OUT_DIR = os.path.join(
    ROOT,
    "results",
    "physical_state_definition"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

stats_path = os.path.join(
    OUT_DIR,
    "physical_state_statistics.csv"
)

stats_df.to_csv(
    stats_path,
    index=False
)

spec = {

    "physical_state_candidate":
        "ion concentration / transport state field c(x,t)",

    "primary_state_array":
        "state_t",

    "target_state_array":
        "state_t1",

    "state_transition":
        "delta_state = state_t1 - state_t",

    "spatial_resolution":
        128,

    "environment_variables": [
        "gamma",
        "noise",
        "dissipation",
    ],

    "additional_observables": [
        "phi",
        "flux",
        "entropy",
        "instability",
    ],

    "scientific_objective":
        "learn environment-driven non-equilibrium "
        "evolution of the internal ion-transport state",

    "note":
        "This specification intentionally does not "
        "identify the neural latent representation with "
        "the physical state.",
}

spec_path = os.path.join(
    OUT_DIR,
    "physical_state_specification.json"
)

import json

with open(
    spec_path,
    "w"
) as f:

    json.dump(
        spec,
        f,
        indent=2
    )

print(
    "\n[+] Statistics saved:",
    stats_path
)

print(
    "[+] State specification saved:",
    spec_path
)

print("=" * 90)
print("STEP 1 COMPLETE")
print("=" * 90)
