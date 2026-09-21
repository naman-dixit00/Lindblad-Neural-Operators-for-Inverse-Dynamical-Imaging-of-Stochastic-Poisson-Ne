# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 29
# Step            : STEP_03
# Step Heading    : # STEP 3 — EXACT PAIRED OPEN/REFERENCE TRAJECTORY SETUP
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 3 — EXACT PAIRED OPEN/REFERENCE TRAJECTORY SETUP
#
# PURPOSE:
#   Determine how to replay the SAME initial condition and
#   SAME stochastic realization while changing only the
#   environment coupling gamma.
#
# NO TRAINING
# NO DATASET MODIFICATION
# ============================================================

import os
import inspect
import json
import numpy as np

from pathlib import Path

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

print("=" * 100)
print("STEP 3 — PAIRED OPEN / REFERENCE TRAJECTORY API AUDIT")
print("=" * 100)

# ============================================================
# 1. IMPORT
# ============================================================

from data.pnp_simulator import PhysicsOperatorSimulator
from data.regime_configs import REGIMES

# Existing project config
import yaml

CONFIG_PATH = ROOT / "config.yaml"

assert CONFIG_PATH.is_file(), CONFIG_PATH

with open(
    CONFIG_PATH,
    "r"
) as f:

    config = yaml.safe_load(f)

sim = PhysicsOperatorSimulator(
    config
)

print(
    "[+] Simulator:",
    type(sim).__name__
)

# ============================================================
# 2. SHOW SIMULATOR METHODS
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "SIMULATOR METHODS"
)

print(
    "=" * 100
)

methods = []

for name in dir(sim):

    if name.startswith(
        "__"
    ):

        continue

    try:

        attr = getattr(
            sim,
            name
        )

    except Exception:

        continue

    if callable(attr):

        methods.append(
            name
        )

for name in methods:

    print(
        f"{name:40s}",
        inspect.signature(
            getattr(
                sim,
                name
            )
        )
    )

# ============================================================
# 3. SHOW KEY ATTRIBUTES
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "SIMULATOR ATTRIBUTES"
)

print(
    "=" * 100
)

for name in [
    "nx",
    "nt",
    "dx",
    "dt",
    "D",
    "mu",
    "epsilon",
]:

    if hasattr(
        sim,
        name
    ):

        print(
            f"{name:15s} =",
            getattr(
                sim,
                name
            )
        )

# ============================================================
# 4. SOURCE OF compute_trajectory
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "EXACT compute_trajectory SOURCE"
)

print(
    "=" * 100
)

try:

    trajectory_source = inspect.getsource(
        sim.compute_trajectory
    )

    print(
        trajectory_source
    )

except Exception as exc:

    trajectory_source = ""

    print(
        "[INFO] Could not retrieve source:",
        repr(exc)
    )

# ============================================================
# 5. SEARCH FOR INITIAL-STATE GENERATION
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "INITIAL-STATE / NOISE GENERATION CANDIDATES"
)

print(
    "=" * 100
)

source_terms = [

    "initial",
    "state",
    "np.random",
    "random",
    "noise",
    "trajectory",
    "seed",
    "default_rng",
]

for line_no, line in enumerate(
    trajectory_source.splitlines(),
    start=1
):

    lower = line.lower()

    if any(
        term in lower
        for term in source_terms
    ):

        print(
            f"{line_no:04d}: {line}"
        )

# ============================================================
# 6. IDENTIFY POSSIBLE INITIAL STATE METHODS
# ============================================================

candidate_methods = [

    name
    for name in methods

    if any(
        token in name.lower()
        for token in [
            "initial",
            "state",
            "reset",
            "simulate",
            "evolve",
            "step",
            "noise",
        ]
    )
]

print(
    "\n" + "=" * 100
)

print(
    "CANDIDATE LOW-LEVEL PHYSICS METHODS"
)

print(
    "=" * 100
)

for name in candidate_methods:

    try:

        source = inspect.getsource(
            getattr(
                sim,
                name
            )
        )

        print(
            "\n###",
            name
        )

        print(
            source[:8000]
        )

    except Exception as exc:

        print(
            "\n###",
            name,
            "source unavailable:",
            repr(exc)
        )

# ============================================================
# 7. IMPORTANT: CURRENT TRAJECTORY CONVENTION
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "CURRENT TRAJECTORY CONVENTION"
)

print(
    "=" * 100
)

print(
    "Available regimes:"
)

for name, cfg in REGIMES.items():

    print(
        f"  {name:20s}"
        f" noise={cfg['noise']}"
        f" gamma={cfg['gamma']}"
    )

print(
    "\nReference candidate: gamma = 0"
)

print(
    "Open candidates: gamma > 0"
)

print(
    "\nRequired pairing conditions:"
)

print(
    "  1. SAME initial physical state"
)

print(
    "  2. SAME stochastic realization where possible"
)

print(
    "  3. DIFFERENT environment coupling gamma"
)

# ============================================================
# 8. DO NOT CONSTRUCT PAIRS YET IF API IS UNKNOWN
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "STEP 3 DECISION"
)

print(
    "=" * 100
)

if (
    "trajectory_source" in locals()
    and
    trajectory_source
):

    # Check for explicit initial-state arguments
    explicit_initial_state = (
        "initial_state" in trajectory_source
        or
        "state0" in trajectory_source
        or
        "state_0" in trajectory_source
        or
        "initial" in trajectory_source
    )

    print(
        "[+] compute_trajectory inspected."
    )

    print(
        "[+] Initial-state handling detected:",
        explicit_initial_state
    )

    print(
        "\n[INFO] We intentionally do NOT generate"
    )

    print(
        "paired trajectories until the simulator's"
    )

    print(
        "initial-state replay mechanism is verified."
    )

else:

    print(
        "[INFO] Source unavailable."
    )

    print(
        "Use the printed low-level methods to identify "
        "the exact replay path."
    )

# ============================================================
# 9. SAVE API AUDIT
# ============================================================

OUT_DIR = (
    ROOT
    / "results"
    / "step3_paired_trajectory_audit"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

audit = {

    "constructor":
        str(
            inspect.signature(
                PhysicsOperatorSimulator
            )
        ),

    "compute_trajectory":
        str(
            inspect.signature(
                PhysicsOperatorSimulator.compute_trajectory
            )
        ),

    "regimes":
        {
            name: {
                "noise":
                    float(
                        cfg["noise"]
                    ),
                "gamma":
                    float(
                        cfg["gamma"]
                    ),
            }
            for name, cfg
            in REGIMES.items()
        },

    "reference_gamma":
        0.0,

    "required_pairing": [

        "same_initial_state",

        "same_stochastic_realization",

        "different_environment_gamma",
    ],
}

with open(
    OUT_DIR
    / "step3_api_audit.json",
    "w"
) as f:

    json.dump(
        audit,
        f,
        indent=2
    )

with open(
    OUT_DIR
    / "compute_trajectory_source.txt",
    "w"
) as f:

    f.write(
        trajectory_source
    )

print(
    "\n[+] Saved API audit:",
    OUT_DIR
)

print("=" * 100)
print("STEP 3 API AUDIT COMPLETE")
print("=" * 100)
