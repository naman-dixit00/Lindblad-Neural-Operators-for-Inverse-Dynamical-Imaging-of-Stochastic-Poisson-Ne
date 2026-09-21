# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 40
# Step            : STEP_09
# Step Heading    : # STEP 9 — TRAJECTORY-AWARE TRAIN / VAL / TEST SPLIT
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 9 — TRAJECTORY-AWARE TRAIN / VAL / TEST SPLIT
#
# PURPOSE:
#   Build a leakage-safe split for the current Dataset V3.
#
# IMPORTANT:
#   We DO NOT randomly split individual temporal transitions.
#
#   Each regime contains one 100-step trajectory.
#   Therefore all transitions from one trajectory stay together.
#
#   Current prototype has only 5 trajectories:
#       low_noise
#       stochastic
#       heavy_dissipation
#       collapse
#       metastable
#
#   With only five trajectories, a conventional 80/10/10
#   trajectory split is impossible without leaving regimes
#   completely absent from train/validation/test.
#
#   Therefore this cell creates:
#
#   1. PRIMARY REGIME-HOLDOUT EVALUATION
#      ---------------------------------
#      Each regime can be held out in turn.
#
#   2. PROTOTYPE TRAIN/VAL/TEST SPLIT
#      --------------------------------
#      Within the current testing dataset, transitions are
#      split by TIMESTEP BLOCKS, not random rows, for a
#      controlled development experiment.
#
#   Final expanded dataset will contain MANY independent
#   trajectories, after which a true trajectory-level
#   train/val/test split will be used.
#
# NO MODEL TRAINING
# NO CHECKPOINT MODIFICATION
# NO DATASET V3 OVERWRITE
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
print("STEP 9 — TRAJECTORY-AWARE TRAIN / VAL / TEST SPLIT")
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

# ============================================================
# 3. LOAD ALL LEARNING ARRAYS
# ============================================================

X_R = data[
    "X_R"
].astype(
    np.float32
)

Y_R = data[
    "Y_R"
].astype(
    np.float32
)

X_A = data[
    "X_amplitude"
].astype(
    np.float32
)

Y_A = data[
    "Y_amplitude"
].astype(
    np.float32
)

X_Z = data[
    "X_information_z"
].astype(
    np.float32
)

Y_Z = data[
    "Y_information_z"
].astype(
    np.float32
)

X_STATE = data[
    "X_state"
].astype(
    np.float32
)

Y_STATE = data[
    "Y_state"
].astype(
    np.float32
)

ENV = data[
    "environment"
].astype(
    np.float32
)

DELTA_R_ENV = data[
    "delta_R_environment"
].astype(
    np.float32
)

DELTA_A_ENV = data[
    "delta_amplitude_environment"
].astype(
    np.float32
)

DELTA_STATE_ENV = data[
    "delta_state_environment"
].astype(
    np.float32
)

# ============================================================
# 4. GLOBAL CHECK
# ============================================================

N = X_R.shape[0]
NX = X_R.shape[1]

assert len(metadata) == N

print(
    "[+] Total transitions:",
    N
)

print(
    "[+] Spatial points:",
    NX
)

# ============================================================
# 5. DISCOVER TRAJECTORY GROUPS
#
# Each regime currently corresponds to one complete
# 100-timestep trajectory.
# ============================================================

trajectory_groups = []

for regime in sorted(
    metadata[
        "regime"
    ].unique()
):

    mask = (
        metadata[
            "regime"
        ].to_numpy()
        ==
        regime
    )

    indices = np.where(
        mask
    )[0]

    assert len(indices) == 99, (
        f"Unexpected number of transitions for {regime}: "
        f"{len(indices)}"
    )

    trajectory_groups.append({
        "trajectory_id":
            regime,

        "regime":
            regime,

        "indices":
            indices,

        "num_transitions":
            int(
                len(indices)
            ),
    })

print(
    "\n" + "=" * 100
)

print(
    "TRAJECTORY INVENTORY"
)

print(
    "=" * 100
)

for group in trajectory_groups:

    print(
        f"{group['trajectory_id']:20s}"
        f" | transitions={group['num_transitions']}"
    )

assert len(
    trajectory_groups
) == 5

# ============================================================
# 6. PROTOTYPE DEVELOPMENT SPLIT
#
# Since only 99 transitions exist per trajectory, we use
# contiguous time blocks.
#
# 60% train
# 20% validation
# 20% test
#
# This avoids random temporal leakage.
#
# NOTE:
# This is NOT the final scientific generalization split.
# The final expanded dataset will use independent trajectories.
# ============================================================

TRAIN_FRACTION = 0.60
VAL_FRACTION = 0.20
TEST_FRACTION = 0.20

assert abs(
    TRAIN_FRACTION
    +
    VAL_FRACTION
    +
    TEST_FRACTION
    -
    1.0
) < 1e-12

split_rows = []

train_indices = []
val_indices = []
test_indices = []

for group in trajectory_groups:

    indices = group[
        "indices"
    ]

    n_transitions = len(
        indices
    )

    n_train = int(
        np.floor(
            n_transitions
            *
            TRAIN_FRACTION
        )
    )

    n_val = int(
        np.floor(
            n_transitions
            *
            VAL_FRACTION
        )
    )

    # Remaining samples go to test
    n_test = (
        n_transitions
        -
        n_train
        -
        n_val
    )

    assert n_train > 0
    assert n_val > 0
    assert n_test > 0

    train_part = indices[
        :n_train
    ]

    val_part = indices[
        n_train:
        n_train + n_val
    ]

    test_part = indices[
        n_train + n_val:
    ]

    assert (
        len(train_part)
        ==
        n_train
    )

    assert (
        len(val_part)
        ==
        n_val
    )

    assert (
        len(test_part)
        ==
        n_test
    )

    train_indices.extend(
        train_part.tolist()
    )

    val_indices.extend(
        val_part.tolist()
    )

    test_indices.extend(
        test_part.tolist()
    )

    # --------------------------------------------------------
    # Save per-trajectory split metadata
    # --------------------------------------------------------

    for idx in train_part:

        row = metadata.iloc[
            int(idx)
        ]

        split_rows.append({

            "dataset_index":
                int(idx),

            "regime":
                str(
                    row["regime"]
                ),

            "timestep":
                int(
                    row["timestep"]
                ),

            "split":
                "train",

            "gamma":
                float(
                    row["gamma"]
                ),

            "noise_sigma":
                float(
                    row["noise_sigma"]
                ),
        })

    for idx in val_part:

        row = metadata.iloc[
            int(idx)
        ]

        split_rows.append({

            "dataset_index":
                int(idx),

            "regime":
                str(
                    row["regime"]
                ),

            "timestep":
                int(
                    row["timestep"]
                ),

            "split":
                "validation",

            "gamma":
                float(
                    row["gamma"]
                ),

            "noise_sigma":
                float(
                    row["noise_sigma"]
                ),
        })

    for idx in test_part:

        row = metadata.iloc[
            int(idx)
        ]

        split_rows.append({

            "dataset_index":
                int(idx),

            "regime":
                str(
                    row["regime"]
                ),

            "timestep":
                int(
                    row["timestep"]
                ),

            "split":
                "test",

            "gamma":
                float(
                    row["gamma"]
                ),

            "noise_sigma":
                float(
                    row["noise_sigma"]
                ),
        })

# ============================================================
# 7. CONVERT TO INTEGER ARRAYS
# ============================================================

train_indices = np.asarray(
    train_indices,
    dtype=np.int64
)

val_indices = np.asarray(
    val_indices,
    dtype=np.int64
)

test_indices = np.asarray(
    test_indices,
    dtype=np.int64
)

# ============================================================
# 8. NO-OVERLAP CHECK
# ============================================================

train_set = set(
    train_indices.tolist()
)

val_set = set(
    val_indices.tolist()
)

test_set = set(
    test_indices.tolist()
)

assert train_set.isdisjoint(
    val_set
)

assert train_set.isdisjoint(
    test_set
)

assert val_set.isdisjoint(
    test_set
)

assert (
    len(train_set)
    +
    len(val_set)
    +
    len(test_set)
    ==
    N
)

print(
    "\n[PASS] No dataset-index overlap."
)

# ============================================================
# 9. TEMPORAL LEAKAGE CHECK
#
# Ensure each trajectory's timestep blocks remain contiguous.
# ============================================================

for group in trajectory_groups:

    regime = group[
        "regime"
    ]

    group_indices = group[
        "indices"
    ]

    local_train = [
        i
        for i in group_indices
        if i in train_set
    ]

    local_val = [
        i
        for i in group_indices
        if i in val_set
    ]

    local_test = [
        i
        for i in group_indices
        if i in test_set
    ]

    # Get actual timesteps
    train_times = metadata.iloc[
        local_train
    ][
        "timestep"
    ].to_numpy()

    val_times = metadata.iloc[
        local_val
    ][
        "timestep"
    ].to_numpy()

    test_times = metadata.iloc[
        local_test
    ][
        "timestep"
    ].to_numpy()

    # Contiguous ordering
    if len(train_times) > 1:

        assert np.all(
            np.diff(
                train_times
            )
            == 1
        )

    if len(val_times) > 1:

        assert np.all(
            np.diff(
                val_times
            )
            == 1
        )

    if len(test_times) > 1:

        assert np.all(
            np.diff(
                test_times
            )
            == 1
        )

    assert (
        max(
            train_times
        )
        <
        min(
            val_times
        )
    )

    assert (
        max(
            val_times
        )
        <
        min(
            test_times
        )
    )

print(
    "[PASS] Temporal blocks are contiguous and non-overlapping."
)

# ============================================================
# 10. SPLIT COUNTS
# ============================================================

split_metadata = pd.DataFrame(
    split_rows
)

print(
    "\n" + "=" * 100
)

print(
    "PROTOTYPE SPLIT COUNTS"
)

print(
    "=" * 100
)

print(
    split_metadata[
        "split"
    ].value_counts()
)

print(
    "\nBy regime:"
)

display(
    pd.crosstab(
        split_metadata[
            "regime"
        ],
        split_metadata[
            "split"
        ]
    )
)

# ============================================================
# 11. ACTUAL ARRAY SPLITS
# ============================================================

X_R_train = X_R[
    train_indices
]

Y_R_train = Y_R[
    train_indices
]

X_A_train = X_A[
    train_indices
]

Y_A_train = Y_A[
    train_indices
]

X_Z_train = X_Z[
    train_indices
]

Y_Z_train = Y_Z[
    train_indices
]

X_STATE_train = X_STATE[
    train_indices
]

Y_STATE_train = Y_STATE[
    train_indices
]

ENV_train = ENV[
    train_indices
]

DELTA_R_train = DELTA_R_ENV[
    train_indices
]

DELTA_A_train = DELTA_A_ENV[
    train_indices
]

DELTA_STATE_train = DELTA_STATE_ENV[
    train_indices
]

# ------------------------------------------------------------

X_R_val = X_R[
    val_indices
]

Y_R_val = Y_R[
    val_indices
]

X_A_val = X_A[
    val_indices
]

Y_A_val = Y_A[
    val_indices
]

X_Z_val = X_Z[
    val_indices
]

Y_Z_val = Y_Z[
    val_indices
]

X_STATE_val = X_STATE[
    val_indices
]

Y_STATE_val = Y_STATE[
    val_indices
]

ENV_val = ENV[
    val_indices
]

DELTA_R_val = DELTA_R_ENV[
    val_indices
]

DELTA_A_val = DELTA_A_ENV[
    val_indices
]

DELTA_STATE_val = DELTA_STATE_ENV[
    val_indices
]

# ------------------------------------------------------------

X_R_test = X_R[
    test_indices
]

Y_R_test = Y_R[
    test_indices
]

X_A_test = X_A[
    test_indices
]

Y_A_test = Y_A[
    test_indices
]

X_Z_test = X_Z[
    test_indices
]

Y_Z_test = Y_Z[
    test_indices
]

X_STATE_test = X_STATE[
    test_indices
]

Y_STATE_test = Y_STATE[
    test_indices
]

ENV_test = ENV[
    test_indices
]

DELTA_R_test = DELTA_R_ENV[
    test_indices
]

DELTA_A_test = DELTA_A_ENV[
    test_indices
]

DELTA_STATE_test = DELTA_STATE_ENV[
    test_indices
]

# ============================================================
# 12. SHAPE CHECK
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "SPLIT ARRAY SHAPES"
)

print(
    "=" * 100
)

print(
    "TRAIN"
)

print(
    "  X_R        :",
    X_R_train.shape
)

print(
    "  X_A        :",
    X_A_train.shape
)

print(
    "  Environment:",
    ENV_train.shape
)

print(
    "VALIDATION"
)

print(
    "  X_R        :",
    X_R_val.shape
)

print(
    "  X_A        :",
    X_A_val.shape
)

print(
    "  Environment:",
    ENV_val.shape
)

print(
    "TEST"
)

print(
    "  X_R        :",
    X_R_test.shape
)

print(
    "  X_A        :",
    X_A_test.shape
)

print(
    "  Environment:",
    ENV_test.shape
)

# ============================================================
# 13. FINITE CHECK
# ============================================================

for split_name, arrays_split in {

    "train": [
        X_R_train,
        Y_R_train,
        X_A_train,
        Y_A_train,
        ENV_train,
    ],

    "validation": [
        X_R_val,
        Y_R_val,
        X_A_val,
        Y_A_val,
        ENV_val,
    ],

    "test": [
        X_R_test,
        Y_R_test,
        X_A_test,
        Y_A_test,
        ENV_test,
    ],

}.items():

    for arr in arrays_split:

        assert np.isfinite(
            arr
        ).all()

print(
    "[PASS] Train/validation/test arrays finite."
)

# ============================================================
# 14. ENVIRONMENT COVERAGE
#
# For the prototype development split, every split contains
# all five regimes because temporal blocks are split inside
# every trajectory.
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "ENVIRONMENT COVERAGE"
)

print(
    "=" * 100
)

for split_name in [
    "train",
    "validation",
    "test",
]:

    subset = split_metadata[
        split_metadata[
            "split"
        ]
        ==
        split_name
    ]

    print(
        f"\n{split_name.upper()}"
    )

    print(
        "  gamma:",
        sorted(
            subset[
                "gamma"
            ].unique().tolist()
        )
    )

    print(
        "  noise_sigma:",
        sorted(
            subset[
                "noise_sigma"
            ].unique().tolist()
        )
    )

    print(
        "  regimes:",
        sorted(
            subset[
                "regime"
            ].unique().tolist()
        )
    )

# ============================================================
# 15. REGIME HOLDOUT MANIFEST
#
# This is the important future generalization test.
#
# Example:
#
#   train on four regimes
#   test on the fifth regime
#
# This will later be used for OOD/collapse/generalization.
# ============================================================

regime_holdout_rows = []

all_regimes = sorted(
    metadata[
        "regime"
    ].unique()
)

for held_out in all_regimes:

    test_mask = (
        metadata[
            "regime"
        ].to_numpy()
        ==
        held_out
    )

    train_mask = (
        ~test_mask
    )

    train_ids = np.where(
        train_mask
    )[0]

    test_ids = np.where(
        test_mask
    )[0]

    regime_holdout_rows.append({

        "held_out_regime":
            held_out,

        "train_transitions":
            int(
                len(train_ids)
            ),

        "test_transitions":
            int(
                len(test_ids)
            ),

        "train_regimes":
            ",".join(
                [
                    r
                    for r in all_regimes
                    if r != held_out
                ]
            ),

        "test_regime":
            held_out,
    })

regime_holdout_df = pd.DataFrame(
    regime_holdout_rows
)

print(
    "\n" + "=" * 100
)

print(
    "REGIME-HOLDOUT EVALUATION MANIFEST"
)

print(
    "=" * 100
)

display(
    regime_holdout_df
)

# ============================================================
# 16. SAVE SPLIT DATA
# ============================================================

OUT_DIR = (
    ROOT
    / "results"
    / "dataset_v3"
    / "step9_splits"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# Development split
# ------------------------------------------------------------

np.savez_compressed(

    OUT_DIR
    / "prototype_train_val_test_split.npz",

    train_indices=
        train_indices,

    val_indices=
        val_indices,

    test_indices=
        test_indices,

    X_R_train=
        X_R_train,

    Y_R_train=
        Y_R_train,

    X_amplitude_train=
        X_A_train,

    Y_amplitude_train=
        Y_A_train,

    environment_train=
        ENV_train,

    X_R_val=
        X_R_val,

    Y_R_val=
        Y_R_val,

    X_amplitude_val=
        X_A_val,

    Y_amplitude_val=
        Y_A_val,

    environment_val=
        ENV_val,

    X_R_test=
        X_R_test,

    Y_R_test=
        Y_R_test,

    X_amplitude_test=
        X_A_test,

    Y_amplitude_test=
        Y_A_test,

    environment_test=
        ENV_test,

)

# ============================================================
# 17. SAVE METADATA
# ============================================================

split_metadata_path = (
    OUT_DIR
    / "prototype_split_metadata.csv"
)

split_metadata.to_csv(
    split_metadata_path,
    index=False
)

regime_holdout_path = (
    OUT_DIR
    / "regime_holdout_manifest.csv"
)

regime_holdout_df.to_csv(
    regime_holdout_path,
    index=False
)

# ============================================================
# 18. SAVE JSON SUMMARY
# ============================================================

split_summary = {

    "dataset":
        str(
            DATASET_PATH
        ),

    "prototype_dataset_size":
        int(N),

    "spatial_points":
        int(NX),

    "split_strategy":
        "contiguous temporal blocks within each trajectory",

    "train_fraction":
        TRAIN_FRACTION,

    "validation_fraction":
        VAL_FRACTION,

    "test_fraction":
        TEST_FRACTION,

    "train_transitions":
        int(
            len(train_indices)
        ),

    "validation_transitions":
        int(
            len(val_indices)
        ),

    "test_transitions":
        int(
            len(test_indices)
        ),

    "regime_count":
        len(
            all_regimes
        ),

    "regimes":
        all_regimes,

    "trajectory_level_leakage":
        False,

    "row_overlap":
        False,

    "regime_holdout_available":
        True,

    "note":
        "Prototype development split. Final expanded dataset "
        "must use independent trajectories for scientific "
        "train/validation/test generalization.",

    "artifacts":
        {

            "split_npz":
                str(
                    OUT_DIR
                    / "prototype_train_val_test_split.npz"
                ),

            "split_metadata":
                str(
                    split_metadata_path
                ),

            "regime_holdout_manifest":
                str(
                    regime_holdout_path
                ),
        },
}

summary_path = (
    OUT_DIR
    / "step9_split_summary.json"
)

with open(
    summary_path,
    "w"
) as f:

    json.dump(
        split_summary,
        f,
        indent=2,
        allow_nan=False
    )

# ============================================================
# 19. FINAL
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "STEP 9 COMPLETE"
)

print(
    "=" * 100
)

print(
    "Train transitions:",
    len(
        train_indices
    )
)

print(
    "Validation transitions:",
    len(
        val_indices
    )
)

print(
    "Test transitions:",
    len(
        test_indices
    )
)

print(
    "No row overlap:",
    True
)

print(
    "No temporal leakage:",
    True
)

print(
    "Regime-holdout manifest:",
    True
)

print(
    "\nSplit artifacts:",
    OUT_DIR
)

print(
    "[+] IMPORTANT:"
)

print(
    "This is still the 495-transition prototype."
)

print(
    "Final training requires expanded independent trajectories."
)

print("=" * 100)
