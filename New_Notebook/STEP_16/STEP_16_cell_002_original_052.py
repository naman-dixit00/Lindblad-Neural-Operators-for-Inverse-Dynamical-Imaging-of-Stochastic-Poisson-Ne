# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 52
# Step            : STEP_16
# Step Heading    : # STEP 16 — FINAL TRAJECTORY-LEVEL TRAIN / VAL / TEST SPLIT
# Step Cell No.   : 2
# ============================================================

# ============================================================
# STEP 16 — FINAL TRAJECTORY-LEVEL TRAIN / VAL / TEST SPLIT
#
# DATA:
#   Dataset V3 Expanded
#   100 independent trajectories
#   9,900 transitions
#
# SPLIT:
#   70 trajectories -> TRAIN
#   15 trajectories -> VALIDATION
#   15 trajectories -> TEST
#
# CRITICAL:
#   Split is performed at TRAJECTORY level.
#   No trajectory can appear in more than one split.
#
# GOAL:
#   Prevent temporal leakage and provide a clean final
#   generalization benchmark for FNO vs repaired LNO.
#
# ============================================================

import os
import json
import random

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
print("STEP 16 — FINAL TRAJECTORY-LEVEL TRAIN / VAL / TEST SPLIT")
print("=" * 100)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

SEED = 2026

random.seed(
    SEED
)

np.random.seed(
    SEED
)


# ============================================================
# 3. PATHS
# ============================================================

DATASET_PATH = (
    ROOT
    / "results"
    / "dataset_v3_expanded"
    / "environment_conditioned_lno_dataset_v3_expanded.npz"
)

METADATA_PATH = (
    ROOT
    / "results"
    / "dataset_v3_expanded"
    / "dataset_v3_expanded_metadata.csv"
)

TRAJECTORY_METADATA_PATH = (
    ROOT
    / "results"
    / "dataset_v3_expanded"
    / "trajectory_metadata.csv"
)

OUT_DIR = (
    ROOT
    / "results"
    / "step16_final_splits"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

assert DATASET_PATH.is_file(), (
    f"Expanded dataset missing:\n{DATASET_PATH}"
)

assert METADATA_PATH.is_file(), (
    f"Transition metadata missing:\n{METADATA_PATH}"
)

assert TRAJECTORY_METADATA_PATH.is_file(), (
    f"Trajectory metadata missing:\n{TRAJECTORY_METADATA_PATH}"
)


# ============================================================
# 4. LOAD
# ============================================================

data = np.load(
    DATASET_PATH
)

metadata = pd.read_csv(
    METADATA_PATH
)

trajectory_metadata = pd.read_csv(
    TRAJECTORY_METADATA_PATH
)

print(
    "[+] Dataset loaded."
)

print(
    "[+] Dataset:",
    DATASET_PATH
)

print(
    "[+] Transition metadata rows:",
    len(metadata)
)

print(
    "[+] Trajectory metadata rows:",
    len(trajectory_metadata)
)


# ============================================================
# 5. BASIC DATASET CHECK
# ============================================================

X_R = data[
    "X_R"
]

Y_R = data[
    "Y_R"
]

X_A = data[
    "X_amplitude"
]

Y_A = data[
    "Y_amplitude"
]

X_Z = data[
    "X_information_z"
]

Y_Z = data[
    "Y_information_z"
]

X_STATE = data[
    "X_state"
]

Y_STATE = data[
    "Y_state"
]

ENV = data[
    "environment"
]

DELTA_R_ENV = data[
    "delta_R_environment"
]

DELTA_A_ENV = data[
    "delta_amplitude_environment"
]

DELTA_STATE_ENV = data[
    "delta_state_environment"
]

N = X_R.shape[
    0
]

NX = X_R.shape[
    1
]

DIM = X_R.shape[
    2
]

assert X_R.shape == (
    N,
    NX,
    6,
    6
)

assert Y_R.shape == X_R.shape

assert X_A.shape == (
    N,
    NX
)

assert Y_A.shape == (
    N,
    NX
)

assert X_Z.shape == (
    N,
    NX,
    6
)

assert Y_Z.shape == (
    N,
    NX,
    6
)

assert ENV.shape == (
    N,
    NX,
    2
)

assert metadata.shape[
    0
] == N

print(
    "\n[+] Total transitions:",
    N
)

print(
    "[+] Spatial points:",
    NX
)

print(
    "[+] Information dimension:",
    DIM
)


# ============================================================
# 6. TRAJECTORY INVENTORY
# ============================================================

required_metadata_columns = [

    "dataset_index",
    "trajectory_id",
    "regime",
    "trajectory_index",
    "timestep",
    "noise_seed",
    "gamma",
    "noise_sigma",
]

missing_columns = [
    c
    for c in required_metadata_columns
    if c not in metadata.columns
]

assert not missing_columns, (
    f"Missing metadata columns: {missing_columns}"
)

trajectory_ids = (
    metadata[
        "trajectory_id"
    ]
    .astype(str)
    .unique()
)

trajectory_ids = sorted(
    trajectory_ids
)

num_trajectories = len(
    trajectory_ids
)

print(
    "\n" + "=" * 100
)

print(
    "TRAJECTORY INVENTORY"
)

print(
    "=" * 100
)

print(
    "[+] Unique trajectories:",
    num_trajectories
)

assert num_trajectories == 100


# ============================================================
# 7. VERIFY 99 TRANSITIONS PER TRAJECTORY
# ============================================================

trajectory_counts = (
    metadata
    .groupby(
        "trajectory_id"
    )
    .size()
)

print(
    "\nTransitions per trajectory:"
)

print(
    trajectory_counts
    .value_counts()
    .sort_index()
)

assert (
    trajectory_counts.min()
    ==
    99
)

assert (
    trajectory_counts.max()
    ==
    99
)

assert (
    trajectory_counts.shape[
        0
    ]
    ==
    100
)

print(
    "[PASS] Every trajectory contains exactly 99 transitions."
)


# ============================================================
# 8. VERIFY TIMESTEPS
# ============================================================

timestep_problems = []

for trajectory_id, group in metadata.groupby(
    "trajectory_id",
    sort=False
):

    timesteps = (
        group[
            "timestep"
        ]
        .astype(int)
        .to_numpy()
    )

    expected = np.arange(
        99,
        dtype=int
    )

    if not np.array_equal(
        np.sort(timesteps),
        expected
    ):

        timestep_problems.append(
            trajectory_id
        )

assert not timestep_problems, (
    f"Bad timestep sequence: {timestep_problems[:10]}"
)

print(
    "[PASS] All trajectories contain contiguous timesteps 0..98."
)


# ============================================================
# 9. REGIME INVENTORY
# ============================================================

regimes = sorted(
    metadata[
        "regime"
    ]
    .astype(str)
    .unique()
)

print(
    "\n" + "=" * 100
)

print(
    "REGIME INVENTORY"
)

print(
    "=" * 100
)

display(
    metadata[
        [
            "trajectory_id",
            "regime",
            "trajectory_index",
            "gamma",
            "noise_sigma",
        ]
    ]
    .drop_duplicates(
        "trajectory_id"
    )
    .sort_values(
        [
            "regime",
            "trajectory_index",
        ]
    )
    .reset_index(
        drop=True
    )
)

print(
    "\nRegimes:",
    regimes
)

assert regimes == [
    "collapse",
    "heavy_dissipation",
    "low_noise",
    "metastable",
    "stochastic",
]

# ============================================================
# 10. TRAJECTORY-LEVEL STRATIFIED SPLIT
#
# Each regime:
#   20 trajectories
#
# Split:
#   14 train
#    3 validation
#    3 test
#
# Total:
#   70 / 15 / 15 trajectories
# ============================================================

TRAIN_TRAJECTORIES = []
VAL_TRAJECTORIES = []
TEST_TRAJECTORIES = []

split_rows = []

trajectory_table = (
    metadata[
        [
            "trajectory_id",
            "regime",
            "trajectory_index",
            "gamma",
            "noise_sigma",
            "noise_seed",
        ]
    ]
    .drop_duplicates(
        "trajectory_id"
    )
    .copy()
)

# deterministic regime-wise selection
for regime in regimes:

    regime_table = (
        trajectory_table[
            trajectory_table[
                "regime"
            ]
            ==
            regime
        ]
        .sort_values(
            "trajectory_index"
        )
        .reset_index(
            drop=True
        )
    )

    assert len(
        regime_table
    ) == 20

    # fixed deterministic permutation within each regime
    local_rng = np.random.default_rng(
        SEED
        +
        regimes.index(
            regime
        )
    )

    perm = local_rng.permutation(
        len(
            regime_table
        )
    )

    shuffled = (
        regime_table
        .iloc[
            perm
        ]
        .reset_index(
            drop=True
        )
    )

    train_part = shuffled.iloc[
        :14
    ]

    val_part = shuffled.iloc[
        14:17
    ]

    test_part = shuffled.iloc[
        17:20
    ]

    TRAIN_TRAJECTORIES.extend(
        train_part[
            "trajectory_id"
        ]
        .tolist()
    )

    VAL_TRAJECTORIES.extend(
        val_part[
            "trajectory_id"
        ]
        .tolist()
    )

    TEST_TRAJECTORIES.extend(
        test_part[
            "trajectory_id"
        ]
        .tolist()
    )

# ============================================================
# 11. SPLIT DISJOINTNESS
# ============================================================

train_set = set(
    TRAIN_TRAJECTORIES
)

val_set = set(
    VAL_TRAJECTORIES
)

test_set = set(
    TEST_TRAJECTORIES
)

assert len(
    train_set
) == 70

assert len(
    val_set
) == 15

assert len(
    test_set
) == 15

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
    len(
        train_set
        |
        val_set
        |
        test_set
    )
    ==
    100
)

print(
    "\n[PASS] Trajectory sets are completely disjoint."
)


# ============================================================
# 12. BUILD TRANSITION INDICES
# ============================================================

train_mask = metadata[
    "trajectory_id"
].astype(str).isin(
    train_set
)

val_mask = metadata[
    "trajectory_id"
].astype(str).isin(
    val_set
)

test_mask = metadata[
    "trajectory_id"
].astype(str).isin(
    test_set
)

train_indices = (
    metadata.index[
        train_mask
    ]
    .to_numpy(
        dtype=np.int64
    )
)

val_indices = (
    metadata.index[
        val_mask
    ]
    .to_numpy(
        dtype=np.int64
    )
)

test_indices = (
    metadata.index[
        test_mask
    ]
    .to_numpy(
        dtype=np.int64
    )
)

assert len(
    train_indices
) == 70 * 99

assert len(
    val_indices
) == 15 * 99

assert len(
    test_indices
) == 15 * 99

assert (
    len(
        train_indices
    )
    +
    len(
        val_indices
    )
    +
    len(
        test_indices
    )
    ==
    N
)

print(
    "\n" + "=" * 100
)

print(
    "FINAL SPLIT COUNTS"
)

print(
    "=" * 100
)

print(
    "Train trajectories:",
    len(train_set)
)

print(
    "Validation trajectories:",
    len(val_set)
)

print(
    "Test trajectories:",
    len(test_set)
)

print(
    "\nTrain transitions:",
    len(train_indices)
)

print(
    "Validation transitions:",
    len(val_indices)
)

print(
    "Test transitions:",
    len(test_indices)
)


# ============================================================
# 13. REGIME BALANCE
# ============================================================

def regime_split_counts(
    indices
):

    return (
        metadata.iloc[
            indices
        ]
        .groupby(
            "regime"
        )
        .size()
        .reindex(
            regimes,
            fill_value=0
        )
    )


train_regime_counts = (
    regime_split_counts(
        train_indices
    )
)

val_regime_counts = (
    regime_split_counts(
        val_indices
    )
)

test_regime_counts = (
    regime_split_counts(
        test_indices
    )
)

regime_balance = pd.DataFrame({

    "train_trajectories":
        [
            sum(
                metadata.iloc[
                    train_indices
                ][
                    "regime"
                ]
                ==
                r
            )
            // 99
            for r in regimes
        ],

    "validation_trajectories":
        [
            sum(
                metadata.iloc[
                    val_indices
                ][
                    "regime"
                ]
                ==
                r
            )
            // 99
            for r in regimes
        ],

    "test_trajectories":
        [
            sum(
                metadata.iloc[
                    test_indices
                ][
                    "regime"
                ]
                ==
                r
            )
            // 99
            for r in regimes
        ],
}, index=regimes)

print(
    "\n" + "=" * 100
)

print(
    "REGIME BALANCE"
)

print(
    "=" * 100
)

display(
    regime_balance
)

assert (
    regime_balance[
        "train_trajectories"
    ]
    ==
    14
).all()

assert (
    regime_balance[
        "validation_trajectories"
    ]
    ==
    3
).all()

assert (
    regime_balance[
        "test_trajectories"
    ]
    ==
    3
).all()

print(
    "[PASS] Each regime has 14/3/3 trajectories."
)


# ============================================================
# 14. ENVIRONMENT COVERAGE
# ============================================================

def environment_report(
    indices
):

    subset = metadata.iloc[
        indices
    ]

    return {

        "gamma_unique":
            sorted(
                subset[
                    "gamma"
                ]
                .round(6)
                .unique()
                .tolist()
            ),

        "noise_sigma_unique":
            sorted(
                subset[
                    "noise_sigma"
                ]
                .round(6)
                .unique()
                .tolist()
            ),

        "regimes":
            sorted(
                subset[
                    "regime"
                ]
                .unique()
                .tolist()
            ),
    }


train_env = environment_report(
    train_indices
)

val_env = environment_report(
    val_indices
)

test_env = environment_report(
    test_indices
)

print(
    "\n" + "=" * 100
)

print(
    "ENVIRONMENT COVERAGE"
)

print(
    "=" * 100
)

print(
    "TRAIN:",
    train_env
)

print(
    "VALIDATION:",
    val_env
)

print(
    "TEST:",
    test_env
)

assert len(
    train_env[
        "gamma_unique"
    ]
) == 5

assert len(
    val_env[
        "gamma_unique"
    ]
) == 5

assert len(
    test_env[
        "gamma_unique"
    ]
) == 5

assert len(
    train_env[
        "noise_sigma_unique"
    ]
) == 5

assert len(
    val_env[
        "noise_sigma_unique"
    ]
) == 5

assert len(
    test_env[
        "noise_sigma_unique"
    ]
) == 5

print(
    "[PASS] All splits cover all five environment regimes."
)


# ============================================================
# 15. ACTUAL TRAJECTORY LEAKAGE CHECK
# ============================================================

def get_trajectory_set(
    indices
):

    return set(
        metadata.iloc[
            indices
        ][
            "trajectory_id"
        ]
        .astype(str)
        .unique()
    )

assert get_trajectory_set(
    train_indices
) == train_set

assert get_trajectory_set(
    val_indices
) == val_set

assert get_trajectory_set(
    test_indices
) == test_set

print(
    "\n[PASS] Transition indices exactly match trajectory sets."
)


# ============================================================
# 16. TEMPORAL ORDER CHECK
# ============================================================

for split_name, indices in {

    "train":
        train_indices,

    "validation":
        val_indices,

    "test":
        test_indices,

}.items():

    subset = metadata.iloc[
        indices
    ]

    for trajectory_id, group in subset.groupby(
        "trajectory_id"
    ):

        timesteps = (
            group[
                "timestep"
            ]
            .sort_values()
            .to_numpy()
        )

        assert np.array_equal(
            timesteps,
            np.arange(
                99
            )
        )

print(
    "[PASS] Each trajectory remains temporally complete within its split."
)


# ============================================================
# 17. BUILD SPLIT ARRAYS
# ============================================================

def gather(
    indices
):

    return {

        "X_R":
            X_R[
                indices
            ],

        "Y_R":
            Y_R[
                indices
            ],

        "X_amplitude":
            X_A[
                indices
            ],

        "Y_amplitude":
            Y_A[
                indices
            ],

        "X_information_z":
            X_Z[
                indices
            ],

        "Y_information_z":
            Y_Z[
                indices
            ],

        "X_state":
            X_STATE[
                indices
            ],

        "Y_state":
            Y_STATE[
                indices
            ],

        "environment":
            ENV[
                indices
            ],

        "delta_R_environment":
            DELTA_R_ENV[
                indices
            ],

        "delta_amplitude_environment":
            DELTA_A_ENV[
                indices
            ],

        "delta_state_environment":
            DELTA_STATE_ENV[
                indices
            ],
    }


train = gather(
    train_indices
)

val = gather(
    val_indices
)

test = gather(
    test_indices
)


# ============================================================
# 18. FINITE CHECK
# ============================================================

for split_name, split_data in {

    "train":
        train,

    "validation":
        val,

    "test":
        test,

}.items():

    for name, arr in split_data.items():

        ok = bool(
            np.isfinite(
                arr
            ).all()
        )

        assert ok, (
            f"Non-finite {name} in {split_name}"
        )

print(
    "[PASS] All final split tensors are finite."
)


# ============================================================
# 19. SAVE FINAL SPLIT NPZ
# ============================================================

SPLIT_PATH = (
    OUT_DIR
    / "final_trajectory_level_split.npz"
)

np.savez_compressed(

    SPLIT_PATH,

    train_indices=
        train_indices,

    val_indices=
        val_indices,

    test_indices=
        test_indices,

    # TRAIN
    X_R_train=
        train["X_R"],

    Y_R_train=
        train["Y_R"],

    X_amplitude_train=
        train["X_amplitude"],

    Y_amplitude_train=
        train["Y_amplitude"],

    X_information_z_train=
        train["X_information_z"],

    Y_information_z_train=
        train["Y_information_z"],

    X_state_train=
        train["X_state"],

    Y_state_train=
        train["Y_state"],

    environment_train=
        train["environment"],

    delta_R_environment_train=
        train[
            "delta_R_environment"
        ],

    delta_amplitude_environment_train=
        train[
            "delta_amplitude_environment"
        ],

    delta_state_environment_train=
        train[
            "delta_state_environment"
        ],

    # VALIDATION
    X_R_val=
        val["X_R"],

    Y_R_val=
        val["Y_R"],

    X_amplitude_val=
        val["X_amplitude"],

    Y_amplitude_val=
        val["Y_amplitude"],

    X_information_z_val=
        val["X_information_z"],

    Y_information_z_val=
        val["Y_information_z"],

    X_state_val=
        val["X_state"],

    Y_state_val=
        val["Y_state"],

    environment_val=
        val["environment"],

    delta_R_environment_val=
        val[
            "delta_R_environment"
        ],

    delta_amplitude_environment_val=
        val[
            "delta_amplitude_environment"
        ],

    delta_state_environment_val=
        val[
            "delta_state_environment"
        ],

    # TEST
    X_R_test=
        test["X_R"],

    Y_R_test=
        test["Y_R"],

    X_amplitude_test=
        test["X_amplitude"],

    Y_amplitude_test=
        test["Y_amplitude"],

    X_information_z_test=
        test["X_information_z"],

    Y_information_z_test=
        test["Y_information_z"],

    X_state_test=
        test["X_state"],

    Y_state_test=
        test["Y_state"],

    environment_test=
        test["environment"],

    delta_R_environment_test=
        test[
            "delta_R_environment"
        ],

    delta_amplitude_environment_test=
        test[
            "delta_amplitude_environment"
        ],

    delta_state_environment_test=
        test[
            "delta_state_environment"
        ],
)

print(
    "\n[+] Final split saved:",
    SPLIT_PATH
)


# ============================================================
# 20. SAVE TRANSITION MANIFEST
# ============================================================

split_label = np.full(
    N,
    "",
    dtype=object
)

split_label[
    train_indices
] = "train"

split_label[
    val_indices
] = "validation"

split_label[
    test_indices
] = "test"

final_transition_manifest = metadata.copy()

final_transition_manifest[
    "split"
] = split_label

MANIFEST_PATH = (
    OUT_DIR
    / "final_transition_manifest.csv"
)

final_transition_manifest.to_csv(
    MANIFEST_PATH,
    index=False
)


# ============================================================
# 21. SAVE TRAJECTORY MANIFEST
# ============================================================

trajectory_manifest = trajectory_table.copy()

trajectory_manifest[
    "split"
] = "unassigned"

trajectory_manifest.loc[
    trajectory_manifest[
        "trajectory_id"
    ].isin(
        train_set
    ),
    "split"
] = "train"

trajectory_manifest.loc[
    trajectory_manifest[
        "trajectory_id"
    ].isin(
        val_set
    ),
    "split"
] = "validation"

trajectory_manifest.loc[
    trajectory_manifest[
        "trajectory_id"
    ].isin(
        test_set
    ),
    "split"
] = "test"

TRAJECTORY_MANIFEST_PATH = (
    OUT_DIR
    / "final_trajectory_manifest.csv"
)

trajectory_manifest.to_csv(
    TRAJECTORY_MANIFEST_PATH,
    index=False
)


# ============================================================
# 22. SUMMARY JSON
# ============================================================

summary = {

    "step":
        16,

    "dataset":
        str(
            DATASET_PATH
        ),

    "total_transitions":
        int(N),

    "total_trajectories":
        int(num_trajectories),

    "spatial_points":
        int(NX),

    "information_dimension":
        int(DIM),

    "seed":
        int(SEED),

    "split_policy":
        "trajectory-level stratified split",

    "train_trajectories":
        int(len(train_set)),

    "validation_trajectories":
        int(len(val_set)),

    "test_trajectories":
        int(len(test_set)),

    "train_transitions":
        int(len(train_indices)),

    "validation_transitions":
        int(len(val_indices)),

    "test_transitions":
        int(len(test_indices)),

    "regime_trajectories_per_split":
        {
            regime: {

                "train":
                    int(
                        regime_balance.loc[
                            regime,
                            "train_trajectories"
                        ]
                    ),

                "validation":
                    int(
                        regime_balance.loc[
                            regime,
                            "validation_trajectories"
                        ]
                    ),

                "test":
                    int(
                        regime_balance.loc[
                            regime,
                            "test_trajectories"
                        ]
                    ),
            }

            for regime in regimes
        },

    "trajectory_overlap":
        False,

    "temporal_leakage":
        False,

    "environment_all_splits":
        True,

    "outputs":
        {

            "split_npz":
                str(
                    SPLIT_PATH
                ),

            "transition_manifest":
                str(
                    MANIFEST_PATH
                ),

            "trajectory_manifest":
                str(
                    TRAJECTORY_MANIFEST_PATH
                ),
        },
}

SUMMARY_PATH = (
    OUT_DIR
    / "step16_summary.json"
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
# 23. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "STEP 16 COMPLETE — FINAL TRAJECTORY-LEVEL SPLIT"
)

print(
    "=" * 100
)

print(
    "[+] Train trajectories:",
    len(train_set)
)

print(
    "[+] Validation trajectories:",
    len(val_set)
)

print(
    "[+] Test trajectories:",
    len(test_set)
)

print(
    "[+] Train transitions:",
    len(train_indices)
)

print(
    "[+] Validation transitions:",
    len(val_indices)
)

print(
    "[+] Test transitions:",
    len(test_indices)
)

print(
    "[+] Total transitions:",
    N
)

print(
    "[+] Trajectory overlap:",
    False
)

print(
    "[+] Temporal leakage:",
    False
)

print(
    "[+] All five regimes in every split:",
    True
)

print(
    "[+] Split:",
    SPLIT_PATH
)

print(
    "[+] Transition manifest:",
    MANIFEST_PATH
)

print(
    "[+] Trajectory manifest:",
    TRAJECTORY_MANIFEST_PATH
)

print(
    "[+] Summary:",
    SUMMARY_PATH
)

print(
    "\n[READY] Final FNO/LNO training can now use this split."
)

print(
    "[IMPORTANT] This is the 9,900-transition expanded dataset."
)

print("=" * 100)
