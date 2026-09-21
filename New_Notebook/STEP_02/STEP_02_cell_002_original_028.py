# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 28
# Step            : STEP_02
# Step Heading    : # STEP 2 — SIMULATOR → TRAJECTORY / OBSERVABLE AUDIT
# Step Cell No.   : 2
# ============================================================

# ============================================================
# STEP 2 — SIMULATOR → TRAJECTORY / OBSERVABLE AUDIT
# FULLY CORRECTED PATH VERSION
#
# NO TRAINING
# NO MODEL CHANGES
# NO DATASET MODIFICATION
# ============================================================

import os
import json
import inspect
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
print("STEP 2 — SIMULATOR / TRAJECTORY / OBSERVABLE AUDIT")
print("=" * 100)

# ============================================================
# 2. IMPORT
# ============================================================

from data.pnp_simulator import PhysicsOperatorSimulator
from data.regime_configs import REGIMES

print(
    "[+] PhysicsOperatorSimulator imported."
)

print(
    "[+] Constructor:",
    inspect.signature(
        PhysicsOperatorSimulator
    )
)

print(
    "[+] compute_trajectory:",
    inspect.signature(
        PhysicsOperatorSimulator.compute_trajectory
    )
)

# ============================================================
# 3. FIND PROJECT CONFIG
# ============================================================

candidate_configs = [

    ROOT / "configs" / "config.yaml",
    ROOT / "configs" / "config.yml",

    ROOT / "config.yaml",
    ROOT / "config.yml",

    ROOT / "configs" / "default.yaml",
    ROOT / "configs" / "default.yml",

    ROOT / "configs" / "physics.yaml",
    ROOT / "configs" / "physics.yml",
]

config_path = None

for path in candidate_configs:

    if path.is_file():

        config_path = path
        break

# ------------------------------------------------------------
# If standard locations fail, search project files.
# ------------------------------------------------------------

if config_path is None:

    possible = []

    for pattern in [
        "*.yaml",
        "*.yml",
        "*.json",
    ]:

        possible.extend(
            ROOT.rglob(pattern)
        )

    # Don't accidentally inspect generated experiment files
    possible = [
        p
        for p in possible
        if "results" not in p.parts
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
    ]

    ranked = []

    for path in possible:

        name = path.name.lower()

        score = 0

        for token in [
            "config",
            "physics",
            "pnp",
            "sim",
            "default",
        ]:

            if token in name:

                score += 1

        ranked.append(
            (
                score,
                path
            )
        )

    ranked.sort(
        key=lambda item: item[0],
        reverse=True
    )

    if ranked:

        config_path = ranked[0][1]

# ============================================================
# 4. CONFIG CHECK
# ============================================================

if config_path is None:

    print(
        "\n[WARNING] No config file found automatically."
    )

    print(
        "[+] Searching Python source for an existing "
        "physics configuration dictionary..."
    )

    config_candidates = []

    for py_file in ROOT.rglob("*.py"):

        if (
            ".git" in py_file.parts
            or
            "__pycache__" in py_file.parts
            or
            "results" in py_file.parts
        ):

            continue

        try:

            text = py_file.read_text(
                errors="ignore"
            )

        except Exception:

            continue

        lower = text.lower()

        if (
            '"physics"' in lower
            or
            "'physics'" in lower
        ):

            if (
                '"nx"' in lower
                and
                '"dt"' in lower
            ):

                config_candidates.append(
                    py_file
                )

    print(
        "[+] Candidate Python config files:"
    )

    for p in config_candidates[:20]:

        print(
            "   ",
            p.relative_to(ROOT)
        )

    raise RuntimeError(
        "\nNo existing config file could be located "
        "automatically.\n"
        "I am intentionally stopping instead of inventing "
        "physics parameters."
    )

print(
    "\n[+] Config found:",
    config_path.relative_to(ROOT)
)

# ============================================================
# 5. LOAD CONFIG
# ============================================================

if config_path.suffix.lower() in [
    ".yaml",
    ".yml",
]:

    import yaml

    with open(
        config_path,
        "r"
    ) as f:

        config = yaml.safe_load(
            f
        )

elif config_path.suffix.lower() == ".json":

    with open(
        config_path,
        "r"
    ) as f:

        config = json.load(
            f
        )

else:

    raise RuntimeError(
        "Unsupported config format: "
        + str(config_path.suffix)
    )

assert isinstance(
    config,
    dict
)

assert "physics" in config, (
    "Existing config does not contain "
    "'physics' section."
)

print(
    "\n[+] Physics configuration:"
)

print(
    json.dumps(
        config["physics"],
        indent=2
    )
)

# ============================================================
# 6. CREATE SIMULATOR
# ============================================================

sim = PhysicsOperatorSimulator(
    config
)

print(
    "\n[PASS] Simulator constructed successfully."
)

# ============================================================
# 7. OUTPUT DIRECTORY
# ============================================================

OUT_DIR = (
    ROOT
    / "results"
    / "step2_trajectory_audit"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 8. AVAILABLE REGIMES
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "AVAILABLE REGIMES"
)

print(
    "=" * 100
)

for name, cfg in REGIMES.items():

    print(
        f"{name:20s} | "
        f"noise={cfg['noise']} | "
        f"gamma={cfg['gamma']}"
    )

# ============================================================
# 9. TRAJECTORY ANALYSIS
# ============================================================

trajectory_manifest = []

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
        f"noise_sigma = {noise_sigma}"
    )

    print(
        f"gamma       = {gamma}"
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # Reproducible stochastic realization
    # --------------------------------------------------------

    np.random.seed(
        2026
    )

    # --------------------------------------------------------
    # Actual project API
    # --------------------------------------------------------

    result = sim.compute_trajectory(
        noise_sigma=noise_sigma,
        gamma=gamma
    )

    assert isinstance(
        result,
        tuple
    )

    print(
        "[+] Returned objects:",
        len(result)
    )

    # --------------------------------------------------------
    # Returned object inspection
    # --------------------------------------------------------

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
    # 10. EXPECTED PROJECT RETURN ORDER
    #
    # From the existing simulator:
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
    # ========================================================

    assert len(result) >= 9

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

    # ========================================================
    # 11. SHAPES
    # ========================================================

    T = states_t.shape[0]
    NX = states_t.shape[-1]

    print(
        "\n[+] STRUCTURED TRAJECTORY FIELDS"
    )

    for name, arr in arrays.items():

        print(
            f"    {name:15s}: {arr.shape}"
        )

    assert states_t.ndim == 2
    assert states_t1.ndim == 2

    assert (
        states_t.shape
        ==
        states_t1.shape
    )

    for name in [
        "phi",
        "flux",
        "noise",
        "dissipation",
        "gamma",
        "instability",
    ]:

        assert arrays[name].shape == (
            T,
            NX
        ), (
            f"{name} mismatch: "
            f"{arrays[name].shape}"
        )

    assert (
        entropies_t.shape[0]
        ==
        T
    )

    print(
        "\n[PASS] All fields temporally aligned."
    )

    print(
        "[+] Timesteps:",
        T
    )

    print(
        "[+] Spatial resolution:",
        NX
    )

    # ========================================================
    # 12. FINITE CHECK
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
        "[+] All finite:",
        all_finite
    )

    # ========================================================
    # 13. STATE EVOLUTION
    # ========================================================

    delta_state = (
        states_t1
        -
        states_t
    )

    delta_state_l2 = np.linalg.norm(
        delta_state,
        axis=-1
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "STATE EVOLUTION"
    )

    print(
        "=" * 100
    )

    print(
        "Mean ||state(t+1)-state(t)||:",
        f"{delta_state_l2.mean():.12e}"
    )

    print(
        "Median:",
        f"{np.median(delta_state_l2):.12e}"
    )

    print(
        "Maximum:",
        f"{delta_state_l2.max():.12e}"
    )

    # ========================================================
    # 14. INTERNAL INFORMATION VECTOR
    #
    # Physical observables:
    #
    #   state_t
    #   phi
    #   flux
    #   noise
    #   dissipation
    #   instability
    #
    # gamma remains an external environmental control.
    # ========================================================

    feature_names = [
        "state_t",
        "phi",
        "flux",
        "noise",
        "dissipation",
        "instability",
    ]

    information_vector = np.stack(
        [
            arrays[name]
            for name in feature_names
        ],
        axis=1
    )

    print(
        "\n[+] Internal information vector:",
        information_vector.shape
    )

    # ========================================================
    # 15. NORMALIZATION
    # ========================================================

    normalized = np.zeros_like(
        information_vector,
        dtype=np.float64
    )

    normalization = {}

    for channel_idx, name in enumerate(
        feature_names
    ):

        values = information_vector[
            :,
            channel_idx,
            :
        ]

        mean = float(
            np.mean(values)
        )

        std = float(
            np.std(values)
        )

        if std < 1e-12:

            std = 1.0

        normalized[
            :,
            channel_idx,
            :
        ] = (
            values - mean
        ) / std

        normalization[
            name
        ] = {

            "mean":
                mean,

            "std":
                std,
        }

    # ========================================================
    # 16. INFORMATION-STATE MATRIX
    #
    # R(t,x) = z z^T / Tr(z z^T)
    #
    # Mathematical information representation only.
    # ========================================================

    z = np.transpose(
        normalized,
        (
            0,
            2,
            1
        )
    )

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

    print(
        "[+] R(t) shape:",
        R.shape
    )

    # ========================================================
    # 17. MATRIX VALIDATION
    # ========================================================

    R_trace = np.trace(
        R,
        axis1=-2,
        axis2=-1
    )

    trace_error = float(
        np.max(
            np.abs(
                R_trace - 1.0
            )
        )
    )

    symmetry_error = float(
        np.max(
            np.abs(
                R
                -
                np.swapaxes(
                    R,
                    -1,
                    -2
                )
            )
        )
    )

    # Select a small subset for eigenvalue calculation
    subset = R[
        :min(T, 32),
        ::max(1, NX // 16),
        :,
        :
    ]

    eigenvalues = np.linalg.eigvalsh(
        subset
    )

    min_eigenvalue = float(
        np.min(
            eigenvalues
        )
    )

    print(
        "\n[+] R trace error:",
        f"{trace_error:.12e}"
    )

    print(
        "[+] R symmetry error:",
        f"{symmetry_error:.12e}"
    )

    print(
        "[+] Minimum eigenvalue:",
        f"{min_eigenvalue:.12e}"
    )

    assert trace_error < 1e-6
    assert symmetry_error < 1e-6
    assert min_eigenvalue >= -1e-8

    print(
        "[PASS] R(t) numerical sanity checks passed."
    )

    # ========================================================
    # 18. INFORMATION EVOLUTION
    # ========================================================

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

    mean_R_change = float(
        np.mean(
            R_delta_norm
        )
    )

    median_R_change = float(
        np.median(
            R_delta_norm
        )
    )

    max_R_change = float(
        np.max(
            R_delta_norm
        )
    )

    print(
        "\n" + "=" * 100
    )

    print(
        "INTERNAL INFORMATION EVOLUTION"
    )

    print(
        "=" * 100
    )

    print(
        "Mean ||R(t+1)-R(t)||:",
        f"{mean_R_change:.12e}"
    )

    print(
        "Median:",
        f"{median_R_change:.12e}"
    )

    print(
        "Maximum:",
        f"{max_R_change:.12e}"
    )

    # ========================================================
    # 19. ENVIRONMENT COUPLING
    # ========================================================

    gamma_mean_per_t = np.mean(
        gammas_t,
        axis=-1
    )

    R_change_per_t = np.mean(
        R_delta_norm,
        axis=-1
    )

    gamma_change_corr = np.nan

    if (
        len(gamma_mean_per_t) > 1
        and
        np.std(
            gamma_mean_per_t[1:]
        ) > 1e-12
        and
        np.std(
            R_change_per_t
        ) > 1e-12
    ):

        gamma_change_corr = float(
            np.corrcoef(
                gamma_mean_per_t[1:],
                R_change_per_t
            )[0, 1]
        )

    print(
        "\n" + "=" * 100
    )

    print(
        "ENVIRONMENT → INFORMATION RESPONSE"
    )

    print(
        "=" * 100
    )

    print(
        "Gamma trajectory min:",
        float(
            np.min(
                gamma_mean_per_t
            )
        )
    )

    print(
        "Gamma trajectory max:",
        float(
            np.max(
                gamma_mean_per_t
            )
        )
    )

    print(
        "corr(gamma, ||ΔR||):",
        f"{gamma_change_corr:+.8f}"
    )

    # ========================================================
    # 20. SAVE TRAJECTORY
    # ========================================================

    trajectory_file = (
        OUT_DIR
        / f"{regime_name}_trajectory.npz"
    )

    np.savez_compressed(

        trajectory_file,

        state_t=states_t,

        state_t1=states_t1,

        phi=phis_t,

        flux=fluxes_t,

        noise=noises_t,

        dissipation=dissipations_t,

        gamma=gammas_t,

        entropy=entropies_t,

        instability=instabilities_t,

        information_state_R=
            R.astype(
                np.float32
            ),
    )

    print(
        "[+] Saved trajectory:",
        trajectory_file
    )

    # ========================================================
    # 21. MANIFEST
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

        "mean_state_transition_l2":
            float(
                np.mean(
                    delta_state_l2
                )
            ),

        "max_state_transition_l2":
            float(
                np.max(
                    delta_state_l2
                )
            ),

        "mean_information_change":
            mean_R_change,

        "median_information_change":
            median_R_change,

        "max_information_change":
            max_R_change,

        "gamma_information_change_correlation":
            gamma_change_corr,

        "trajectory_file":
            str(
                trajectory_file
            ),
    })

# ============================================================
# 22. SAVE MANIFESTS
# ============================================================

manifest_df = pd.DataFrame(
    trajectory_manifest
)

MANIFEST_CSV = (
    OUT_DIR
    / "trajectory_manifest.csv"
)

MANIFEST_JSON = (
    OUT_DIR
    / "trajectory_manifest.json"
)

manifest_df.to_csv(
    MANIFEST_CSV,
    index=False
)

with open(
    MANIFEST_JSON,
    "w"
) as f:

    json.dump(
        trajectory_manifest,
        f,
        indent=2
    )

print(
    "\n" + "=" * 100
)

print(
    "STEP 2 COMPLETE"
)

print(
    "[+] Successful regimes:",
    len(trajectory_manifest)
)

print(
    "[+] Manifest CSV:",
    MANIFEST_CSV
)

print(
    "[+] Manifest JSON:",
    MANIFEST_JSON
)

print(
    "[+] Output:",
    OUT_DIR
)

print("=" * 100)
