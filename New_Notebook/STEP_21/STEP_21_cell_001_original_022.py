# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 22
# Step            : STEP_21
# Step Heading    : # STEP 21 — FINAL FIGURES + CSV/JSON SUBMISSION BUNDLE
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 21 — FINAL FIGURES + CSV/JSON SUBMISSION BUNDLE
#
# Purpose:
#   Collect all verified final experiment artifacts into one
#   clean directory and ZIP archive.
#
# NO TRAINING
# NO MODEL MODIFICATION
# NO CHECKPOINT MODIFICATION
# ============================================================

import os
import shutil
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

print("=" * 90)
print("STEP 21 — FINAL FIGURES + CSV/JSON BUNDLE")
print("=" * 90)

# ============================================================
# 1. ROOT
# ============================================================

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

print(
    "[+] Project root:",
    ROOT
)

# ============================================================
# 2. BUNDLE DIRECTORY
# ============================================================

BUNDLE_DIR = (
    ROOT
    / "results"
    / "FINAL_SUBMISSION_BUNDLE"
)

FIGURES_DIR = (
    BUNDLE_DIR
    / "figures"
)

TABLES_DIR = (
    BUNDLE_DIR
    / "tables"
)

METRICS_DIR = (
    BUNDLE_DIR
    / "metrics"
)

MANIFEST_DIR = (
    BUNDLE_DIR
    / "manifest"
)

for directory in [
    BUNDLE_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    METRICS_DIR,
    MANIFEST_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

print(
    "[+] Bundle directory:",
    BUNDLE_DIR
)

# ============================================================
# 3. ARTIFACT REGISTRY
#
# source = existing experiment artifact
# destination = clean submission location
# ============================================================

ARTIFACTS = {

    # --------------------------------------------------------
    # FINAL LNO TEST
    # --------------------------------------------------------

    "final_lno_metrics":
        (
            ROOT
            / "results"
            / "final_lno"
            / "original_lno"
            / "lno_test_metrics.json",
            METRICS_DIR
            / "lno_test_metrics.json"
        ),

    "final_lno_summary":
        (
            ROOT
            / "results"
            / "final_lno"
            / "original_lno"
            / "lno_test_summary.json",
            METRICS_DIR
            / "lno_test_summary.json"
        ),

    # --------------------------------------------------------
    # FNO vs LNO
    # --------------------------------------------------------

    "fno_lno_comparison_csv":
        (
            ROOT
            / "results"
            / "final_comparison"
            / "fno_vs_lno_final.csv",
            TABLES_DIR
            / "fno_vs_lno_final.csv"
        ),

    "fno_lno_comparison_json":
        (
            ROOT
            / "results"
            / "final_comparison"
            / "fno_vs_lno_final.json",
            METRICS_DIR
            / "fno_vs_lno_final.json"
        ),

    # --------------------------------------------------------
    # FINAL OOD
    # --------------------------------------------------------

    "final_ood_csv":
        (
            ROOT
            / "results"
            / "final_ood"
            / "final_fno_vs_lno_ood_sigma_sweep.csv",
            TABLES_DIR
            / "final_fno_vs_lno_ood_sigma_sweep.csv"
        ),

    "final_ood_json":
        (
            ROOT
            / "results"
            / "final_ood"
            / "final_fno_vs_lno_ood_sigma_sweep.json",
            METRICS_DIR
            / "final_fno_vs_lno_ood_sigma_sweep.json"
        ),

    # --------------------------------------------------------
    # LONG HORIZON
    # --------------------------------------------------------

    "long_horizon_csv":
        (
            ROOT
            / "results"
            / "final_long_horizon"
            / "long_horizon_rollout.csv",
            TABLES_DIR
            / "long_horizon_rollout.csv"
        ),

    "long_horizon_json":
        (
            ROOT
            / "results"
            / "final_long_horizon"
            / "long_horizon_summary.json",
            METRICS_DIR
            / "long_horizon_summary.json"
        ),

    # --------------------------------------------------------
    # LONG HORIZON FIGURES
    # --------------------------------------------------------

    "fig_energy":
        (
            ROOT
            / "results"
            / "final_long_horizon"
            / "fig_energy_vs_step.png",
            FIGURES_DIR
            / "fig_energy_vs_step.png"
        ),

    "fig_energy_drift":
        (
            ROOT
            / "results"
            / "final_long_horizon"
            / "fig_energy_drift.png",
            FIGURES_DIR
            / "fig_energy_drift.png"
        ),

    "fig_fno_lno_separation":
        (
            ROOT
            / "results"
            / "final_long_horizon"
            / "fig_fno_lno_separation.png",
            FIGURES_DIR
            / "fig_fno_lno_separation.png"
        ),

    # --------------------------------------------------------
    # SPECTRAL STABILITY
    # --------------------------------------------------------

    "spectral_timestep_csv":
        (
            ROOT
            / "results"
            / "final_spectral_stability"
            / "spectral_stability_vs_timestep.csv",
            TABLES_DIR
            / "spectral_stability_vs_timestep.csv"
        ),

    "spectral_mode_csv":
        (
            ROOT
            / "results"
            / "final_spectral_stability"
            / "spectral_energy_vs_mode.csv",
            TABLES_DIR
            / "spectral_energy_vs_mode.csv"
        ),

    "spectral_summary_json":
        (
            ROOT
            / "results"
            / "final_spectral_stability"
            / "spectral_stability_summary.json",
            METRICS_DIR
            / "spectral_stability_summary.json"
        ),

    # --------------------------------------------------------
    # SPECTRAL FIGURES
    # --------------------------------------------------------

    "fig_spectral_mode":
        (
            ROOT
            / "results"
            / "final_spectral_stability"
            / "fig_mean_spectral_energy_vs_mode.png",
            FIGURES_DIR
            / "fig_mean_spectral_energy_vs_mode.png"
        ),

    "fig_high_frequency":
        (
            ROOT
            / "results"
            / "final_spectral_stability"
            / "fig_high_frequency_fraction_vs_timestep.png",
            FIGURES_DIR
            / "fig_high_frequency_fraction_vs_timestep.png"
        ),

    "fig_total_spectral":
        (
            ROOT
            / "results"
            / "final_spectral_stability"
            / "fig_total_spectral_energy_vs_timestep.png",
            FIGURES_DIR
            / "fig_total_spectral_energy_vs_timestep.png"
        ),
}

# ============================================================
# 4. COPY ARTIFACTS
# ============================================================

copied = []
missing = []
errors = []

print("\n" + "=" * 90)
print("COLLECTING ARTIFACTS")
print("=" * 90)

for name, (
    source,
    destination
) in ARTIFACTS.items():

    source = Path(source)
    destination = Path(destination)

    if not source.is_file():

        missing.append({
            "name":
                name,

            "source":
                str(source),
        })

        print(
            "[MISSING]",
            name,
            "->",
            source
        )

        continue

    try:

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            source,
            destination
        )

        copied.append({
            "name":
                name,

            "source":
                str(source),

            "destination":
                str(destination),

            "size_bytes":
                destination.stat().st_size,
        })

        print(
            "[COPIED]",
            name
        )

    except Exception as exc:

        errors.append({
            "name":
                name,

            "source":
                str(source),

            "error":
                repr(exc),
        })

        print(
            "[ERROR]",
            name,
            repr(exc)
        )

# ============================================================
# 5. ALSO COPY AVAILABLE SVG / PDF FIGURES
#
# These are optional publication-quality versions generated
# by earlier experiment cells.
# ============================================================

OPTIONAL_PUBLICATION_FIGURES = [

    # Long horizon
    ROOT
    / "results"
    / "phase1_zero_shot"
    / "long_horizon"
    / "fig_long_horizon_energy.pdf",

    ROOT
    / "results"
    / "phase1_zero_shot"
    / "long_horizon"
    / "fig_long_horizon_energy.svg",

    ROOT
    / "results"
    / "phase1_zero_shot"
    / "long_horizon"
    / "fig_long_horizon_energy_drift.pdf",

    ROOT
    / "results"
    / "phase1_zero_shot"
    / "long_horizon"
    / "fig_long_horizon_energy_drift.svg",

    ROOT
    / "results"
    / "phase1_zero_shot"
    / "long_horizon"
    / "fig_fno_lno_rollout_separation.pdf",

    ROOT
    / "results"
    / "phase1_zero_shot"
    / "long_horizon"
    / "fig_fno_lno_rollout_separation.svg",

    # Original spectral publication artifacts
    ROOT
    / "results"
    / "phase1_zero_shot"
    / "long_horizon"
    / "spectral_stability"
    / "fig_mean_spectral_energy_vs_mode.pdf",

    ROOT
    / "results"
    / "phase1_zero_shot"
    / "long_horizon"
    / "spectral_stability"
    / "fig_high_frequency_fraction_vs_timestep.pdf",

    ROOT
    / "results"
    / "phase1_zero_shot"
    / "long_horizon"
    / "spectral_stability"
    / "fig_total_spectral_energy_vs_timestep.pdf",
]

print(
    "\n" + "=" * 90
)

print(
    "COLLECTING OPTIONAL PUBLICATION FIGURES"
)

print(
    "=" * 90
)

optional_copied = []

for source in OPTIONAL_PUBLICATION_FIGURES:

    source = Path(source)

    if not source.is_file():

        continue

    destination = (
        FIGURES_DIR
        / source.name
    )

    try:

        shutil.copy2(
            source,
            destination
        )

        optional_copied.append(
            str(destination)
        )

        print(
            "[PUBLICATION FIGURE]",
            source.name
        )

    except Exception as exc:

        errors.append({
            "name":
                source.name,

            "source":
                str(source),

            "error":
                repr(exc),
        })

# ============================================================
# 6. GENERATE HUMAN-READABLE ARTIFACT INDEX
# ============================================================

INDEX_PATH = (
    MANIFEST_DIR
    / "artifact_index.csv"
)

index_rows = []

for item in copied:

    destination = Path(
        item["destination"]
    )

    index_rows.append({

        "artifact":
            item["name"],

        "path":
            str(
                destination.relative_to(
                    BUNDLE_DIR
                )
            ),

        "size_bytes":
            item["size_bytes"],

        "type":
            destination.suffix.lower(),

    })

for destination_str in optional_copied:

    destination = Path(
        destination_str
    )

    index_rows.append({

        "artifact":
            "publication_figure",

        "path":
            str(
                destination.relative_to(
                    BUNDLE_DIR
                )
            ),

        "size_bytes":
            destination.stat().st_size,

        "type":
            destination.suffix.lower(),

    })

index_df = pd.DataFrame(
    index_rows
)

index_df.to_csv(
    INDEX_PATH,
    index=False
)

# ============================================================
# 7. CREATE BUNDLE MANIFEST
# ============================================================

MANIFEST_PATH = (
    MANIFEST_DIR
    / "final_submission_manifest.json"
)

manifest = {

    "created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "project_root":
        str(ROOT),

    "bundle_directory":
        str(BUNDLE_DIR),

    "copied_artifact_count":
        len(copied),

    "optional_publication_figure_count":
        len(optional_copied),

    "missing_artifact_count":
        len(missing),

    "error_count":
        len(errors),

    "artifacts":
        copied,

    "optional_publication_figures":
        optional_copied,

    "missing":
        missing,

    "errors":
        errors,
}

with open(
    MANIFEST_PATH,
    "w"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )

# ============================================================
# 8. CREATE ZIP
# ============================================================

ZIP_BASE = (
    ROOT
    / "results"
    / "FINAL_SUBMISSION_BUNDLE"
)

ZIP_PATH = shutil.make_archive(
    str(ZIP_BASE),
    "zip",
    root_dir=str(
        BUNDLE_DIR
    )
)

# ============================================================
# 9. SHA256 OF ZIP
# ============================================================

def sha256_file(
    path
):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        for chunk in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b""
        ):

            h.update(
                chunk
            )

    return h.hexdigest()


zip_sha256 = sha256_file(
    ZIP_PATH
)

# Save ZIP hash
ZIP_HASH_PATH = (
    MANIFEST_DIR
    / "bundle_sha256.txt"
)

with open(
    ZIP_HASH_PATH,
    "w"
) as f:

    f.write(
        zip_sha256
    )

# ============================================================
# 10. FINAL REPORT
# ============================================================

print("\n" + "=" * 90)
print("FINAL SUBMISSION BUNDLE COMPLETE")
print("=" * 90)

print(
    "[+] Core artifacts copied:",
    len(copied)
)

print(
    "[+] Optional publication figures:",
    len(optional_copied)
)

print(
    "[+] Missing artifacts:",
    len(missing)
)

print(
    "[+] Errors:",
    len(errors)
)

print(
    "[+] Bundle directory:",
    BUNDLE_DIR
)

print(
    "[+] Artifact index:",
    INDEX_PATH
)

print(
    "[+] Manifest:",
    MANIFEST_PATH
)

print(
    "[+] ZIP:",
    ZIP_PATH
)

print(
    "[+] ZIP SHA256:",
    zip_sha256
)

# ============================================================
# 11. ASSERT CORE BUNDLE EXISTS
# ============================================================

assert os.path.isdir(
    BUNDLE_DIR
)

assert os.path.isfile(
    INDEX_PATH
)

assert os.path.isfile(
    MANIFEST_PATH
)

assert os.path.isfile(
    ZIP_PATH
)

assert os.path.isfile(
    ZIP_HASH_PATH
)

print(
    "\n[PASS] Final artifact bundle verified."
)

print("=" * 90)
