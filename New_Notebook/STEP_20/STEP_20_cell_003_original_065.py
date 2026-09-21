# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 65
# Step            : STEP_20
# Step Heading    : # STEP 20 — SAVE COMPLETE OOD SIGMA OUTPUT
# Step Cell No.   : 3
# ============================================================

# ============================================================
# STEP 20 — SAVE COMPLETE OOD SIGMA OUTPUT
# ============================================================

from pathlib import Path
import json
import zipfile
from google.colab import files

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

OUT_DIR = (
    ROOT
    / "results"
    / "step20_ood_sigma"
)

CSV_PATH = (
    OUT_DIR
    / "ood_sigma_results.csv"
)

JSON_PATH = (
    OUT_DIR
    / "ood_sigma_summary.json"
)

PLOT_PATH = (
    OUT_DIR
    / "ood_sigma_stability.png"
)

REPORT_PATH = (
    OUT_DIR
    / "step20_ood_sigma_report.txt"
)

ZIP_PATH = (
    OUT_DIR
    / "step20_ood_sigma_complete.zip"
)


# ============================================================
# CHECK
# ============================================================

required = {
    "CSV": CSV_PATH,
    "JSON": JSON_PATH,
    "Plot": PLOT_PATH,
}

print("=" * 100)
print("STEP 20 — SAVING COMPLETE OUTPUT")
print("=" * 100)

for name, path in required.items():

    exists = path.is_file()

    print(
        f"{name:10s}: {exists}"
    )

    if not exists:
        raise FileNotFoundError(
            f"Missing Step-20 output:\n{path}"
        )


# ============================================================
# LOAD RESULTS
# ============================================================

results_df = __import__(
    "pandas"
).read_csv(
    CSV_PATH
)

with open(
    JSON_PATH,
    "r"
) as f:

    summary = json.load(
        f
    )


# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

lines = []

lines.append(
    "=" * 100
)

lines.append(
    "STEP 20 — OOD SIGMA SWEEP REPORT"
)

lines.append(
    "=" * 100
)

lines.append(
    ""
)

lines.append(
    f"Training sigma maximum: "
    f"{summary['training_sigma_max']}"
)

lines.append(
    f"Sigma values tested: "
    f"{summary['sigma_values']}"
)

lines.append(
    ""
)

lines.append(
    "IMPORTANT:"
)

lines.append(
    summary["description"]
)

lines.append(
    ""
)

lines.append(
    "FNO OOD DIAGNOSTICS"
)

lines.append(
    "-" * 60
)

lines.append(
    f"Minimum eigenvalue: "
    f"{summary['FNO']['ood_minimum_eigenvalue']:.12e}"
)

lines.append(
    f"Maximum trace error: "
    f"{summary['FNO']['ood_max_trace_error']:.12e}"
)

lines.append(
    f"Maximum negative-eigenvalue fraction: "
    f"{summary['FNO']['ood_max_negative_eigenvalue_fraction']:.8f}"
)

lines.append(
    f"All finite: "
    f"{summary['FNO']['all_finite']}"
)

lines.append(
    f"Response monotonic: "
    f"{summary['FNO']['response_monotonic']}"
)

lines.append(
    ""
)

lines.append(
    "LNO OOD DIAGNOSTICS"
)

lines.append(
    "-" * 60
)

lines.append(
    f"Minimum eigenvalue: "
    f"{summary['LNO']['ood_minimum_eigenvalue']:.12e}"
)

lines.append(
    f"Maximum trace error: "
    f"{summary['LNO']['ood_max_trace_error']:.12e}"
)

lines.append(
    f"Maximum negative-eigenvalue fraction: "
    f"{summary['LNO']['ood_max_negative_eigenvalue_fraction']:.8f}"
)

lines.append(
    f"All finite: "
    f"{summary['LNO']['all_finite']}"
)

lines.append(
    f"Response monotonic: "
    f"{summary['LNO']['response_monotonic']}"
)

lines.append(
    ""
)

lines.append(
    "FULL RESULT TABLE"
)

lines.append(
    "-" * 60
)

lines.append(
    results_df.to_string(
        index=False
    )
)

lines.append(
    ""
)

lines.append(
    "=" * 100
)

lines.append(
    "END OF STEP 20 REPORT"
)

lines.append(
    "=" * 100
)


with open(
    REPORT_PATH,
    "w"
) as f:

    f.write(
        "\n".join(
            lines
        )
    )


# ============================================================
# CREATE ZIP
# ============================================================

with zipfile.ZipFile(
    ZIP_PATH,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as zf:

    zf.write(
        CSV_PATH,
        arcname=CSV_PATH.name
    )

    zf.write(
        JSON_PATH,
        arcname=JSON_PATH.name
    )

    zf.write(
        PLOT_PATH,
        arcname=PLOT_PATH.name
    )

    zf.write(
        REPORT_PATH,
        arcname=REPORT_PATH.name
    )


# ============================================================
# VERIFY ZIP
# ============================================================

assert ZIP_PATH.is_file()

zip_size_mb = (
    ZIP_PATH.stat().st_size
    /
    (1024 ** 2)
)

print(
    "\n[PASS] Complete Step-20 ZIP created."
)

print(
    "ZIP:",
    ZIP_PATH
)

print(
    f"Size: {zip_size_mb:.3f} MB"
)

print(
    "\nContents:"
)

with zipfile.ZipFile(
    ZIP_PATH,
    "r"
) as zf:

    for name in zf.namelist():

        print(
            "  ",
            name
        )


# ============================================================
# DOWNLOAD
# ============================================================

print(
    "\n[+] Starting browser download..."
)

files.download(
    str(
        ZIP_PATH
    )
)

print(
    "[PASS] Step-20 output download initiated."
)

print("=" * 100)
