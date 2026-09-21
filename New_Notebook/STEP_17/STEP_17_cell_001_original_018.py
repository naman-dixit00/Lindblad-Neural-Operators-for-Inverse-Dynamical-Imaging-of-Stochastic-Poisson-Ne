# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 18
# Step            : STEP_17
# Step Heading    : # STEP 17 — FINAL FNO vs LNO COMPARISON
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 17 — FINAL FNO vs LNO COMPARISON
#
# SAME TEST SET
# SAME METRICS
# SAME TARGETS
#
# Default:
#   FNO               = best_fno.pt
#   LNO               = original best_lno.pt
#
# Optional:
#   USE_PILOT = True
#   -> learnable-dissipation LNO checkpoint
#
# NO TRAINING
# NO CHECKPOINT MODIFICATION
# ============================================================

import os
import json
import hashlib
import numpy as np
import pandas as pd

print("=" * 90)
print("STEP 17 — FINAL FNO vs LNO COMPARISON")
print("=" * 90)

# ============================================================
# 1. SETTINGS
# ============================================================

USE_PILOT = False

ROOT = (
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

# ============================================================
# 2. CHECKPOINTS
# ============================================================

FNO_CKPT = os.path.join(
    ROOT,
    "results",
    "check_points",
    "best_fno.pt"
)

LNO_ORIGINAL_CKPT = os.path.join(
    ROOT,
    "results",
    "check_points",
    "best_lno.pt"
)

LNO_PILOT_CKPT = os.path.join(
    ROOT,
    "results",
    "lno_learnable_dissipation",
    "best_lno_alpha.pt"
)

assert os.path.isfile(
    FNO_CKPT
), FNO_CKPT

if USE_PILOT:

    assert os.path.isfile(
        LNO_PILOT_CKPT
    ), LNO_PILOT_CKPT

    LNO_CKPT = LNO_PILOT_CKPT
    LNO_NAME = "LNO_learnable_dissipation"

else:

    assert os.path.isfile(
        LNO_ORIGINAL_CKPT
    ), LNO_ORIGINAL_CKPT

    LNO_CKPT = LNO_ORIGINAL_CKPT
    LNO_NAME = "LNO_original"

print(
    "[+] FNO checkpoint:",
    FNO_CKPT
)

print(
    "[+] LNO checkpoint:",
    LNO_CKPT
)

# ============================================================
# 3. SHA256
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):

            h.update(chunk)

    return h.hexdigest()


fno_sha = sha256_file(
    FNO_CKPT
)

lno_sha = sha256_file(
    LNO_CKPT
)

print(
    "\n[+] FNO SHA256:",
    fno_sha
)

print(
    "[+] LNO SHA256:",
    lno_sha
)

# ============================================================
# 4. FIND TEST ARTIFACTS
# ============================================================

FNO_METRICS = os.path.join(
    ROOT,
    "results",
    "phase1_zero_shot",
    "fno_test",
    "fno_test_metrics.json"
)

LNO_METRICS = os.path.join(
    ROOT,
    "results",
    "final_lno",
    (
        "learnable_dissipation_lno"
        if USE_PILOT
        else "original_lno"
    ),
    "lno_test_metrics.json"
)

# ------------------------------------------------------------
# If the final LNO evaluation cell has not been run in the
# current branch, stop with a clear message.
# ------------------------------------------------------------

assert os.path.isfile(
    FNO_METRICS
), (
    "FNO test metrics not found:\n"
    + FNO_METRICS
)

assert os.path.isfile(
    LNO_METRICS
), (
    "LNO test metrics not found.\n"
    "Run STEP 16 first:\n"
    + LNO_METRICS
)

print(
    "\n[+] FNO metrics:",
    FNO_METRICS
)

print(
    "[+] LNO metrics:",
    LNO_METRICS
)

# ============================================================
# 5. LOAD METRICS
# ============================================================

with open(
    FNO_METRICS,
    "r"
) as f:

    fno = json.load(f)

with open(
    LNO_METRICS,
    "r"
) as f:

    lno = json.load(f)

print(
    "\n[+] FNO test samples:",
    fno.get("test_samples")
)

print(
    "[+] LNO test samples:",
    lno.get("test_samples")
)

# ============================================================
# 6. SAME-TEST-SET VERIFICATION
# ============================================================

assert (
    int(fno["test_samples"])
    == int(lno["test_samples"])
    == 3000
), (
    "FNO/LNO test sample count mismatch."
)

fno_shape = tuple(
    fno["prediction_shape"]
)

lno_shape = tuple(
    lno["prediction_shape"]
)

assert fno_shape == lno_shape, (
    f"Prediction shape mismatch: "
    f"FNO={fno_shape}, LNO={lno_shape}"
)

print(
    "[PASS] Same test-set sample count"
)

print(
    "[PASS] Prediction geometry:",
    fno_shape
)

# ============================================================
# 7. METRICS TO COMPARE
#
# For ALL metrics here, LOWER is better.
# ============================================================

metric_names = [
    "RMSE",
    "Relative_L2",
    "MAE",
    "Mass_Error",
    "Entropy_Error",
    "Instability_Index",
    "Dissipation_Rate",
    "Spectral_Energy",
]

# ============================================================
# 8. COMPARISON
# ============================================================

rows = []

for metric in metric_names:

    if (
        metric not in fno
        or metric not in lno
    ):

        print(
            f"[WARNING] Missing metric: {metric}"
        )

        continue

    fno_value = float(
        fno[metric]
    )

    lno_value = float(
        lno[metric]
    )

    # Positive = LNO is better
    # Negative = FNO is better
    #
    # Since lower is better:
    #
    # improvement =
    # (FNO - LNO) / FNO * 100

    improvement = (
        (
            fno_value
            - lno_value
        )
        /
        (
            abs(fno_value)
            + 1e-30
        )
        * 100.0
    )

    if (
        lno_value
        < fno_value
    ):

        winner = "LNO"

    elif (
        fno_value
        < lno_value
    ):

        winner = "FNO"

    else:

        winner = "Tie"

    rows.append({

        "Metric":
            metric,

        "FNO":
            fno_value,

        "LNO":
            lno_value,

        "LNO_improvement_percent":
            improvement,

        "Winner":
            winner,
    })

comparison_df = pd.DataFrame(
    rows
)

# ============================================================
# 9. PRINT MAIN TABLE
# ============================================================

print("\n" + "=" * 90)
print("FINAL FNO vs LNO — SAME 3000 TEST SAMPLES")
print("=" * 90)

display(
    comparison_df.round(10)
)

# ============================================================
# 10. PRIMARY RMSE RESULT
# ============================================================

rmse_row = comparison_df.loc[
    comparison_df["Metric"] == "RMSE"
]

assert len(
    rmse_row
) == 1

rmse_row = rmse_row.iloc[0]

fno_rmse = float(
    rmse_row["FNO"]
)

lno_rmse = float(
    rmse_row["LNO"]
)

rmse_improvement = float(
    rmse_row[
        "LNO_improvement_percent"
    ]
)

rmse_winner = (
    rmse_row["Winner"]
)

print("\n" + "=" * 90)
print("PRIMARY RESULT — RMSE")
print("=" * 90)

print(
    "FNO RMSE:",
    f"{fno_rmse:.12e}"
)

print(
    "LNO RMSE:",
    f"{lno_rmse:.12e}"
)

print(
    "LNO vs FNO improvement:",
    f"{rmse_improvement:+.6f}%"
)

print(
    "Winner:",
    rmse_winner
)

# ============================================================
# 11. COUNT METRIC WINS
# ============================================================

lno_wins = int(
    (
        comparison_df["Winner"]
        == "LNO"
    ).sum()
)

fno_wins = int(
    (
        comparison_df["Winner"]
        == "FNO"
    ).sum()
)

ties = int(
    (
        comparison_df["Winner"]
        == "Tie"
    ).sum()
)

print("\n" + "=" * 90)
print("METRIC WIN COUNT")
print("=" * 90)

print(
    "[+] LNO wins:",
    lno_wins
)

print(
    "[+] FNO wins:",
    fno_wins
)

print(
    "[+] Ties:",
    ties
)

# ============================================================
# 12. OOD / ROBUSTNESS CONTEXT
#
# This does NOT mix OOD values into the clean-test table.
# It simply records where the existing OOD artifact is.
# ============================================================

OOD_CSV = os.path.join(
    ROOT,
    "results",
    "phase1_zero_shot",
    "ood_sigma_sweep",
    "ood_sigma_sweep.csv"
)

if os.path.isfile(
    OOD_CSV
):

    ood_status = "available"

else:

    ood_status = "not_found"

print(
    "\n[+] Existing OOD artifact:",
    ood_status
)

# ============================================================
# 13. FINAL DATA FRAME
# ============================================================

final_table = comparison_df.copy()

# ============================================================
# 14. SAVE
# ============================================================

OUT_DIR = os.path.join(
    ROOT,
    "results",
    "final_comparison"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

CSV_PATH = os.path.join(
    OUT_DIR,
    "fno_vs_lno_final.csv"
)

JSON_PATH = os.path.join(
    OUT_DIR,
    "fno_vs_lno_final.json"
)

final_table.to_csv(
    CSV_PATH,
    index=False
)

comparison_json = {

    "status":
        "COMPLETE",

    "test_samples":
        3000,

    "fno_checkpoint":
        FNO_CKPT,

    "fno_sha256":
        fno_sha,

    "lno_checkpoint":
        LNO_CKPT,

    "lno_sha256":
        lno_sha,

    "lno_model":
        LNO_NAME,

    "metrics":
        rows,

    "primary_metric":
        "RMSE",

    "primary_fno_rmse":
        fno_rmse,

    "primary_lno_rmse":
        lno_rmse,

    "primary_lno_improvement_percent":
        rmse_improvement,

    "primary_winner":
        rmse_winner,

    "lno_metric_wins":
        lno_wins,

    "fno_metric_wins":
        fno_wins,

    "ties":
        ties,

    "ood_artifact":
        OOD_CSV
        if ood_status == "available"
        else None,
}

with open(
    JSON_PATH,
    "w"
) as f:

    json.dump(
        comparison_json,
        f,
        indent=2
    )

# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 90)
print("STEP 17 COMPLETE")
print("=" * 90)

print(
    "[+] Model:",
    LNO_NAME
)

print(
    "[+] Same test samples:",
    3000
)

print(
    "[+] Primary FNO RMSE:",
    f"{fno_rmse:.12e}"
)

print(
    "[+] Primary LNO RMSE:",
    f"{lno_rmse:.12e}"
)

print(
    "[+] LNO RMSE improvement:",
    f"{rmse_improvement:+.6f}%"
)

print(
    "[+] RMSE winner:",
    rmse_winner
)

print(
    "[+] CSV:",
    CSV_PATH
)

print(
    "[+] JSON:",
    JSON_PATH
)

print("=" * 90)
