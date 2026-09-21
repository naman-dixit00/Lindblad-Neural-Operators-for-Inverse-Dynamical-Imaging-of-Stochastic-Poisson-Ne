# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 16
# Step            : STEP_01
# Step Heading    : # PHASE 1 — STEP 1: REPOSITORY + TENSOR GEOMETRY AUDIT
# Step Cell No.   : 12
# ============================================================

# ============================================================
# FINAL LNO CHECKPOINT SELECTOR + VERIFICATION
# ============================================================

import os
import hashlib
import torch
import json

ROOT = (
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

ORIGINAL = os.path.join(
    ROOT,
    "results",
    "check_points",
    "best_lno.pt"
)

PILOT = os.path.join(
    ROOT,
    "results",
    "lno_learnable_dissipation",
    "best_lno_alpha.pt"
)

print("=" * 90)
print("LNO FINAL CHECKPOINT VERIFICATION")
print("=" * 90)

# ------------------------------------------------------------
# Candidate selection
# ------------------------------------------------------------

candidates = []

if os.path.isfile(ORIGINAL):
    candidates.append(
        ("original_lno", ORIGINAL)
    )

if os.path.isfile(PILOT):
    candidates.append(
        ("learnable_dissipation_lno", PILOT)
    )

print("[+] Available checkpoints:")

for name, path in candidates:
    print(
        f"    {name}: {path}"
    )

# ------------------------------------------------------------
# Read pilot metadata if available
# ------------------------------------------------------------

selected_name = "original_lno"
selected_path = ORIGINAL

if os.path.isfile(PILOT):

    pilot = torch.load(
        PILOT,
        map_location="cpu"
    )

    pilot_val = pilot.get(
        "best_val_loss",
        float("inf")
    )

    print(
        "\n[+] Pilot best validation MSE:",
        f"{pilot_val:.12e}"
    )

    selected_name = (
        "learnable_dissipation_lno"
    )

    selected_path = PILOT

else:

    print(
        "\n[+] Pilot checkpoint not found."
    )

# ------------------------------------------------------------
# SHA256
# ------------------------------------------------------------

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


sha = sha256_file(
    selected_path
)

print(
    "\n[+] SELECTED:",
    selected_name
)

print(
    "[+] PATH:",
    selected_path
)

print(
    "[+] SHA256:",
    sha
)

# ------------------------------------------------------------
# Load verification
# ------------------------------------------------------------

selected_ckpt = torch.load(
    selected_path,
    map_location="cpu"
)

if (
    "model_state_dict"
    in selected_ckpt
):

    state_dict = selected_ckpt[
        "model_state_dict"
    ]

else:

    state_dict = selected_ckpt

# ------------------------------------------------------------
# Basic tensor integrity
# ------------------------------------------------------------

all_finite = True
parameter_count = 0

for name, tensor in state_dict.items():

    if torch.is_tensor(tensor):

        parameter_count += tensor.numel()

        if not torch.isfinite(
            tensor
        ).all():

            all_finite = False

print(
    "[+] Parameter count:",
    f"{parameter_count:,}"
)

print(
    "[+] All tensors finite:",
    all_finite
)

assert all_finite

# ------------------------------------------------------------
# Save final manifest
# ------------------------------------------------------------

FINAL_DIR = os.path.join(
    ROOT,
    "results",
    "final_lno"
)

os.makedirs(
    FINAL_DIR,
    exist_ok=True
)

manifest = {

    "selected_checkpoint":
        selected_name,

    "path":
        selected_path,

    "sha256":
        sha,

    "parameter_count":
        parameter_count,

    "all_finite":
        all_finite,

    "selection_rule":
        "pilot checkpoint selected after training; "
        "original checkpoint remains untouched",

}

with open(
    os.path.join(
        FINAL_DIR,
        "final_lno_manifest.json"
    ),
    "w"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )

print(
    "\n[+] Final manifest saved:"
)

print(
    os.path.join(
        FINAL_DIR,
        "final_lno_manifest.json"
    )
)

print("=" * 90)
print("CHECKPOINT VERIFICATION COMPLETE")
print("=" * 90)
