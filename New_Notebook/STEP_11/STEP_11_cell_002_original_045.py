# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 45
# Step            : STEP_11
# Step Heading    : # RECOVER STEP-11 LNO CHECKPOINT FROM GITHUB
# Step Cell No.   : 2
# ============================================================

# ============================================================
# RECOVER STEP-11 LNO CHECKPOINT FROM GITHUB
# ============================================================

import os
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/naman-dixit00/Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne.git"

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

EXPECTED = (
    ROOT
    / "results"
    / "step11_lno_prototype"
    / "lno_prototype_best.pt"
)

print("=" * 100)
print("RECOVERING LNO CHECKPOINT")
print("=" * 100)

print("[+] Repo root:", ROOT)
print("[+] Expected checkpoint:", EXPECTED)

# ------------------------------------------------------------
# If repository disappeared, clone it.
# ------------------------------------------------------------

if not ROOT.exists():

    print("\n[+] Repository not found. Cloning...")

    subprocess.run(
        [
            "git",
            "clone",
            REPO_URL,
            str(ROOT),
        ],
        check=True
    )

else:

    print(
        "\n[+] Repository directory already exists."
    )

    # Pull latest committed version
    try:

        subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "pull",
            ],
            check=True
        )

        print(
            "[PASS] Latest committed repository state pulled."
        )

    except Exception as e:

        print(
            "[WARNING] git pull failed:",
            repr(e)
        )

# ------------------------------------------------------------
# Search for checkpoint anywhere inside repo
# ------------------------------------------------------------

print(
    "\n[+] Searching repository for LNO checkpoint..."
)

matches = list(
    ROOT.rglob(
        "lno_prototype_best.pt"
    )
)

if not matches:

    # broader search
    matches = list(
        ROOT.rglob(
            "*.pt"
        )
    )

print(
    "[+] .pt files found:",
    len(matches)
)

for p in matches[:30]:
    print(
        "   ",
        p
    )

# ------------------------------------------------------------
# Recover exact expected checkpoint
# ------------------------------------------------------------

if EXPECTED.is_file():

    print(
        "\n[PASS] Expected LNO checkpoint already exists:"
    )

    print(
        EXPECTED
    )

else:

    # Look for exact basename
    candidates = [
        p
        for p in matches
        if p.name == "lno_prototype_best.pt"
    ]

    if candidates:

        source = candidates[0]

        EXPECTED.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            source,
            EXPECTED
        )

        print(
            "\n[PASS] Checkpoint recovered:"
        )

        print(
            "Source:",
            source
        )

        print(
            "Target:",
            EXPECTED
        )

    else:

        print(
            "\n[FAIL] lno_prototype_best.pt "
            "was not found in the repository."
        )

        print(
            "\nThis means the checkpoint was probably "
            "NOT included in the Git commit."
        )

# ------------------------------------------------------------
# Final verification
# ------------------------------------------------------------

print(
    "\n" + "=" * 100
)

print(
    "CHECKPOINT VERIFICATION"
)

print(
    "=" * 100
)

print(
    "Exists:",
    EXPECTED.exists()
)

if EXPECTED.exists():

    print(
        "Size:",
        EXPECTED.stat().st_size,
        "bytes"
    )

    print(
        "[PASS] Step-11 checkpoint is available."
    )

    print(
        "\nYou can now run Step 13."
    )

else:

    print(
        "[FAIL] Step-11 checkpoint still unavailable."
    )

    print(
        "Do NOT run Step 13 yet."
    )

print("=" * 100)
