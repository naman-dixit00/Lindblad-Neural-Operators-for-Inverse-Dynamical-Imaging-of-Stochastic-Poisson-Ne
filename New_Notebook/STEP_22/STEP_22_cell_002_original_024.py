# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 24
# Step            : STEP_22
# Step Heading    : # STEP 22 — FINAL PAPER FIGURES + IEEE-READY TABLES
# Step Cell No.   : 2
# ============================================================

# ============================================================
# LNO — PHYSICS / TARGET SOURCE AUDIT
#
# PURPOSE:
#   Automatically locate:
#     1. Dataset generation
#     2. Target construction
#     3. Lindblad / dissipative dynamics
#     4. State(t) -> State(t+1)
#     5. Any decomposition of physical dynamics
#
# NO TRAINING
# NO MODEL CHANGES
# ============================================================

import os
import re
from pathlib import Path

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

print("=" * 100)
print("LNO — PHYSICS / TARGET SOURCE AUDIT")
print("=" * 100)

# ============================================================
# 1. SEARCH ROOTS
# ============================================================

SEARCH_DIRS = [
    ROOT / "data",
    ROOT / "dataset",
    ROOT / "src",
    ROOT / "models",
    ROOT / "scripts",
    ROOT / "notebooks",
]

SEARCH_DIRS = [
    p for p in SEARCH_DIRS
    if p.exists()
]

print(
    "[+] Search directories:"
)

for p in SEARCH_DIRS:
    print(
        "   ",
        p
    )

# ============================================================
# 2. KEYWORDS
# ============================================================

KEYWORDS = [

    # Target generation
    "target",
    "next_state",
    "state_t1",
    "state_t_plus_1",
    "nextstate",
    "y_target",

    # Dynamics
    "state_t",
    "dt",
    "delta_t",
    "timestep",
    "evolve",
    "evolution",

    # Lindblad
    "lindblad",
    "dissipative",
    "dissipation",
    "jump_operator",
    "jump_transform",
    "gamma",
    "collapse",

    # Open systems
    "environment",
    "decoherence",
    "dephasing",
    "thermal",

    # Physics components
    "coherent",
    "hamiltonian",
    "diffusion",
    "transport",
    "noise",
]

# ============================================================
# 3. PY FILE DISCOVERY
# ============================================================

python_files = []

for search_dir in SEARCH_DIRS:

    for path in search_dir.rglob("*.py"):

        if (
            "__pycache__"
            in path.parts
        ):
            continue

        python_files.append(
            path
        )

# Remove duplicates
python_files = sorted(
    set(python_files)
)

print(
    "\n[+] Python files found:",
    len(python_files)
)

# ============================================================
# 4. SEARCH
# ============================================================

matches = []

for path in python_files:

    try:

        text = path.read_text(
            errors="ignore"
        )

    except Exception:

        continue

    lines = text.splitlines()

    for line_no, line in enumerate(
        lines,
        start=1
    ):

        lower = line.lower()

        hit_keywords = [
            kw
            for kw in KEYWORDS
            if kw.lower() in lower
        ]

        if hit_keywords:

            matches.append({

                "file":
                    str(
                        path.relative_to(
                            ROOT
                        )
                    ),

                "line":
                    line_no,

                "keywords":
                    hit_keywords,

                "text":
                    line.strip(),
            })

# ============================================================
# 5. RANK IMPORTANT FILES
# ============================================================

from collections import Counter

file_scores = Counter()

for match in matches:

    score = len(
        match["keywords"]
    )

    important_terms = [
        "lindblad",
        "dissipative",
        "target",
        "next_state",
        "evolve",
        "state_t1",
        "environment",
    ]

    for term in important_terms:

        if term in match["keywords"]:

            score += 5

    file_scores[
        match["file"]
    ] += score

print(
    "\n" + "=" * 100
)

print(
    "TOP PHYSICS / TARGET FILES"
)

print(
    "=" * 100
)

for file_name, score in (
    file_scores
    .most_common(20)
):

    print(
        f"{score:5d}  {file_name}"
    )

# ============================================================
# 6. PRINT RELEVANT MATCHES
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "RELEVANT SOURCE MATCHES"
)

print(
    "=" * 100
)

# Limit output so Colab doesn't become huge
MAX_MATCHES = 200

for match in matches[
    :MAX_MATCHES
]:

    print(
        f"\n[{match['file']}:"
        f"{match['line']}]"
    )

    print(
        "Keywords:",
        ", ".join(
            match["keywords"]
        )
    )

    print(
        match["text"]
    )

# ============================================================
# 7. TARGET-FOCUSED SEARCH
# ============================================================

TARGET_PATTERNS = [

    r"target\s*=",
    r"next[_ ]state",
    r"state[_ ]t.?1",
    r"y\s*=",
    r"targets?",
    r"return.*target",
]

print(
    "\n" + "=" * 100
)

print(
    "TARGET-CONSTRUCTION CANDIDATES"
)

print(
    "=" * 100
)

target_hits = []

for path in python_files:

    try:

        lines = path.read_text(
            errors="ignore"
        ).splitlines()

    except Exception:

        continue

    for line_no, line in enumerate(
        lines,
        start=1
    ):

        lower = line.lower()

        if any(
            re.search(
                pattern,
                lower
            )
            for pattern in TARGET_PATTERNS
        ):

            target_hits.append(
                (
                    path,
                    line_no,
                    line.strip()
                )
            )

for path, line_no, line in (
    target_hits[:150]
):

    print(
        f"\n{path.relative_to(ROOT)}:"
        f"{line_no}"
    )

    print(
        line
    )

# ============================================================
# 8. PHYSICS-FORMULA CANDIDATES
# ============================================================

PHYSICS_PATTERNS = [

    r"lindblad",
    r"commutator",
    r"anticommutator",
    r"l\s*@\s*rho",
    r"rho\s*@\s*l",
    r"l.*t.*@.*l",
    r"jump",
    r"gamma",
    r"dissipat",
    r"dephas",
    r"decoher",
]

print(
    "\n" + "=" * 100
)

print(
    "LINDblad / DISSIPATIVE FORMULA CANDIDATES"
)

print(
    "=" * 100
)

physics_hits = []

for path in python_files:

    try:

        lines = path.read_text(
            errors="ignore"
        ).splitlines()

    except Exception:

        continue

    for line_no, line in enumerate(
        lines,
        start=1
    ):

        lower = line.lower()

        if any(
            re.search(
                pattern,
                lower
            )
            for pattern in PHYSICS_PATTERNS
        ):

            physics_hits.append(
                (
                    path,
                    line_no,
                    line.strip()
                )
            )

for path, line_no, line in (
    physics_hits[:200]
):

    print(
        f"\n{path.relative_to(ROOT)}:"
        f"{line_no}"
    )

    print(
        line
    )

# ============================================================
# 9. AUTOMATIC REPORT
# ============================================================

REPORT_DIR = (
    ROOT
    / "results"
    / "lno_physics_audit"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

report_path = (
    REPORT_DIR
    / "source_audit.txt"
)

with open(
    report_path,
    "w"
) as f:

    f.write(
        "LNO PHYSICS / TARGET SOURCE AUDIT\n"
    )

    f.write(
        "=" * 80
        + "\n\n"
    )

    f.write(
        "TOP FILES\n"
    )

    for file_name, score in (
        file_scores.most_common(50)
    ):

        f.write(
            f"{score:5d}  {file_name}\n"
        )

    f.write(
        "\n\nRELEVANT MATCHES\n"
    )

    for match in matches:

        f.write(
            f"\n[{match['file']}:"
            f"{match['line']}]\n"
        )

        f.write(
            "Keywords: "
            + ", ".join(
                match["keywords"]
            )
            + "\n"
        )

        f.write(
            match["text"]
            + "\n"
        )

    f.write(
        "\n\nTARGET CANDIDATES\n"
    )

    for path, line_no, line in (
        target_hits
    ):

        f.write(
            f"{path.relative_to(ROOT)}:"
            f"{line_no}\n"
        )

        f.write(
            line
            + "\n"
        )

    f.write(
        "\n\nPHYSICS CANDIDATES\n"
    )

    for path, line_no, line in (
        physics_hits
    ):

        f.write(
            f"{path.relative_to(ROOT)}:"
            f"{line_no}\n"
        )

        f.write(
            line
            + "\n"
        )

print(
    "\n" + "=" * 100
)

print(
    "SOURCE AUDIT COMPLETE"
)

print(
    "[+] Report:",
    report_path
)

print(
    "[+] Top candidate files:"
)

for file_name, score in (
    file_scores.most_common(10)
):

    print(
        "   ",
        file_name
    )

print("=" * 100)
