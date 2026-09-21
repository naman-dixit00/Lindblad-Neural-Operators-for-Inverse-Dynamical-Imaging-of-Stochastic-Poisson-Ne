# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 91
# Step            : STEP_31
# Step Heading    : # STEP 31 — CORRECTED FINAL PUBLICATION PACKAGE AUDIT
# Step Cell No.   : 2
# ============================================================

# ======================================================================================
# STEP 31 — CORRECTED FINAL PUBLICATION PACKAGE AUDIT
# ======================================================================================
#
# AUDITS THE EXISTING STEP 30 PACKAGE ONLY.
#
# NO:
#   - training
#   - model inference
#   - checkpoint modification
#   - figure regeneration
#   - evidence modification
#
# IMPORTANT:
#   Step 30 stores the 8 core figures in:
#       SQUARE/
#       TWO_COLUMN/
#       WIDE/
#
#   Step 30 does NOT store flat copies such as:
#       PNG/01_overall_accuracy.png
#       SVG/01_overall_accuracy.svg
#       PDF/01_overall_accuracy.pdf
#
#   Therefore this audit checks the ACTUAL Step 30 structure.
# ======================================================================================


# ======================================================================================
# 0. IMPORTS
# ======================================================================================

import os
import gc
import json
import hashlib
import zipfile
import shutil
from pathlib import Path

from PIL import Image


# ======================================================================================
# 1. ROOT AUTO-DETECTION
# ======================================================================================

CONTENT_ROOT = Path("/content")

repo_candidates = [
    p
    for p in CONTENT_ROOT.iterdir()
    if (
        p.is_dir()
        and (p / "results").is_dir()
        and (p / "physics").is_dir()
        and (p / "training").is_dir()
    )
]

assert repo_candidates, "Repository root not found."

ROOT = repo_candidates[0]

os.chdir(ROOT)

print("=" * 110)
print("STEP 31 — CORRECTED FINAL IEEE / JOURNAL PUBLICATION PACKAGE AUDIT")
print("=" * 110)

print("[+] Root:", ROOT)


# ======================================================================================
# 2. STEP 30 PACKAGE PATHS
# ======================================================================================

OUT = (
    ROOT
    / "results"
    / "step30_ieee_publication_package"
)

ZIP_PATH = (
    ROOT
    / "results"
    / "STEP30_IEEE_PUBLICATION_PACKAGE.zip"
)

PNG_DIR = OUT / "PNG"
SVG_DIR = OUT / "SVG"
PDF_DIR = OUT / "PDF"

SQUARE_DIR = OUT / "SQUARE"
TWO_COLUMN_DIR = OUT / "TWO_COLUMN"
WIDE_DIR = OUT / "WIDE"

THREED_DIR = OUT / "3D"
GIF_DIR = OUT / "GIF"
ARCH_DIR = OUT / "ARCHITECTURE"

EVIDENCE_DIR = OUT / "validated_evidence"

TEMP_DIR = OUT / "_temp_gif_frames"


# ======================================================================================
# 3. AUDIT RESULT STORAGE
# ======================================================================================

audit_results = []
warnings = []
errors = []


def record(check, passed, detail=""):

    status = "PASS" if passed else "FAIL"

    audit_results.append({
        "check": check,
        "status": status,
        "detail": str(detail)
    })

    if passed:

        print(
            f"[PASS] {check}"
            + (
                f" — {detail}"
                if detail
                else ""
            )
        )

    else:

        print(
            f"[FAIL] {check}"
            + (
                f" — {detail}"
                if detail
                else ""
            )
        )

        errors.append(
            f"{check}: {detail}"
        )


def warning(check, detail=""):

    warnings.append(
        f"{check}: {detail}"
    )

    print(
        f"[WARN] {check}"
        + (
            f" — {detail}"
            if detail
            else ""
        )
    )


# ======================================================================================
# 4. PACKAGE EXISTENCE
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 1 — PACKAGE EXISTENCE")
print("=" * 110)

record(
    "Step 30 output directory exists",
    OUT.is_dir(),
    OUT
)

record(
    "Step 30 ZIP exists",
    ZIP_PATH.is_file(),
    ZIP_PATH
)


# ======================================================================================
# 5. REQUIRED DIRECTORY STRUCTURE
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 2 — REQUIRED DIRECTORY STRUCTURE")
print("=" * 110)

required_dirs = [
    PNG_DIR,
    SVG_DIR,
    PDF_DIR,
    SQUARE_DIR,
    TWO_COLUMN_DIR,
    WIDE_DIR,
    THREED_DIR,
    GIF_DIR,
    ARCH_DIR,
    EVIDENCE_DIR,
]

for directory in required_dirs:

    record(
        f"Directory exists: {directory.name}",
        directory.is_dir(),
        directory
    )


# ======================================================================================
# 6. CORE FIGURE DEFINITIONS
# ======================================================================================

CORE_FIGURES = [
    "01_overall_accuracy",
    "02_regime_wise_R_accuracy",
    "03_structural_preservation",
    "04_C_LNO_100step_trace_stability",
    "05_C_LNO_100step_PSD_stability",
    "06_C_LNO_R_norm_stability",
    "07_regime_structural_stability",
    "08_final_paper_summary",
]


# ======================================================================================
# 7. ACTUAL STEP 30 ASPECT-RATIO FILES
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 3 — CORE FIGURE ASPECT-RATIO PACKAGE")
print("=" * 110)

aspect_specs = {
    "square": SQUARE_DIR,
    "two_column": TWO_COLUMN_DIR,
    "wide": WIDE_DIR,
}

for figure in CORE_FIGURES:

    for aspect, directory in aspect_specs.items():

        path = (
            directory
            /
            f"{figure}_{aspect}.png"
        )

        record(
            f"{figure} — {aspect} PNG",
            path.is_file(),
            path
        )


# ======================================================================================
# 8. IMPORTANT: ROOT FORMAT DIRECTORIES
# ======================================================================================
#
# Step 30 created these directories, but they are not the canonical storage location
# for the 8 core aspect-ratio figures.
#
# Therefore:
#   - empty is acceptable
#   - populated is acceptable
#   - absence would have been structurally odd, but directory existence is already
#     checked above.
#
# We do NOT treat missing flat core files as failures.
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 4 — ROOT FORMAT DIRECTORIES")
print("=" * 110)

for directory in [
    PNG_DIR,
    SVG_DIR,
    PDF_DIR,
]:

    if directory.is_dir():

        file_count = sum(
            p.is_file()
            for p in directory.iterdir()
        )

        print(
            f"[INFO] {directory.name}/ contains {file_count} file(s)."
        )


# ======================================================================================
# 9. 3D FIGURE AUDIT
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 5 — 3D FIGURES")
print("=" * 110)

THREED_FIGURES = [
    "09_3D_C_LNO_R_norm_stability",
    "10_3D_R_matrix_structure",
]

for figure in THREED_FIGURES:

    for extension in [
        "png",
        "svg",
        "pdf",
    ]:

        path = (
            THREED_DIR
            /
            f"{figure}.{extension}"
        )

        record(
            f"{figure} — {extension.upper()}",
            path.is_file(),
            path
        )


# ======================================================================================
# 10. ARCHITECTURE AUDIT
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 6 — ARCHITECTURE")
print("=" * 110)

ARCH_FILES = [
    "11_LNO_architecture_publication.png",
    "11_LNO_architecture_publication.svg",
    "11_LNO_architecture_publication.pdf",
]

for filename in ARCH_FILES:

    path = ARCH_DIR / filename

    record(
        f"Architecture — {filename}",
        path.is_file(),
        path
    )


# ======================================================================================
# 11. GIF AUDIT
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 7 — GIF FILES")
print("=" * 110)

ARCH_GIF = (
    GIF_DIR
    /
    "12_LNO_architecture_working.gif"
)

R_NORM_GIF = (
    GIF_DIR
    /
    "13_C_LNO_3D_R_norm_evolution.gif"
)


def inspect_gif(
    label,
    path
):

    record(
        f"{label} exists",
        path.is_file(),
        path
    )

    if not path.is_file():

        return None

    try:

        with Image.open(path) as img:

            frames = int(
                getattr(
                    img,
                    "n_frames",
                    1
                )
            )

            width, height = img.size

            duration = img.info.get(
                "duration",
                None
            )

        record(
            f"{label} readable",
            True,
            f"{width}x{height}"
        )

        record(
            f"{label} multiple frames",
            frames >= 2,
            frames
        )

        if duration is None:

            warning(
                f"{label} duration metadata",
                "Not present."
            )

        else:

            record(
                f"{label} duration metadata",
                duration > 0,
                f"{duration} ms"
            )

        return {
            "frames": frames,
            "width": width,
            "height": height,
            "duration_ms": duration,
            "size_bytes": path.stat().st_size,
        }

    except Exception as exc:

        record(
            f"{label} readable",
            False,
            repr(exc)
        )

        return None


arch_gif_info = inspect_gif(
    "Architecture GIF",
    ARCH_GIF
)

rnorm_gif_info = inspect_gif(
    "3D R-norm GIF",
    R_NORM_GIF
)


# ======================================================================================
# 12. MANIFEST / README
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 8 — MANIFEST / README")
print("=" * 110)

MANIFEST_PATH = (
    OUT
    /
    "STEP30_FIGURE_MANIFEST.json"
)

README_PATH = (
    OUT
    /
    "README.txt"
)

record(
    "Figure manifest exists",
    MANIFEST_PATH.is_file(),
    MANIFEST_PATH
)

record(
    "README exists",
    README_PATH.is_file(),
    README_PATH
)


# ======================================================================================
# 13. MANIFEST VALIDATION
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 9 — MANIFEST CONTENT")
print("=" * 110)

manifest_data = None

if MANIFEST_PATH.is_file():

    try:

        with open(
            MANIFEST_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            manifest_data = json.load(f)

        record(
            "Figure manifest is valid JSON",
            True
        )

    except Exception as exc:

        record(
            "Figure manifest is valid JSON",
            False,
            repr(exc)
        )


if manifest_data is not None:

    record(
        "Manifest step = 30",
        manifest_data.get("step") == 30,
        manifest_data.get("step")
    )

    record(
        "Manifest training_performed = False",
        manifest_data.get(
            "training_performed"
        ) is False,
        manifest_data.get(
            "training_performed"
        )
    )

    record(
        "Manifest new_model_inference = False",
        manifest_data.get(
            "new_model_inference"
        ) is False,
        manifest_data.get(
            "new_model_inference"
        )
    )

    record(
        "Manifest checkpoint_modified = False",
        manifest_data.get(
            "checkpoint_modified"
        ) is False,
        manifest_data.get(
            "checkpoint_modified"
        )
    )

    manifest_core = set(
        manifest_data.get(
            "core_figures",
            []
        )
    )

    record(
        "Manifest contains all 8 core figures",
        set(CORE_FIGURES).issubset(
            manifest_core
        ),
        f"{len(set(CORE_FIGURES) & manifest_core)}/8"
    )

    manifest_3d = set(
        manifest_data.get(
            "three_d_figures",
            []
        )
    )

    record(
        "Manifest contains 2 3D figures",
        set(THREED_FIGURES).issubset(
            manifest_3d
        ),
        f"{len(set(THREED_FIGURES) & manifest_3d)}/2"
    )

    manifest_architecture = set(
        manifest_data.get(
            "architecture",
            []
        )
    )

    record(
        "Manifest contains architecture figure",
        "11_LNO_architecture_publication"
        in manifest_architecture,
        manifest_architecture
    )

    manifest_gifs = set(
        manifest_data.get(
            "gifs",
            []
        )
    )

    expected_gifs = {
        ARCH_GIF.name,
        R_NORM_GIF.name,
    }

    record(
        "Manifest contains both GIFs",
        expected_gifs.issubset(
            manifest_gifs
        ),
        f"{len(expected_gifs & manifest_gifs)}/2"
    )


# ======================================================================================
# 14. VALIDATED EVIDENCE AUDIT
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 10 — VALIDATED EVIDENCE")
print("=" * 110)

STEP27_DIR = (
    ROOT
    /
    "results"
    /
    "step27_c_lno_independent_validation"
)

STEP28_DIR = (
    ROOT
    /
    "results"
    /
    "step28_c_lno_long_horizon"
)

STEP29_DIR = (
    ROOT
    /
    "results"
    /
    "step29_final_c_lno_evaluation"
)

EVIDENCE_FILES = [

    STEP27_DIR
    /
    "three_model_comparison.csv",

    STEP27_DIR
    /
    "regime_comparison.csv",

    STEP28_DIR
    /
    "c_lno_rollout_details.csv",

    STEP28_DIR
    /
    "c_lno_regime_summary.csv",

    STEP28_DIR
    /
    "c_lno_global_summary.csv",

    STEP29_DIR
    /
    "step29_final_summary.json",
]

for source_path in EVIDENCE_FILES:

    packaged_path = (
        EVIDENCE_DIR
        /
        source_path.name
    )

    record(
        f"Source evidence exists — {source_path.name}",
        source_path.is_file(),
        source_path
    )

    record(
        f"Packaged evidence exists — {source_path.name}",
        packaged_path.is_file(),
        packaged_path
    )


# ======================================================================================
# 15. BYTE-LEVEL EVIDENCE INTEGRITY
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 11 — EVIDENCE BYTE INTEGRITY")
print("=" * 110)


def sha256_file(path):

    digest = hashlib.sha256()

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

            digest.update(
                chunk
            )

    return digest.hexdigest()


for source_path in EVIDENCE_FILES:

    packaged_path = (
        EVIDENCE_DIR
        /
        source_path.name
    )

    if (
        source_path.is_file()
        and packaged_path.is_file()
    ):

        source_hash = sha256_file(
            source_path
        )

        packaged_hash = sha256_file(
            packaged_path
        )

        record(
            f"Byte integrity — {source_path.name}",
            source_hash == packaged_hash,
            source_hash
        )


# ======================================================================================
# 16. TEMP DIRECTORY HANDLING
# ======================================================================================
#
# Step 30 creates TEMP_DIR as a parent workspace and removes subdirectories after
# successful GIF creation. Depending on execution history, the parent directory itself
# may remain empty.
#
# An EMPTY temp directory is therefore NOT an audit failure.
#
# We clean it now if it contains no files.
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 12 — TEMPORARY ARTIFACTS")
print("=" * 110)

if TEMP_DIR.exists():

    remaining_temp_files = [
        p
        for p in TEMP_DIR.rglob("*")
        if p.is_file()
    ]

    if len(remaining_temp_files) == 0:

        shutil.rmtree(
            TEMP_DIR,
            ignore_errors=True
        )

        record(
            "No temporary GIF frame files remain",
            True,
            "Empty temp directory removed."
        )

    else:

        record(
            "No temporary GIF frame files remain",
            False,
            f"{len(remaining_temp_files)} file(s) remain."
        )

else:

    record(
        "No temporary GIF frame files remain",
        True,
        "Temporary directory absent."
    )


# ======================================================================================
# 17. ACTUAL PACKAGE COUNTS
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 13 — PACKAGE COUNTS")
print("=" * 110)

package_files = []

for root, dirs, files in os.walk(
    OUT
):

    for filename in files:

        full_path = (
            Path(root)
            /
            filename
        )

        if TEMP_DIR in full_path.parents:
            continue

        package_files.append(
            full_path
        )


png_files = [
    p for p in package_files
    if p.suffix.lower() == ".png"
]

svg_files = [
    p for p in package_files
    if p.suffix.lower() == ".svg"
]

pdf_files = [
    p for p in package_files
    if p.suffix.lower() == ".pdf"
]

gif_files = [
    p for p in package_files
    if p.suffix.lower() == ".gif"
]

print(
    f"[INFO] PNG files: {len(png_files)}"
)

print(
    f"[INFO] SVG files: {len(svg_files)}"
)

print(
    f"[INFO] PDF files: {len(pdf_files)}"
)

print(
    f"[INFO] GIF files: {len(gif_files)}"
)

record(
    "Expected 27 PNG files",
    len(png_files) == 27,
    len(png_files)
)

record(
    "Expected 27 SVG files",
    len(svg_files) == 27,
    len(svg_files)
)

record(
    "Expected 27 PDF files",
    len(pdf_files) == 27,
    len(pdf_files)
)

record(
    "Expected exactly 2 GIF files",
    len(gif_files) == 2,
    len(gif_files)
)


# ======================================================================================
# 18. ASPECT-RATIO COUNTS
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 14 — ASPECT-RATIO COUNTS")
print("=" * 110)

square_pngs = list(
    SQUARE_DIR.glob("*.png")
)

two_column_pngs = list(
    TWO_COLUMN_DIR.glob("*.png")
)

wide_pngs = list(
    WIDE_DIR.glob("*.png")
)

record(
    "Exactly 8 square PNG figures",
    len(square_pngs) == 8,
    len(square_pngs)
)

record(
    "Exactly 8 two-column PNG figures",
    len(two_column_pngs) == 8,
    len(two_column_pngs)
)

record(
    "Exactly 8 wide PNG figures",
    len(wide_pngs) == 8,
    len(wide_pngs)
)


# ======================================================================================
# 19. ZIP INTEGRITY
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 15 — ZIP INTEGRITY")
print("=" * 110)

zip_names = []

if ZIP_PATH.is_file():

    try:

        with zipfile.ZipFile(
            ZIP_PATH,
            "r"
        ) as z:

            bad_member = z.testzip()

            zip_names = z.namelist()

            record(
                "ZIP opens successfully",
                True,
                ZIP_PATH
            )

            record(
                "ZIP CRC integrity",
                bad_member is None,
                bad_member
            )

            record(
                "ZIP contains files",
                len(zip_names) > 0,
                len(zip_names)
            )

    except Exception as exc:

        record(
            "ZIP opens successfully",
            False,
            repr(exc)
        )


# ======================================================================================
# 20. ZIP CONTENT AUDIT
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 16 — ZIP CONTENT")
print("=" * 110)


def zip_contains(relative_path):

    expected = (
        OUT.name
        /
        Path(relative_path)
    )

    expected = str(
        expected
    ).replace(
        os.sep,
        "/"
    )

    return expected in zip_names


zip_required = [

    "STEP30_FIGURE_MANIFEST.json",
    "README.txt",

    "ARCHITECTURE/11_LNO_architecture_publication.png",
    "ARCHITECTURE/11_LNO_architecture_publication.svg",
    "ARCHITECTURE/11_LNO_architecture_publication.pdf",

    "GIF/12_LNO_architecture_working.gif",
    "GIF/13_C_LNO_3D_R_norm_evolution.gif",

    "3D/09_3D_C_LNO_R_norm_stability.png",
    "3D/09_3D_C_LNO_R_norm_stability.svg",
    "3D/09_3D_C_LNO_R_norm_stability.pdf",

    "3D/10_3D_R_matrix_structure.png",
    "3D/10_3D_R_matrix_structure.svg",
    "3D/10_3D_R_matrix_structure.pdf",
]

for relative_path in zip_required:

    record(
        f"ZIP contains {relative_path}",
        zip_contains(relative_path),
        relative_path
    )


# Core aspect-ratio PNGs in ZIP

for figure in CORE_FIGURES:

    for aspect in [
        "square",
        "two_column",
        "wide",
    ]:

        relative_path = (
            f"{aspect.upper()}/"
            f"{figure}_{aspect}.png"
        )

        record(
            f"ZIP contains {relative_path}",
            zip_contains(relative_path),
            relative_path
        )


# ======================================================================================
# 21. ZIP TEMP FILE AUDIT
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 17 — ZIP TEMPORARY FILE CHECK")
print("=" * 110)

zip_temp_members = [
    name
    for name in zip_names
    if "_temp_gif_frames" in name
]

record(
    "ZIP contains no temporary GIF frames",
    len(zip_temp_members) == 0,
    len(zip_temp_members)
)


# ======================================================================================
# 22. ZIP EVIDENCE AUDIT
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 18 — ZIP VALIDATED EVIDENCE")
print("=" * 110)

for source_path in EVIDENCE_FILES:

    relative_path = (
        "validated_evidence/"
        +
        source_path.name
    )

    record(
        f"ZIP contains evidence — {source_path.name}",
        zip_contains(relative_path),
        relative_path
    )


# ======================================================================================
# 23. FILE SIZE SANITY
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 19 — FILE SIZE SANITY")
print("=" * 110)


def size_mb(path):

    return (
        path.stat().st_size
        /
        (1024 ** 2)
    )


zip_size_mb = size_mb(
    ZIP_PATH
)

record(
    "ZIP size > 0",
    zip_size_mb > 0,
    f"{zip_size_mb:.2f} MB"
)

for label, path in [
    (
        "Architecture GIF",
        ARCH_GIF
    ),
    (
        "3D R-norm GIF",
        R_NORM_GIF
    ),
]:

    if path.is_file():

        record(
            f"{label} size > 0",
            path.stat().st_size > 0,
            f"{size_mb(path):.3f} MB"
        )


# ======================================================================================
# 24. GIF FRAME COUNTS
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 20 — GIF FRAME COUNTS")
print("=" * 110)

if arch_gif_info is not None:

    record(
        "Architecture GIF >= 50 frames",
        arch_gif_info["frames"] >= 50,
        arch_gif_info["frames"]
    )

if rnorm_gif_info is not None:

    record(
        "3D R-norm GIF = 100 frames",
        rnorm_gif_info["frames"] == 100,
        rnorm_gif_info["frames"]
    )


# ======================================================================================
# 25. README SANITY
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 21 — README CONTENT")
print("=" * 110)

if README_PATH.is_file():

    try:

        readme_text = README_PATH.read_text(
            encoding="utf-8"
        )

        required_phrases = [
            "NO:",
            "model retraining",
            "checkpoint modification",
            "new model inference",
            "Step 27",
            "Step 28",
            "Step 29",
            "architecture",
            "C-LNO",
        ]

        for phrase in required_phrases:

            record(
                f"README contains '{phrase}'",
                phrase in readme_text
            )

    except Exception as exc:

        record(
            "README readable",
            False,
            repr(exc)
        )


# ======================================================================================
# 26. SCIENTIFIC-INTEGRITY DECLARATION
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 22 — SCIENTIFIC INTEGRITY")
print("=" * 110)

record(
    "No training performed during Step 31",
    True
)

record(
    "No model inference performed during Step 31",
    True
)

record(
    "No checkpoint modification performed during Step 31",
    True
)

record(
    "No source evidence modification performed during Step 31",
    True
)


# ======================================================================================
# 27. FINAL SUMMARY
# ======================================================================================

print("\n" + "=" * 110)
print("STEP 31 — FINAL AUDIT SUMMARY")
print("=" * 110)

total_checks = len(
    audit_results
)

passed_checks = sum(
    result["status"] == "PASS"
    for result in audit_results
)

failed_checks = sum(
    result["status"] == "FAIL"
    for result in audit_results
)

audit_status = (
    "PASS"
    if failed_checks == 0
    else "FAIL"
)

print(
    f"\n[INFO] Total checks: {total_checks}"
)

print(
    f"[INFO] Passed:       {passed_checks}"
)

print(
    f"[INFO] Failed:       {failed_checks}"
)

print(
    f"[INFO] Warnings:     {len(warnings)}"
)

print(
    f"[INFO] FINAL STATUS: {audit_status}"
)


# ======================================================================================
# 28. WRITE FINAL AUDIT JSON
# ======================================================================================

AUDIT_JSON = (
    OUT
    /
    "STEP31_PUBLICATION_AUDIT.json"
)

audit_payload = {

    "step":
        31,

    "title":
        "Corrected Final IEEE / Journal Publication Package Audit",

    "status":
        audit_status,

    "root":
        str(ROOT),

    "package_directory":
        str(OUT),

    "zip_path":
        str(ZIP_PATH),

    "total_checks":
        total_checks,

    "passed_checks":
        passed_checks,

    "failed_checks":
        failed_checks,

    "warnings":
        len(warnings),

    "package_counts": {
        "png":
            len(png_files),
        "svg":
            len(svg_files),
        "pdf":
            len(pdf_files),
        "gif":
            len(gif_files),
    },

    "aspect_ratio_counts": {
        "square":
            len(square_pngs),
        "two_column":
            len(two_column_pngs),
        "wide":
            len(wide_pngs),
    },

    "gif_info": {
        "architecture":
            arch_gif_info,
        "r_norm_3d":
            rnorm_gif_info,
    },

    "zip_size_mb":
        round(
            zip_size_mb,
            4
        ),

    "failed_checks": [
        result
        for result in audit_results
        if result["status"] == "FAIL"
    ],

    "warnings_detail":
        warnings,

    "checks":
        audit_results,

    "scientific_integrity": {
        "training_performed":
            False,

        "new_model_inference":
            False,

        "checkpoint_modified":
            False,

        "source_evidence_modified":
            False,
    },
}

with open(
    AUDIT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        audit_payload,
        f,
        indent=2
    )

print(
    "\n[PASS] Audit JSON written:"
)

print(
    AUDIT_JSON
)


# ======================================================================================
# 29. WRITE HUMAN-READABLE REPORT
# ======================================================================================

AUDIT_TXT = (
    OUT
    /
    "STEP31_PUBLICATION_AUDIT.txt"
)

with open(
    AUDIT_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 31 — CORRECTED FINAL IEEE / JOURNAL PUBLICATION PACKAGE AUDIT\n"
    )

    f.write(
        "=" * 76
        +
        "\n\n"
    )

    f.write(
        f"FINAL STATUS: {audit_status}\n"
    )

    f.write(
        f"Total checks: {total_checks}\n"
    )

    f.write(
        f"Passed: {passed_checks}\n"
    )

    f.write(
        f"Failed: {failed_checks}\n"
    )

    f.write(
        f"Warnings: {len(warnings)}\n\n"
    )

    f.write(
        "PACKAGE COUNTS\n"
    )

    f.write(
        "-" * 76
        +
        "\n"
    )

    f.write(
        f"PNG: {len(png_files)}\n"
    )

    f.write(
        f"SVG: {len(svg_files)}\n"
    )

    f.write(
        f"PDF: {len(pdf_files)}\n"
    )

    f.write(
        f"GIF: {len(gif_files)}\n"
    )

    f.write(
        f"ZIP size: {zip_size_mb:.2f} MB\n\n"
    )

    f.write(
        "ASPECT-RATIO COUNTS\n"
    )

    f.write(
        "-" * 76
        +
        "\n"
    )

    f.write(
        f"Square: {len(square_pngs)}\n"
    )

    f.write(
        f"Two-column: {len(two_column_pngs)}\n"
    )

    f.write(
        f"Wide: {len(wide_pngs)}\n\n"
    )

    f.write(
        "GIF INFORMATION\n"
    )

    f.write(
        "-" * 76
        +
        "\n"
    )

    f.write(
        f"Architecture GIF: {arch_gif_info}\n"
    )

    f.write(
        f"3D R-norm GIF: {rnorm_gif_info}\n\n"
    )

    f.write(
        "FAILED CHECKS\n"
    )

    f.write(
        "-" * 76
        +
        "\n"
    )

    if failed_checks == 0:

        f.write(
            "None\n"
        )

    else:

        for result in audit_results:

            if result["status"] == "FAIL":

                f.write(
                    f"- {result['check']}: "
                    f"{result['detail']}\n"
                )

    f.write(
        "\nWARNINGS\n"
    )

    f.write(
        "-" * 76
        +
        "\n"
    )

    if not warnings:

        f.write(
            "None\n"
        )

    else:

        for item in warnings:

            f.write(
                f"- {item}\n"
            )


print(
    "[PASS] Human-readable audit written:"
)

print(
    AUDIT_TXT
)


# ======================================================================================
# 30. FINAL STATUS
# ======================================================================================

print("\n" + "=" * 110)

if audit_status == "PASS":

    print(
        "STEP 31 COMPLETE — PUBLICATION PACKAGE AUDIT PASSED"
    )

else:

    print(
        "STEP 31 COMPLETE — PUBLICATION PACKAGE AUDIT FAILED"
    )

print("=" * 110)

print(
    "\nAudit JSON:"
)

print(
    AUDIT_JSON
)

print(
    "\nAudit TXT:"
)

print(
    AUDIT_TXT
)

print(
    "\nPublication package:"
)

print(
    OUT
)

print(
    "\nZIP:"
)

print(
    ZIP_PATH
)

print(
    "\n[INFO] No training performed."
)

print(
    "[INFO] No model inference performed."
)

print(
    "[INFO] No checkpoint modified."
)

print(
    "[INFO] No source evidence modified."
)

print(
    "\n[PASS] STEP 31 AUDIT FINISHED."
)

print("=" * 110)

gc.collect()
