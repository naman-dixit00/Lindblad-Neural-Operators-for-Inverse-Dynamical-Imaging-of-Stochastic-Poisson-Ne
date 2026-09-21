# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 90
# Step            : STEP_31
# Step Heading    : # STEP 31 — FINAL PUBLICATION PACKAGE AUDIT
# Step Cell No.   : 1
# ============================================================

# ======================================================================================
# STEP 31 — FINAL PUBLICATION PACKAGE AUDIT
# ======================================================================================
#
# PURPOSE
#   Independently audit the complete Step 30 IEEE / Journal publication package.
#
# CHECKS
#   1. Step 30 output directory exists
#   2. ZIP package exists
#   3. All 8 core figures exist in:
#        - SQUARE
#        - TWO_COLUMN
#        - WIDE
#        - PNG
#        - SVG
#        - PDF
#   4. 3D Figures 09 and 10 exist in PNG/SVG/PDF
#   5. Architecture Figure 11 exists in PNG/SVG/PDF
#   6. Architecture GIF exists and is readable
#   7. 3D R-norm GIF exists and is readable
#   8. Validated Step 27/28/29 evidence exists
#   9. Manifest and README exist
#  10. No temporary GIF frames remain
#  11. ZIP is readable
#  12. ZIP contains expected publication artifacts
#  13. ZIP does not contain temporary frame directory
#  14. Source evidence copied to package matches original files byte-for-byte
#  15. Final counts are internally consistent
#
# IMPORTANT
#   - NO model training
#   - NO model inference
#   - NO checkpoint modification
#   - NO source evidence modification
#   - NO figure regeneration
#   - READ / VERIFY ONLY
# ======================================================================================


# ======================================================================================
# 0. IMPORTS
# ======================================================================================

import os
import gc
import json
import hashlib
import zipfile
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
print("STEP 31 — FINAL IEEE / JOURNAL PUBLICATION PACKAGE AUDIT")
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
# 3. AUDIT RESULT STRUCTURE
# ======================================================================================

audit_results = []

errors = []

warnings = []


def record(
    check,
    passed,
    detail=""
):
    """
    Record a single audit result.
    """

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


def record_warning(
    check,
    detail=""
):
    """
    Record a non-fatal warning.
    """

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
# 4. BASIC PACKAGE EXISTENCE
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
# 5. REQUIRED SUBDIRECTORIES
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 2 — REQUIRED DIRECTORIES")
print("=" * 110)

required_directories = [
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

for directory in required_directories:

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
# 7. CORE FIGURE AUDIT — SQUARE / TWO-COLUMN / WIDE
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 3 — CORE FIGURE ASPECT-RATIO PACKAGE")
print("=" * 110)

aspect_directories = {
    "square": SQUARE_DIR,
    "two_column": TWO_COLUMN_DIR,
    "wide": WIDE_DIR,
}

aspect_failures = []

for figure in CORE_FIGURES:

    for aspect_name, directory in aspect_directories.items():

        path = (
            directory
            /
            f"{figure}_{aspect_name}.png"
        )

        passed = path.is_file()

        record(
            f"{figure} — {aspect_name} PNG",
            passed,
            path
        )

        if not passed:
            aspect_failures.append(path)


# ======================================================================================
# 8. CORE FIGURE VECTOR / PDF PACKAGE
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 4 — CORE FIGURE PNG / SVG / PDF EXPORTS")
print("=" * 110)

core_export_missing = []

for figure in CORE_FIGURES:

    for directory, extension in [
        (PNG_DIR, "png"),
        (SVG_DIR, "svg"),
        (PDF_DIR, "pdf"),
    ]:

        path = (
            directory
            /
            f"{figure}.{extension}"
        )

        passed = path.is_file()

        record(
            f"{figure} — {extension.upper()} export",
            passed,
            path
        )

        if not passed:
            core_export_missing.append(
                path
            )


# ======================================================================================
# 9. 3D FIGURE AUDIT
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 5 — 3D FIGURES")
print("=" * 110)

THREED_FILES = [
    "09_3D_C_LNO_R_norm_stability",
    "10_3D_R_matrix_structure",
]

for figure in THREED_FILES:

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
# 10. ARCHITECTURE FIGURE AUDIT
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 6 — ARCHITECTURE FIGURE")
print("=" * 110)

ARCH_FILES = [
    "11_LNO_architecture_publication.png",
    "11_LNO_architecture_publication.svg",
    "11_LNO_architecture_publication.pdf",
]

for filename in ARCH_FILES:

    path = ARCH_DIR / filename

    record(
        f"Architecture file — {filename}",
        path.is_file(),
        path
    )


# ======================================================================================
# 11. GIF EXISTENCE + READABILITY
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


def audit_gif(
    name,
    path
):

    record(
        f"{name} exists",
        path.is_file(),
        path
    )

    if not path.is_file():
        return None

    try:

        with Image.open(path) as img:

            frame_count = int(
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
            f"{name} readable",
            True,
            f"{width}x{height}"
        )

        record(
            f"{name} contains multiple frames",
            frame_count >= 2,
            f"{frame_count} frames"
        )

        if duration is None:

            record_warning(
                f"{name} duration metadata",
                "No explicit duration metadata found."
            )

        else:

            record(
                f"{name} frame duration available",
                duration > 0,
                f"{duration} ms"
            )

        return {
            "frames": frame_count,
            "width": width,
            "height": height,
            "duration_ms": duration,
        }

    except Exception as exc:

        record(
            f"{name} readable",
            False,
            repr(exc)
        )

        return None


arch_gif_info = audit_gif(
    "Architecture GIF",
    ARCH_GIF
)

rnorm_gif_info = audit_gif(
    "3D R-norm GIF",
    R_NORM_GIF
)


# ======================================================================================
# 12. MANIFEST / README AUDIT
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


# ======================================================================================
# 13. MANIFEST CONTENT AUDIT
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 9 — MANIFEST CONTENT")
print("=" * 110)

if manifest_data is not None:

    record(
        "Manifest step equals 30",
        manifest_data.get("step") == 30,
        manifest_data.get("step")
    )

    record(
        "Manifest says no training",
        manifest_data.get("training_performed") is False,
        manifest_data.get("training_performed")
    )

    record(
        "Manifest says no new inference",
        manifest_data.get("new_model_inference") is False,
        manifest_data.get("new_model_inference")
    )

    record(
        "Manifest says checkpoint unmodified",
        manifest_data.get("checkpoint_modified") is False,
        manifest_data.get("checkpoint_modified")
    )

    manifest_core = set(
        manifest_data.get(
            "core_figures",
            []
        )
    )

    expected_core = set(
        CORE_FIGURES
    )

    record(
        "Manifest contains all 8 core figures",
        expected_core.issubset(manifest_core),
        f"{len(manifest_core & expected_core)}/8"
    )

    manifest_3d = set(
        manifest_data.get(
            "three_d_figures",
            []
        )
    )

    record(
        "Manifest contains both 3D figures",
        set(THREED_FILES).issubset(
            manifest_3d
        ),
        f"{len(set(THREED_FILES) & manifest_3d)}/2"
    )

    manifest_arch = set(
        manifest_data.get(
            "architecture",
            []
        )
    )

    record(
        "Manifest contains architecture figure",
        "11_LNO_architecture_publication"
        in manifest_arch,
        manifest_arch
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
    (
        STEP27_DIR
        /
        "three_model_comparison.csv"
    ),
    (
        STEP27_DIR
        /
        "regime_comparison.csv"
    ),
    (
        STEP28_DIR
        /
        "c_lno_rollout_details.csv"
    ),
    (
        STEP28_DIR
        /
        "c_lno_regime_summary.csv"
    ),
    (
        STEP28_DIR
        /
        "c_lno_global_summary.csv"
    ),
    (
        STEP29_DIR
        /
        "step29_final_summary.json"
    ),
]

for source_path in EVIDENCE_FILES:

    package_path = (
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
        package_path.is_file(),
        package_path
    )


# ======================================================================================
# 15. BYTE-FOR-BYTE EVIDENCE INTEGRITY
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
            lambda: f.read(
                1024 * 1024
            ),
            b""
        ):

            digest.update(
                chunk
            )

    return digest.hexdigest()


for source_path in EVIDENCE_FILES:

    package_path = (
        EVIDENCE_DIR
        /
        source_path.name
    )

    if (
        source_path.is_file()
        and package_path.is_file()
    ):

        source_hash = sha256_file(
            source_path
        )

        package_hash = sha256_file(
            package_path
        )

        record(
            f"Evidence byte integrity — {source_path.name}",
            source_hash == package_hash,
            source_hash
        )


# ======================================================================================
# 16. TEMPORARY DIRECTORY AUDIT
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 12 — TEMPORARY ARTIFACTS")
print("=" * 110)

temp_exists = TEMP_DIR.exists()

record(
    "Temporary GIF frame directory absent",
    not temp_exists,
    TEMP_DIR
)

if temp_exists:

    temp_files = [
        p
        for p in TEMP_DIR.rglob("*")
        if p.is_file()
    ]

    record_warning(
        "Temporary files remain",
        f"{len(temp_files)} file(s)"
    )


# ======================================================================================
# 17. PACKAGE FILE COUNTS
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 13 — PACKAGE FILE COUNTS")
print("=" * 110)

package_files = []

if OUT.is_dir():

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
    p
    for p in package_files
    if p.suffix.lower() == ".png"
]

svg_files = [
    p
    for p in package_files
    if p.suffix.lower() == ".svg"
]

pdf_files = [
    p
    for p in package_files
    if p.suffix.lower() == ".pdf"
]

gif_files = [
    p
    for p in package_files
    if p.suffix.lower() == ".gif"
]

json_files = [
    p
    for p in package_files
    if p.suffix.lower() == ".json"
]

txt_files = [
    p
    for p in package_files
    if p.suffix.lower() == ".txt"
]

print(
    f"[INFO] Package PNG count:  {len(png_files)}"
)

print(
    f"[INFO] Package SVG count:  {len(svg_files)}"
)

print(
    f"[INFO] Package PDF count:  {len(pdf_files)}"
)

print(
    f"[INFO] Package GIF count:  {len(gif_files)}"
)

record(
    "At least 27 PNG files present",
    len(png_files) >= 27,
    len(png_files)
)

record(
    "At least 27 SVG files present",
    len(svg_files) >= 27,
    len(svg_files)
)

record(
    "At least 27 PDF files present",
    len(pdf_files) >= 27,
    len(pdf_files)
)

record(
    "Exactly 2 GIF files present",
    len(gif_files) == 2,
    len(gif_files)
)


# ======================================================================================
# 18. ASPECT-RATIO COUNT AUDIT
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
    "8 square core PNG figures",
    len(square_pngs) == 8,
    len(square_pngs)
)

record(
    "8 two-column core PNG figures",
    len(two_column_pngs) == 8,
    len(two_column_pngs)
)

record(
    "8 wide core PNG figures",
    len(wide_pngs) == 8,
    len(wide_pngs)
)


# ======================================================================================
# 19. ZIP STRUCTURAL AUDIT
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 15 — ZIP PACKAGE")
print("=" * 110)

zip_names = []

if ZIP_PATH.is_file():

    try:

        with zipfile.ZipFile(
            ZIP_PATH,
            "r"
        ) as z:

            bad_member = z.testzip()

            record(
                "ZIP archive opens successfully",
                True,
                ZIP_PATH
            )

            record(
                "ZIP internal CRC integrity",
                bad_member is None,
                bad_member
            )

            zip_names = z.namelist()

            record(
                "ZIP contains files",
                len(zip_names) > 0,
                len(zip_names)
            )

    except Exception as exc:

        record(
            "ZIP archive opens successfully",
            False,
            repr(exc)
        )


# ======================================================================================
# 20. ZIP EXPECTED ARTIFACT AUDIT
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 16 — ZIP CONTENT")
print("=" * 110)


def zip_contains(
    relative_path
):

    # ZIP entries use POSIX separators.
    relative_path = str(
        relative_path
    ).replace(
        os.sep,
        "/"
    )

    expected = (
        OUT.name
        +
        "/"
        +
        relative_path
    )

    return expected in zip_names


zip_expected_files = [

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

for expected in zip_expected_files:

    record(
        f"ZIP contains {expected}",
        zip_contains(expected),
        expected
    )


# Core ZIP content checks

for figure in CORE_FIGURES:

    for aspect_name in [
        "square",
        "two_column",
        "wide",
    ]:

        expected = (
            f"{aspect_name.upper()}/"
            f"{figure}_{aspect_name}.png"
        )

        record(
            f"ZIP core figure — {expected}",
            zip_contains(expected),
            expected
        )


# ======================================================================================
# 21. ZIP TEMPORARY FILE AUDIT
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

    expected = (
        f"validated_evidence/"
        f"{source_path.name}"
    )

    record(
        f"ZIP contains evidence — {source_path.name}",
        zip_contains(expected),
        expected
    )


# ======================================================================================
# 23. OUTPUT SIZE AUDIT
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 19 — FILE SIZE SANITY")
print("=" * 110)


def file_size_mb(path):

    if not path.is_file():
        return 0.0

    return (
        path.stat().st_size
        /
        (1024 ** 2)
    )


zip_size_mb = file_size_mb(
    ZIP_PATH
)

record(
    "ZIP package has non-zero size",
    zip_size_mb > 0,
    f"{zip_size_mb:.2f} MB"
)

for gif_name, gif_path in [
    (
        "Architecture GIF",
        ARCH_GIF
    ),
    (
        "3D R-norm GIF",
        R_NORM_GIF
    ),
]:

    size_mb = file_size_mb(
        gif_path
    )

    record(
        f"{gif_name} has non-zero size",
        size_mb > 0,
        f"{size_mb:.3f} MB"
    )


# ======================================================================================
# 24. GIF FRAME EXPECTATION
# ======================================================================================

print("\n" + "=" * 110)
print("AUDIT 20 — GIF FRAME COUNTS")
print("=" * 110)

if arch_gif_info is not None:

    record(
        "Architecture GIF contains expected multi-frame animation",
        arch_gif_info["frames"] >= 50,
        arch_gif_info["frames"]
    )

if rnorm_gif_info is not None:

    record(
        "3D R-norm GIF contains exactly 100 frames",
        rnorm_gif_info["frames"] == 100,
        rnorm_gif_info["frames"]
    )


# ======================================================================================
# 25. READ README SANITY
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
# 26. FINAL AUDIT SUMMARY
# ======================================================================================

print("\n" + "=" * 110)
print("STEP 31 — FINAL AUDIT SUMMARY")
print("=" * 110)


total_checks = len(
    audit_results
)

passed_checks = sum(
    r["status"] == "PASS"
    for r in audit_results
)

failed_checks = sum(
    r["status"] == "FAIL"
    for r in audit_results
)

warning_count = len(
    warnings
)


audit_status = (
    "PASS"
    if failed_checks == 0
    else "FAIL"
)


print(
    f"\n[INFO] Total checks:   {total_checks}"
)

print(
    f"[INFO] Passed:         {passed_checks}"
)

print(
    f"[INFO] Failed:         {failed_checks}"
)

print(
    f"[INFO] Warnings:       {warning_count}"
)

print(
    f"[INFO] FINAL STATUS:   {audit_status}"
)


# ======================================================================================
# 27. WRITE MACHINE-READABLE AUDIT JSON
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
        "Final IEEE / Journal Publication Package Audit",

    "root":
        str(ROOT),

    "package_directory":
        str(OUT),

    "zip_path":
        str(ZIP_PATH),

    "status":
        audit_status,

    "total_checks":
        total_checks,

    "passed_checks":
        passed_checks,

    "failed_checks":
        failed_checks,

    "warnings":
        warning_count,

    "package_counts": {
        "png":
            len(png_files),
        "svg":
            len(svg_files),
        "pdf":
            len(pdf_files),
        "gif":
            len(gif_files),
        "json":
            len(json_files),
        "txt":
            len(txt_files),
    },

    "aspect_ratio_counts": {
        "square_png":
            len(square_pngs),
        "two_column_png":
            len(two_column_pngs),
        "wide_png":
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
        r
        for r in audit_results
        if r["status"] == "FAIL"
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
    }
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
    "\n[PASS] Machine-readable audit written:"
)

print(
    AUDIT_JSON
)


# ======================================================================================
# 28. WRITE HUMAN-READABLE AUDIT REPORT
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
        "STEP 31 — FINAL IEEE / JOURNAL PUBLICATION PACKAGE AUDIT\n"
    )

    f.write(
        "=" * 72
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
        f"Warnings: {warning_count}\n\n"
    )

    f.write(
        "PACKAGE COUNTS\n"
    )

    f.write(
        "-" * 72
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
        "ASPECT RATIO COUNTS\n"
    )

    f.write(
        "-" * 72
        +
        "\n"
    )

    f.write(
        f"Square PNG: {len(square_pngs)}\n"
    )

    f.write(
        f"Two-column PNG: {len(two_column_pngs)}\n"
    )

    f.write(
        f"Wide PNG: {len(wide_pngs)}\n\n"
    )

    f.write(
        "GIF INFORMATION\n"
    )

    f.write(
        "-" * 72
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
        "SCIENTIFIC INTEGRITY\n"
    )

    f.write(
        "-" * 72
        +
        "\n"
    )

    f.write(
        "Training performed: False\n"
    )

    f.write(
        "New model inference: False\n"
    )

    f.write(
        "Checkpoint modified: False\n"
    )

    f.write(
        "Source evidence modified: False\n\n"
    )

    f.write(
        "FAILED CHECKS\n"
    )

    f.write(
        "-" * 72
        +
        "\n"
    )

    if failed_checks == 0:

        f.write(
            "None\n"
        )

    else:

        for item in audit_results:

            if item["status"] == "FAIL":

                f.write(
                    f"- {item['check']}: "
                    f"{item['detail']}\n"
                )

    f.write(
        "\nWARNINGS\n"
    )

    f.write(
        "-" * 72
        +
        "\n"
    )

    if not warnings:

        f.write(
            "None\n"
        )

    else:

        for warning in warnings:

            f.write(
                f"- {warning}\n"
            )


print(
    "[PASS] Human-readable audit written:"
)

print(
    AUDIT_TXT
)


# ======================================================================================
# 29. FINAL STATUS
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
    f"\nAudit JSON:"
)

print(
    AUDIT_JSON
)

print(
    f"\nAudit TXT:"
)

print(
    AUDIT_TXT
)

print(
    f"\nPublication package:"
)

print(
    OUT
)

print(
    f"\nZIP:"
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


# ======================================================================================
# 30. CLEANUP
# ======================================================================================

gc.collect()
