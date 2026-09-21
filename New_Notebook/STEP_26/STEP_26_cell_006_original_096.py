# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 96
# Step            : STEP_26
# Step Heading    : # Step 26 C-LNO checkpoint from the validated R-focused + soft-physics ablation.
# Step Cell No.   : 6
# ============================================================

# ======================================================================================
# PYPI PRODUCTION PUBLISH — lno_ion_transport 0.1.3
# ======================================================================================
#
# PURPOSE:
#   Publish the already-built Python package to REAL PyPI.
#
# PACKAGE:
#   lno_ion_transport
#
# VERSION:
#   0.1.3
#
# IMPORTANT:
#   - REAL PyPI, NOT GitHub Packages
#   - No GitHub Release modification
#   - No git commit
#   - No git push
#   - No source modification
#   - No setup.py modification
#   - No model training
#   - No model inference
#   - No checkpoint modification
#
# AUTH:
#   PyPI API token
#   Username must be: __token__
#
# ======================================================================================


# ======================================================================================
# 0. IMPORTS
# ======================================================================================

import os
import sys
import re
import gc
import subprocess
import getpass
from pathlib import Path


# ======================================================================================
# 1. CONFIG
# ======================================================================================

PACKAGE_NAME = "lno_ion_transport"

EXPECTED_VERSION = "0.1.3"

PYPI_UPLOAD_URL = "https://upload.pypi.org/legacy/"


# ======================================================================================
# 2. FIND PROJECT ROOT
# ======================================================================================

CONTENT_ROOT = Path("/content")

repo_candidates = [
    p
    for p in CONTENT_ROOT.iterdir()
    if (
        p.is_dir()
        and (p / "results").is_dir()
    )
]

assert repo_candidates, (
    "Project repository not found under /content."
)


preferred = [
    p
    for p in repo_candidates
    if p.name.startswith(
        "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging"
    )
]

ROOT = (
    preferred[0]
    if preferred
    else repo_candidates[0]
)

os.chdir(ROOT)


print("=" * 110)
print("PYPI PRODUCTION PUBLISH")
print("=" * 110)

print(
    "[+] Repository:",
    ROOT
)

print(
    "[+] Package:",
    PACKAGE_NAME
)

print(
    "[+] Target version:",
    EXPECTED_VERSION
)

print(
    "[+] PyPI upload endpoint:",
    PYPI_UPLOAD_URL
)


# ======================================================================================
# 3. VERIFY setup.py
# ======================================================================================

SETUP_PY = ROOT / "setup.py"

assert SETUP_PY.is_file(), (
    f"setup.py not found: {SETUP_PY}"
)


setup_text = SETUP_PY.read_text(
    encoding="utf-8"
)


version_match = re.search(
    r"version\s*=\s*['\"]([^'\"]+)['\"]",
    setup_text
)


assert version_match, (
    "Could not read package version from setup.py."
)


SETUP_VERSION = version_match.group(1)


print(
    "\n[PASS] setup.py version:",
    SETUP_VERSION
)


assert SETUP_VERSION == EXPECTED_VERSION, (

    "\nVersion mismatch.\n"

    f"setup.py = {SETUP_VERSION}\n"

    f"expected  = {EXPECTED_VERSION}\n"

    "This script will NOT modify setup.py."

)


print(
    "[PASS] setup.py matches version 0.1.3."
)


# ======================================================================================
# 4. LOCATE DIST
# ======================================================================================

DIST_DIR = ROOT / "dist"

assert DIST_DIR.is_dir(), (
    f"dist directory not found: {DIST_DIR}"
)


wheel_files = sorted(
    DIST_DIR.glob(
        f"{PACKAGE_NAME}-{EXPECTED_VERSION}-*.whl"
    )
)


sdist_files = sorted(
    DIST_DIR.glob(
        f"{PACKAGE_NAME}-{EXPECTED_VERSION}.tar.gz"
    )
)


assert wheel_files, (
    "Expected 0.1.3 wheel not found."
)


assert sdist_files, (
    "Expected 0.1.3 source distribution not found."
)


built_files = [
    *wheel_files,
    *sdist_files,
]


print(
    "\n[PASS] Expected PyPI artifacts found:"
)


for p in built_files:

    print(
        "   ",
        p.name,
        f"({p.stat().st_size / 1024:.1f} KB)"
    )


# ======================================================================================
# 5. ENSURE NO OTHER VERSION IS BEING UPLOADED
# ======================================================================================

all_dist_files = [
    p
    for p in DIST_DIR.iterdir()
    if p.is_file()
]


unexpected = [
    p
    for p in all_dist_files
    if (
        p not in built_files
        and
        p.name.startswith(
            PACKAGE_NAME + "-"
        )
    )
]


assert not unexpected, (
    "Additional package versions/files found in dist/:\n"
    +
    "\n".join(
        f"  - {p.name}"
        for p in unexpected
    )
)


print(
    "[PASS] dist/ contains only the intended package version."
)


# ======================================================================================
# 6. INSTALL / UPDATE BUILD TOOLS
# ======================================================================================

print(
    "\n[INFO] Installing build validation tools..."
)


pip_result = subprocess.run(

    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--upgrade",
        "twine",
        "build",
    ],

    text=True,

    stdout=subprocess.PIPE,

    stderr=subprocess.STDOUT,

    check=False,
)


assert pip_result.returncode == 0, (
    pip_result.stdout
)


print(
    "[PASS] build and twine are available."
)


# ======================================================================================
# 7. TWINE CHECK
# ======================================================================================
#
# This checks that package metadata and README rendering are valid
# before anything is uploaded.
#
# ======================================================================================

print(
    "\n" + "=" * 110
)

print(
    "PYPI PACKAGE VALIDATION"
)

print(
    "=" * 110
)


check_result = subprocess.run(

    [
        sys.executable,
        "-m",
        "twine",
        "check",
        *[
            str(p)
            for p in built_files
        ],
    ],

    cwd=str(ROOT),

    text=True,

    stdout=subprocess.PIPE,

    stderr=subprocess.STDOUT,

    check=False,
)


print(
    check_result.stdout
)


assert check_result.returncode == 0, (

    "\nTwine package validation failed.\n\n"

    "Fix the package metadata/README before publishing."

)


print(
    "[PASS] Twine metadata/README validation passed."
)


# ======================================================================================
# 8. READ PYPI TOKEN
# ======================================================================================

print(
    "\n" + "=" * 110
)

print(
    "PYPI AUTHENTICATION"
)

print(
    "=" * 110
)


print(
    "Create a PyPI API token from your PyPI account."
)

print(
    "Do NOT send the token to me."
)

print(
    "The username for a PyPI API token upload is:"
)

print(
    "    __token__"
)


PYPI_TOKEN = getpass.getpass(
    "\nPyPI API token: "
).strip()


assert PYPI_TOKEN, (
    "No PyPI API token entered."
)


# ======================================================================================
# 9. TOKEN SANITY
# ======================================================================================

try:

    PYPI_TOKEN.encode(
        "ascii"
    )

except UnicodeEncodeError as e:

    raise RuntimeError(

        "\nThe PyPI token contains non-ASCII characters.\n\n"

        "Paste the token directly from PyPI without quotes,\n"

        "spaces, or explanatory text."

    ) from e


assert PYPI_TOKEN.startswith(
    "pypi-"
), (

    "\nThis does not look like a PyPI API token.\n\n"

    "A PyPI API token normally begins with:\n"

    "    pypi-\n"

)


print(
    "[PASS] PyPI API token format detected."
)


# ======================================================================================
# 10. PYPI ENVIRONMENT
# ======================================================================================

env = os.environ.copy()


env["TWINE_USERNAME"] = "__token__"

env["TWINE_PASSWORD"] = PYPI_TOKEN

env["TWINE_REPOSITORY_URL"] = PYPI_UPLOAD_URL


# ======================================================================================
# 11. FINAL CONFIRMATION
# ======================================================================================

print(
    "\n" + "=" * 110
)

print(
    "READY TO PUBLISH TO REAL PYPI"
)

print(
    "=" * 110
)


print(
    "Package:",
    PACKAGE_NAME
)

print(
    "Version:",
    EXPECTED_VERSION
)

print(
    "Artifacts:",
    len(built_files)
)

print(
    "Destination:",
    "https://pypi.org/"
)


print(
    "\nThe following will be uploaded:"
)


for p in built_files:

    print(
        "   ",
        p.name
    )


# ======================================================================================
# 12. PUBLISH
# ======================================================================================

print(
    "\n[+] Uploading to production PyPI..."
)


upload_command = [

    sys.executable,

    "-m",

    "twine",

    "upload",

    "--non-interactive",

    "--repository-url",

    PYPI_UPLOAD_URL,

    *[
        str(p)
        for p in built_files
    ],

]


upload_result = subprocess.run(

    upload_command,

    cwd=str(ROOT),

    env=env,

    text=True,

    stdout=subprocess.PIPE,

    stderr=subprocess.STDOUT,

    check=False,

)


print(
    upload_result.stdout
)


# ======================================================================================
# 13. HANDLE FAILURE
# ======================================================================================

if upload_result.returncode != 0:

    output = upload_result.stdout

    lower = output.lower()


    # Duplicate package release
    if (
        "already exists"
        in lower
        or
        "file already exists"
        in lower
        or
        "400" in lower
    ):

        raise RuntimeError(

            "\nPyPI rejected the upload because the package/version "
            "may already exist.\n\n"

            f"Package: {PACKAGE_NAME}\n"

            f"Version: {EXPECTED_VERSION}\n\n"

            "PyPI releases are immutable; do not repeatedly upload "
            "the same version."

        )


    # Authentication failure
    if (
        "401" in lower
        or
        "403" in lower
        or
        "unauthorized" in lower
        or
        "forbidden" in lower
        or
        "invalid authentication" in lower
    ):

        raise RuntimeError(

            "\nPyPI authentication/permission failed.\n\n"

            "Verify:\n"

            "  - PyPI account is verified\n"

            "  - API token is valid\n"

            "  - username is __token__\n"

            "  - token begins with pypi-\n"

            "  - token has permission to upload this project"

        )


    raise RuntimeError(

        "\nPyPI upload failed.\n\n"

        +
        output

    )


# ======================================================================================
# 14. CLEAR SECRET
# ======================================================================================

env.pop(
    "TWINE_USERNAME",
    None
)

env.pop(
    "TWINE_PASSWORD",
    None
)

env.pop(
    "TWINE_REPOSITORY_URL",
    None
)

PYPI_TOKEN = None


gc.collect()


# ======================================================================================
# 15. FINAL
# ======================================================================================

print(
    "\n" + "=" * 110
)

print(
    "PYPI PUBLISH SUCCESSFUL"
)

print(
    "=" * 110
)


print(
    "\nPackage:"
)

print(
    PACKAGE_NAME
)


print(
    "\nVersion:"
)

print(
    EXPECTED_VERSION
)


print(
    "\nPyPI page:"
)

print(
    f"https://pypi.org/project/"
    f"{PACKAGE_NAME}/"
)


print(
    "\nInstall command:"
)

print(
    f"pip install {PACKAGE_NAME}=={EXPECTED_VERSION}"
)


print(
    "\n[PASS] Package published to production PyPI."
)

print(
    "[PASS] No GitHub Release modified."
)

print(
    "[PASS] No git commit performed."
)

print(
    "[PASS] No git push performed."
)

print(
    "[PASS] No source code modified."
)

print(
    "[PASS] No setup.py modification."
)

print(
    "[PASS] No model training."
)

print(
    "[PASS] No model inference."
)

print(
    "[PASS] No checkpoint modification."
)

print(
    "\n" + "=" * 110
)
