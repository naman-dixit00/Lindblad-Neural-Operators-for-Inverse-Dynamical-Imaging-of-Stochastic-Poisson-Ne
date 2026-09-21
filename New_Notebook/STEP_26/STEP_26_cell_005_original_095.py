# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 95
# Step            : STEP_26
# Step Heading    : # Step 26 C-LNO checkpoint from the validated R-focused + soft-physics ablation.
# Step Cell No.   : 5
# ============================================================

# ======================================================================================
# GITHUB PACKAGES — ROBUST PYTHON PACKAGE PUBLISHER
# ======================================================================================
#
# TARGET:
#   lno_ion_transport==0.1.3
#
# DOES:
#   - Verify setup.py version
#   - Verify wheel + source distribution
#   - Safely extract GitHub PAT from pasted input
#   - Verify GitHub authentication
#   - Publish package to GitHub Packages
#
# DOES NOT:
#   - create/change releases
#   - create tags
#   - git commit
#   - git push
#   - modify setup.py
#   - modify source code
#   - train models
#   - run model inference
#   - modify checkpoints
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
# 1. CONFIGURATION
# ======================================================================================

GITHUB_OWNER = "naman-dixit00"

PACKAGE_NAME = "lno_ion_transport"

EXPECTED_VERSION = "0.1.3"

REGISTRY_URL = (
    f"https://pypi.pkg.github.com/{GITHUB_OWNER}/"
)


# ======================================================================================
# 2. FIND REPOSITORY
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
    "Could not find repository under /content."
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
print("GITHUB PACKAGES — ROBUST PYTHON PACKAGE PUBLISHER")
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
    "[+] Expected version:",
    EXPECTED_VERSION
)

print(
    "[+] Registry:",
    REGISTRY_URL
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


setup_version_match = re.search(
    r"version\s*=\s*['\"]([^'\"]+)['\"]",
    setup_text,
)


assert setup_version_match, (
    "Could not determine package version from setup.py."
)


SETUP_VERSION = setup_version_match.group(1)


print(
    "\n[PASS] setup.py located."
)

print(
    "[INFO] setup.py version:",
    SETUP_VERSION
)


assert SETUP_VERSION == EXPECTED_VERSION, (

    "\nVersion mismatch.\n"

    f"setup.py version: {SETUP_VERSION}\n"

    f"Expected version: {EXPECTED_VERSION}\n\n"

    "This script will NOT modify setup.py."

)


print(
    "[PASS] setup.py version is 0.1.3."
)


# ======================================================================================
# 4. LOCATE PACKAGE BUILD FILES
# ======================================================================================

DIST_DIR = ROOT / "dist"


assert DIST_DIR.is_dir(), (
    f"dist directory not found: {DIST_DIR}"
)


wheel_files = sorted(
    DIST_DIR.glob(
        f"{PACKAGE_NAME}-*.whl"
    )
)


sdist_files = sorted(
    DIST_DIR.glob(
        f"{PACKAGE_NAME}-*.tar.gz"
    )
)


assert wheel_files, (
    "No wheel found in dist/."
)


assert sdist_files, (
    "No source distribution found in dist/."
)


built_files = [
    *wheel_files,
    *sdist_files,
]


print(
    "\n[PASS] Built package files found:"
)


for p in built_files:

    print(
        "   ",
        p.name
    )


# ======================================================================================
# 5. ROBUST VERSION PARSING
# ======================================================================================

detected_versions = set()


# --------------------------
# Wheel
# --------------------------

wheel_pattern = re.compile(
    rf"^{re.escape(PACKAGE_NAME)}-"
    r"([0-9]+(?:\.[0-9]+)*)-"
)


for wheel in wheel_files:

    match = wheel_pattern.match(
        wheel.name
    )

    assert match, (
        f"Could not parse wheel filename: {wheel.name}"
    )

    detected_versions.add(
        match.group(1)
    )


# --------------------------
# Source distribution
# --------------------------

sdist_pattern = re.compile(
    rf"^{re.escape(PACKAGE_NAME)}-"
    r"([0-9]+(?:\.[0-9]+)*)\.tar\.gz$"
)


for sdist in sdist_files:

    match = sdist_pattern.match(
        sdist.name
    )

    assert match, (
        f"Could not parse source distribution: {sdist.name}"
    )

    detected_versions.add(
        match.group(1)
    )


print(
    "\n[INFO] Detected package versions:"
)

for v in sorted(
    detected_versions
):

    print(
        "   ",
        v
    )


assert detected_versions == {
    EXPECTED_VERSION
}, (

    "\nPackage artifact version mismatch.\n"

    f"Detected: {sorted(detected_versions)}\n"

    f"Expected: {EXPECTED_VERSION}"

)


print(
    "[PASS] Wheel and source distribution are both version 0.1.3."
)


# ======================================================================================
# 6. VERIFY NON-EMPTY FILES
# ======================================================================================

for p in built_files:

    assert p.stat().st_size > 0, (
        f"Empty package artifact: {p}"
    )


print(
    "[PASS] Package artifacts are non-empty."
)


# ======================================================================================
# 7. INSTALL TWINE
# ======================================================================================

print(
    "\n" + "=" * 110
)

print(
    "CHECK TWINE"
)

print(
    "=" * 110
)


pip_result = subprocess.run(

    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "twine",
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
    "[PASS] Twine available."
)


# ======================================================================================
# 8. READ PAT
# ======================================================================================

print(
    "\n" + "=" * 110
)

print(
    "GITHUB AUTHENTICATION"
)

print(
    "=" * 110
)


print(
    "Paste your GitHub Personal Access Token."
)

print(
    "The input may accidentally contain extra copied text."
)

print(
    "This script will extract a GitHub token automatically."
)

print(
    "Token is NOT printed back to the screen."
)


raw_token_input = getpass.getpass(
    "\nGitHub PAT: "
)


assert raw_token_input.strip(), (
    "No token input received."
)


# ======================================================================================
# 9. EXTRACT ACTUAL GITHUB TOKEN
# ======================================================================================
#
# Handles both common GitHub PAT families:
#
#   ghp_...
#   github_pat_...
#
# This avoids the exact problem where pasted explanatory text
# containing an em dash was accidentally included.
# ======================================================================================

token_candidates = []


# Classic PAT
classic_matches = re.findall(
    r"ghp_[A-Za-z0-9]+",
    raw_token_input
)


# Fine-grained PAT
fine_matches = re.findall(
    r"github_pat_[A-Za-z0-9_]+",
    raw_token_input
)


token_candidates.extend(
    classic_matches
)

token_candidates.extend(
    fine_matches
)


# Remove duplicates while preserving order.
token_candidates = list(
    dict.fromkeys(
        token_candidates
    )
)


if len(token_candidates) == 0:

    raise RuntimeError(

        "\nCould not identify a GitHub PAT in the pasted input.\n\n"

        "Please paste the token directly from GitHub.\n"

        "Do not type it manually with quotation marks.\n\n"

        "Expected token prefixes are normally:\n"

        "  ghp_\n"
        "  github_pat_\n"

    )


if len(token_candidates) > 1:

    raise RuntimeError(

        "\nMultiple GitHub-token-like strings were detected.\n"

        "Paste only one GitHub PAT."

    )


GITHUB_TOKEN = token_candidates[0]


print(
    "\n[PASS] GitHub token pattern detected."
)


# ======================================================================================
# 10. TOKEN ASCII CHECK
# ======================================================================================

try:

    GITHUB_TOKEN.encode(
        "ascii"
    )

except UnicodeEncodeError as e:

    raise RuntimeError(

        "\nThe extracted token unexpectedly contains non-ASCII "
        "characters.\n"

        "Please generate/copy the token again directly from GitHub."

    ) from e


print(
    "[PASS] Token encoding is valid ASCII."
)


# ======================================================================================
# 11. VERIFY TOKEN AGAINST GITHUB API
# ======================================================================================

print(
    "\n" + "=" * 110
)

print(
    "VERIFY GITHUB TOKEN"
)

print(
    "=" * 110
)


# We deliberately use requests via a subprocess-free Python call here.
# No secret is printed.

import requests


api_headers = {

    "Accept":
        "application/vnd.github+json",

    "Authorization":
        f"Bearer {GITHUB_TOKEN}",

    "X-GitHub-Api-Version":
        "2022-11-28",

}


user_response = requests.get(

    "https://api.github.com/user",

    headers=api_headers,

    timeout=30,

)


if user_response.status_code != 200:

    raise RuntimeError(

        "\nGitHub token verification failed.\n"

        f"HTTP status: {user_response.status_code}\n\n"

        "GitHub response:\n"

        +
        user_response.text

    )


github_user = user_response.json().get(
    "login"
)


print(
    "[PASS] GitHub authentication succeeded."
)

print(
    "[INFO] Authenticated account:",
    github_user
)


assert github_user, (
    "GitHub API did not return a login."
)


# ======================================================================================
# 12. VERIFY AUTHENTICATED USER
# ======================================================================================

assert (
    github_user.lower()
    ==
    GITHUB_OWNER.lower()
), (

    "\nAuthenticated GitHub user does not match the expected owner.\n"

    f"Expected: {GITHUB_OWNER}\n"

    f"Authenticated: {github_user}\n"

)


print(
    "[PASS] Authenticated account matches package owner."
)


# ======================================================================================
# 13. PUBLISH PACKAGE
# ======================================================================================

print(
    "\n" + "=" * 110
)

print(
    "PUBLISH PYTHON PACKAGE TO GITHUB PACKAGES"
)

print(
    "=" * 110
)


env = os.environ.copy()


env["TWINE_USERNAME"] = (
    GITHUB_OWNER
)


env["TWINE_PASSWORD"] = (
    GITHUB_TOKEN
)


env["TWINE_REPOSITORY_URL"] = (
    REGISTRY_URL
)


twine_command = [

    sys.executable,

    "-m",

    "twine",

    "upload",

    "--non-interactive",

    *[
        str(p)
        for p in built_files
    ],

]


publish_result = subprocess.run(

    twine_command,

    cwd=str(ROOT),

    env=env,

    text=True,

    stdout=subprocess.PIPE,

    stderr=subprocess.STDOUT,

    check=False,

)


print(
    publish_result.stdout
)


# ======================================================================================
# 14. HANDLE PUBLISH RESULT
# ======================================================================================

if publish_result.returncode != 0:

    output_lower = (
        publish_result.stdout
        .lower()
    )


    # Duplicate version
    if (
        "already exists"
        in output_lower
        or
        "file already exists"
        in output_lower
    ):

        raise RuntimeError(

            "\nGitHub Packages reports that version "
            f"{EXPECTED_VERSION} already exists.\n\n"

            "The existing package was NOT overwritten.\n"

            "Use the next version, e.g. 0.1.4."

        )


    # Authentication/permission
    if (
        "401"
        in output_lower
        or
        "403"
        in output_lower
        or
        "unauthorized"
        in output_lower
        or
        "forbidden"
        in output_lower
        or
        "permission"
        in output_lower
    ):

        raise RuntimeError(

            "\nGitHub Packages rejected the upload because "
            "of authentication or permissions.\n\n"

            "Check that the PAT has package write permission "
            "and that the authenticated account can publish "
            "packages for this repository."

        )


    raise RuntimeError(

        "\nGitHub Packages upload failed.\n\n"

        +
        publish_result.stdout

    )


# ======================================================================================
# 15. CLEAR SECRET VARIABLES
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

GITHUB_TOKEN = None

raw_token_input = None

token_candidates = None


gc.collect()


# ======================================================================================
# 16. FINAL
# ======================================================================================

print(
    "\n" + "=" * 110
)

print(
    "GITHUB PACKAGE PUBLISHED SUCCESSFULLY"
)

print(
    "=" * 110
)

print(
    "\nPackage:",
    PACKAGE_NAME
)

print(
    "Version:",
    EXPECTED_VERSION
)

print(
    "Registry:",
    REGISTRY_URL
)

print(
    "\nPackage page:"
)

print(
    f"https://github.com/"
    f"{GITHUB_OWNER}"
    f"?tab=packages"
)

print(
    "\n[PASS] Package publication complete."
)

print(
    "[PASS] Existing GitHub release was not modified."
)

print(
    "[PASS] No tag was created."
)

print(
    "[PASS] No git commit was performed."
)

print(
    "[PASS] No git push was performed."
)

print(
    "[PASS] setup.py was not modified."
)

print(
    "[PASS] No model training performed."
)

print(
    "[PASS] No model inference performed."
)

print(
    "[PASS] No checkpoint modified."
)

print(
    "=" * 110
)
