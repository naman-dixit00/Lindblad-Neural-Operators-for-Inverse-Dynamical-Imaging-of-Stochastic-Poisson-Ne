# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 19
# Step            : STEP_18
# Step Heading    : # STEP 18 — FINAL OOD SIGMA SWEEP
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 18 — FINAL OOD SIGMA SWEEP
#
# FNO vs FINAL/SELECTED LNO
#
# Sigma values:
#   0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5
#
# SAME TEST SET
# SAME INPUT GEOMETRY
# SAME TARGETS
# SAME RANDOM SEED
#
# NO TRAINING
# NO CHECKPOINT MODIFICATION
# ============================================================

import os
import json
import hashlib
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader

print("=" * 90)
print("STEP 18 — FINAL SYSTEMATIC OOD SIGMA SWEEP")
print("=" * 90)

# ============================================================
# 1. SETTINGS
# ============================================================

USE_PILOT = False

SEED = 2026

SIGMA_VALUES = [
    0.0,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
]

BATCH_SIZE = 32

ROOT = (
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "[+] Device:",
    DEVICE
)

print(
    "[+] Seed:",
    SEED
)

print(
    "[+] Sigma values:",
    SIGMA_VALUES
)

# ============================================================
# 2. CHECKPOINT PATHS
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
    LNO_NAME = (
        "learnable_dissipation_lno"
    )

else:

    assert os.path.isfile(
        LNO_ORIGINAL_CKPT
    ), LNO_ORIGINAL_CKPT

    LNO_CKPT = LNO_ORIGINAL_CKPT
    LNO_NAME = (
        "original_lno"
    )

print(
    "\n[+] FNO checkpoint:",
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
    "[+] FNO SHA256:",
    fno_sha
)

print(
    "[+] LNO SHA256:",
    lno_sha
)

# ============================================================
# 4. TEST DATASET
# ============================================================

TEST_DIR = os.path.join(
    ROOT,
    "dataset ",
    "test"
)

assert os.path.isdir(
    TEST_DIR
)

from data.dataset_loader import IonTransportDataset

test_dataset = IonTransportDataset(
    root_dir=TEST_DIR
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

assert len(test_dataset) == 3000

print(
    "\n[+] Test samples:",
    len(test_dataset)
)

print(
    "[+] Test batches:",
    len(test_loader)
)

# ============================================================
# 5. LOAD FNO
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "LOADING FNO"
)

print(
    "=" * 90
)

from models.fno_model import FNOBaseline

fno = FNOBaseline(
    modes=16,
    width=64,
    depth=4,
    in_channels=6,
    grid_size=128,
).to(DEVICE)

fno_ckpt = torch.load(
    FNO_CKPT,
    map_location=DEVICE
)

if (
    "model_state_dict"
    in fno_ckpt
):

    fno.load_state_dict(
        fno_ckpt[
            "model_state_dict"
        ],
        strict=True
    )

else:

    fno.load_state_dict(
        fno_ckpt,
        strict=True
    )

fno.eval()

print(
    "[+] FNO loaded."
)

# ============================================================
# 6. LOAD LNO
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "LOADING LNO"
)

print(
    "=" * 90
)

from models.lno_model import (
    LindbladNeuralOperator
)

lno = LindbladNeuralOperator(
    modes=16,
    width=64,
    depth=4,
    in_channels=6
).to(DEVICE)

lno_ckpt = torch.load(
    LNO_CKPT,
    map_location=DEVICE
)

if USE_PILOT:

    # Learnable alpha parameter must exist
    # BEFORE loading the pilot checkpoint.

    lno.raw_dissipative_alpha = (
        torch.nn.Parameter(
            torch.zeros(
                4,
                dtype=torch.float32,
                device=DEVICE
            )
        )
    )

if (
    "model_state_dict"
    in lno_ckpt
):

    lno.load_state_dict(
        lno_ckpt[
            "model_state_dict"
        ],
        strict=True
    )

else:

    lno.load_state_dict(
        lno_ckpt,
        strict=True
    )

lno.eval()

print(
    "[+] LNO loaded:",
    LNO_NAME
)

# ============================================================
# 7. FNO FORWARD
# ============================================================

def fno_forward(
    inputs
):

    # FNO baseline accepts packed
    # [B, 6, 128] according to the
    # verified test geometry.

    output = fno(
        inputs
    )

    if isinstance(
        output,
        dict
    ):

        if "next_state" in output:

            output = output[
                "next_state"
            ]

        elif "prediction" in output:

            output = output[
                "prediction"
            ]

        else:

            raise KeyError(
                "Unknown FNO output keys: "
                + str(
                    output.keys()
                )
            )

    if output.ndim == 3:

        if output.shape[-1] == 1:

            output = output.squeeze(-1)

        elif output.shape[1] == 1:

            output = output.squeeze(1)

    return output

# ============================================================
# 8. LNO FORWARD
# ============================================================

def lno_forward(
    inputs
):

    state = inputs[:, 0, :]
    phi = inputs[:, 1, :]
    flux = inputs[:, 2, :]
    noise = inputs[:, 3, :]
    dissipation = inputs[:, 4, :]
    gamma = inputs[:, 5, :]

    # --------------------------------------------------------
    # Original LNO
    # --------------------------------------------------------

    if not USE_PILOT:

        output = lno(
            state=state,
            phi=phi,
            flux=flux,
            noise=noise,
            dissipation=dissipation,
            gamma=gamma,
        )

        if "next_state" in output:

            return output[
                "next_state"
            ]

        if "prediction" in output:

            return output[
                "prediction"
            ]

        raise KeyError(
            "Unknown LNO output keys."
        )

    # --------------------------------------------------------
    # Learnable-alpha LNO
    # --------------------------------------------------------

    if gamma.dim() == 1:

        gamma = gamma.unsqueeze(
            -1
        )

    x = torch.stack(
        [
            state,
            phi,
            flux,
            noise,
            dissipation,
            gamma.repeat(
                1,
                state.shape[-1]
            ),
        ],
        dim=-1
    )

    x = lno.input_projection(
        x
    )

    x = x.permute(
        0,
        2,
        1
    )

    alphas = F.softplus(
        lno.raw_dissipative_alpha
    )

    for k in range(
        lno.depth
    ):

        spectral_update = (
            lno.spectral_layers[k](x)
        )

        pointwise_update = (
            lno.pointwise_layers[k](x)
        )

        x_linear = (
            spectral_update
            + pointwise_update
        )

        layer = (
            lno.dissipative_layers[k]
        )

        L = layer.jump_transform

        xp = x_linear.permute(
            0,
            2,
            1
        )

        LTL = (
            L.t()
            @ L
        )

        D = (
            (xp @ L.t()) @ L
            - 0.5 * (
                xp @ LTL
                + xp @ LTL.t()
            )
        )

        D = D.permute(
            0,
            2,
            1
        )

        D = (
            gamma
            .view(
                -1,
                1,
                1
            )
            * D
        )

        padded = F.pad(
            x_linear,
            (2, 2),
            mode="circular"
        )

        coherent = F.conv1d(
            padded,
            layer.coherent_kernel
        )

        left = torch.roll(
            x_linear,
            -1,
            dims=-1
        )

        right = torch.roll(
            x_linear,
            1,
            dims=-1
        )

        spatial = (
            layer.spatial_diffusion_weight
            * (
                left
                - 2.0 * x_linear
                + right
            )
        )

        x = (
            x_linear
            + coherent
            + alphas[k] * D
            + spatial
        )

        if k < lno.depth - 1:

            x = F.gelu(
                x
            )

    prediction = (
        lno.output_projection(
            x.permute(
                0,
                2,
                1
            )
        )
        .squeeze(-1)
    )

    return prediction

# ============================================================
# 9. SANITY CHECK
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "FORWARD SANITY CHECK"
)

print(
    "=" * 90
)

with torch.no_grad():

    sx, sy = next(
        iter(test_loader)
    )

    sx = sx.float().to(
        DEVICE
    )

    sy = sy.float().to(
        DEVICE
    )

    fp = fno_forward(
        sx
    )

    lp = lno_forward(
        sx
    )

assert fp.shape == sy.shape
assert lp.shape == sy.shape

assert torch.isfinite(
    fp
).all()

assert torch.isfinite(
    lp
).all()

print(
    "[+] FNO shape:",
    tuple(fp.shape)
)

print(
    "[+] LNO shape:",
    tuple(lp.shape)
)

print(
    "[PASS] Forward sanity check."
)

# ============================================================
# 10. OOD SWEEP
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "BEGIN FINAL OOD SWEEP"
)

print(
    "=" * 90
)

rows = []

for sigma in SIGMA_VALUES:

    # Reinitialize deterministic RNG for
    # each sigma so FNO and LNO receive
    # EXACTLY the same perturbations.

    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        SEED
        +
        int(
            round(
                sigma * 1000
            )
        )
    )

    fno_sq_error = 0.0
    lno_sq_error = 0.0

    fno_abs_error = 0.0
    lno_abs_error = 0.0

    total_elements = 0

    print(
        f"\n--- sigma={sigma:.2f} ---"
    )

    for batch_idx, (
        clean_inputs,
        targets
    ) in enumerate(
        test_loader,
        start=1
    ):

        clean_inputs = (
            clean_inputs
            .float()
        )

        targets = (
            targets
            .float()
            .to(DEVICE)
        )

        # ----------------------------------------------------
        # Same perturbation for both models
        # ----------------------------------------------------

        noise = torch.randn(
            clean_inputs.shape,
            generator=generator,
            dtype=clean_inputs.dtype
        )

        perturbed_inputs = (
            clean_inputs
            + sigma * noise
        )

        perturbed_inputs = (
            perturbed_inputs
            .to(DEVICE)
        )

        # ----------------------------------------------------
        # FNO
        # ----------------------------------------------------

        with torch.no_grad():

            fno_pred = fno_forward(
                perturbed_inputs
            )

            # ------------------------------------------------
            # LNO
            # ------------------------------------------------

            lno_pred = lno_forward(
                perturbed_inputs
            )

        # ----------------------------------------------------
        # Finite
        # ----------------------------------------------------

        if not torch.isfinite(
            fno_pred
        ).all():

            raise RuntimeError(
                f"Non-finite FNO output at "
                f"sigma={sigma}, "
                f"batch={batch_idx}"
            )

        if not torch.isfinite(
            lno_pred
        ).all():

            raise RuntimeError(
                f"Non-finite LNO output at "
                f"sigma={sigma}, "
                f"batch={batch_idx}"
            )

        # ----------------------------------------------------
        # Errors
        # ----------------------------------------------------

        fno_error = (
            fno_pred
            - targets
        )

        lno_error = (
            lno_pred
            - targets
        )

        fno_sq_error += (
            torch.sum(
                fno_error
                ** 2
            ).item()
        )

        lno_sq_error += (
            torch.sum(
                lno_error
                ** 2
            ).item()
        )

        fno_abs_error += (
            torch.sum(
                torch.abs(
                    fno_error
                )
            ).item()
        )

        lno_abs_error += (
            torch.sum(
                torch.abs(
                    lno_error
                )
            ).item()
        )

        total_elements += (
            fno_error.numel()
        )

    # --------------------------------------------------------
    # Global metrics
    # --------------------------------------------------------

    fno_mse = (
        fno_sq_error
        /
        total_elements
    )

    lno_mse = (
        lno_sq_error
        /
        total_elements
    )

    fno_rmse = (
        fno_mse
        ** 0.5
    )

    lno_rmse = (
        lno_mse
        ** 0.5
    )

    fno_mae = (
        fno_abs_error
        /
        total_elements
    )

    lno_mae = (
        lno_abs_error
        /
        total_elements
    )

    # --------------------------------------------------------
    # Improvement
    #
    # Positive = LNO better
    # --------------------------------------------------------

    lno_improvement = (
        (
            fno_rmse
            - lno_rmse
        )
        /
        (
            fno_rmse
            + 1e-30
        )
        * 100.0
    )

    winner = (
        "LNO"
        if lno_rmse < fno_rmse
        else
        "FNO"
        if fno_rmse < lno_rmse
        else
        "Tie"
    )

    rows.append({

        "sigma":
            sigma,

        "FNO_RMSE":
            fno_rmse,

        "LNO_RMSE":
            lno_rmse,

        "FNO_MAE":
            fno_mae,

        "LNO_MAE":
            lno_mae,

        "LNO_improvement_percent":
            lno_improvement,

        "Winner":
            winner,

        "FNO_finite":
            True,

        "LNO_finite":
            True,
    })

    print(
        f"sigma={sigma:>4.2f} | "
        f"FNO RMSE={fno_rmse:.10e} | "
        f"LNO RMSE={lno_rmse:.10e} | "
        f"LNO improvement={lno_improvement:+.4f}% | "
        f"Winner={winner}"
    )

# ============================================================
# 11. SUMMARY TABLE
# ============================================================

ood_df = pd.DataFrame(
    rows
)

print(
    "\n" + "=" * 90
)

print(
    "FINAL OOD SIGMA SWEEP SUMMARY"
)

print(
    "=" * 90
)

display(
    ood_df.round(10)
)

# ============================================================
# 12. BEST / WORST OOD POINTS
# ============================================================

best_row = ood_df.loc[
    ood_df[
        "LNO_improvement_percent"
    ].idxmax()
]

worst_row = ood_df.loc[
    ood_df[
        "LNO_improvement_percent"
    ].idxmin()
]

print(
    "\n[+] Maximum LNO improvement:"
)

print(
    f"    sigma={best_row['sigma']}"
)

print(
    f"    improvement="
    f"{best_row['LNO_improvement_percent']:+.6f}%"
)

print(
    "\n[+] Minimum LNO improvement:"
)

print(
    f"    sigma={worst_row['sigma']}"
)

print(
    f"    improvement="
    f"{worst_row['LNO_improvement_percent']:+.6f}%"
)

# ============================================================
# 13. WIN COUNTS
# ============================================================

lno_wins = int(
    (
        ood_df["Winner"]
        == "LNO"
    ).sum()
)

fno_wins = int(
    (
        ood_df["Winner"]
        == "FNO"
    ).sum()
)

ties = int(
    (
        ood_df["Winner"]
        == "Tie"
    ).sum()
)

print(
    "\n[+] OOD win count:"
)

print(
    "    LNO:",
    lno_wins
)

print(
    "    FNO:",
    fno_wins
)

print(
    "    Tie:",
    ties
)

# ============================================================
# 14. SAVE
# ============================================================

OUT_DIR = os.path.join(
    ROOT,
    "results",
    "final_ood"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

CSV_PATH = os.path.join(
    OUT_DIR,
    "final_fno_vs_lno_ood_sigma_sweep.csv"
)

JSON_PATH = os.path.join(
    OUT_DIR,
    "final_fno_vs_lno_ood_sigma_sweep.json"
)

ood_df.to_csv(
    CSV_PATH,
    index=False
)

summary = {

    "status":
        "COMPLETE",

    "seed":
        SEED,

    "sigma_values":
        SIGMA_VALUES,

    "test_samples":
        len(test_dataset),

    "batch_size":
        BATCH_SIZE,

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

    "results":
        rows,

    "lno_wins":
        lno_wins,

    "fno_wins":
        fno_wins,

    "ties":
        ties,
}

with open(
    JSON_PATH,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

# ============================================================
# 15. FINAL
# ============================================================

print(
    "\n" + "=" * 90
)

print(
    "STEP 18 — FINAL OOD SIGMA SWEEP COMPLETE"
)

print(
    "=" * 90
)

print(
    "[+] CSV:",
    CSV_PATH
)

print(
    "[+] JSON:",
    JSON_PATH
)

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

print("=" * 90)
