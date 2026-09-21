# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 20
# Step            : STEP_19
# Step Heading    : # STEP 19 — FINAL LONG-HORIZON ROLLOUT
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 19 — FINAL LONG-HORIZON ROLLOUT
#
# FNO vs SELECTED LNO
#
# Horizon:
#   40 autoregressive steps
#
# SAME TEST SAMPLES
# SAME INITIAL CONDITIONS
#
# Measures:
#   - Energy drift
#   - Relative energy drift
#   - Max |state|
#   - Numerical finiteness
#   - FNO/LNO separation over time
#
# IMPORTANT:
#   The current dataset contains one-step targets only.
#   Therefore TRUE RMSE(t) against future ground truth is
#   NOT computed here.
#
# NO TRAINING
# NO CHECKPOINT MODIFICATION
# ============================================================

import os
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

print("=" * 90)
print("STEP 19 — FINAL FNO vs LNO LONG-HORIZON ROLLOUT")
print("=" * 90)

# ============================================================
# 1. SETTINGS
# ============================================================

USE_PILOT = False

HORIZON = 40
BATCH_SIZE = 32
SEED = 2026

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
    "[+] Rollout steps:",
    HORIZON
)

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
)

if USE_PILOT:

    assert os.path.isfile(
        LNO_PILOT_CKPT
    )

    LNO_CKPT = LNO_PILOT_CKPT
    LNO_NAME = "learnable_dissipation_lno"

else:

    assert os.path.isfile(
        LNO_ORIGINAL_CKPT
    )

    LNO_CKPT = LNO_ORIGINAL_CKPT
    LNO_NAME = "original_lno"

print(
    "[+] FNO checkpoint:",
    FNO_CKPT
)

print(
    "[+] LNO checkpoint:",
    LNO_CKPT
)

# ============================================================
# 3. TEST DATA
# ============================================================

TEST_DIR = os.path.join(
    ROOT,
    "dataset ",
    "test"
)

from data.dataset_loader import IonTransportDataset

test_dataset = IonTransportDataset(
    root_dir=TEST_DIR
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

assert len(test_dataset) == 3000

print(
    "[+] Test samples:",
    len(test_dataset)
)

# ============================================================
# 4. FNO
# ============================================================

from models.fno_model import FNOBaseline

fno = FNOBaseline(
    modes=16,
    width=64,
    depth=4,
    in_channels=6,
    grid_size=128
).to(DEVICE)

fno_ckpt = torch.load(
    FNO_CKPT,
    map_location=DEVICE
)

if "model_state_dict" in fno_ckpt:

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
# 5. LNO
# ============================================================

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

    lno.raw_dissipative_alpha = (
        torch.nn.Parameter(
            torch.zeros(
                4,
                dtype=torch.float32,
                device=DEVICE
            )
        )
    )

if "model_state_dict" in lno_ckpt:

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
# 6. FORWARD FUNCTIONS
# ============================================================

def fno_forward(
    x
):

    out = fno(
        x
    )

    if isinstance(
        out,
        dict
    ):

        if "next_state" in out:

            out = out[
                "next_state"
            ]

        elif "prediction" in out:

            out = out[
                "prediction"
            ]

        else:

            raise KeyError(
                str(out.keys())
            )

    if out.ndim == 3:

        if out.shape[-1] == 1:

            out = out.squeeze(-1)

        elif out.shape[1] == 1:

            out = out.squeeze(1)

    return out


def lno_forward(
    x
):

    state = x[:, 0, :]
    phi = x[:, 1, :]
    flux = x[:, 2, :]
    noise = x[:, 3, :]
    dissipation = x[:, 4, :]
    gamma = x[:, 5, :]

    # --------------------------------------------------------
    # Original LNO
    # --------------------------------------------------------

    if not USE_PILOT:

        out = lno(
            state=state,
            phi=phi,
            flux=flux,
            noise=noise,
            dissipation=dissipation,
            gamma=gamma,
        )

        if "next_state" in out:

            return out[
                "next_state"
            ]

        if "prediction" in out:

            return out[
                "prediction"
            ]

        raise KeyError(
            str(out.keys())
        )

    # --------------------------------------------------------
    # Learnable-alpha LNO
    # --------------------------------------------------------

    if gamma.dim() == 1:

        gamma = gamma.unsqueeze(-1)

    h = torch.stack(
        [
            state,
            phi,
            flux,
            noise,
            dissipation,
            gamma.repeat(
                1,
                state.shape[-1]
            )
        ],
        dim=-1
    )

    h = lno.input_projection(
        h
    )

    h = h.permute(
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

        spectral = (
            lno
            .spectral_layers[k](h)
        )

        pointwise = (
            lno
            .pointwise_layers[k](h)
        )

        x_linear = (
            spectral
            + pointwise
        )

        layer = (
            lno
            .dissipative_layers[k]
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

        h = (
            x_linear
            + coherent
            + alphas[k] * D
            + spatial
        )

        if k < lno.depth - 1:

            h = F.gelu(
                h
            )

    prediction = (
        lno
        .output_projection(
            h.permute(
                0,
                2,
                1
            )
        )
        .squeeze(-1)
    )

    return prediction

# ============================================================
# 7. ONE-STEP → ROLLOUT INPUT CONSTRUCTION
# ============================================================

def build_next_input(
    previous_input,
    prediction
):

    # The original input has 6 channels.
    #
    # We update the state channel with the predicted state.
    #
    # Other conditioning fields remain fixed for this rollout.
    #
    # This matches the existing autoregressive rollout protocol
    # used in the earlier analysis.

    next_input = (
        previous_input
        .clone()
    )

    next_input[:, 0, :] = (
        prediction
    )

    return next_input

# ============================================================
# 8. ENERGY
# ============================================================

def state_energy(
    state
):

    return torch.mean(
        state ** 2,
        dim=-1
    )

# ============================================================
# 9. STORAGE
# ============================================================

fno_energy_all = []
lno_energy_all = []

fno_states_all = []
lno_states_all = []

fno_finite_all = []
lno_finite_all = []

# ============================================================
# 10. ROLLOUT
# ============================================================

print("\n" + "=" * 90)
print("RUNNING LONG-HORIZON AUTOREGRESSIVE ROLLOUT")
print("=" * 90)

for batch_idx, (
    inputs,
    targets
) in enumerate(
    test_loader,
    start=1
):

    inputs = inputs.float().to(
        DEVICE
    )

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    fno_current = inputs.clone()
    lno_current = inputs.clone()

    # --------------------------------------------------------
    # Initial energy
    # --------------------------------------------------------

    fno_energy = [
        state_energy(
            fno_current[:, 0, :]
        )
        .detach()
        .cpu()
    ]

    lno_energy = [
        state_energy(
            lno_current[:, 0, :]
        )
        .detach()
        .cpu()
    ]

    # --------------------------------------------------------
    # Initial states
    # --------------------------------------------------------

    fno_states = [
        fno_current[:, 0, :]
        .detach()
        .cpu()
    ]

    lno_states = [
        lno_current[:, 0, :]
        .detach()
        .cpu()
    ]

    fno_finite = True
    lno_finite = True

    # --------------------------------------------------------
    # Autoregressive rollout
    # --------------------------------------------------------

    with torch.no_grad():

        for step in range(
            HORIZON
        ):

            # ------------------------------------------------
            # FNO
            # ------------------------------------------------

            fno_prediction = (
                fno_forward(
                    fno_current
                )
            )

            # ------------------------------------------------
            # LNO
            # ------------------------------------------------

            lno_prediction = (
                lno_forward(
                    lno_current
                )
            )

            # ------------------------------------------------
            # Finiteness
            # ------------------------------------------------

            if not torch.isfinite(
                fno_prediction
            ).all():

                fno_finite = False

            if not torch.isfinite(
                lno_prediction
            ).all():

                lno_finite = False

            # ------------------------------------------------
            # Update state
            # ------------------------------------------------

            fno_current = build_next_input(
                fno_current,
                fno_prediction
            )

            lno_current = build_next_input(
                lno_current,
                lno_prediction
            )

            # ------------------------------------------------
            # Record
            # ------------------------------------------------

            fno_energy.append(
                state_energy(
                    fno_current[:, 0, :]
                )
                .detach()
                .cpu()
            )

            lno_energy.append(
                state_energy(
                    lno_current[:, 0, :]
                )
                .detach()
                .cpu()
            )

            fno_states.append(
                fno_current[:, 0, :]
                .detach()
                .cpu()
            )

            lno_states.append(
                lno_current[:, 0, :]
                .detach()
                .cpu()
            )

    # --------------------------------------------------------
    # Store batch trajectories
    # --------------------------------------------------------

    fno_energy_all.append(
        torch.stack(
            fno_energy,
            dim=0
        )
    )

    lno_energy_all.append(
        torch.stack(
            lno_energy,
            dim=0
        )
    )

    fno_states_all.append(
        torch.stack(
            fno_states,
            dim=0
        )
    )

    lno_states_all.append(
        torch.stack(
            lno_states,
            dim=0
        )
    )

    fno_finite_all.append(
        fno_finite
    )

    lno_finite_all.append(
        lno_finite
    )

    if (
        batch_idx == 1
        or batch_idx % 25 == 0
        or batch_idx == len(test_loader)
    ):

        print(
            f"    Batch "
            f"{batch_idx:03d}/"
            f"{len(test_loader)}"
        )

# ============================================================
# 11. ASSEMBLE
# ============================================================

fno_energy = torch.cat(
    fno_energy_all,
    dim=1
)

lno_energy = torch.cat(
    lno_energy_all,
    dim=1
)

fno_states = torch.cat(
    fno_states_all,
    dim=1
)

lno_states = torch.cat(
    lno_states_all,
    dim=1
)

# Shapes:
#
# energy = [41, 3000]
# states = [41, 3000, 128]

print(
    "\n[+] FNO energy shape:",
    tuple(fno_energy.shape)
)

print(
    "[+] LNO energy shape:",
    tuple(lno_energy.shape)
)

print(
    "[+] FNO states shape:",
    tuple(fno_states.shape)
)

print(
    "[+] LNO states shape:",
    tuple(lno_states.shape)
)

# ============================================================
# 12. MEAN ENERGY CURVES
# ============================================================

mean_fno_energy = (
    fno_energy
    .mean(dim=1)
    .numpy()
)

mean_lno_energy = (
    lno_energy
    .mean(dim=1)
    .numpy()
)

initial_fno_energy = (
    mean_fno_energy[0]
)

initial_lno_energy = (
    mean_lno_energy[0]
)

final_fno_energy = (
    mean_fno_energy[-1]
)

final_lno_energy = (
    mean_lno_energy[-1]
)

fno_abs_drift = abs(
    final_fno_energy
    - initial_fno_energy
)

lno_abs_drift = abs(
    final_lno_energy
    - initial_lno_energy
)

fno_relative_drift = (
    fno_abs_drift
    /
    (
        abs(
            initial_fno_energy
        )
        + 1e-30
    )
)

lno_relative_drift = (
    lno_abs_drift
    /
    (
        abs(
            initial_lno_energy
        )
        + 1e-30
    )
)

# ============================================================
# 13. MAX STATE
# ============================================================

fno_max_abs = (
    torch.max(
        torch.abs(
            fno_states
        )
    )
    .item()
)

lno_max_abs = (
    torch.max(
        torch.abs(
            lno_states
        )
    )
    .item()
)

fno_numerically_finite = all(
    fno_finite_all
)

lno_numerically_finite = all(
    lno_finite_all
)

# ============================================================
# 14. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("FINAL LONG-HORIZON ROLLOUT RESULTS")
print("=" * 90)

print("\n[FNO]")

print(
    "  Initial energy:",
    f"{initial_fno_energy:.12e}"
)

print(
    "  Final energy:",
    f"{final_fno_energy:.12e}"
)

print(
    "  Final abs drift:",
    f"{fno_abs_drift:.12e}"
)

print(
    "  Final relative drift:",
    f"{fno_relative_drift:.12e}"
)

print(
    "  Max |state|:",
    f"{fno_max_abs:.12e}"
)

print(
    "  Numerically finite:",
    fno_numerically_finite
)

print("\n[LNO]")

print(
    "  Initial energy:",
    f"{initial_lno_energy:.12e}"
)

print(
    "  Final energy:",
    f"{final_lno_energy:.12e}"
)

print(
    "  Final abs drift:",
    f"{lno_abs_drift:.12e}"
)

print(
    "  Final relative drift:",
    f"{lno_relative_drift:.12e}"
)

print(
    "  Max |state|:",
    f"{lno_max_abs:.12e}"
)

print(
    "  Numerically finite:",
    lno_numerically_finite
)

# ============================================================
# 15. MODEL SEPARATION
# ============================================================

fno_mean_state = (
    fno_states
    .mean(dim=1)
)

lno_mean_state = (
    lno_states
    .mean(dim=1)
)

relative_separation = (
    torch.linalg.vector_norm(
        fno_mean_state
        - lno_mean_state,
        dim=-1
    )
    /
    (
        torch.linalg.vector_norm(
            fno_mean_state,
            dim=-1
        )
        + 1e-30
    )
).numpy()

# ============================================================
# 16. SAVE OUTPUT
# ============================================================

OUT_DIR = os.path.join(
    ROOT,
    "results",
    "final_long_horizon"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

# ------------------------------------------------------------
# CSV
# ------------------------------------------------------------

rollout_steps = np.arange(
    HORIZON + 1
)

rollout_df = pd.DataFrame({

    "step":
        rollout_steps,

    "FNO_mean_energy":
        mean_fno_energy,

    "LNO_mean_energy":
        mean_lno_energy,

    "FNO_abs_energy_drift":
        np.abs(
            mean_fno_energy
            - initial_fno_energy
        ),

    "LNO_abs_energy_drift":
        np.abs(
            mean_lno_energy
            - initial_lno_energy
        ),

    "relative_FNO_LNO_separation":
        relative_separation,
})

CSV_PATH = os.path.join(
    OUT_DIR,
    "long_horizon_rollout.csv"
)

rollout_df.to_csv(
    CSV_PATH,
    index=False
)

# ------------------------------------------------------------
# JSON
# ------------------------------------------------------------

SUMMARY_PATH = os.path.join(
    OUT_DIR,
    "long_horizon_summary.json"
)

summary = {

    "status":
        "COMPLETE",

    "horizon":
        HORIZON,

    "test_samples":
        len(test_dataset),

    "fno_checkpoint":
        FNO_CKPT,

    "lno_checkpoint":
        LNO_CKPT,

    "lno_model":
        LNO_NAME,

    "FNO": {

        "initial_energy":
            float(initial_fno_energy),

        "final_energy":
            float(final_fno_energy),

        "absolute_drift":
            float(fno_abs_drift),

        "relative_drift":
            float(fno_relative_drift),

        "max_abs_state":
            float(fno_max_abs),

        "numerically_finite":
            bool(fno_numerically_finite),
    },

    "LNO": {

        "initial_energy":
            float(initial_lno_energy),

        "final_energy":
            float(final_lno_energy),

        "absolute_drift":
            float(lno_abs_drift),

        "relative_drift":
            float(lno_relative_drift),

        "max_abs_state":
            float(lno_max_abs),

        "numerically_finite":
            bool(lno_numerically_finite),
    },

    "ground_truth_rmse_available":
        False,

    "ground_truth_note":
        "Current dataset stores one-step targets, "
        "not complete multi-step trajectories.",

}

with open(
    SUMMARY_PATH,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

# ============================================================
# 17. FIGURE 1 — ENERGY
# ============================================================

FIG1 = os.path.join(
    OUT_DIR,
    "fig_energy_vs_step.png"
)

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    rollout_steps,
    mean_fno_energy,
    label="FNO"
)

plt.plot(
    rollout_steps,
    mean_lno_energy,
    label="LNO"
)

plt.xlabel(
    "Autoregressive step"
)

plt.ylabel(
    "Mean state energy"
)

plt.title(
    "Long-Horizon Energy Evolution"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIG1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 18. FIGURE 2 — ENERGY DRIFT
# ============================================================

FIG2 = os.path.join(
    OUT_DIR,
    "fig_energy_drift.png"
)

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    rollout_steps,
    np.abs(
        mean_fno_energy
        - initial_fno_energy
    ),
    label="FNO"
)

plt.plot(
    rollout_steps,
    np.abs(
        mean_lno_energy
        - initial_lno_energy
    ),
    label="LNO"
)

plt.xlabel(
    "Autoregressive step"
)

plt.ylabel(
    "Absolute energy drift"
)

plt.title(
    "Long-Horizon Energy Drift"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIG2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 19. FIGURE 3 — FNO/LNO SEPARATION
# ============================================================

FIG3 = os.path.join(
    OUT_DIR,
    "fig_fno_lno_separation.png"
)

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    rollout_steps,
    relative_separation
)

plt.xlabel(
    "Autoregressive step"
)

plt.ylabel(
    "Relative FNO-LNO separation"
)

plt.title(
    "FNO vs LNO Long-Horizon Separation"
)

plt.tight_layout()

plt.savefig(
    FIG3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 20. FINAL
# ============================================================

print("\n" + "=" * 90)
print("STEP 19 — FINAL LONG-HORIZON ROLLOUT COMPLETE")
print("=" * 90)

print(
    "[+] Common horizon:",
    HORIZON
)

print(
    "[+] FNO relative drift:",
    f"{fno_relative_drift:.12e}"
)

print(
    "[+] LNO relative drift:",
    f"{lno_relative_drift:.12e}"
)

print(
    "[+] Ground-truth RMSE(t): NOT COMPUTED"
)

print(
    "[+] CSV:",
    CSV_PATH
)

print(
    "[+] JSON:",
    SUMMARY_PATH
)

print(
    "[+] Figure:",
    FIG1
)

print(
    "[+] Figure:",
    FIG2
)

print(
    "[+] Figure:",
    FIG3
)

print("=" * 90)
