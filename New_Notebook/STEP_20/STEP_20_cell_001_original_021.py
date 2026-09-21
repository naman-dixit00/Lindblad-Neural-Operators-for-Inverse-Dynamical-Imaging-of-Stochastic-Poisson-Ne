# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 21
# Step            : STEP_20
# Step Heading    : # STEP 20 — FINAL SPECTRAL STABILITY ANALYSIS
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 20 — FINAL SPECTRAL STABILITY ANALYSIS
#
# Uses the rollout produced by STEP 19.
#
# FNO vs LNO
# Horizon = 40
#
# Measures:
#   - Total spectral energy vs timestep
#   - High-frequency spectral growth
#   - High-frequency energy fraction
#   - Per-mode mean spectral energy
#
# NO TRAINING
# NO CHECKPOINT MODIFICATION
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

print("=" * 90)
print("STEP 20 — FINAL SPECTRAL STABILITY ANALYSIS")
print("=" * 90)

# ============================================================
# 1. SETTINGS
# ============================================================

USE_PILOT = False
HORIZON = 40
SEED = 2026

ROOT = (
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

# ============================================================
# 2. EXPECTED STEP-19 OUTPUTS
# ============================================================

ROLLOUT_DIR = os.path.join(
    ROOT,
    "results",
    "final_long_horizon"
)

ROLLOUT_CSV = os.path.join(
    ROLLOUT_DIR,
    "long_horizon_rollout.csv"
)

ROLLOUT_JSON = os.path.join(
    ROLLOUT_DIR,
    "long_horizon_summary.json"
)

assert os.path.isfile(
    ROLLOUT_CSV
), (
    "STEP 19 CSV not found:\n"
    + ROLLOUT_CSV
)

assert os.path.isfile(
    ROLLOUT_JSON
), (
    "STEP 19 summary not found:\n"
    + ROLLOUT_JSON
)

print(
    "[+] Rollout directory:",
    ROLLOUT_DIR
)

# ============================================================
# 3. LOAD TEST DATA
#
# We regenerate the same deterministic initial states used
# in the rollout.
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
    batch_size=32,
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
# 4. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "[+] Device:",
    DEVICE
)

# ============================================================
# 5. CHECKPOINTS
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
    LNO_NAME = (
        "learnable_dissipation_lno"
    )

else:

    assert os.path.isfile(
        LNO_ORIGINAL_CKPT
    )

    LNO_CKPT = LNO_ORIGINAL_CKPT
    LNO_NAME = (
        "original_lno"
    )

print(
    "[+] FNO:",
    FNO_CKPT
)

print(
    "[+] LNO:",
    LNO_CKPT
)

# ============================================================
# 6. LOAD MODELS
# ============================================================

from models.fno_model import FNOBaseline
from models.lno_model import LindbladNeuralOperator

# ------------------------------------------------------------
# FNO
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# LNO
# ------------------------------------------------------------

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
    "[+] FNO loaded."
)

print(
    "[+] LNO loaded."
)

# ============================================================
# 7. FORWARD FUNCTIONS
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

        gamma = gamma.unsqueeze(
            -1
        )

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

    alphas = torch.nn.functional.softplus(
        lno.raw_dissipative_alpha
    )

    for k in range(
        lno.depth
    ):

        spectral = (
            lno.spectral_layers[k](h)
        )

        pointwise = (
            lno.pointwise_layers[k](h)
        )

        x_linear = (
            spectral
            + pointwise
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

        LTL = L.t() @ L

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
            gamma.view(
                -1,
                1,
                1
            )
            * D
        )

        padded = torch.nn.functional.pad(
            x_linear,
            (2, 2),
            mode="circular"
        )

        coherent = torch.nn.functional.conv1d(
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

            h = torch.nn.functional.gelu(
                h
            )

    prediction = (
        lno.output_projection(
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
# 8. REGENERATE ROLLOUT STATES
#
# We store:
#   [timestep, sample, spatial_position]
#
# Shape:
#   (41, 3000, 128)
# ============================================================

print("\n" + "=" * 90)
print("REGENERATING ROLLOUT STATES FOR SPECTRAL ANALYSIS")
print("=" * 90)

fno_state_batches = []
lno_state_batches = []

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

    fno_current = inputs.clone()
    lno_current = inputs.clone()

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

    with torch.no_grad():

        for step in range(
            HORIZON
        ):

            fno_pred = fno_forward(
                fno_current
            )

            lno_pred = lno_forward(
                lno_current
            )

            fno_current = (
                fno_current.clone()
            )

            lno_current = (
                lno_current.clone()
            )

            fno_current[:, 0, :] = (
                fno_pred
            )

            lno_current[:, 0, :] = (
                lno_pred
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

    fno_state_batches.append(
        torch.stack(
            fno_states,
            dim=0
        )
    )

    lno_state_batches.append(
        torch.stack(
            lno_states,
            dim=0
        )
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

fno_states = torch.cat(
    fno_state_batches,
    dim=1
)

lno_states = torch.cat(
    lno_state_batches,
    dim=1
)

print(
    "\n[+] FNO states:",
    tuple(fno_states.shape)
)

print(
    "[+] LNO states:",
    tuple(lno_states.shape)
)

assert fno_states.shape == (
    HORIZON + 1,
    3000,
    128
)

assert lno_states.shape == (
    HORIZON + 1,
    3000,
    128
)

# ============================================================
# 9. FOURIER TRANSFORM
# ============================================================

print(
    "\n[+] Computing Fourier spectra..."
)

with torch.no_grad():

    fno_fft = torch.fft.rfft(
        fno_states,
        dim=-1
    )

    lno_fft = torch.fft.rfft(
        lno_states,
        dim=-1
    )

# Spectral energy
fno_spectral = (
    torch.abs(
        fno_fft
    ) ** 2
)

lno_spectral = (
    torch.abs(
        lno_fft
    ) ** 2
)

# ============================================================
# 10. TOTAL SPECTRAL ENERGY
# ============================================================

fno_total_energy = (
    fno_spectral
    .sum(dim=-1)
    .mean(dim=1)
    .numpy()
)

lno_total_energy = (
    lno_spectral
    .sum(dim=-1)
    .mean(dim=1)
    .numpy()
)

# Normalize to timestep 0
fno_total_growth = (
    fno_total_energy
    /
    (
        fno_total_energy[0]
        + 1e-30
    )
)

lno_total_growth = (
    lno_total_energy
    /
    (
        lno_total_energy[0]
        + 1e-30
    )
)

# ============================================================
# 11. HIGH-FREQUENCY REGION
# ============================================================

num_modes = (
    fno_spectral.shape[-1]
)

# Upper half of nonzero Fourier modes
high_start = max(
    1,
    num_modes // 2
)

high_slice = slice(
    high_start,
    num_modes
)

fno_high_energy = (
    fno_spectral[
        ...,
        high_slice
    ]
    .sum(dim=-1)
    .mean(dim=1)
    .numpy()
)

lno_high_energy = (
    lno_spectral[
        ...,
        high_slice
    ]
    .sum(dim=-1)
    .mean(dim=1)
    .numpy()
)

# Normalize high-frequency growth
fno_high_growth = (
    fno_high_energy
    /
    (
        fno_high_energy[0]
        + 1e-30
    )
)

lno_high_growth = (
    lno_high_energy
    /
    (
        lno_high_energy[0]
        + 1e-30
    )
)

# High-frequency fraction of total spectral energy
fno_high_fraction = (
    fno_high_energy
    /
    (
        fno_total_energy
        + 1e-30
    )
)

lno_high_fraction = (
    lno_high_energy
    /
    (
        lno_total_energy
        + 1e-30
    )
)

# ============================================================
# 12. PER-MODE MEAN SPECTRAL ENERGY
# ============================================================

fno_mean_by_mode = (
    fno_spectral
    .mean(dim=(0, 1))
    .numpy()
)

lno_mean_by_mode = (
    lno_spectral
    .mean(dim=(0, 1))
    .numpy()
)

modes = np.arange(
    num_modes
)

# ============================================================
# 13. SUMMARY VALUES
# ============================================================

fno_max_total_growth = (
    np.max(
        fno_total_growth
    )
)

lno_max_total_growth = (
    np.max(
        lno_total_growth
    )
)

fno_final_total_growth = (
    fno_total_growth[-1]
)

lno_final_total_growth = (
    lno_total_growth[-1]
)

fno_max_high_growth = (
    np.max(
        fno_high_growth
    )
)

lno_max_high_growth = (
    np.max(
        lno_high_growth
    )
)

fno_final_high_fraction = (
    fno_high_fraction[-1]
)

lno_final_high_fraction = (
    lno_high_fraction[-1]
)

# ============================================================
# 14. PRINT
# ============================================================

print("\n" + "=" * 90)
print("FINAL SPECTRAL STABILITY SUMMARY")
print("=" * 90)

print("\n[FNO]")

print(
    "  Max total spectral growth:",
    f"{fno_max_total_growth:.12e}"
)

print(
    "  Final total spectral growth:",
    f"{fno_final_total_growth:.12e}"
)

print(
    "  Max high-frequency growth:",
    f"{fno_max_high_growth:.12e}"
)

print(
    "  Final high-frequency fraction:",
    f"{fno_final_high_fraction:.12e}"
)

print("\n[LNO]")

print(
    "  Max total spectral growth:",
    f"{lno_max_total_growth:.12e}"
)

print(
    "  Final total spectral growth:",
    f"{lno_final_total_growth:.12e}"
)

print(
    "  Max high-frequency growth:",
    f"{lno_max_high_growth:.12e}"
)

print(
    "  Final high-frequency fraction:",
    f"{lno_final_high_fraction:.12e}"
)

# ============================================================
# 15. SAVE DIRECTORY
# ============================================================

OUT_DIR = os.path.join(
    ROOT,
    "results",
    "final_spectral_stability"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

# ============================================================
# 16. TIMESTEP CSV
# ============================================================

timestep_df = pd.DataFrame({

    "timestep":
        np.arange(
            HORIZON + 1
        ),

    "FNO_total_spectral_energy":
        fno_total_energy,

    "LNO_total_spectral_energy":
        lno_total_energy,

    "FNO_total_spectral_growth":
        fno_total_growth,

    "LNO_total_spectral_growth":
        lno_total_growth,

    "FNO_high_frequency_energy":
        fno_high_energy,

    "LNO_high_frequency_energy":
        lno_high_energy,

    "FNO_high_frequency_growth":
        fno_high_growth,

    "LNO_high_frequency_growth":
        lno_high_growth,

    "FNO_high_frequency_fraction":
        fno_high_fraction,

    "LNO_high_frequency_fraction":
        lno_high_fraction,
})

TIMESTEP_CSV = os.path.join(
    OUT_DIR,
    "spectral_stability_vs_timestep.csv"
)

timestep_df.to_csv(
    TIMESTEP_CSV,
    index=False
)

# ============================================================
# 17. MODE CSV
# ============================================================

mode_df = pd.DataFrame({

    "fourier_mode":
        modes,

    "FNO_mean_spectral_energy":
        fno_mean_by_mode,

    "LNO_mean_spectral_energy":
        lno_mean_by_mode,
})

MODE_CSV = os.path.join(
    OUT_DIR,
    "spectral_energy_vs_mode.csv"
)

mode_df.to_csv(
    MODE_CSV,
    index=False
)

# ============================================================
# 18. SUMMARY JSON
# ============================================================

SUMMARY_JSON = os.path.join(
    OUT_DIR,
    "spectral_stability_summary.json"
)

summary = {

    "status":
        "COMPLETE",

    "horizon":
        HORIZON,

    "test_samples":
        3000,

    "fourier_modes":
        int(num_modes),

    "high_frequency_start_mode":
        int(high_start),

    "model":
        LNO_NAME,

    "FNO": {

        "max_total_spectral_growth":
            float(
                fno_max_total_growth
            ),

        "final_total_spectral_growth":
            float(
                fno_final_total_growth
            ),

        "max_high_frequency_growth":
            float(
                fno_max_high_growth
            ),

        "final_high_frequency_fraction":
            float(
                fno_final_high_fraction
            ),
    },

    "LNO": {

        "max_total_spectral_growth":
            float(
                lno_max_total_growth
            ),

        "final_total_spectral_growth":
            float(
                lno_final_total_growth
            ),

        "max_high_frequency_growth":
            float(
                lno_max_high_growth
            ),

        "final_high_frequency_fraction":
            float(
                lno_final_high_fraction
            ),
    },

    "ground_truth_status":
        "Missing",

    "ground_truth_note":
        "Current dataset stores one-step targets "
        "rather than complete 40-step trajectories.",
}

with open(
    SUMMARY_JSON,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

# ============================================================
# 19. FIGURE 1 — SPECTRAL ENERGY BY MODE
# ============================================================

FIG1 = os.path.join(
    OUT_DIR,
    "fig_mean_spectral_energy_vs_mode.png"
)

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    modes,
    fno_mean_by_mode,
    label="FNO"
)

plt.plot(
    modes,
    lno_mean_by_mode,
    label="LNO"
)

plt.xlabel(
    "Fourier mode"
)

plt.ylabel(
    "Mean spectral energy"
)

plt.title(
    "Mean Spectral Energy vs Fourier Mode"
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
# 20. FIGURE 2 — HIGH FREQUENCY FRACTION
# ============================================================

FIG2 = os.path.join(
    OUT_DIR,
    "fig_high_frequency_fraction_vs_timestep.png"
)

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    np.arange(
        HORIZON + 1
    ),
    fno_high_fraction,
    label="FNO"
)

plt.plot(
    np.arange(
        HORIZON + 1
    ),
    lno_high_fraction,
    label="LNO"
)

plt.xlabel(
    "Autoregressive timestep"
)

plt.ylabel(
    "High-frequency spectral fraction"
)

plt.title(
    "High-Frequency Spectral Fraction"
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
# 21. FIGURE 3 — TOTAL SPECTRAL ENERGY
# ============================================================

FIG3 = os.path.join(
    OUT_DIR,
    "fig_total_spectral_energy_vs_timestep.png"
)

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    np.arange(
        HORIZON + 1
    ),
    fno_total_growth,
    label="FNO"
)

plt.plot(
    np.arange(
        HORIZON + 1
    ),
    lno_total_growth,
    label="LNO"
)

plt.xlabel(
    "Autoregressive timestep"
)

plt.ylabel(
    "Normalized total spectral energy"
)

plt.title(
    "Total Spectral Energy Growth"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIG3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 22. FINAL
# ============================================================

print("\n" + "=" * 90)
print("STEP 20 — SPECTRAL STABILITY ANALYSIS COMPLETE")
print("=" * 90)

print(
    "[+] FNO states:",
    tuple(fno_states.shape)
)

print(
    "[+] LNO states:",
    tuple(lno_states.shape)
)

print(
    "[+] Fourier modes:",
    num_modes
)

print(
    "[+] Timestep CSV:",
    TIMESTEP_CSV
)

print(
    "[+] Mode CSV:",
    MODE_CSV
)

print(
    "[+] Summary JSON:",
    SUMMARY_JSON
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

print(
    "\n[!] Scientific note:"
)

print(
    "    True multi-step RMSE against future ground truth "
    "is not computed because the current dataset stores "
    "one-step targets rather than complete trajectories."
)

print("=" * 90)
