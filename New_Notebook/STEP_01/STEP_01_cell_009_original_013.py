# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 13
# Step            : STEP_01
# Step Heading    : # PHASE 1 — STEP 1: REPOSITORY + TENSOR GEOMETRY AUDIT
# Step Cell No.   : 9
# ============================================================

# ============================================================
# LNO — DISSIPATION SCALE SWEEP
# Purpose:
#   Determine whether the very small dissipative contribution
#   is the reason for the clean-test RMSE gap.
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

print("=" * 90)
print("ORIGINAL LNO — DISSIPATION SCALE SWEEP")
print("=" * 90)

model = old_lno
model.eval()

device = next(
    model.parameters()
).device

# ------------------------------------------------------------
# test batch
# ------------------------------------------------------------

safe_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

inputs, targets = next(
    iter(safe_loader)
)

inputs = inputs.float().to(device)
targets = targets.float().to(device)

state = inputs[:, 0, :]
phi = inputs[:, 1, :]
flux = inputs[:, 2, :]
noise = inputs[:, 3, :]
dissipation = inputs[:, 4, :]
base_gamma = inputs[:, 5, 0]

# ------------------------------------------------------------
# Baseline input
# ------------------------------------------------------------

gamma_input = (
    base_gamma
    .unsqueeze(-1)
    .repeat(
        1,
        128
    )
)

packed = torch.stack(
    [
        state,
        phi,
        flux,
        noise,
        dissipation,
        gamma_input
    ],
    dim=-1
)

# ------------------------------------------------------------
# Sweep values
# ------------------------------------------------------------

scales = [
    0.0,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
]

results = []

# ------------------------------------------------------------
# forward using manually modified dissipative scale
# ------------------------------------------------------------

def forward_with_diss_scale(scale):

    with torch.no_grad():

        h = (
            model
            .input_projection(packed)
            .permute(
                0,
                2,
                1
            )
        )

        for layer_idx in range(4):

            layer = (
                model
                .dissipative_layers[layer_idx]
            )

            spectral = (
                model
                .spectral_layers[layer_idx](h)
            )

            pointwise = (
                model
                .pointwise_layers[layer_idx](h)
            )

            x_linear = (
                spectral
                + pointwise
            )

            # ------------------------------------------------
            # Lindblad term
            # ------------------------------------------------

            L = (
                layer.jump_transform
            )

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
                base_gamma
                .view(-1, 1, 1)
                * D
            )

            # ------------------------------------------------
            # Coherent
            # ------------------------------------------------

            padded = F.pad(
                x_linear,
                (2, 2),
                mode="circular"
            )

            coherent = F.conv1d(
                padded,
                layer.coherent_kernel
            )

            # ------------------------------------------------
            # Spatial
            # ------------------------------------------------

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

            # ------------------------------------------------
            # SCALED DISSIPATION
            # ------------------------------------------------

            h = (
                x_linear
                + coherent
                + scale * D
                + spatial
            )

            if layer_idx < 3:

                h = F.gelu(
                    h
                )

        # ----------------------------------------------------
        # Actual output projection
        # ----------------------------------------------------

        output = model.output_projection(
            h.permute(
                0,
                2,
                1
            )
        )

        # Normalize output shape
        if output.ndim == 3:

            if output.shape[-1] == 1:

                output = output.squeeze(
                    -1
                )

            elif output.shape[1] == 1:

                output = output.squeeze(
                    1
                )

        return output


# ------------------------------------------------------------
# Baseline reference
# ------------------------------------------------------------

baseline_prediction = forward_with_diss_scale(
    1.0
)

baseline_mse = F.mse_loss(
    baseline_prediction,
    targets
).item()

baseline_rmse = (
    baseline_mse ** 0.5
)

print(
    "\n[+] Baseline scale=1 MSE:",
    f"{baseline_mse:.12e}"
)

print(
    "[+] Baseline scale=1 RMSE:",
    f"{baseline_rmse:.12e}"
)

# ------------------------------------------------------------
# Sweep
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("SWEEP")
print("=" * 90)

for scale in scales:

    prediction = forward_with_diss_scale(
        scale
    )

    mse = F.mse_loss(
        prediction,
        targets
    ).item()

    rmse = (
        mse ** 0.5
    )

    output_change = (
        torch.linalg.vector_norm(
            prediction
            - baseline_prediction
        )
        /
        (
            torch.linalg.vector_norm(
                baseline_prediction
            )
            + 1e-30
        )
    ).item()

    results.append({

        "dissipation_scale":
            scale,

        "mse":
            mse,

        "rmse":
            rmse,

        "delta_mse":
            mse
            - baseline_mse,

        "relative_output_change":
            output_change,
    })

    print(
        f"scale={scale:7.1f} | "
        f"RMSE={rmse:.12e} | "
        f"dMSE={mse-baseline_mse:+.12e} | "
        f"output_change={output_change:.12e}"
    )

# ------------------------------------------------------------
# Table
# ------------------------------------------------------------

df = pd.DataFrame(
    results
)

print("\n" + "=" * 90)
print("DISSIPATION SCALE SWEEP SUMMARY")
print("=" * 90)

display(
    df.round(12)
)

# ------------------------------------------------------------
# Best scale
# ------------------------------------------------------------

best = df.loc[
    df["mse"].idxmin()
]

print(
    "\nBest scale:",
    best["dissipation_scale"]
)

print(
    "Best RMSE:",
    f"{best['rmse']:.12e}"
)

print(
    "Baseline RMSE:",
    f"{baseline_rmse:.12e}"
)

improvement = (
    (
        baseline_rmse
        - best["rmse"]
    )
    /
    (
        baseline_rmse
        + 1e-30
    )
    * 100.0
)

print(
    "RMSE improvement:",
    f"{improvement:.6f}%"
)

print("=" * 90)
print("DISSIPATION SCALE SWEEP COMPLETE")
print("=" * 90)
