# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 14
# Step            : STEP_01
# Step Heading    : # PHASE 1 — STEP 1: REPOSITORY + TENSOR GEOMETRY AUDIT
# Step Cell No.   : 10
# ============================================================

# ============================================================
# LNO — EXTENDED DISSIPATION SCALE SWEEP
# Purpose:
#   Locate the useful alpha_diss range around the current
#   best tested value (1000).
# ============================================================

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

print("=" * 90)
print("ORIGINAL LNO — EXTENDED DISSIPATION SCALE SWEEP")
print("=" * 90)

model = old_lno
model.eval()

device = next(
    model.parameters()
).device

# ------------------------------------------------------------
# Safe test batch
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

gamma_input = (
    base_gamma
    .unsqueeze(-1)
    .repeat(1, 128)
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
# Extended range around current best
# ------------------------------------------------------------

scales = [
    300.0,
    500.0,
    750.0,
    1000.0,
    1500.0,
    2000.0,
    3000.0,
    5000.0,
]

def forward_with_diss_scale(
    scale
):

    with torch.no_grad():

        h = (
            model
            .input_projection(
                packed
            )
            .permute(
                0,
                2,
                1
            )
        )

        for layer_idx in range(4):

            layer = (
                model
                .dissipative_layers[
                    layer_idx
                ]
            )

            spectral = (
                model
                .spectral_layers[
                    layer_idx
                ](h)
            )

            pointwise = (
                model
                .pointwise_layers[
                    layer_idx
                ](h)
            )

            x_linear = (
                spectral
                + pointwise
            )

            # Lindblad dissipator
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
                base_gamma
                .view(-1, 1, 1)
                * D
            )

            # Coherent
            padded = F.pad(
                x_linear,
                (2, 2),
                mode="circular"
            )

            coherent = F.conv1d(
                padded,
                layer.coherent_kernel
            )

            # Spatial diffusion
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

            # Controlled dissipative coupling
            h = (
                x_linear
                + coherent
                + scale * D
                + spatial
            )

            if layer_idx < 3:

                h = F.gelu(h)

        output = model.output_projection(
            h.permute(
                0,
                2,
                1
            )
        )

        if output.ndim == 3:

            if output.shape[-1] == 1:
                output = output.squeeze(-1)

            elif output.shape[1] == 1:
                output = output.squeeze(1)

        return output


# ------------------------------------------------------------
# Evaluate scales
# ------------------------------------------------------------

results = []

for scale in scales:

    prediction = (
        forward_with_diss_scale(
            scale
        )
    )

    mse = F.mse_loss(
        prediction,
        targets
    ).item()

    rmse = mse ** 0.5

    finite = bool(
        torch.isfinite(
            prediction
        ).all().item()
    )

    results.append({
        "alpha_diss": scale,
        "MSE": mse,
        "RMSE": rmse,
        "finite": finite,
    })

    print(
        f"alpha={scale:7.1f} | "
        f"RMSE={rmse:.12e} | "
        f"MSE={mse:.12e} | "
        f"finite={finite}"
    )

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

df = pd.DataFrame(
    results
)

print("\n" + "=" * 90)
print("EXTENDED SCALE SWEEP SUMMARY")
print("=" * 90)

display(
    df.round(12)
)

finite_df = df[
    df["finite"]
]

best = finite_df.loc[
    finite_df["MSE"].idxmin()
]

print(
    "\n[+] Best tested alpha:",
    best["alpha_diss"]
)

print(
    "[+] Best tested RMSE:",
    f"{best['RMSE']:.12e}"
)

print("=" * 90)
print("EXTENDED SCALE SWEEP COMPLETE")
print("=" * 90)
