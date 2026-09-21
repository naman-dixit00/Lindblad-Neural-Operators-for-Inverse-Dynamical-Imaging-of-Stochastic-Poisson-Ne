# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 12
# Step            : STEP_01
# Step Heading    : # PHASE 1 — STEP 1: REPOSITORY + TENSOR GEOMETRY AUDIT
# Step Cell No.   : 8
# ============================================================

# ============================================================
# LNO — FORWARD PATH / WEAK-POINT DIAGNOSTIC
#
# PURPOSE:
#   1. Measure layer/component scales
#   2. Measure original prediction error
#   3. Disable dissipation / coherent / spatial diffusion
#      one at a time and evaluate the REAL LNO forward pass
# ============================================================

import os
import copy
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

print("=" * 90)
print("ORIGINAL LNO — FORWARD PATH / WEAK-POINT DIAGNOSTIC")
print("=" * 90)

# ============================================================
# 1. MODEL
# ============================================================

assert "old_lno" in globals(), (
    "`old_lno` is not available in the current runtime."
)

model = old_lno
model.eval()

device = next(
    model.parameters()
).device

print("[+] Device:", device)

# ============================================================
# 2. TEST DATA — SAFE CPU WORKER CONFIG
# ============================================================

if "test_dataset" in globals():

    dataset = test_dataset

else:

    assert "TEST_DIR" in globals(), (
        "`TEST_DIR` is not defined."
    )

    dataset = IonTransportDataset(
        root_dir=TEST_DIR
    )

safe_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

inputs, targets = next(
    iter(safe_loader)
)

inputs = inputs.float().to(device)
targets = targets.float().to(device)

print(
    "[+] Input shape :",
    tuple(inputs.shape)
)

print(
    "[+] Target shape:",
    tuple(targets.shape)
)

# ============================================================
# 3. PHYSICAL INPUT FIELDS
# ============================================================

state = inputs[:, 0, :]
phi = inputs[:, 1, :]
flux = inputs[:, 2, :]
noise = inputs[:, 3, :]
dissipation = inputs[:, 4, :]
base_gamma = inputs[:, 5, 0]

# ============================================================
# 4. ACTUAL ORIGINAL LNO FORWARD
# ============================================================

def run_actual_lno():

    with torch.no_grad():

        output = model(
            state=state,
            phi=phi,
            flux=flux,
            noise=noise,
            dissipation=dissipation,
            gamma=base_gamma,
        )

    if not isinstance(
        output,
        dict
    ):

        raise TypeError(
            "Expected LNO forward() to return a dict, "
            f"got {type(output)}"
        )

    if "next_state" in output:

        prediction = output[
            "next_state"
        ]

    elif "prediction" in output:

        prediction = output[
            "prediction"
        ]

    else:

        raise KeyError(
            "LNO output keys are "
            f"{list(output.keys())}; "
            "expected `next_state` or `prediction`."
        )

    if prediction.ndim == 3:

        # Handle possible singleton channel dimension
        if prediction.shape[1] == 1:

            prediction = prediction.squeeze(1)

        elif prediction.shape[-1] == 1:

            prediction = prediction.squeeze(-1)

    if prediction.shape != targets.shape:

        raise RuntimeError(
            f"Prediction shape {tuple(prediction.shape)} "
            f"does not match target shape "
            f"{tuple(targets.shape)}"
        )

    return prediction


# ============================================================
# 5. BASELINE
# ============================================================

prediction = run_actual_lno()

baseline_mse = F.mse_loss(
    prediction,
    targets
).item()

baseline_rmse = (
    baseline_mse ** 0.5
)

prediction_l2 = torch.linalg.vector_norm(
    prediction
).item()

target_l2 = torch.linalg.vector_norm(
    targets
).item()

error = (
    targets
    - prediction
)

error_l2 = torch.linalg.vector_norm(
    error
).item()

print("\n" + "=" * 90)
print("BASELINE LNO")
print("=" * 90)

print(
    "[+] Prediction shape:",
    tuple(prediction.shape)
)

print(
    "[+] Baseline MSE:",
    f"{baseline_mse:.12e}"
)

print(
    "[+] Baseline RMSE:",
    f"{baseline_rmse:.12e}"
)

print(
    "[+] Prediction L2:",
    f"{prediction_l2:.12e}"
)

print(
    "[+] Target L2:",
    f"{target_l2:.12e}"
)

print(
    "[+] Prediction error L2:",
    f"{error_l2:.12e}"
)

# ============================================================
# 6. MANUAL LAYER-WISE COMPONENT SCALE
# ============================================================

print("\n" + "=" * 90)
print("LAYER-WISE COMPONENT SCALE")
print("=" * 90)

# Build same input representation used by the LNO
gamma_input = (
    base_gamma
    .unsqueeze(-1)
    .repeat(
        1,
        inputs.shape[-1]
    )
)

packed_input = torch.stack(
    [
        state,
        phi,
        flux,
        noise,
        dissipation,
        gamma_input,
    ],
    dim=-1
)

with torch.no_grad():

    h = (
        model
        .input_projection(
            packed_input
        )
        .permute(
            0,
            2,
            1
        )
    )

layer_rows = []

with torch.no_grad():

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

        # ----------------------------------------------------
        # Original Lindblad dissipator
        # ----------------------------------------------------

        L = (
            layer
            .jump_transform
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

        weighted_D = (
            base_gamma
            .view(-1, 1, 1)
            * D
        )

        # ----------------------------------------------------
        # Coherent component
        # ----------------------------------------------------

        padded = F.pad(
            x_linear,
            (2, 2),
            mode="circular"
        )

        coherent = F.conv1d(
            padded,
            layer.coherent_kernel
        )

        # ----------------------------------------------------
        # Spatial diffusion component
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Norms
        # ----------------------------------------------------

        x_norm = torch.linalg.vector_norm(
            x_linear
        ).item()

        spectral_norm = torch.linalg.vector_norm(
            spectral
        ).item()

        pointwise_norm = torch.linalg.vector_norm(
            pointwise
        ).item()

        coherent_norm = torch.linalg.vector_norm(
            coherent
        ).item()

        dissipative_norm = torch.linalg.vector_norm(
            weighted_D
        ).item()

        spatial_norm = torch.linalg.vector_norm(
            spatial
        ).item()

        full_update = (
            x_linear
            + coherent
            + weighted_D
            + spatial
        )

        layer_rows.append({

            "layer":
                layer_idx,

            "x_linear_l2":
                x_norm,

            "spectral_l2":
                spectral_norm,

            "pointwise_l2":
                pointwise_norm,

            "coherent_l2":
                coherent_norm,

            "dissipative_l2":
                dissipative_norm,

            "spatial_l2":
                spatial_norm,

            "D_over_x":
                dissipative_norm
                / (x_norm + 1e-30),

            "coherent_over_x":
                coherent_norm
                / (x_norm + 1e-30),

            "spatial_over_x":
                spatial_norm
                / (x_norm + 1e-30),

            "full_update_l2":
                torch.linalg.vector_norm(
                    full_update
                ).item(),
        })

        # ----------------------------------------------------
        # Continue along original forward pathway
        # ----------------------------------------------------

        h = full_update

        if layer_idx < 3:

            h = F.gelu(
                h
            )

layer_df = pd.DataFrame(
    layer_rows
)

display(
    layer_df.round(12)
)

# ============================================================
# 7. SAFE MODEL SNAPSHOT
# ============================================================

original_state = {
    k: v.detach().cpu().clone()
    for k, v in model.state_dict().items()
}

# ============================================================
# 8. COUNTERFACTUAL FORWARD TEST
#
# We temporarily zero ONE physical component at a time:
#
#   dissipation:
#       jump_transform = 0
#
#   coherent:
#       coherent_kernel = 0
#
#   spatial:
#       spatial_diffusion_weight = 0
#
# Then restore the complete original state.
# ============================================================

print("\n" + "=" * 90)
print("COUNTERFACTUAL COMPONENT ABLATION")
print("=" * 90)

def restore_original_state():

    model.load_state_dict(
        original_state,
        strict=True
    )

    model.eval()


def evaluate_prediction():

    pred = run_actual_lno()

    mse = F.mse_loss(
        pred,
        targets
    ).item()

    rmse = (
        mse ** 0.5
    )

    relative_prediction_change = (
        torch.linalg.vector_norm(
            pred
            - prediction
        )
        /
        (
            torch.linalg.vector_norm(
                prediction
            )
            + 1e-30
        )
    ).item()

    return (
        mse,
        rmse,
        relative_prediction_change
    )


counterfactual_rows = []

# ------------------------------------------------------------
# Baseline row
# ------------------------------------------------------------

counterfactual_rows.append({

    "disabled":
        "none",

    "mse":
        baseline_mse,

    "rmse":
        baseline_rmse,

    "delta_mse":
        0.0,

    "delta_rmse":
        0.0,

    "relative_output_change":
        0.0,
})

print(
    f"Disable {'none':12s} | "
    f"MSE={baseline_mse:.12e} | "
    f"dMSE={0.0:+.12e} | "
    f"output_change={0.0:.12e}"
)

# ============================================================
# A. Disable DISSIPATION
# ============================================================

restore_original_state()

with torch.no_grad():

    for layer in (
        model.dissipative_layers
    ):

        layer.jump_transform.zero_()

diss_mse, diss_rmse, diss_change = (
    evaluate_prediction()
)

counterfactual_rows.append({

    "disabled":
        "dissipation",

    "mse":
        diss_mse,

    "rmse":
        diss_rmse,

    "delta_mse":
        diss_mse
        - baseline_mse,

    "delta_rmse":
        diss_rmse
        - baseline_rmse,

    "relative_output_change":
        diss_change,
})

print(
    f"Disable {'dissipation':12s} | "
    f"MSE={diss_mse:.12e} | "
    f"dMSE={diss_mse-baseline_mse:+.12e} | "
    f"output_change={diss_change:.12e}"
)

# ============================================================
# B. Disable COHERENT TERM
# ============================================================

restore_original_state()

with torch.no_grad():

    for layer in (
        model.dissipative_layers
    ):

        layer.coherent_kernel.zero_()

coh_mse, coh_rmse, coh_change = (
    evaluate_prediction()
)

counterfactual_rows.append({

    "disabled":
        "coherent",

    "mse":
        coh_mse,

    "rmse":
        coh_rmse,

    "delta_mse":
        coh_mse
        - baseline_mse,

    "delta_rmse":
        coh_rmse
        - baseline_rmse,

    "relative_output_change":
        coh_change,
})

print(
    f"Disable {'coherent':12s} | "
    f"MSE={coh_mse:.12e} | "
    f"dMSE={coh_mse-baseline_mse:+.12e} | "
    f"output_change={coh_change:.12e}"
)

# ============================================================
# C. Disable SPATIAL DIFFUSION
# ============================================================

restore_original_state()

with torch.no_grad():

    for layer in (
        model.dissipative_layers
    ):

        layer.spatial_diffusion_weight.zero_()

spatial_mse, spatial_rmse, spatial_change = (
    evaluate_prediction()
)

counterfactual_rows.append({

    "disabled":
        "spatial",

    "mse":
        spatial_mse,

    "rmse":
        spatial_rmse,

    "delta_mse":
        spatial_mse
        - baseline_mse,

    "delta_rmse":
        spatial_rmse
        - baseline_rmse,

    "relative_output_change":
        spatial_change,
})

print(
    f"Disable {'spatial':12s} | "
    f"MSE={spatial_mse:.12e} | "
    f"dMSE={spatial_mse-baseline_mse:+.12e} | "
    f"output_change={spatial_change:.12e}"
)

# ============================================================
# 9. RESTORE ORIGINAL MODEL — IMPORTANT
# ============================================================

restore_original_state()

# Verify restoration
restored_prediction = run_actual_lno()

restored_mse = F.mse_loss(
    restored_prediction,
    targets
).item()

restoration_error = (
    torch.max(
        torch.abs(
            restored_prediction
            - prediction
        )
    )
    .item()
)

print("\n" + "=" * 90)
print("MODEL RESTORATION")
print("=" * 90)

print(
    "Original MSE :",
    f"{baseline_mse:.12e}"
)

print(
    "Restored MSE :",
    f"{restored_mse:.12e}"
)

print(
    "Max prediction difference:",
    f"{restoration_error:.12e}"
)

assert abs(
    restored_mse
    - baseline_mse
) < 1e-12

print(
    "[PASS] Original LNO state restored."
)

# ============================================================
# 10. COUNTERFACTUAL TABLE
# ============================================================

cf_df = pd.DataFrame(
    counterfactual_rows
)

print("\n" + "=" * 90)
print("COUNTERFACTUAL SUMMARY")
print("=" * 90)

display(
    cf_df.round(12)
)

# ============================================================
# 11. AUTOMATIC INTERPRETATION
# ============================================================

print("\n" + "=" * 90)
print("WEAK-POINT INTERPRETATION")
print("=" * 90)

# Find which component removal hurts performance most
non_baseline = cf_df[
    cf_df["disabled"] != "none"
].copy()

non_baseline["mse_increase"] = (
    non_baseline["mse"]
    - baseline_mse
)

worst = non_baseline.loc[
    non_baseline["mse_increase"].idxmax()
]

print(
    "[+] Most damaging component removal:",
    worst["disabled"]
)

print(
    "[+] MSE change:",
    f"{worst['mse_increase']:+.12e}"
)

print(
    "\nInterpretation:"
)

if (
    diss_mse
    > baseline_mse
    * 1.001
):

    print(
        "  [INFO] Dissipation materially helps "
        "the supervised prediction."
    )

elif (
    diss_mse
    < baseline_mse
    * 0.999
):

    print(
        "  [INFO] Removing dissipation improves "
        "the supervised prediction."
    )

else:

    print(
        "  [INFO] Dissipation has little effect "
        "on this batch's supervised prediction."
    )

if (
    coh_mse
    > baseline_mse
    * 1.001
):

    print(
        "  [INFO] Coherent pathway is important "
        "for prediction."
    )

if (
    spatial_mse
    > baseline_mse
    * 1.001
):

    print(
        "  [INFO] Spatial diffusion is important "
        "for prediction."
    )

# ============================================================
# 12. SAVE DIAGNOSTIC
# ============================================================

OUT_DIR = os.path.join(
    ROOT,
    "results",
    "phase1_zero_shot",
    "lno_weak_point_diagnostic"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

layer_df.to_csv(
    os.path.join(
        OUT_DIR,
        "layer_component_scale.csv"
    ),
    index=False
)

cf_df.to_csv(
    os.path.join(
        OUT_DIR,
        "component_counterfactuals.csv"
    ),
    index=False
)

summary = {

    "baseline_mse":
        baseline_mse,

    "baseline_rmse":
        baseline_rmse,

    "baseline_prediction_l2":
        prediction_l2,

    "target_l2":
        target_l2,

    "error_l2":
        error_l2,

    "counterfactuals":
        counterfactual_rows,

    "most_damaging_component":
        str(
            worst["disabled"]
        ),

    "most_damaging_mse_change":
        float(
            worst["mse_increase"]
        ),

    "restoration_max_prediction_difference":
        restoration_error,
}

import json

with open(
    os.path.join(
        OUT_DIR,
        "weak_point_summary.json"
    ),
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

print(
    "\n[+] Saved diagnostic artifacts to:",
    OUT_DIR
)

print("=" * 90)
print("ORIGINAL LNO WEAK-POINT DIAGNOSTIC COMPLETE")
print("=" * 90)
