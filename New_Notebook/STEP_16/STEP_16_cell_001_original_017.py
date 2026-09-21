# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 17
# Step            : STEP_16
# Step Heading    : # STEP 16 — FINAL 3000-TEST LNO EVALUATION
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 16 — FINAL 3000-TEST LNO EVALUATION
#
# Evaluates the ORIGINAL VERIFIED LNO checkpoint by default.
#
# Set:
#     USE_PILOT = True
#
# only AFTER the learnable-dissipation pilot has completed
# and we decide to use its checkpoint.
#
# ============================================================

import os
import json
import hashlib
import time

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# ============================================================
# 1. SETTINGS
# ============================================================

USE_PILOT = False

ROOT = (
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 90)
print("STEP 16 — FINAL 3000-TEST LNO EVALUATION")
print("=" * 90)

print(
    "[+] Device:",
    DEVICE
)

# ============================================================
# 2. CHECKPOINT
# ============================================================

ORIGINAL_CKPT = os.path.join(
    ROOT,
    "results",
    "check_points",
    "best_lno.pt"
)

PILOT_CKPT = os.path.join(
    ROOT,
    "results",
    "lno_learnable_dissipation",
    "best_lno_alpha.pt"
)

if USE_PILOT:

    assert os.path.isfile(
        PILOT_CKPT
    ), (
        "Pilot checkpoint not found:\n"
        + PILOT_CKPT
    )

    CKPT = PILOT_CKPT
    MODEL_TYPE = "learnable_dissipation_lno"

else:

    assert os.path.isfile(
        ORIGINAL_CKPT
    ), (
        "Original LNO checkpoint not found:\n"
        + ORIGINAL_CKPT
    )

    CKPT = ORIGINAL_CKPT
    MODEL_TYPE = "original_lno"

print(
    "\n[+] Model type:",
    MODEL_TYPE
)

print(
    "[+] Checkpoint:",
    CKPT
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


checkpoint_sha = sha256_file(
    CKPT
)

print(
    "[+] SHA256:",
    checkpoint_sha
)

# ============================================================
# 4. DATASET
# ============================================================

TEST_DIR = os.path.join(
    ROOT,
    "dataset ",
    "test"
)

assert os.path.isdir(
    TEST_DIR
), TEST_DIR

from data.dataset_loader import IonTransportDataset

test_dataset = IonTransportDataset(
    root_dir=TEST_DIR
)

TEST_BATCH_SIZE = 32

test_loader = DataLoader(
    test_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

print(
    "\n[+] Test directory:",
    TEST_DIR
)

print(
    "[+] Test samples:",
    len(test_dataset)
)

print(
    "[+] Test batches:",
    len(test_loader)
)

assert len(test_dataset) == 3000

# ============================================================
# 5. MODEL
# ============================================================

from models.lno_model import LindbladNeuralOperator

model = LindbladNeuralOperator(
    modes=16,
    width=64,
    depth=4,
    in_channels=6
).to(DEVICE)

checkpoint = torch.load(
    CKPT,
    map_location=DEVICE
)

# ------------------------------------------------------------
# Original LNO
# ------------------------------------------------------------

if MODEL_TYPE == "original_lno":

    if (
        "model_state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    print(
        "[+] Original LNO loaded."
    )

# ------------------------------------------------------------
# Learnable-alpha LNO
# ------------------------------------------------------------

else:

    # Add the extra alpha parameter BEFORE loading.
    #
    # The exact parameterization used in the pilot:
    #
    #     alpha = softplus(raw_alpha)
    #
    # with four layer-wise values.

    model.raw_dissipative_alpha = torch.nn.Parameter(
        torch.zeros(
            4,
            device=DEVICE,
            dtype=torch.float32
        )
    )

    if (
        "model_state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    print(
        "[+] Learnable-dissipation LNO loaded."
    )

# ============================================================
# 6. LEARNABLE-ALPHA FORWARD
# ============================================================

def positive_alpha(model):

    return F.softplus(
        model.raw_dissipative_alpha
    )


def model_forward(inputs):

    state = inputs[:, 0, :]
    phi = inputs[:, 1, :]
    flux = inputs[:, 2, :]
    noise = inputs[:, 3, :]
    dissipation = inputs[:, 4, :]
    gamma = inputs[:, 5, :]

    # --------------------------------------------------------
    # ORIGINAL LNO PATH
    # --------------------------------------------------------

    if MODEL_TYPE == "original_lno":

        output = model(
            state=state,
            phi=phi,
            flux=flux,
            noise=noise,
            dissipation=dissipation,
            gamma=gamma,
        )

        return output

    # --------------------------------------------------------
    # LEARNABLE-DISSIPATION PATH
    # --------------------------------------------------------

    if gamma.dim() == 1:

        gamma = gamma.unsqueeze(-1)

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

    x = model.input_projection(
        x
    )

    x = x.permute(
        0,
        2,
        1
    )

    alphas = positive_alpha(
        model
    )

    spectral_energy_history = []

    for k in range(
        model.depth
    ):

        spectral_update = (
            model
            .spectral_layers[k](x)
        )

        pointwise_update = (
            model
            .pointwise_layers[k](x)
        )

        x_linear = (
            spectral_update
            + pointwise_update
        )

        layer = (
            model
            .dissipative_layers[k]
        )

        # ----------------------------------------------------
        # Lindblad dissipator
        # ----------------------------------------------------

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
            gamma.view(
                -1,
                1,
                1
            )
            * D
        )

        # ----------------------------------------------------
        # Coherent
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
        # Spatial diffusion
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
        # Learnable dissipative coupling
        # ----------------------------------------------------

        x = (
            x_linear
            + coherent
            + alphas[k] * D
            + spatial
        )

        if k < model.depth - 1:

            x = F.gelu(
                x
            )

        # spectral diagnostic
        try:

            spectral_energy_history.append(
                model.compute_spectral_energy(x)
            )

        except Exception:

            pass

    instability_map = (
        model
        .instability_head(x)
    )

    x_out = x.permute(
        0,
        2,
        1
    )

    predicted = (
        model
        .output_projection(
            x_out
        )
        .squeeze(-1)
    )

    return {
        "next_state":
            predicted,

        "instability_map":
            instability_map.squeeze(1),

        "spectral_energy":
            spectral_energy_history,
    }


# ============================================================
# 7. MODEL INTEGRITY
# ============================================================

model.eval()

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "\n[+] Parameter count:",
    f"{parameter_count:,}"
)

with torch.no_grad():

    first_inputs, first_targets = next(
        iter(test_loader)
    )

    first_inputs = first_inputs.float().to(
        DEVICE
    )

    first_targets = first_targets.float().to(
        DEVICE
    )

    first_output = model_forward(
        first_inputs
    )

    first_prediction = first_output[
        "next_state"
    ]

assert first_prediction.shape == (
    32,
    128
)

assert torch.isfinite(
    first_prediction
).all()

print(
    "[+] Forward check: PASSED"
)

print(
    "[+] Input geometry:",
    tuple(first_inputs.shape)
)

print(
    "[+] Output geometry:",
    tuple(first_prediction.shape)
)

# ============================================================
# 8. COMPLETE TEST-SET EVALUATION
# ============================================================

print("\n" + "=" * 90)
print("RUNNING COMPLETE 3000-SAMPLE TEST SET")
print("=" * 90)

all_predictions = []
all_targets = []
all_inputs = []

all_instability = []

start_time = time.time()

with torch.no_grad():

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

        targets = targets.float().to(
            DEVICE
        )

        output = model_forward(
            inputs
        )

        predictions = output[
            "next_state"
        ]

        all_predictions.append(
            predictions.cpu()
        )

        all_targets.append(
            targets.cpu()
        )

        all_inputs.append(
            inputs.cpu()
        )

        if (
            "instability_map"
            in output
        ):

            all_instability.append(
                output[
                    "instability_map"
                ].cpu()
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

predictions = torch.cat(
    all_predictions,
    dim=0
)

targets = torch.cat(
    all_targets,
    dim=0
)

inputs_all = torch.cat(
    all_inputs,
    dim=0
)

elapsed = (
    time.time()
    - start_time
)

print(
    "\n[+] Prediction tensor:",
    tuple(predictions.shape)
)

print(
    "[+] Target tensor:",
    tuple(targets.shape)
)

print(
    "[+] Input tensor:",
    tuple(inputs_all.shape)
)

assert predictions.shape == (
    3000,
    128
)

assert targets.shape == (
    3000,
    128
)

# ============================================================
# 9. CORE METRICS
# ============================================================

error = (
    predictions
    - targets
)

# RMSE
mse = torch.mean(
    error ** 2
).item()

rmse = (
    mse ** 0.5
)

# MAE
mae = torch.mean(
    torch.abs(error)
).item()

# Relative L2
relative_l2 = (
    torch.linalg.vector_norm(
        error
    )
    /
    (
        torch.linalg.vector_norm(
            targets
        )
        + 1e-30
    )
).item()

# ============================================================
# 10. PHYSICAL / STRUCTURAL METRICS
# ============================================================

# Mass error:
# mean absolute difference of spatial integral/sum
pred_mass = predictions.sum(
    dim=-1
)

true_mass = targets.sum(
    dim=-1
)

mass_error = torch.mean(
    torch.abs(
        pred_mass
        - true_mass
    )
).item()

# ------------------------------------------------------------
# Entropy error
# ------------------------------------------------------------

eps = 1e-12

pred_prob = (
    torch.softmax(
        predictions,
        dim=-1
    )
    + eps
)

true_prob = (
    torch.softmax(
        targets,
        dim=-1
    )
    + eps
)

pred_entropy = -torch.sum(
    pred_prob
    * torch.log(pred_prob),
    dim=-1
)

true_entropy = -torch.sum(
    true_prob
    * torch.log(true_prob),
    dim=-1
)

entropy_error = torch.mean(
    torch.abs(
        pred_entropy
        - true_entropy
    )
).item()

# ------------------------------------------------------------
# Instability index
# ------------------------------------------------------------

max_abs_state = torch.max(
    torch.abs(
        predictions
    )
).item()

instability_index = max_abs_state

# ------------------------------------------------------------
# Dissipation rate
# ------------------------------------------------------------

# Current observable proxy:
# normalized one-step change magnitude.

input_state = inputs_all[:, 0, :]

dissipation_rate = torch.mean(
    torch.abs(
        predictions
        - input_state
    )
).item()

# ------------------------------------------------------------
# Spectral energy
# ------------------------------------------------------------

pred_fft = torch.fft.rfft(
    predictions,
    dim=-1
)

spectral_energy = torch.mean(
    torch.abs(
        pred_fft
    ) ** 2
).item()

# ============================================================
# 11. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 90)
print(
    f"{MODEL_TYPE.upper()} — FINAL TEST RESULTS"
)
print("=" * 90)

print(
    f"RMSE              : {rmse:.10e}"
)

print(
    f"Relative_L2       : {relative_l2:.10e}"
)

print(
    f"MAE               : {mae:.10e}"
)

print(
    f"Mass_Error        : {mass_error:.10e}"
)

print(
    f"Entropy_Error     : {entropy_error:.10e}"
)

print(
    f"Instability_Index : {instability_index:.10e}"
)

print(
    f"Dissipation_Rate  : {dissipation_rate:.10e}"
)

print(
    f"Spectral_Energy   : {spectral_energy:.10e}"
)

print("=" * 90)

# ============================================================
# 12. SAVE FINAL ARTIFACTS
# ============================================================

FINAL_DIR = os.path.join(
    ROOT,
    "results",
    "final_lno",
    MODEL_TYPE
)

os.makedirs(
    FINAL_DIR,
    exist_ok=True
)

pred_path = os.path.join(
    FINAL_DIR,
    "lno_test_predictions.npy"
)

target_path = os.path.join(
    FINAL_DIR,
    "lno_test_targets.npy"
)

input_path = os.path.join(
    FINAL_DIR,
    "lno_test_inputs.npy"
)

metrics_path = os.path.join(
    FINAL_DIR,
    "lno_test_metrics.json"
)

summary_path = os.path.join(
    FINAL_DIR,
    "lno_test_summary.json"
)

np.save(
    pred_path,
    predictions.numpy()
)

np.save(
    target_path,
    targets.numpy()
)

np.save(
    input_path,
    inputs_all.numpy()
)

metrics = {

    "model_type":
        MODEL_TYPE,

    "checkpoint":
        CKPT,

    "checkpoint_sha256":
        checkpoint_sha,

    "test_samples":
        int(len(test_dataset)),

    "batch_size":
        TEST_BATCH_SIZE,

    "input_shape":
        list(inputs_all.shape),

    "target_shape":
        list(targets.shape),

    "prediction_shape":
        list(predictions.shape),

    "RMSE":
        rmse,

    "Relative_L2":
        relative_l2,

    "MAE":
        mae,

    "Mass_Error":
        mass_error,

    "Entropy_Error":
        entropy_error,

    "Instability_Index":
        instability_index,

    "Dissipation_Rate":
        dissipation_rate,

    "Spectral_Energy":
        spectral_energy,

    "evaluation_seconds":
        elapsed,
}

with open(
    metrics_path,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )

summary = {

    "status":
        "COMPLETE",

    "model":
        MODEL_TYPE,

    "checkpoint":
        CKPT,

    "sha256":
        checkpoint_sha,

    "best_test_rmse":
        rmse,

    "metrics_file":
        metrics_path,

    "prediction_file":
        pred_path,

    "target_file":
        target_path,

    "input_file":
        input_path,
}

with open(
    summary_path,
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )

# ============================================================
# 13. INTEGRITY
# ============================================================

assert os.path.isfile(
    pred_path
)

assert os.path.isfile(
    target_path
)

assert os.path.isfile(
    input_path
)

assert os.path.isfile(
    metrics_path
)

assert os.path.isfile(
    summary_path
)

# Check saved arrays
saved_pred = np.load(
    pred_path
)

saved_target = np.load(
    target_path
)

saved_input = np.load(
    input_path
)

assert saved_pred.shape == (
    3000,
    128
)

assert saved_target.shape == (
    3000,
    128
)

assert saved_input.shape == (
    3000,
    6,
    128
)

assert np.isfinite(
    saved_pred
).all()

assert np.isfinite(
    saved_target
).all()

print("\n" + "=" * 90)
print("FINAL 3000-TEST LNO EVALUATION COMPLETE")
print("=" * 90)

print(
    "[+] Test samples evaluated:",
    len(test_dataset)
)

print(
    "[+] RMSE:",
    f"{rmse:.10e}"
)

print(
    "[+] Relative_L2:",
    f"{relative_l2:.10e}"
)

print(
    "[+] MAE:",
    f"{mae:.10e}"
)

print(
    "[+] Predictions:",
    pred_path
)

print(
    "[+] Targets:",
    target_path
)

print(
    "[+] Inputs:",
    input_path
)

print(
    "[+] Metrics:",
    metrics_path
)

print(
    "[+] Summary:",
    summary_path
)

print("=" * 90)
