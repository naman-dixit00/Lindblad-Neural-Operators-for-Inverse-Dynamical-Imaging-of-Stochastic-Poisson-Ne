# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 15
# Step            : STEP_01
# Step Heading    : # PHASE 1 — STEP 1: REPOSITORY + TENSOR GEOMETRY AUDIT
# Step Cell No.   : 11
# ============================================================

# ============================================================
# LNO — LEARNABLE DISSIPATIVE COUPLING PILOT
#
# New model:
#   alpha_0 ... alpha_3
#   alpha_l > 0
#   initial alpha = 3000
#
# Pilot:
#   20 epochs
#
# Saves:
#   best checkpoint
#   latest checkpoint
#   history
# ============================================================

import os
import json
import random
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

print("=" * 90)
print("LNO — LEARNABLE DISSIPATIVE COUPLING PILOT")
print("=" * 90)

# ============================================================
# 1. ROOT
# ============================================================

ROOT = (
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("[+] Device:", device)

if device.type == "cuda":

    print(
        "[+] GPU:",
        torch.cuda.get_device_name(0)
    )

# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

SEED = 2026

random.seed(SEED)
torch.manual_seed(SEED)

if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ============================================================
# 4. DATASET
# ============================================================

DATASET_ROOT = os.path.join(
    ROOT,
    "dataset "
)

TRAIN_DIR = os.path.join(
    DATASET_ROOT,
    "train"
)

VAL_DIR = os.path.join(
    DATASET_ROOT,
    "val"
)

assert os.path.isdir(TRAIN_DIR)
assert os.path.isdir(VAL_DIR)

from data.dataset_loader import IonTransportDataset

train_dataset = IonTransportDataset(
    root_dir=TRAIN_DIR
)

val_dataset = IonTransportDataset(
    root_dir=VAL_DIR
)

BATCH_SIZE = 32

NUM_WORKERS = (
    4 if device.type == "cuda"
    else 2
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(NUM_WORKERS > 0),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(NUM_WORKERS > 0),
)

print(
    "[+] Train samples:",
    len(train_dataset)
)

print(
    "[+] Val samples:",
    len(val_dataset)
)

print(
    "[+] Train batches:",
    len(train_loader)
)

print(
    "[+] Val batches:",
    len(val_loader)
)

# ============================================================
# 5. LOAD ORIGINAL LNO
# ============================================================

ORIGINAL_CKPT = os.path.join(
    ROOT,
    "results",
    "check_points",
    "best_lno.pt"
)

assert os.path.isfile(
    ORIGINAL_CKPT
)

print(
    "\n[+] Original checkpoint:",
    ORIGINAL_CKPT
)

# ------------------------------------------------------------
# Import the actual class from your project.
# ------------------------------------------------------------

from models.lno_model import (
    LindbladNeuralOperator
)

model = LindbladNeuralOperator(
    modes=16,
    width=64,
    depth=4,
    in_channels=6
).to(device)

ckpt = torch.load(
    ORIGINAL_CKPT,
    map_location=device
)

# Support both checkpoint formats
if "model_state_dict" in ckpt:

    model.load_state_dict(
        ckpt["model_state_dict"],
        strict=True
    )

else:

    model.load_state_dict(
        ckpt,
        strict=True
    )

print(
    "[+] Original LNO weights loaded."
)

# ============================================================
# 6. LEARNABLE POSITIVE ALPHA
# ============================================================

INITIAL_ALPHA = 3000.0

# softplus^-1(alpha)
# Stable for large positive alpha:
def inverse_softplus(x):

    if x > 20.0:
        return x

    return math.log(
        math.expm1(x)
    )

initial_raw_alpha = inverse_softplus(
    INITIAL_ALPHA
)

model.raw_dissipative_alpha = nn.Parameter(
    torch.full(
        (4,),
        initial_raw_alpha,
        dtype=torch.float32,
        device=device
    )
)

def get_positive_alpha():

    return F.softplus(
        model.raw_dissipative_alpha
    )

print(
    "\n[+] Initial alpha:",
    INITIAL_ALPHA
)

print(
    "[+] Raw alpha initialization:",
    initial_raw_alpha
)

# ============================================================
# 7. EXACT MODIFIED FORWARD
# ============================================================

def forward_with_learnable_alpha(
    model,
    state,
    phi,
    flux,
    noise,
    dissipation,
    gamma
):

    # --------------------------------------------------------
    # Gamma alignment
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

    alphas = get_positive_alpha()

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

        # ====================================================
        # DISSIPATIVE LAYER
        # ====================================================

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
            gamma
            .view(
                -1,
                1,
                1
            )
            * D
        )

        # ----------------------------------------------------
        # Coherent term
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
        # LEARNABLE DISSIPATIVE COUPLING
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

    # ========================================================
    # OUTPUT HEAD
    # ========================================================

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
        .output_projection(x_out)
        .squeeze(-1)
    )

    return {
        "next_state":
            predicted,

        "instability_map":
            instability_map.squeeze(1)
    }

# ============================================================
# 8. FORWARD SANITY CHECK
# ============================================================

safe_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

test_x, test_y = next(
    iter(safe_loader)
)

test_x = test_x.float().to(device)

test_state = test_x[:, 0, :]
test_phi = test_x[:, 1, :]
test_flux = test_x[:, 2, :]
test_noise = test_x[:, 3, :]
test_dissipation = test_x[:, 4, :]
test_gamma = test_x[:, 5, 0]

with torch.no_grad():

    sanity_out = forward_with_learnable_alpha(
        model,
        test_state,
        test_phi,
        test_flux,
        test_noise,
        test_dissipation,
        test_gamma
    )

sanity_pred = sanity_out[
    "next_state"
]

assert sanity_pred.shape == (
    4,
    128
)

assert torch.isfinite(
    sanity_pred
).all()

print(
    "\n[PASS] Modified LNO forward check:"
)

print(
    "  Output shape:",
    tuple(sanity_pred.shape)
)

print(
    "  Alpha:",
    get_positive_alpha()
    .detach()
    .cpu()
    .numpy()
)

# ============================================================
# 9. LOSS
# ============================================================

criterion = nn.MSELoss()

# ============================================================
# 10. OPTIMIZER
# ============================================================

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

EPOCHS = 20

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print(
    "\n[+] Loss:",
    "MSE"
)

print(
    "[+] Optimizer:",
    "AdamW"
)

print(
    "[+] Learning rate:",
    LEARNING_RATE
)

print(
    "[+] Weight decay:",
    WEIGHT_DECAY
)

print(
    "[+] Epochs:",
    EPOCHS
)

# ============================================================
# 11. OUTPUTS
# ============================================================

OUT_DIR = os.path.join(
    ROOT,
    "results",
    "lno_learnable_dissipation"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

BEST_CKPT = os.path.join(
    OUT_DIR,
    "best_lno_alpha.pt"
)

LATEST_CKPT = os.path.join(
    OUT_DIR,
    "latest_lno_alpha.pt"
)

HISTORY_PATH = os.path.join(
    OUT_DIR,
    "training_history.json"
)

# ============================================================
# 12. TRAINING
# ============================================================

history = []

best_val_loss = float(
    "inf"
)

best_epoch = None

training_start = time.time()

print("\n" + "=" * 90)
print("STARTING LEARNABLE-DISSIPATION LNO PILOT")
print("=" * 90)

for epoch in range(
    1,
    EPOCHS + 1
):

    epoch_start = time.time()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_sum = 0.0
    train_count = 0

    for inputs, targets in train_loader:

        inputs = inputs.float().to(
            device,
            non_blocking=True
        )

        targets = targets.float().to(
            device,
            non_blocking=True
        )

        state = inputs[:, 0, :]
        phi = inputs[:, 1, :]
        flux = inputs[:, 2, :]
        noise = inputs[:, 3, :]
        dissipation = inputs[:, 4, :]
        gamma = inputs[:, 5, 0]

        optimizer.zero_grad(
            set_to_none=True
        )

        output = forward_with_learnable_alpha(
            model,
            state,
            phi,
            flux,
            noise,
            dissipation,
            gamma
        )

        prediction = output[
            "next_state"
        ]

        loss = criterion(
            prediction,
            targets
        )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        train_sum += loss.item()
        train_count += 1

    train_loss = (
        train_sum
        /
        max(
            train_count,
            1
        )
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_sum = 0.0
    val_count = 0

    with torch.no_grad():

        for inputs, targets in val_loader:

            inputs = inputs.float().to(
                device,
                non_blocking=True
            )

            targets = targets.float().to(
                device,
                non_blocking=True
            )

            state = inputs[:, 0, :]
            phi = inputs[:, 1, :]
            flux = inputs[:, 2, :]
            noise = inputs[:, 3, :]
            dissipation = inputs[:, 4, :]
            gamma = inputs[:, 5, 0]

            output = forward_with_learnable_alpha(
                model,
                state,
                phi,
                flux,
                noise,
                dissipation,
                gamma
            )

            prediction = output[
                "next_state"
            ]

            loss = criterion(
                prediction,
                targets
            )

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    f"Non-finite validation loss "
                    f"at epoch {epoch}"
                )

            val_sum += loss.item()
            val_count += 1

    val_loss = (
        val_sum
        /
        max(
            val_count,
            1
        )
    )

    # --------------------------------------------------------
    # ALPHA
    # --------------------------------------------------------

    with torch.no_grad():

        alpha_values = (
            get_positive_alpha()
            .detach()
            .cpu()
            .tolist()
        )

    # --------------------------------------------------------
    # ALPHA GRADIENT
    # --------------------------------------------------------

    if (
        model.raw_dissipative_alpha.grad
        is not None
    ):

        alpha_grad = (
            torch.linalg.vector_norm(
                model
                .raw_dissipative_alpha
                .grad
            )
            .item()
        )

    else:

        alpha_grad = 0.0

    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    epoch_time = (
        time.time()
        - epoch_start
    )

    record = {

        "epoch":
            epoch,

        "train_loss":
            float(train_loss),

        "val_loss":
            float(val_loss),

        "alpha":
            alpha_values,

        "alpha_grad_norm":
            float(alpha_grad),

        "epoch_seconds":
            float(epoch_time)
    }

    history.append(
        record
    )

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"\nEpoch {epoch:02d}/{EPOCHS}"
    )

    print(
        f"  Train MSE : "
        f"{train_loss:.10e}"
    )

    print(
        f"  Val MSE   : "
        f"{val_loss:.10e}"
    )

    print(
        "  Alpha     :",
        [
            f"{a:.6e}"
            for a in alpha_values
        ]
    )

    print(
        "  Alpha grad:",
        f"{alpha_grad:.12e}"
    )

    print(
        f"  Time      : "
        f"{epoch_time:.1f}s"
    )

    # --------------------------------------------------------
    # BEST
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = float(
            val_loss
        )

        best_epoch = epoch

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "epoch":
                    epoch,

                "best_val_loss":
                    best_val_loss,

                "best_epoch":
                    best_epoch,

                "alpha":
                    alpha_values,

                "architecture": {
                    "name":
                        "LindbladNeuralOperatorLearnableDissipation",

                    "modes":
                        16,

                    "width":
                        64,

                    "depth":
                        4,

                    "in_channels":
                        6,
                },

                "training": {
                    "seed":
                        SEED,

                    "batch_size":
                        BATCH_SIZE,

                    "learning_rate":
                        LEARNING_RATE,

                    "weight_decay":
                        WEIGHT_DECAY,

                    "optimizer":
                        "AdamW",

                    "epochs":
                        EPOCHS,

                    "initial_alpha":
                        INITIAL_ALPHA,

                },

                "history":
                    history,
            },
            BEST_CKPT
        )

        print(
            "  >>> NEW BEST CHECKPOINT SAVED"
        )

    # --------------------------------------------------------
    # LATEST / RESUME
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "epoch":
                epoch,

            "best_val_loss":
                best_val_loss,

            "best_epoch":
                best_epoch,

            "alpha":
                alpha_values,

            "history":
                history,
        },
        LATEST_CKPT
    )

    with open(
        HISTORY_PATH,
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

# ============================================================
# 13. FINAL
# ============================================================

print("\n" + "=" * 90)
print("LEARNABLE-DISSIPATION PILOT COMPLETE")
print("=" * 90)

print(
    "[+] Best epoch:",
    best_epoch
)

print(
    "[+] Best validation MSE:",
    f"{best_val_loss:.12e}"
)

print(
    "[+] Final alpha:",
    [
        f"{a:.8e}"
        for a in get_positive_alpha()
        .detach()
        .cpu()
        .tolist()
    ]
)

print(
    "[+] Best checkpoint:",
    BEST_CKPT
)

print(
    "[+] Latest checkpoint:",
    LATEST_CKPT
)

print(
    "[+] History:",
    HISTORY_PATH
)

print(
    "[+] Total time:",
    f"{(time.time() - training_start)/60:.2f} min"
)

print(
    "[+] Original best_lno.pt untouched: TRUE"
)

print("=" * 90)
