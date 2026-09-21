# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 46
# Step            : STEP_11
# Step Heading    : # RECOVER STEP-11 LNO CHECKPOINT FROM GITHUB
# Step Cell No.   : 3
# ============================================================

import torch
from pathlib import Path

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

candidates = [
    ROOT / "results" / "step12_ablation" / "D_Full_LNO.pt",
    ROOT / "results" / "check_points" / "best_lno.pt",
]

print("=" * 100)
print("CHECKPOINT INSPECTION")
print("=" * 100)

for path in candidates:

    print("\nFile:", path)
    print("Exists:", path.is_file())

    if not path.is_file():
        continue

    ckpt = torch.load(
        path,
        map_location="cpu"
    )

    print("Checkpoint type:", type(ckpt))

    if isinstance(ckpt, dict):

        print(
            "Keys:",
            list(ckpt.keys())
        )

        if "label" in ckpt:
            print(
                "Label:",
                ckpt["label"]
            )

        if "prototype" in ckpt:
            print(
                "Prototype:",
                ckpt["prototype"]
            )

        if "best_epoch" in ckpt:
            print(
                "Best epoch:",
                ckpt["best_epoch"]
            )

        if "best_validation_loss" in ckpt:
            print(
                "Best validation loss:",
                ckpt["best_validation_loss"]
            )

        if "model_state_dict" in ckpt:

            state = ckpt[
                "model_state_dict"
            ]

            print(
                "State-dict tensors:",
                len(state)
            )

            print(
                "First keys:"
            )

            for key in list(
                state.keys()
            )[:15]:

                print(
                    "   ",
                    key
                )

    print("-" * 80)
