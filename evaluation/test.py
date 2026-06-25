import torch
import numpy as np
from torch.utils.data import DataLoader
import os

from data.dataset_loader import IonTransportDataset
from models.lno_model import LindbladNeuralOperator
from evaluation.metrics import rmse # Assuming rmse is available in this path

# ===============================
# DEVICE
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# SAFE LOAD MODEL
# ===============================
model = LindbladNeuralOperator().to(device)

ckpt_path = "results/check_points/best_lno.pt"

if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    print("[+] Model loaded successfully")
else:
    print("[!] No checkpoint found at specified path, using random weights. Path: ", ckpt_path)

model.eval()


# ===============================
# DATASET
# ===============================
dataset = IonTransportDataset("dataset/test")
loader = DataLoader(dataset, batch_size=32, shuffle=False)

print("[+] Dataset loaded. Total samples:", len(dataset))


# ===============================
# TEST LOOP
# ===============================
total_rmse = 0
count = 0

print("\n[+] Running Evaluation...")

with torch.no_grad():
    for x, y_true in loader:

        x = x.to(device).float()
        y_true = y_true.to(device).float()

        # Unpack the 6-channel input tensor into individual components
        state = x[:, 0, :]
        phi = x[:, 1, :]
        flux = x[:, 2, :]
        noise = x[:, 3, :]
        dissipation = x[:, 4, :]
        gamma = x[:, 5, :] # This gamma is [Batch, Nx]

        outputs = model(
            state=state,
            phi=phi,
            flux=flux,
            noise=noise,
            dissipation=dissipation,
            gamma=gamma
        )

        predicted_next_state = outputs["next_state"]

        loss_rmse = rmse(predicted_next_state, y_true)
        total_rmse += loss_rmse
        count += 1

        print(f"  Batch {count} RMSE: {loss_rmse:.6f}")


print("\n=============================")
print(f"FINAL AVERAGE TEST RMSE: {total_rmse / count:.6f}")
print("=============================")
