
import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader

from data.dataset_loader import IonTransportDataset
from models.lno_model import LindbladNeuralOperator
from evaluation.metrics import rmse

def compute_ablation_score(loader, model, device):
    total_rmse_stoch = 0.0
    total_rmse_det = 0.0
    count = 0

    # 1. Stochastic Run (Noise ON - Normal Evaluation)
    print("[+] Running Base Stochastic Inference (Noise ON)...")
    with torch.no_grad():
        for x, y_true in loader:
            x, y_true = x.to(device).float(), y_true.to(device).float()
            outputs = model(
                state=x[:, 0, :], phi=x[:, 1, :], flux=x[:, 2, :],
                noise=x[:, 3, :], dissipation=x[:, 4, :], gamma=x[:, 5, :]
            )
            total_rmse_stoch += rmse(outputs["next_state"], y_true)
            count += 1

    E_stochastic = total_rmse_stoch / count

    # 2. Fake Ablation Run (Noise OFF - Deterministic Operator)
    print("[+] Applying Zero-Shot Fake Ablation (Clamping Dissipation Net)...")
    deterministic_model = deepcopy(model)

    # Force the structural gates to absolute zero explicitly
    for layer in deterministic_model.dissipative_layers:
        layer.raw_gamma.data.fill_(-1e4) # Sigmoid(-10000) -> 0.0
        layer.jump_transform.data.zero_() # Wipe the L matrix

    print("[+] Running Deterministic Inference (Noise OFF)...")
    with torch.no_grad():
        for x, y_true in loader:
            x, y_true = x.to(device).float(), y_true.to(device).float()
            outputs = deterministic_model(
                state=x[:, 0, :], phi=x[:, 1, :], flux=x[:, 2, :],
                noise=x[:, 3, :], dissipation=x[:, 4, :], gamma=x[:, 5, :]
            )
            total_rmse_det += rmse(outputs["next_state"], y_true)

    E_deterministic = total_rmse_det / count

    # 3. Compute Phase 1 Equation Metric
    noise_score = (E_deterministic - E_stochastic) / E_stochastic
    return E_stochastic, E_deterministic, noise_score

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LindbladNeuralOperator(in_channels=6).to(device)

    # Load Best Model Checkpoint
    ckpt = torch.load("results/checkpoints/best_lno.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model.eval()

    dataset = IonTransportDataset("dataset/test")
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    E_stoch, E_det, score = compute_ablation_score(loader, model, device)

    print("\n" + "="*50)
    print("PHASE 1: 3-FIGURE PROOF (METRIC A) METRICS")
    print(f"E_stochastic (Noise ON)   : {E_stoch:.6f}")
    print(f"E_deterministic (Noise OFF): {E_det:.6f}")
    print(f"Noise Score               : {score:.4f}")
    print("="*50)
