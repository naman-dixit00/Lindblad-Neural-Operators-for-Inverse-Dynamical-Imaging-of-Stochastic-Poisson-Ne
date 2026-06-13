r"""
================================================================================
STOCHASTIC THERMAL PERTURBATION ANALYSIS
File: experiments/exp_noise_robustness.py
Description: Evaluates LNO structural stability under out-of-distribution thermal fluctuations.
================================================================================
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_loader import IonTransportDataset
from models.lno_model import LindbladNeuralOperator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def run_noise_sweep(model, loader, noise_levels):
    accumulated_errors = []

    # Get a fixed batch from verification stream to apply controlled noise
    for batch_tuple in loader:
        inputs_stacked, targets_true = batch_tuple
        inputs_stacked, targets_true = inputs_stacked.to(DEVICE), targets_true.to(DEVICE)
        break

    print("[*] Launching Perturbation Loop across Noise Spectrum...")
    for idx, sigma in enumerate(noise_levels):
        # Inject Gaussian Noise into the initial state channel (Channel 0)
        perturbed_inputs = inputs_stacked.clone()
        perturbed_inputs[:, 0, :] += torch.randn_like(perturbed_inputs[:, 0, :]) * sigma

        # Exact Gemini Unpacking Engine
        state_t = perturbed_inputs[:, 0, :]
        phi = perturbed_inputs[:, 1, :]
        flux = perturbed_inputs[:, 2, :]
        noise = perturbed_inputs[:, 3, :]
        dissipation = perturbed_inputs[:, 4, :]
        gamma = perturbed_inputs[:, 5, 0]

        pred = model(state_t, phi, flux, noise, dissipation, gamma)
        if isinstance(pred, dict) and "next_state" in pred:
            pred = pred["next_state"]

        # Compute Root Mean Squared Error (RMSE)
        rmse = torch.sqrt(torch.mean((pred - targets_true) ** 2)).item()
        accumulated_errors.append(rmse)
        print(f"    -> Sigma [{sigma:.3f}] | Bound Error (RMSE): {rmse:.6f}")

    return accumulated_errors

def main():
    os.makedirs("results/noise_robustness", exist_ok=True)
    print("\n" + "="*70 + "\n[+] INITIATING STOCHASTIC THERMAL ROBUSTNESS SWEEP\n" + "="*70)

    # Load Model with exact parameters matched by Gemini
    model = LindbladNeuralOperator(in_channels=6, width=64, modes=16).to(DEVICE)
    checkpoint_path = "results/checkpoints/best_lno.pt"

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print("[+] Successfully loaded trained weights for LNO Profile.")
    else:
        print("[!] Warning: Checkpoint missing. Running fallback.")
    model.eval()

    # Mount verification data stream
    dataset = IonTransportDataset(root_dir="dataset/val")
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    noise_levels = np.linspace(0.0, 1.5, 20)
    errors = run_noise_sweep(model, loader, noise_levels)

    np.save("results/noise_robustness/errors.npy", np.array(errors))

    # Leica-Inspired Minimalist Visual Plot
    plt.figure(figsize=(7, 4.5), dpi=300)
    plt.plot(noise_levels, errors, color='#111111', linewidth=2, marker='o', markersize=3, label='LNO Error Bound')
    plt.xlabel("Injected Thermal Fluctuation Strength ($\\sigma$)", fontsize=9, fontweight='medium')
    plt.ylabel("State Vector Forecast RMSE", fontsize=9, fontweight='medium')
    plt.title("OOD Generalization: LNO Stability under Non-Equilibrium Noise", fontsize=10, pad=12, fontweight='bold')
    plt.grid(True, linestyle='--', color='#f0f0f0', alpha=0.8)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("results/noise_robustness/noise_curve.png", bbox_inches='tight')
    plt.close()
    print("[++++] Noise Evaluation Complete. Visual Diagnostics saved to results/noise_robustness/")

if __name__ == "__main__":
    main()
