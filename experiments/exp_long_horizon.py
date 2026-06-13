"""
================================================================================
AUTOREGRESSIVE LONG-HORIZON DRIFT TRACKING
File: experiments/exp_long_horizon.py
Description: Tracks systemic error accumulation across recursive multi-step unrolling.
================================================================================
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_loader import IonTransportDataset
from models.fno_baseline import FNOBaseline
from models.lno_model import LindbladNeuralOperator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_autoregressive_rollout(model, loader, steps=50):
    for batch_tuple in loader:
        inputs_stacked, _ = batch_tuple
        inputs_stacked = inputs_stacked.to(DEVICE)
        break # Extract a clean verification batch anchor

    # Extract static environment states
    phi = inputs_stacked[:, 1, :]
    flux = inputs_stacked[:, 2, :]
    noise = inputs_stacked[:, 3, :]
    dissipation = inputs_stacked[:, 4, :]
    gamma = inputs_stacked[:, 5, 0]

    # Initial state tracking
    current_state = inputs_stacked[:, 0, :].clone()
    energy_history = []

    for step in range(steps):
        pred = model(current_state, phi, flux, noise, dissipation, gamma)
        if isinstance(pred, dict) and "next_state" in pred:
            pred = pred["next_state"]
        
        # Autoregressively feed back prediction as next input state
        current_state = pred.detach()
        
        # Track Kinetic Energy Norm to monitor explosion / decay boundaries
        kinetic_energy = torch.mean(current_state ** 2).item()
        energy_history.append(kinetic_energy)
        
    return energy_history

def main():
    os.makedirs("results/long_horizon", exist_ok=True)
    print("\n" + "="*70 + "\n[+] INITIALIZING AUTOREGRESSIVE LONG-HORIZON STABILITY EXPERIMENT\n" + "="*70)

    dataset = IonTransportDataset(root_dir="dataset/val")
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Initializing matched classes
    fno = FNOBaseline(in_channels=6, width=64, modes=16).to(DEVICE)
    lno = LindbladNeuralOperator(in_channels=6, width=64, modes=16).to(DEVICE)

    if os.path.exists("results/checkpoints/best_lno.pt"):
        lno.load_state_dict(torch.load("results/checkpoints/best_lno.pt", map_location=DEVICE))
        print("[+] Loaded pre-trained weights for LNO Target Layers.")

    fno.eval()
    lno.eval()

    print("[*] Simulating FNO Unbounded Horizon...")
    fno_energy = run_autoregressive_rollout(fno, loader, steps=40)
    
    print("[*] Simulating LNO Dissipation-Constrained Horizon...")
    lno_energy = run_autoregressive_rollout(lno, loader, steps=40)

    # Leica Design Language Plot
    plt.figure(figsize=(7.5, 4), dpi=300)
    plt.plot(fno_energy, label='Standard FNO (Unconstrained Operator)', color='#e74c3c', linewidth=1.5, linestyle='--')
    plt.plot(lno_energy, label='Lindblad Neural Operator (LNO Boundary)', color='#111111', linewidth=2)
    plt.xlabel("Autoregressive Unrolling Horizons (t -> t+N)", fontsize=9)
    plt.ylabel("Systemic State Trajectory Energy Norm", fontsize=9)
    plt.title("Thermodynamic Trajectory Invariance and Open System Stability Bounds", fontsize=10, pad=10, fontweight='bold')
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.grid(True, linestyle=':', color='#e0e0e0')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig("results/long_horizon/energy_stability.png", bbox_inches='tight')
    plt.close()

    print("[++++] Stability Rollouts Compiled. Diagnostic plots committed to results/long_horizon/ plots.")

if __name__ == "__main__":
    main()
