"""
================================================================================
PHYSICAL STRUCTURE ABLATION MATRIX
File: experiments/exp_dissipation_ablation.py
Description: Evaluates the specific mathematical impact of individual channels.
================================================================================
"""

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset_loader import IonTransportDataset
from models.lno_model import LindbladNeuralOperator
from evaluation.metrics import compute_all_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate_ablated_lno(model, loader, mode="none"):
    batch_logs = []
    for batch_tuple in loader:
        inputs_stacked, targets_true = batch_tuple
        
        state_t = inputs_stacked[:, 0, :].to(DEVICE)
        phi = inputs_stacked[:, 1, :].to(DEVICE)
        flux = inputs_stacked[:, 2, :].to(DEVICE)
        noise = inputs_stacked[:, 3, :].to(DEVICE)
        
        # Ablation Strategy Selection
        if mode == "ablate_dissipation":
            dissipation = torch.zeros_like(inputs_stacked[:, 4, :]).to(DEVICE) # Kill dissipation profile
        else:
            dissipation = inputs_stacked[:, 4, :].to(DEVICE)
            
        if mode == "ablate_gamma":
            gamma = torch.zeros_like(inputs_stacked[:, 5, 0]).to(DEVICE) # Kill coupling scale
        else:
            gamma = inputs_stacked[:, 5, 0].to(DEVICE)

        targets_true = targets_true.to(DEVICE)
        pred = model(state_t, phi, flux, noise, dissipation, gamma)
        
        if isinstance(pred, dict) and "next_state" in pred:
            pred = pred["next_state"]

        metrics = compute_all_metrics(pred, targets_true)
        batch_logs.append(metrics)
        
    return pd.DataFrame(batch_logs).mean().to_dict()

def main():
    os.makedirs("results/ablation", exist_ok=True)
    print("\n" + "="*70 + "\n[+] RUNNING CRITICAL LAYER ABLATION MATRIX\n" + "="*70)

    dataset = IonTransportDataset(root_dir="dataset/val")
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = LindbladNeuralOperator(in_channels=6, width=64, modes=16).to(DEVICE)
    checkpoint = "results/checkpoints/best_lno.pt"
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()

    ablation_results = {}
    
    print("[*] Evaluating Full LNO Framework Architecture...")
    ablation_results["Full_LNO_Framework"] = evaluate_ablated_lno(model, loader, mode="none")
    
    print("[*] Evaluating Ablated Dissipation Space Module...")
    ablation_results["Ablated_Dissipation"] = evaluate_ablated_lno(model, loader, mode="ablate_dissipation")
    
    print("[*] Evaluating Ablated Gamma Coupling Coefficients...")
    ablation_results["Ablated_Gamma_Coupling"] = evaluate_ablated_lno(model, loader, mode="ablate_gamma")

    df = pd.DataFrame(ablation_results).T
    print("\n" + "="*20 + " ABLATION MATRIX RESULTS " + "="*20)
    print(df[['rmse', 'mae', 'entropy_error']])
    print("="*65 + "\n")
    
    df.to_csv("results/ablation/dissipation_ablation.csv")
    print("[++++] Ablation Matrix committed to disk successfully.")

if __name__ == "__main__":
    main()
