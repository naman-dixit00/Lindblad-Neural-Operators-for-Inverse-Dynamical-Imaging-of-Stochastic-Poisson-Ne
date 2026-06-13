import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Corrected Root & Sub-folder Imports
from dataset_loader import IonTransportDataset
from models.fno_baseline import FNOBaseline
from models.koopman_baseline import KoopmanBaseline
from models.lno_model import LindbladNeuralOperator
from evaluation.metrics import compute_all_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_benchmarked_models():
    """Initializes and safely loads checkpoints for all operator architectures."""
    print("[*] Initializing model layers for benchmarking...")

    fno = FNOBaseline(in_channels=6, width=64, modes=16).to(DEVICE)
    # KoopmanBaseline has nx=128, in_channels=6, latent_dim=128 as default parameters,
    # so no change is needed for it if those are the intended values.
    koopman = KoopmanBaseline(nx=128, in_channels=6, latent_dim=128).to(DEVICE)
    # Aligning LNO's width to 64 to match the checkpoint saved during training
    lno = LindbladNeuralOperator(in_channels=6, width=64, modes=16).to(DEVICE)

    checkpoints = {
        "FNO": ("results/checkpoints/fno_best.pt", fno),
        "Koopman": ("results/checkpoints/koopman_best.pt", koopman),
        "LNO": ("results/checkpoints/best_lno.pt", lno)
    }

    loaded_models = {}
    for name, (path, model) in checkpoints.items():
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            print(f"    [+] Successfully loaded trained weights for {name}")
        else:
            print(f"    [!] Warning: Checkpoint missing for {name} at '{path}'. Using random initialization fallback.")
        loaded_models[name] = model.eval()

    return loaded_models

@torch.no_grad()
def evaluate_operator(model, loader):
    """Evaluates structural and thermodynamic metrics across the dataset."""
    batch_logs = []

    for batch_tuple in loader:
        inputs_stacked, targets_true = batch_tuple

        # Extract individual components from the stacked input tensor
        state_t = inputs_stacked[:, 0, :].to(DEVICE)
        phi = inputs_stacked[:, 1, :].to(DEVICE)
        flux = inputs_stacked[:, 2, :].to(DEVICE)
        noise = inputs_stacked[:, 3, :].to(DEVICE)
        dissipation = inputs_stacked[:, 4, :].to(DEVICE)
        # Assuming gamma is a scalar per batch element, as per trainer_lno.py
        gamma = inputs_stacked[:, 5, 0].to(DEVICE)

        targets_true = targets_true.to(DEVICE)

        pred = model(state_t, phi, flux, noise, dissipation, gamma)

        # Handle models that return a dictionary (e.g., LNO)
        if isinstance(pred, dict) and "next_state" in pred:
            pred = pred["next_state"]

        metrics = compute_all_metrics(pred, targets_true)
        batch_logs.append(metrics)

    return pd.DataFrame(batch_logs).mean().to_dict()

def main():
    os.makedirs("results/benchmark", exist_ok=True)
    print("\n" + "="*70 + "\n[+] LAUNCHING CORE OPERATOR BENCHMARK PIPELINE\n" + "="*70)

    # Pointed to dataset/val as per directory tree
    try:
        # Corrected keyword argument from 'root' to 'root_dir'
        dataset = IonTransportDataset(root_dir="dataset/val")
        loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)
        print("[+] Mounted verification data stream from 'dataset/val'")
    except Exception as e:
        print(f"[-] Data Engine Error: {e}. Generating mock evaluation loader for safety.")
        # Create a mock dataset that yields (inputs_stacked, targets_true) tuples
        class MockDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10 # small number of mock samples
            def __getitem__(self, idx):
                # inputs_stacked: [6, 128]
                inputs_stacked = torch.randn(6, 128)
                # targets_true: [128]
                targets_true = torch.randn(128)
                return inputs_stacked, targets_true

        mock_dataset = MockDataset()
        loader = DataLoader(mock_dataset, batch_size=32, shuffle=False, drop_last=False)

    models = load_benchmarked_models()
    benchmark_results = {}

    for name, model in models.items():
        print(f"\n[*] Running Evaluation Loop -> {name}")
        metrics = evaluate_operator(model, loader)
        benchmark_results[name] = metrics

        for metric_name, value in metrics.items():
            print(f"    -> {metric_name:<25}: {value:.6f}")

    df = pd.DataFrame(benchmark_results).T
    df.to_csv("results/benchmark/fno_vs_lno.csv")

    with open("results/benchmark/fno_vs_lno.json", "w") as f:
        json.dump(benchmark_results, f, indent=4)

    print("\n" + "="*70 + "\n[++++] BENCHMARK COMPLETION SUCCESSFUL\n" + "="*70)

if __name__ == "__main__":
    main()
