"""
LNO-IonTransport Production Pipeline: Master Training Execution Orchestrator
"""

import os
import torch
import yaml
from torch.utils.data import DataLoader

from dataset_loader import IonTransportDataset
from models.lno_model import LindbladNeuralOperator
from training.losses import TotalLNOLoss
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler
from training.trainer_lno import LNOTrainer

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    print("[+] Initializing Yuánlǐ AI LNO-IonTransport Training Orchestrator...")
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Operational Compute Target Selected: Dedicated [{device.upper()}] Hardware Engine.")

    # Multi-regime processing paths alignment
    dataset_cfg = config.get("dataset", {})
    splits_dir = dataset_cfg.get("splits_dir", "dataset")

    # Resolve direct paths to the specific split directories
    train_path = os.path.join(splits_dir, "train")
    val_path = os.path.join(splits_dir, "val")

    # Connect data pipelines interfaces
    print("[+] Configuring Structured Dataset Multi-Channel Streams...")
    train_dataset = IonTransportDataset(train_path)
    val_dataset = IonTransportDataset(val_path)

    # Automatically drop workers if executing on CPU to prevent freezes
    num_workers = 2 if device == "cpu" else config["training"].get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["training"]["batch_size"], 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )

    # Operator setup matching the EXACT inspected signature
    model_cfg = config.get("model", {})
    print(f"[+] Initializing Neural Operator Target Architecture: {model_cfg.get('name', 'LindbladNeuralOperator')}...")
    
    # Explicitly forcing in_channels=6 to match the physical open-system transport vectors layout
    model = LindbladNeuralOperator(
        modes=model_cfg.get("modes", 16),
        width=model_cfg.get("width", 64),
        depth=model_cfg.get("depth", model_cfg.get("num_layers", 4)),
        in_channels=6
    )

    # Physical integration parameters loading to enforce constraint invariants
    physics_cfg = config.get("physics", {})
    criterion = TotalLNOLoss(
        dx=physics_cfg.get("dx", 0.01),
        dt=physics_cfg.get("dt", 0.001)
    )

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    trainer = LNOTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device
    )

    best_val_loss = float("inf")
    total_epochs = config["training"].get("epochs", 150)

    print(f"[+] Training Convergence Cycle Triggered across {total_epochs} Scientific Iterations.\n")
    for epoch in range(total_epochs):
        train_loss, logs = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)

        # Coordinate scheduler behavior
        if scheduler is not None:
            if config["training"].get("scheduler") == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print("=" * 80)
        print(f"Scientific Cycle: [{epoch + 1} / {total_epochs}]")
        print(f"Aggregated Regularization Training Loss     : {train_loss:.6f}")
        print(f"Aggregated Validation Tracking Loss         : {val_loss:.6f}")
        print(f"Continuous Physical Component Metrics Logs  : {logs}")
        print("=" * 80)

        # Secure elite operator model instances checkpoint updates
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Create results/checkpoints directory if it doesn't exist
            os.makedirs("results/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "results/checkpoints/best_lno.pt")
            print(f"[+] Verification Invariant Met. Saved Elite Operator Invariant Checkpoint (Loss: {best_val_loss:.6f})")

    print("\n[+] Yuánlǐ AI Training Suite Pipeline Completed Successfully.")

if __name__ == "__main__":
    main()
