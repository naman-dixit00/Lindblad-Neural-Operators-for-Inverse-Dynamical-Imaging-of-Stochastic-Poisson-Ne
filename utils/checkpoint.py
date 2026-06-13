"""
LNO-IonTransport Pipeline: Model Checkpointing System
"""
import torch
import os

def save_checkpoint(state, checkpoint_dir="results/checkpoints", filename="best_lno_model.pt"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"[+] Production-grade checkpoint securely saved to {filepath}")

def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device="cuda"):
    if not os.path.exists(filepath):
        print(f"[-] No checkpoint found at {filepath}. Starting from scratch.")
        return None

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    print(f"[+] Checkpoint loaded successfully from {filepath} (Epoch {checkpoint.get('epoch', 'unknown')})")
    return checkpoint
