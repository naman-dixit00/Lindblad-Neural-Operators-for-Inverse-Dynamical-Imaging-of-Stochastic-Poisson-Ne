"""
LNO-IonTransport Production Pipeline: Scheduler Builder
"""

import torch

def build_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Configures continuous learning curves adjustments profiles.
    """
    train_cfg = config.get("training", {})
    scheduler_config = train_cfg.get("scheduler", "cosine")

    if isinstance(scheduler_config, dict):
        scheduler_name = scheduler_config.get("name", "cosine").lower()
    else: # Assume it's a string directly
        scheduler_name = str(scheduler_config).lower()

    epochs = train_cfg.get("epochs", 150)

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    else:
        return None
