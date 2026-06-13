"""
LNO-IonTransport Production Pipeline: Optimizer Builder
"""

import torch

def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Parses structural hyperparameters from configuration matrix 
    and returns optimized execution engines.
    """
    train_cfg = config.get("training", {})
    optimizer_name = train_cfg.get("optimizer", "adamw").lower()
    lr = train_cfg.get("learning_rate", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 1e-4)

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"[-] Unsupported optimizer selection configuration: {optimizer_name}")
