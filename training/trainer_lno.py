"""
LNO-IonTransport Production Pipeline: Physics-Informed Operator Training Loop Engine
"""

import torch
from tqdm import tqdm
from collections import defaultdict

class LNOTrainer:
    def __init__(self, model, optimizer, scheduler, criterion, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        epoch_logs = defaultdict(float)
        
        pbar = tqdm(dataloader, desc="[Training Batch Step]", leave=False)
        for batch in pbar:
            if isinstance(batch, (tuple, list)):
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device)
            elif isinstance(batch, dict):
                inputs = batch["inputs"].to(self.device)
                targets = batch["state_t"].to(self.device)

            # Unpack full 6-channel transport field configuration
            state = inputs[:, 0, :]
            phi = inputs[:, 1, :]
            flux = inputs[:, 2, :]
            noise = inputs[:, 3, :]
            dissipation = inputs[:, 4, :]
            gamma = inputs[:, 5, 0] # Uniform scalar coupling field per batch element

            self.optimizer.zero_grad()
            
            # Forward pass through Lindblad Operator Network
            outputs = self.model(
                state=state,
                phi=phi,
                flux=flux,
                noise=noise,
                dissipation=dissipation,
                gamma=gamma
            )
            
            # Compute Multi-Component Open System Physics-Informed Loss Tuple
            loss_val, batch_logs = self.criterion(
                pred=outputs["next_state"],
                target=targets,
                state_t=state,
                flux=flux,
                noise=noise,
                gamma=gamma
            )

            loss_val.backward()
            self.optimizer.step()
            
            # Metric accumulation loop
            total_loss += loss_val.item()
            for k, v in batch_logs.items():
                epoch_logs[k] += v
                
            pbar.set_postfix({
                "total": f"{loss_val.item():.4f}", 
                "residual": f"{batch_logs.get('residual', 0):.4f}"
            })
            
        avg_loss = total_loss / len(dataloader)
        
        # Normalize aggregated physics logs across all training updates
        final_logs = {"running_loss": avg_loss}
        for k, v in epoch_logs.items():
            final_logs[f"avg_{k}"] = v / len(dataloader)
            
        return avg_loss, final_logs

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device)
                elif isinstance(batch, dict):
                    inputs = batch["inputs"].to(self.device)
                    targets = batch["state_t"].to(self.device)
                
                state = inputs[:, 0, :]
                phi = inputs[:, 1, :]
                flux = inputs[:, 2, :]
                noise = inputs[:, 3, :]
                dissipation = inputs[:, 4, :]
                gamma = inputs[:, 5, 0]
                
                outputs = self.model(
                    state=state,
                    phi=phi,
                    flux=flux,
                    noise=noise,
                    dissipation=dissipation,
                    gamma=gamma
                )
                
                loss_val, _ = self.criterion(
                    pred=outputs["next_state"],
                    target=targets,
                    state_t=state,
                    flux=flux,
                    noise=noise,
                    gamma=gamma
                )
                total_loss += loss_val.item()
                
        return total_loss / len(dataloader)
