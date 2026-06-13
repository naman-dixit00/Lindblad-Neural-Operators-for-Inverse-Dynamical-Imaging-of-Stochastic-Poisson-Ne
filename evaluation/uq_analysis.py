"""
LNO-IonTransport Production Pipeline: Monte Carlo Dropout Uncertainty Quantification (UQ)
"""

import torch
import numpy as np

class MonteCarloDropoutUQ:
    def __init__(self, model, samples: int = 20, device: str = "cuda"):
        self.model = model
        self.samples = samples
        self.device = device

    def enable_dropout(self):
        """Forces dropout layers to stay active during stochastic inference inference."""
        for module in self.model.modules():
            if module.__class__.__name__.startswith("Dropout"):
                module.train()

    @torch.no_grad()
    def predict(self, state, phi, flux, noise, dissipation, gamma):
        self.model.eval()
        self.enable_dropout()

        predictions = []
        for _ in range(self.samples):
            outputs = self.model(
                state=state.to(self.device),
                phi=phi.to(self.device),
                flux=flux.to(self.device),
                noise=noise.to(self.device),
                dissipation=dissipation.to(self.device),
                gamma=gamma.to(self.device)
            )
            pred = outputs["next_state"] if isinstance(outputs, dict) else outputs
            predictions.append(pred.unsqueeze(0))

        predictions = torch.cat(predictions, dim=0) # Shape: [Samples, Batch, Spatial_Grid]
        mean_prediction = predictions.mean(dim=0)
        epistemic_uncertainty = predictions.std(dim=0)

        return mean_prediction, epistemic_uncertainty

    def uncertainty_score(self, uncertainty: torch.Tensor) -> float:
        return torch.mean(uncertainty).item()

    def instability_confidence_map(self, uncertainty: torch.Tensor) -> np.ndarray:
        """Generates regularized mapping for hidden transport collapse imaging."""
        normalized = uncertainty / (torch.max(uncertainty) + 1e-8)
        return normalized.cpu().numpy()

    def compute_residual_aleatoric(self, mean_pred: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """Maps data-conditioned variance from empirical validation target residuals."""
        return (ground_truth.to(self.device) - mean_pred) ** 2
