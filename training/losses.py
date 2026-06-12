"""
LNO-IonTransport Production Pipeline: Scientific Loss System

Encodes high-order physical invariant losses, thermodynamic constraints,
relative operator reconstruction targets, and spectral stability regularizers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativeLpLoss(nn.Module):
    """
    Relative L2 Operator Reconstruction Loss.
    Evaluates: ||u_pred - u_true||_2 / ||u_true||_2
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num = torch.norm(pred - target, p=2, dim=-1)
        den = torch.norm(target, p=2, dim=-1) + self.eps
        return torch.mean(num / den)

class DissipativeConstraintLoss(nn.Module):
    """
    Enforces the Second Law of Thermodynamics via soft dissipative decay constraints.
    Penalizes anomalous physical energy growth profiles: E(t+1) > E(t)
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, current_state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        energy_t = torch.mean(current_state ** 2, dim=-1)
        energy_t1 = torch.mean(next_state ** 2, dim=-1)
        
        # Penalize if energy at t+1 is greater than energy at t
        violation = F.relu(energy_t1 - energy_t)
        return self.weight * violation.mean()

class EntropyConsistencyLoss(nn.Module):
    """
    Thermodynamic Entropy Consistency Regularization.
    Prevents unphysical local mass/entropy collapse over advanced transport horizons.
    """
    def __init__(self, eps: float = 1e-12, weight: float = 1.0):
        super().__init__()
        self.eps = eps
        self.weight = weight

    def compute_entropy(self, x: torch.Tensor) -> torch.Tensor:
        x_clamped = torch.clamp(x, min=self.eps)
        return -torch.sum(x_clamped * torch.log(x_clamped), dim=-1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        s_pred = self.compute_entropy(pred)
        s_true = self.compute_entropy(target)
        return self.weight * F.mse_loss(s_pred, s_true)

class SpectralStabilityLoss(nn.Module):
    """
    Fourier Space Spectral Stability Constraints.
    Suppresses high-frequency numerical explosions and chaotic mode amplification.
    """
    def __init__(self, weight: float = 1e-3):
        super().__init__()
        self.weight = weight

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        fft_vals = torch.fft.rfft(field, dim=-1)
        spectral_magnitude = torch.abs(fft_vals)
        
        # Penalize modes exceeding unitary propagation boundary bounds
        unstable_modes = F.relu(spectral_magnitude - 1.0)
        return self.weight * unstable_modes.mean()

class TransportResidualLoss(nn.Module):
    """
    Physics-Informed PDE Operator Residual Loss.
    Evaluates conformity to: \partial_t c + \nabla \cdot J = \eta - \gamma c
    """
    def __init__(self, dx: float = 0.01, dt: float = 0.001, weight: float = 1.0):
        super().__init__()
        self.dx = dx
        self.dt = dt
        self.weight = weight

    def central_gradient_1d(self, x: torch.Tensor) -> torch.Tensor:
        return (x[..., 2:] - x[..., :-2]) / (2.0 * self.dx)

    def forward(self, state_t: torch.Tensor, state_t1: torch.Tensor, 
                flux: torch.Tensor, noise: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        
        dc_dt = (state_t1 - state_t) / self.dt
        flux_grad = self.central_gradient_1d(flux)
        
        # Match dimensions across valid inner spatial domains
        dc_dt_interior = dc_dt[..., 1:-1]
        noise_interior = noise[..., 1:-1]
        
        # Expand gamma if it represents a scalar per batch item
        if gamma.dim() == 1:
            gamma = gamma.view(-1, 1)
        gamma_interior = gamma * state_t[..., 1:-1]
        
        rhs = -flux_grad + noise_interior - gamma_interior
        residual = dc_dt_interior - rhs
        
        return self.weight * torch.mean(residual ** 2)

class TotalLNOLoss(nn.Module):
    """
    Physics-Informed Loss Orchestration System.
    """
    def __init__(self, dx: float = 0.01, dt: float = 0.001,
                 lambda_recon: float = 1.0, lambda_diss: 0.2, 
                 lambda_entropy: float = 0.1, lambda_spectral: float = 1e-4, 
                 lambda_residual: float = 0.2):
        super().__init__()
        self.recon = RelativeLpLoss()
        self.diss = DissipativeConstraintLoss(weight=lambda_diss)
        self.entropy = EntropyConsistencyLoss(weight=lambda_entropy)
        self.spectral = SpectralStabilityLoss(weight=lambda_spectral)
        self.residual = TransportResidualLoss(dx=dx, dt=dt, weight=lambda_residual)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, state_t: torch.Tensor, 
                flux: torch.Tensor, noise: torch.Tensor, gamma: torch.Tensor) -> tuple:
        
        loss_recon = self.recon(pred, target)
        loss_diss = self.diss(state_t, pred)
        loss_entropy = self.entropy(pred, target)
        loss_spectral = self.spectral(pred)
        loss_residual = self.residual(state_t, pred, flux, noise, gamma)
        
        total_loss = (loss_recon + loss_diss + loss_entropy + loss_spectral + loss_residual)
        
        logs = {
            "total": total_loss.item(),
            "recon": loss_recon.item(),
            "diss": loss_diss.item(),
            "entropy": loss_entropy.item(),
            "spectral": loss_spectral.item(),
            "residual": loss_residual.item()
        }
        return total_loss, logs
