"""
LNO-IonTransport Production Pipeline: Physical Invariants & Operator Convergence Metrics
"""

import torch

def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root Mean Squared Error across the batch."""
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()

def relative_l2_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Relative Operator Graph Error: ||u_pred - u_true||_2 / ||u_true||_2
    Calculated per sample and averaged over the batch to ensure dimensional consistency.
    """
    num = torch.norm(pred - target, p=2, dim=-1)
    den = torch.norm(target, p=2, dim=-1) + eps
    return torch.mean(num / den).item()

def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return torch.mean(torch.abs(pred - target)).item()

def mass_conservation_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Evaluates global physical mass conservation drift for the PNP electrodiffusion field:
    Error = |Integral(u_pred) dx - Integral(u_true) dx|
    """
    pred_mass = torch.sum(pred, dim=-1)
    target_mass = torch.sum(target, dim=-1)
    return torch.mean(torch.abs(pred_mass - target_mass)).item()

def transport_entropy(field: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Computes Shannon Spatial Transport Entropy for nonequilibrium configurations:
    S = -Integral(c * log(c) dx)
    """
    field_clamped = torch.clamp(field, min=eps)
    return -torch.sum(field_clamped * torch.log(field_clamped), dim=-1)

def entropy_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Tracks thermodynamic entropy reconstruction mismatch."""
    s_pred = transport_entropy(pred)
    s_true = transport_entropy(target)
    return torch.mean(torch.abs(s_pred - s_true)).item()

def instability_index(field: torch.Tensor) -> float:
    """
    Measures localized gradient spikes to catch high-noise mathematical transport explosion.
    """
    if field.shape[-1] < 2:
        return 0.0
    grad = field[..., 1:] - field[..., :-1]
    return torch.mean(torch.abs(grad)).item()

def dissipation_rate(field: torch.Tensor) -> float:
    """
    Dissipation Energy Metric: D = Integral(|Grad(u)|^2 dx)
    Captures the irreversible open-system decay profile.
    """
    if field.shape[-1] < 2:
        return 0.0
    grad = field[..., 1:] - field[..., :-1]
    return torch.mean(grad ** 2).item()

def spectral_energy(field: torch.Tensor) -> float:
    """Computes total energy in the Fourier space to check spectral damping behavior."""
    fft = torch.fft.rfft(field, dim=-1)
    return torch.mean(torch.abs(fft) ** 2).item()

def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    return {
        "rmse": rmse(pred, target),
        "relative_l2": relative_l2_error(pred, target),
        "mae": mae(pred, target),
        "mass_error": mass_conservation_error(pred, target),
        "entropy_error": entropy_error(pred, target),
        "instability_index": instability_index(pred),
        "dissipation_rate": dissipation_rate(pred),
        "spectral_energy": spectral_energy(pred)
    }
