"""
Yuánlǐ AI Research Laboratory
LNO-IonTransport Production Pipeline: Base Neural Operator Abstract Class

Defines the formal mathematical interface for neural operators mapping non-equilibrium 
infinite-dimensional functional fields across Sobolev spaces G_theta : H^s -> H^s.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseOperator(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Executes forward operator propagation mapping initial conditions 
        and environmental boundary configurations to the next temporal field state.
        """
        pass

    def compute_spectral_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the discrete Power Spectral Density (PSD) energy distribution:
            E(k) = (1 / N_x) * \sum |F(u)(k)|^2
        Used to track spectral decay cascades and verify numerical stability.
        Args:
            x: Tensor of shape [Batch, Channels, Nx] or [Batch, Nx]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Compute real Fast Fourier Transform along spatial dimension
        fft_coeffs = torch.fft.rfft(x, dim=-1)
        energy_density = torch.abs(fft_coeffs) ** 2
        
        # Average across spatial frequencies and channels to get mean scalar energy
        return torch.mean(energy_density)

    def get_operator_frobenius_norm(self) -> torch.Tensor:
        """
        Computes the collective Frobenius parameter norm of the underlying 
        operator layer weights to diagnose gradient structural regularization.
        """
        total_squared_norm = 0.0
        for param in self.parameters():
            if param.requires_grad:
                total_squared_norm += torch.sum(param ** 2)
        return torch.sqrt(total_squared_norm)
