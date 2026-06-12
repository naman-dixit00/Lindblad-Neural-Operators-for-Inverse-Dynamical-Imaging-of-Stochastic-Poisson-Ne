"""
Yuánlǐ AI Research Laboratory
LNO-IonTransport Production Pipeline: Standard Fourier Neural Operator Baseline

Implements a standard conservative Fourier Neural Operator (FNO-1D) architecture 
acting as our primary baseline target control system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_operator import BaseOperator
from models.spectral_kernel import SpectralConv1D

class FNOBaseline(BaseOperator):
    def __init__(self, modes: int = 16, width: int = 64, in_channels: int = 6):
        super().__init__()
        self.modes = modes
        self.width = width
        
        # Input lifter layer mapping point-wise stacked configuration parameters
        self.input_projection = nn.Linear(in_channels, width)
        
        # Sequence of spectral convolutions coupled with pointwise skip projections
        self.conv0 = SpectralConv1D(width, width, modes)
        self.conv1 = SpectralConv1D(width, width, modes)
        self.conv2 = SpectralConv1D(width, width, modes)
        self.conv3 = SpectralConv1D(width, width, modes)
        
        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)
        
        # Output project head layers mapping hidden dimensions back to predicted target scalar field
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state, phi, flux, noise, dissipation, gamma) -> torch.Tensor:
        # Uniform channel stacking initialization to ensure complete evaluation alignment
        if gamma.dim() == 1:
            gamma = gamma.unsqueeze(-1)
            
        x = torch.stack(
            [
                state, 
                phi, 
                flux, 
                noise, 
                dissipation, 
                gamma.repeat(1, state.shape[-1])
            ], 
            dim=-1
        ) # Shape: [Batch, Nx, 6]
        
        # Map to high-dimensional latent space
        x = self.input_projection(x)
        x = x.permute(0, 2, 1) # Shape: [Batch, Width, Nx]
        
        # Layer 1 block iteration
        x = F.gelu(self.conv0(x) + self.w0(x))
        # Layer 2 block iteration
        x = F.gelu(self.conv1(x) + self.w1(x))
        # Layer 3 block iteration
        x = F.gelu(self.conv2(x) + self.w2(x))
        # Layer 4 block iteration
        x = self.conv3(x) + self.w3(x)
        
        # Return coordinates map back to localized point distributions
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        out = self.fc2(x)
        
        return out.squeeze(-1) # Output shape matches next targeted frame state vector
