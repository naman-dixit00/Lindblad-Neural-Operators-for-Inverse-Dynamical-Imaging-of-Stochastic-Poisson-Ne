"""
Yuánlǐ AI Research Laboratory
LNO-IonTransport Production Pipeline: High-Fidelity Fourier Spectral Kernel

Implements the 1D Fourier neural operator kernel parameterized by a truncated 
set of complex-valued modes to perform global spatial convolutions efficiently.
"""

import torch
import torch.nn as nn
import torch.fft

class SpectralConv1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        """
        Mathematical formulation:
            R_R(u)(x) = F^-1 ( R_hat(k) * F(u)(k) )
        where R_hat is a parameterized complex weight tensor restricting high frequencies.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Xavier Glorot initialization scale adapted for complex tensor allocations
        scale = 1.0 / (in_channels * out_channels)
        
        # Native PyTorch complex parameter initialization for seamless backpropagation
        self.weights = nn.Parameter(
            torch.complex(
                torch.randn(in_channels, out_channels, modes) * scale,
                torch.randn(in_channels, out_channels, modes) * scale
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input tensor shape x: [Batch, Channels, Nx]
        batch_size = x.shape[0]
        nx = x.size(-1)
        
        # 1. Transform spatial domain signals to frequency space
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # 2. Allocate complex allocation tensor for output modes matching RFFT tracking
        out_ft = torch.zeros(
            batch_size, 
            self.out_channels, 
            nx // 2 + 1, 
            dtype=torch.cfloat, 
            device=x.device
        )
        
        # 3. Apply parameter weights tensor matrix multiplication over target modes spectrum
        # Tensor index mapping: b=batch, i=in_channels, o=out_channels, x=truncated_modes
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", 
            x_ft[:, :, :self.modes], 
            self.weights
        )
        
        # 4. Inverse Fourier transform to map operators back to continuous spatial coordinates
        return torch.fft.irfft(out_ft, n=nx, dim=-1)
