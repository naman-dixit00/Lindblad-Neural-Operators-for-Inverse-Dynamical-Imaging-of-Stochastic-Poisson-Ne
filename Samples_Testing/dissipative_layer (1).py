import torch
import torch.nn as nn
import torch.nn.functional as F

class DissipativeEvolutionLayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        # Coherent Drift Parameter Kernel for spatial shift transformations
        self.coherent_kernel = nn.Parameter(torch.randn(channels, channels, 5) * 0.02)

        # Multi-Channel Non-Unitary Jump Parameter Operators (L Matrix)
        self.jump_transform = nn.Parameter(torch.randn(channels, channels) * 0.01)

        # Spatial Smoothing Operator Weight to absorb continuous high-frequency noise spikes
        self.spatial_diffusion_weight = nn.Parameter(torch.ones(1, channels, 1) * 0.05)

        # Native 3D Alignment: Matches checkpoint's [64, 64, 5] shape for raw_gamma
        self.raw_gamma = nn.Parameter(torch.randn(channels, channels, 5) * 0.02)

    def get_gated_gamma(self) -> torch.Tensor:
        # Collapse 3D tensor [channels, channels, 5] to channel-wise gating vector [channels]
        channel_gamma = self.raw_gamma.mean(dim=(0, 2))
        return F.softplus(channel_gamma) * torch.sigmoid(channel_gamma)

    def _apply_lindblad_dissipation(self, x: torch.Tensor, environmental_gamma: torch.Tensor) -> torch.Tensor:
        L = self.jump_transform
        L_dag_L = torch.mm(L.t(), L)

        x_permuted = x.permute(0, 2, 1) # [Batch, Nx, Channels]

        sandwiched = torch.matmul(x_permuted, L.t())
        sandwiched = torch.matmul(sandwiched, L)

        anti_comm_left = torch.matmul(x_permuted, L_dag_L)
        anti_comm_right = torch.matmul(x_permuted, L_dag_L.t())
        anti_commutator = 0.5 * (anti_comm_left + anti_comm_right)

        dissipation_tensor = sandwiched - anti_commutator
        dissipation_tensor = dissipation_tensor.permute(0, 2, 1) # [Batch, Channels, Nx]

        # Apply Internal Learnable Structural Gate
        structural_gate = self.get_gated_gamma().view(1, self.channels, 1)
        gated_dissipation = structural_gate * dissipation_tensor

        if environmental_gamma.dim() == 2:
            environmental_gamma = environmental_gamma.unsqueeze(1) # [Batch, 1, Nx]
        elif environmental_gamma.dim() == 1:
            environmental_gamma = environmental_gamma.view(-1, 1, 1)

        return environmental_gamma * gated_dissipation

    def forward(self, x: torch.Tensor, gamma_field: torch.Tensor) -> torch.Tensor:
        padded_x = F.pad(x, (2, 2), mode='circular')
        coherent_out = F.conv1d(padded_x, self.coherent_kernel)

        dissipative_out = self._apply_lindblad_dissipation(x, gamma_field)

        left_shift = torch.roll(x, shifts=-1, dims=-1)
        right_shift = torch.roll(x, shifts=1, dims=-1)
        laplacian_fields = left_shift - 2.0 * x + right_shift
        spatial_stabilization = self.spatial_diffusion_weight * laplacian_fields

        return x + coherent_out + dissipative_out + spatial_stabilization
