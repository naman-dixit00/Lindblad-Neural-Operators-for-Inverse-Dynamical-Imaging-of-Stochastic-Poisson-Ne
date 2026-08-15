r"""
Yuánlǐ AI Research Laboratory
LNO-IonTransport Production Pipeline: Master Lindblad Neural Operator Framework

Combines spatial Fourier spectral transformations with non-unitary open-system
Lindblad dissipation layers to optimize stable, thermodynamically valid predictions.
Includes structural sparsity regularization interface for dynamic channel pruning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
from models.base_operator import BaseOperator
from models.spectral_kernel import SpectralConv1D
from models.dissipative_layer import DissipativeEvolutionLayer

class LindbladNeuralOperator(BaseOperator):
    def __init__(self, modes: int = 16, width: int = 64, depth: int = 4, in_channels: int = 6):
        super().__init__()
        self.width = width
        self.depth = depth
        self.modes = modes

        # 1. Coordinate Parameter Lifter Projection Head
        self.input_projection = nn.Linear(in_channels, width)

        # 2. Interleaved Parallel Spectral Operator Layers Sequence Blocks
        self.spectral_layers = nn.ModuleList([
            SpectralConv1D(width, width, modes) for _ in range(depth)
        ])

        self.pointwise_layers = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(depth)
        ])

        # 3. Explicit Physics-Informed Dissipative Open Matrix Constraints Layers
        self.dissipative_layers = nn.ModuleList([
            DissipativeEvolutionLayer(width) for _ in range(depth)
        ])

        # 4. Standard Forward Space Trajectory Predictor Projection Head
        self.output_projection = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

        # 5. Non-Equilibrium Inverse Dynamical Imaging Diagnostic Functional Head
        self.instability_head = nn.Sequential(
            nn.Conv1d(width, width, 1),
            nn.GELU(),
            nn.Conv1d(width, 1, 1)
        )

    def forward(self, state: torch.Tensor, phi: torch.Tensor, flux: torch.Tensor,
                noise: torch.Tensor, dissipation: torch.Tensor, gamma: torch.Tensor) -> Dict[str, Any]:

        # [FIX]: Bulletproof broadcasting alignment
        if gamma.dim() == 1:
            gamma = gamma.unsqueeze(-1)

        # Only repeat if the spatial grid dimension is NOT already matching the state dimension
        if gamma.shape[-1] != state.shape[-1]:
            gamma = gamma.repeat(1, state.shape[-1])

        x = torch.stack(
            [state, phi, flux, noise, dissipation, gamma],
            dim=-1
        ) # Layout strictly guaranteed: [Batch, Nx, 6]

        x = self.input_projection(x)
        x = x.permute(0, 2, 1) # Shape: [Batch, Width, Nx]

        spectral_energy_history = []

        for k in range(self.depth):
            spectral_update = self.spectral_layers[k](x)
            pointwise_update = self.pointwise_layers[k](x)

            x_linear = spectral_update + pointwise_update
            x = self.dissipative_layers[k](x_linear, gamma)

            if k < self.depth - 1:
                x = F.gelu(x)

            layer_energy = self.compute_spectral_energy(x)
            spectral_energy_history.append(layer_energy)

        instability_map = self.instability_head(x)

        x = x.permute(0, 2, 1)
        predicted_evolution = self.output_projection(x)

        return {
            "next_state": predicted_evolution.squeeze(-1),
            "instability_map": instability_map.squeeze(1),
            "spectral_energy": spectral_energy_history
        }

    def compute_thermodynamic_entropy_loss(self, prediction: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        prob_distribution = F.softmax(prediction, dim=-1)
        entropy_profile = -torch.sum(prob_distribution * torch.log(prob_distribution + eps), dim=-1)
        return -entropy_profile.mean()

    def get_spectral_stability_regularizer(self) -> torch.Tensor:
        total_regularization_penalty = 0.0
        for module in self.modules():
            if isinstance(module, SpectralConv1D):
                total_regularization_penalty += torch.sum(torch.abs(module.weights) ** 2)
        return total_regularization_penalty

    def get_dissipative_coefficient_norm(self) -> torch.Tensor:
        total_decay_norm = 0.0
        for module in self.modules():
            if isinstance(module, DissipativeEvolutionLayer):
                total_decay_norm += torch.sum(torch.abs(module.jump_transform))
        return total_decay_norm

    def get_structural_sparsity_penalty(self) -> torch.Tensor:
        l1_penalty = 0.0
        for layer in self.dissipative_layers:
            l1_penalty += torch.sum(torch.abs(layer.get_gated_gamma()))
        return l1_penalty
