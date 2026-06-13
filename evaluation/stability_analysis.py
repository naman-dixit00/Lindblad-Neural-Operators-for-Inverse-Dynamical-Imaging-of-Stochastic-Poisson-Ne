"""
LNO-IonTransport Production Pipeline: Long-Horizon Stability Analyzer
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

class StabilityAnalyzer:
    def __init__(self, model, device="cuda"):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def rollout(self, initial_state, phi, flux, noise, dissipation, gamma, steps=100):
        """
        Executes long-horizon autoregressive forecasting using the structured
        6-parameter mapping signature required by the Yuánlǐ LNO core layers.
        """
        self.model.eval()
        states = []
        current = initial_state.clone().to(self.device)

        # Static background fields or boundary conditions passed contextually
        phi_dev = phi.to(self.device)
        flux_dev = flux.to(self.device)
        noise_dev = noise.to(self.device)
        diss_dev = dissipation.to(self.device)
        gamma_dev = gamma.to(self.device)

        for _ in range(steps):
            outputs = self.model(
                state=current,
                phi=phi_dev,
                flux=flux_dev,
                noise=noise_dev,
                dissipation=diss_dev,
                gamma=gamma_dev
            )

            # Extracting via signature dictionary format
            pred = outputs["next_state"] if isinstance(outputs, dict) else outputs
            states.append(pred.detach().cpu())
            current = pred

        return torch.stack(states)

    def compute_energy(self, states):
        return torch.mean(states ** 2, dim=(-1, -2)).numpy()

    def compute_spectral_radius(self, states):
        radii = []
        for s in states:
            fft = torch.fft.rfft(s, dim=-1)
            radii.append(torch.max(torch.abs(fft)).item())
        return np.array(radii)

    def detect_explosion(self, energy, threshold=100.0):
        return bool(np.max(energy) > threshold)

    def plot_energy_evolution(self, energy):
        """Minimalist high-end cinematic visualization layout."""
        plt.figure(figsize=(7, 4))
        plt.plot(energy, color='#1A1A1A', linewidth=1.5, label='LNO Field Evolution')
        plt.xlabel("Autoregressive Horizon Steps")
        plt.ylabel("System Energy Invariant")
        plt.title("Long-Horizon Dynamic Operator Stability Bounds", fontsize=10, pad=12)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

    def analyze(self, initial_state, phi, flux, noise, dissipation, gamma, steps=100):
        states = self.rollout(initial_state, phi, flux, noise, dissipation, gamma, steps)
        energy = self.compute_energy(states)
        spectral_radius = self.compute_spectral_radius(states)
        explosion = self.detect_explosion(energy)

        return {
            "energy": energy,
            "spectral_radius": spectral_radius,
            "explosion_detected": explosion
        }
