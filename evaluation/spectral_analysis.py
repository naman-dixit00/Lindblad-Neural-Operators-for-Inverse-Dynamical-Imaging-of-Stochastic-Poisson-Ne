"""

LNO-IonTransport Production Pipeline: Spectral Operator Energy Field Profiler
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

class SpectralAnalyzer:
    def __init__(self):
        pass

    def fft_spectrum(self, field: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.rfft(field, dim=-1)
        return torch.abs(fft)

    def spectral_entropy(self, field: torch.Tensor, eps: float = 1e-8) -> float:
        spectrum = self.fft_spectrum(field)
        p = spectrum / (torch.sum(spectrum, dim=-1, keepdim=True) + eps)
        entropy = -torch.sum(p * torch.log(p + eps), dim=-1)
        return torch.mean(entropy).item()

    def dominant_modes(self, field: torch.Tensor, topk: int = 5):
        spectrum = self.fft_spectrum(field)
        avg_spectrum = torch.mean(spectrum, dim=0)
        if avg_spectrum.ndim > 1:
            avg_spectrum = torch.mean(avg_spectrum, dim=0)
        vals, idx = torch.topk(avg_spectrum, min(topk, avg_spectrum.shape[0]))
        return idx.cpu().numpy(), vals.cpu().numpy()

    def spectral_radius(self, field: torch.Tensor) -> float:
        return torch.max(self.fft_spectrum(field)).item()

    def compare_spectra(self, field_a: torch.Tensor, field_b: torch.Tensor):
        spec_a = torch.mean(self.fft_spectrum(field_a), dim=0)
        spec_b = torch.mean(self.fft_spectrum(field_b), dim=0)
        return spec_a.detach().cpu(), spec_b.detach().cpu()

    def plot_spectra(self, spec_a, spec_b, label_a="Baseline FNO", label_b="Yuánlǐ LNO"):
        plt.figure(figsize=(8, 4))
        plt.plot(spec_a.numpy(), label=label_a, color='#7F7F7F', alpha=0.6, linestyle='--')
        plt.plot(spec_b.numpy(), label=label_b, color='#B8860B', linewidth=1.5) # Premium gold/amber touch
        plt.xlabel("Frequency Eigenmode Configuration")
        plt.ylabel("Spectral Transform Magnitude")
        plt.title("Operator Spectral Field Preservation Analysis", fontsize=10, pad=12)
        plt.legend(frameon=False)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()

    def mode_decay_analysis(self, rollout_states: torch.Tensor) -> np.ndarray:
        decay = []
        for state in rollout_states:
            spectrum = self.fft_spectrum(state)
            high_freq = torch.mean(spectrum[..., -10:])
            decay.append(high_freq.item())
        return np.array(decay)
