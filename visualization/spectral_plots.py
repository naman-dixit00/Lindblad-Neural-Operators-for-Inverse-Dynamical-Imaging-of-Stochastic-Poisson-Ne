import os
import numpy as np
import matplotlib.pyplot as plt

class SpectralVisualizer:
    """
    Advanced spectrum diagnostic engine tracking spectral energy, complex boundaries, and long-horizon stabilization parameters.
    """
    def __init__(self, save_dir="results/spectral"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_spectrum(self, signal, filename="spectral_energy.png"):
        fft_vals = np.fft.fft(signal)
        energy = np.abs(fft_vals)
        freqs = np.fft.fftfreq(len(signal))
        half_len = len(freqs) // 2

        fig, ax = plt.subplots(figsize=(6.5, 2.8), dpi=300)
        ax.plot(freqs[:half_len], energy[:half_len], color='#4A148C', linewidth=1.6)

        ax.set_xlabel("Frequency Mode (k)", fontsize=9)
        ax.set_ylabel("Spectral Power Intensity", fontsize=9)
        ax.set_title("Operator Spectral Energy Distribution", fontsize=10, fontweight='semibold', pad=10, loc='left')
        ax.grid(True, linestyle='--', linewidth=0.5, color='#EAEAEA')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), bbox_inches="tight")
        plt.close()

    def plot_eigenvalue_distribution(self, eigenvalues, filename="eigenvalue_distribution.png"):
        fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=300)

        ax.scatter(np.real(eigenvalues), np.imag(eigenvalues), color='#006064', s=15, alpha=0.75, edgecolors='none', label="Operator Spectrum")

        theta = np.linspace(0, 2*np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), color='#D32F2F', linestyle="--", linewidth=1.0, label="Unit Stability Circle Bound")

        ax.set_xlabel("Real Axis Re(λ)", fontsize=9)
        ax.set_ylabel("Imaginary Axis Im(λ)", fontsize=9)
        ax.set_title("Spectral Stability Complex Plane Mapping", fontsize=10, fontweight='semibold', pad=10)
        ax.grid(True, linestyle='--', linewidth=0.5, color='#EAEAEA')
        ax.legend(frameon=True, fontsize=8, loc='lower left')
        ax.axis("equal")

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), bbox_inches="tight")
        plt.close()

    def plot_long_horizon_energy(self, fno_energy, lno_energy, filename="long_horizon_stability.png"):
        steps = np.arange(len(fno_energy))
        fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=300)

        ax.plot(steps, fno_energy, color='#757575', linestyle="--", linewidth=1.5, label="Fourier Neural Operator (FNO)")
        ax.plot(steps, lno_energy, color='#1A1A1A', linewidth=2.0, label="Lindblad Neural Operator (LNO)")

        ax.set_xlabel("Autoregressive Rollout Horizon Steps", fontsize=9)
        ax.set_ylabel("Total Conserved System Energy Norm", fontsize=9)
        ax.set_title("Long-Horizon Dissipation & Stability Analysis", fontsize=10, fontweight='semibold', pad=10, loc='left')

        ax.grid(True, linestyle='--', linewidth=0.5, color='#EAEAEA')
        ax.legend(frameon=True, facecolor='#FFFFFF', edgecolor='none', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), bbox_inches="tight")
        plt.close()
