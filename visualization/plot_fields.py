import os
import numpy as np
import matplotlib.pyplot as plt

class TransportFieldPlotter:
    """
    Research-grade field visualization engine configured for
    Nature-style editorial layout and high-fidelity text integration.
    """
    def __init__(self, save_dir="results/figures"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self._apply_journal_style()

    def _apply_journal_style(self):
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
            'axes.edgecolor': '#111111',
            'axes.linewidth': 0.8,
            'grid.color': '#EAEAEA',
            'grid.linestyle': '--',
            'grid.linewidth': 0.5,
            'figure.dpi': 300,
            'savefig.dpi': 400
        })

    def plot_prediction_vs_truth(self, ground_truth, prediction, title="LNO Field Prediction", filename="lno_field_prediction.png"):
        x = np.arange(len(ground_truth))
        fig, ax = plt.subplots(figsize=(6.5, 3.5))

        ax.plot(x, ground_truth, color='#1A1A1A', linewidth=2.0, label="Ground Truth")
        ax.plot(x, prediction, color='#D32F2F', linestyle="--", linewidth=1.5, label="LNO Prediction")

        ax.set_xlabel("Spatial Coordinate (x)", fontsize=10, labelpad=6)
        ax.set_ylabel("Ion Density (ρ)", fontsize=10, labelpad=6)
        ax.set_title(title, fontsize=11, fontweight='semibold', pad=10, loc='left')

        ax.legend(frameon=True, facecolor='#FFFFFF', edgecolor='none', fontsize=9)
        ax.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), bbox_inches="tight")
        plt.close()

    def plot_absolute_error(self, ground_truth, prediction, filename="absolute_error_map.png"):
        error = np.abs(ground_truth - prediction)
        x = np.arange(len(error))
        fig, ax = plt.subplots(figsize=(6.5, 2.8))

        ax.fill_between(x, error, color='#0F4C81', alpha=0.15)
        ax.plot(x, error, color='#0F4C81', linewidth=1.5, label="Absolute Error")

        ax.set_xlabel("Spatial Coordinate (x)", fontsize=10, labelpad=6)
        ax.set_ylabel("Reconstruction Error", fontsize=10, labelpad=6)
        ax.set_title("Microscopic Absolute Error Distribution", fontsize=11, fontweight='semibold', pad=10, loc='left')

        ax.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), bbox_inches="tight")
        plt.close()

    def plot_flux_field(self, flux, filename="flux_field.png"):
        x = np.arange(len(flux))
        fig, ax = plt.subplots(figsize=(6.5, 2.8))

        ax.plot(x, flux, color='#2E7D32', linewidth=1.8)
        ax.set_xlabel("Spatial Coordinate (x)", fontsize=10, labelpad=6)
        ax.set_ylabel("Flux Vector (J)", fontsize=10, labelpad=6)
        ax.set_title("Electrodiffusive Flux Field Field-State", fontsize=11, fontweight='semibold', pad=10, loc='left')

        ax.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), bbox_inches="tight")
        plt.close()

    def plot_dissipation_profile(self, dissipation, filename="dissipation_profile.png"):
        x = np.arange(len(dissipation))
        fig, ax = plt.subplots(figsize=(6.5, 2.8))

        ax.plot(x, dissipation, color='#E65100', linewidth=1.8)
        ax.set_xlabel("Spatial Coordinate (x)", fontsize=10, labelpad=6)
        ax.set_ylabel("Dissipation Metric", fontsize=10, labelpad=6)
        ax.set_title("Local Dissipation Energy Functional", fontsize=11, fontweight='semibold', pad=10, loc='left')

        ax.grid(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), bbox_inches="tight")
        plt.close()
