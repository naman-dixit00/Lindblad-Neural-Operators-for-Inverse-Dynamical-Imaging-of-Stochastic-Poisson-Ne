import os
import numpy as np
import matplotlib.pyplot as plt

class HeatmapVisualizer:
    """
    Heatmap rendering engine optimizing continuous structural profiles via advanced perceptually uniform colormaps.
    """
    def __init__(self, save_dir="results/heatmaps"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _render_heatmap(self, matrix, cmap, title, label, filename):
        fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300)

        cax = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, interpolation='nearest')
        cb = fig.colorbar(cax, ax=ax, pad=0.02, shrink=0.85)
        cb.set_label(label, fontsize=9, labelpad=8)
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_visible(False)

        ax.set_xlabel("Spatial Coordinate (x)", fontsize=9, labelpad=4)
        ax.set_ylabel("Time Step (t)", fontsize=9, labelpad=4)
        ax.set_title(title, fontsize=10, fontweight='semibold', pad=10, loc='left')
        ax.tick_params(labelsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), bbox_inches="tight")
        plt.close()

    def plot_dissipation_heatmap(self, dissipation_matrix, filename="dissipation_heatmap.png"):
        self._render_heatmap(dissipation_matrix, "magma", "Dissipation Hotspot Spatiotemporal Imaging", "Dissipation Magnitude Flux", filename)

    def plot_instability_heatmap(self, instability_matrix, filename="instability_heatmap.png"):
        self._render_heatmap(instability_matrix, "inferno", "Metastable Transport Instability Coordinates", "Instability Index Bounds", filename)

    def plot_entropy_heatmap(self, entropy_matrix, filename="entropy_heatmap.png"):
        self._render_heatmap(entropy_matrix, "viridis", "Thermodynamic Local Transport Entropy Map", "Entropy Levels (S)", filename)
