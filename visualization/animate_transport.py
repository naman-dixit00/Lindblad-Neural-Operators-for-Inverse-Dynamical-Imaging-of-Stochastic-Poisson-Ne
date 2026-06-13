import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class TransportAnimator:
    """
    High-fidelity dynamical transport animation engine formatted for clean frame exports.
    """
    def __init__(self, save_dir="results/animations"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def animate_trajectory(self, states, filename="transport_evolution.gif"):
        fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=200)
        x = np.arange(states.shape[1])

        line, = ax.plot([], [], color='#1A1A1A', linewidth=2.0)

        ax.set_xlim(0, states.shape[1] - 1)
        ax.set_ylim(states.min() - 0.05 * np.abs(states.min()), states.max() + 0.05 * np.abs(states.max()))
        ax.set_xlabel("Spatial Coordinate (x)", fontsize=9)
        ax.set_ylabel("Ion Density (ρ)", fontsize=9)
        ax.grid(True, linestyle='--', linewidth=0.5, color='#EAEAEA')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        title_text = ax.text(0.02, 0.92, "", transform=ax.transAxes, fontsize=10, fontweight='semibold')

        def init():
            line.set_data([], [])
            title_text.set_text("")
            return line, title_text

        def update(frame):
            y = states[frame]
            line.set_data(x, y)
            title_text.set_text(f"Transport State Evolution | Epoch t = {frame}")
            return line, title_text

        anim = FuncAnimation(fig, update, frames=len(states), init_func=init, blit=True)
        save_path = os.path.join(self.save_dir, filename)
        anim.save(save_path, writer="pillow", fps=12)
        plt.close()
