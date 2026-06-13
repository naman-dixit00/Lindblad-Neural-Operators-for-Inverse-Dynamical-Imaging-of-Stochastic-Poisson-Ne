"""
LNO-IonTransport Production Pipeline: Stochastic Poisson–Nernst–Planck System

Governs the non-equilibrium electrodiffusive transport of ions, ionic transport,
thermal coupling, stochastic fluctuations, and dissipative transport evolution.
"""

import numpy as np
from typing import Dict, Any

class StochasticPNPSystem:
    """
    Research-Grade Stochastic Poisson–Nernst–Planck System
    
    PDE System:
        \partial c / \partial t = -\nabla \cdot J + \eta(x,t)
        J = -D(\nabla c + \beta \cdot c \cdot \nabla \phi)
        -\epsilon \nabla^2 \phi = \rho
    """

    def __init__(
        self,
        nx: int = 128,
        dx: float = 0.01,
        dt: float = 0.001,
        D: float = 0.1,
        epsilon: float = 0.05,
        beta: float = 1.0
    ):
        self.nx = nx
        self.dx = dx
        self.dt = dt
        self.D = D
        self.epsilon = epsilon
        self.beta = beta

    def solve_poisson(
        self,
        charge_density: np.ndarray,
        left_bc: float = 0.5,
        right_bc: float = 0.0,
        iterations: int = 120
    ) -> np.ndarray:
        """
        Solves the electrostatic potential using Successive Over-Relaxation (SOR).
        """
        phi = np.zeros(self.nx)
        phi[0] = left_bc
        phi[-1] = right_bc

        omega = 1.6

        for _ in range(iterations):
            for i in range(1, self.nx - 1):
                laplace_update = (
                    phi[i + 1]
                    + phi[i - 1]
                    + self.dx**2
                    * charge_density[i]
                    / self.epsilon
                ) / 2.0

                phi[i] = (
                    (1 - omega) * phi[i]
                    + omega * laplace_update
                )

        return phi

    def compute_flux(
        self,
        concentration: np.ndarray,
        potential: np.ndarray
    ) -> np.ndarray:
        """
        Computes the ionic flux combining Fickian diffusion and electrostatic drift.
        """
        grad_c = np.gradient(concentration, self.dx)
        grad_phi = np.gradient(potential, self.dx)

        flux = -self.D * (
            grad_c
            + self.beta
            * concentration
            * grad_phi
        )

        return flux

    def evolve(
        self,
        concentration: np.ndarray,
        noise: np.ndarray,
        gamma: float = 0.05
    ) -> Dict[str, np.ndarray]:
        """
        Advances the stochastic electrodiffusive field by a single temporal integration step.
        """
        phi = self.solve_poisson(concentration)

        flux = self.compute_flux(concentration, phi)

        div_flux = np.gradient(flux, self.dx)

        dissipation = gamma * concentration

        next_state = (
            concentration
            - self.dt * div_flux
            - self.dt * dissipation
            + np.sqrt(self.dt) * noise
        )

        next_state = np.clip(next_state, 1e-8, None)

        return {
            "state_next": next_state,
            "potential": phi,
            "flux": flux,
            "dissipation": dissipation
        }
