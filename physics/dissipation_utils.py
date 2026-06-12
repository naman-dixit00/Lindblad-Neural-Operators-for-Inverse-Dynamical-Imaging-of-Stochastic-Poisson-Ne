"""
LNO-IonTransport Production Pipeline: Dissipation & Geometric Utilities

Provides localized energy signatures, entropy production rates, and stability metrics.
"""

import numpy as np

def compute_gradient_dissipation(concentration: np.ndarray, dx: float, diffusion: float) -> np.ndarray:
    """
    Computes Dissipation Energy Density: D(x,t) = D | \nabla c |^2
    """
    grad_c = np.gradient(concentration, dx)
    dissipation = diffusion * (grad_c**2)
    return dissipation

def compute_transport_energy(concentration: np.ndarray, potential: np.ndarray) -> float:
    """
    Computes Electrochemical Transport Energy: E = \int c(x) \phi(x) dx
    """
    energy = np.sum(concentration * potential)
    return float(energy)

def compute_entropy_production(concentration: np.ndarray, flux: np.ndarray, dx: float) -> float:
    """
    Computes Entropy Production Rate: \sigma = \int J \cdot \nabla \log(c) dx
    """
    eps = 1e-8
    grad_log_c = np.gradient(np.log(concentration + eps), dx)
    sigma = np.sum(flux * grad_log_c)
    return float(sigma)

def compute_transport_instability(flux: np.ndarray, dx: float) -> np.ndarray:
    """
    Computes Instability Metric: I(x) = | \nabla J |
    """
    instability = np.abs(np.gradient(flux, dx))
    return instability

def compute_mass(concentration: np.ndarray, dx: float) -> float:
    return float(np.sum(concentration) * dx)

def compute_mass_error(concentration_t: np.ndarray, concentration_t1: np.ndarray, dx: float) -> float:
    mass_t = compute_mass(concentration_t, dx)
    mass_t1 = compute_mass(concentration_t1, dx)
    return float(np.abs(mass_t1 - mass_t))
