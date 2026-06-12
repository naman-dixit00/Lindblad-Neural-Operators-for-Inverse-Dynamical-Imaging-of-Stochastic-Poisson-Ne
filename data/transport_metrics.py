"""
Physics Analysis Module: Computes high-fidelity statistical signatures, 
entropy tracking, and local instability spatial breakdowns.
"""

import numpy as np

def compute_entropy(c: np.ndarray, dx: float) -> float:
    """
    Computes global Shannon Transport Entropy: S(t) = - ∫ c * ln(c) dx
    Provides continuous non-equilibrium tracking profiles.
    """
    c_safe = np.clip(c, 1e-8, None)
    entropy_value = -np.sum(c_safe * np.log(c_safe)) * dx
    return float(entropy_value)

def compute_flux_energy(flux: np.ndarray) -> float:
    """Computes integrated transport energy variance."""
    return float(np.mean(flux ** 2))

def compute_dissipation_rate(D: float, grad_c: np.ndarray) -> float:
    """Computes spatial thermodynamic energy loss profile means."""
    return float(D * np.mean(grad_c ** 2))

def compute_instability(flux: np.ndarray) -> np.ndarray:
    """
    Generates sharp localized gradients monitoring structural transport breakdown:
    instability = |grad(Flux)|
    """
    return np.abs(np.gradient(flux))

def compute_mass_error(c0: np.ndarray, c1: np.ndarray) -> float:
    """Evaluates total mass retention discrepancy over a system step."""
    return float(np.abs(np.sum(c0) - np.sum(c1)))
