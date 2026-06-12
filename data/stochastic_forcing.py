"""
Stochastic Forcing Layer: Implements non-trivial microscopic fluctuations
representing realistic nanoscale thermal perturbations.
"""

import numpy as np

def generate_gaussian_noise(nx: int, sigma: float = 0.1) -> np.ndarray:
    """Standard spatial white noise wrapper."""
    return sigma * np.random.randn(nx)

def ornstein_uhlenbeck_noise(nx: int, theta: float = 0.2, sigma: float = 0.1, dt: float = 0.001) -> np.ndarray:
    """
    Simulates a continuous temporal Ornstein-Uhlenbeck stochastic relaxation track:
    deta = -theta * eta * dt + sigma * dW_t
    """
    eta = np.zeros(nx)
    spatial_wiener_increment = np.random.randn(nx)
    
    # Solve via Euler-Maruyama discretization scheme
    for i in range(1, nx):
        eta[i] = eta[i-1] - theta * eta[i-1] * dt + sigma * np.sqrt(dt) * spatial_wiener_increment[i]
        
    return eta

def generate_correlated_noise(nx: int, sigma: float = 0.1, correlation_scale: int = 5) -> np.ndarray:
    """Applies a Gaussian kernel smooth convolution to simulate localized spatial correlations."""
    raw_noise = np.random.randn(nx)
    grid_space = np.linspace(-2, 2, correlation_scale)
    kernel = np.exp(-grid_space ** 2)
    kernel /= kernel.sum()
    
    correlated = np.convolve(raw_noise, kernel, mode="same")
    return sigma * correlated
