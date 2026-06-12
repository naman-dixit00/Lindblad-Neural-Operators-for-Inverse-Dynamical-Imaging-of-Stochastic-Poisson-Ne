"""
Data Normalization & Engineering Preprocessing Pipeline components.
"""

import numpy as np

def normalize(x: np.ndarray) -> np.ndarray:
    """Standardizes input fields by tracking mean and variance profiles."""
    mean = np.mean(x)
    std = np.std(x) + 1e-8
    return (x - mean) / std

def clip_outliers(x: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """Caps extreme signal bursts preventing mathematical training overflow spikes."""
    return np.clip(x, -threshold, threshold)

def spectral_normalize(x: np.ndarray) -> np.ndarray:
    """Scales representation coefficients across frequency axes uniformly."""
    fft_vals = np.fft.fft(x)
    fft_vals /= (np.max(np.abs(fft_vals)) + 1e-8)
    return np.real(np.fft.ifft(fft_vals))
