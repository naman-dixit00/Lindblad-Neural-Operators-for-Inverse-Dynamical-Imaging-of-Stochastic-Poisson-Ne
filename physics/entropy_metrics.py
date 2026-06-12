"""
LNO-IonTransport Production Pipeline: Thermodynamic Consistency Analyzer

Tracks structural, spectral, and non-equilibrium free energy tracking systems.
"""

import numpy as np
from typing import Dict, Any

class ThermodynamicEntropyAnalyzer:
    def __init__(self, dx: float = 0.01):
        self.dx = dx

    def shannon_entropy(self, concentration: np.ndarray) -> float:
        eps = 1e-12
        prob = concentration / (np.sum(concentration) + eps)
        entropy = -np.sum(prob * np.log(prob + eps))
        return float(entropy)

    def spectral_entropy(self, signal: np.ndarray) -> float:
        eps = 1e-12
        fft = np.fft.rfft(signal)
        power = np.abs(fft)**2
        power = power / (np.sum(power) + eps)
        entropy = -np.sum(power * np.log(power + eps))
        return float(entropy)

    def entropy_rate(self, entropy_history: list, dt: float) -> np.ndarray:
        entropy_array = np.array(entropy_history)
        rate = np.gradient(entropy_array, dt)
        return rate

    def irreversible_score(self, entropy_history: list) -> float:
        rate = np.diff(entropy_history)
        positive_growth = np.sum(rate > 0)
        return float(positive_growth / len(rate))

    def thermodynamic_consistency(self, entropy_history: list) -> Dict[str, Any]:
        rate = np.diff(entropy_history)
        violations = np.sum(rate < -1e-5)
        return {
            "violations": int(violations),
            "consistency_ratio": float(1.0 - violations / len(rate))
        }

    def transport_free_energy(self, concentration: np.ndarray, potential: np.ndarray) -> float:
        eps = 1e-12
        entropy_term = np.sum(concentration * np.log(concentration + eps))
        electrostatic_term = np.sum(concentration * potential)
        free_energy = electrostatic_term - entropy_term
        return float(free_energy)

    def metastability_index(self, signal: np.ndarray) -> float:
        gradient = np.gradient(signal)
        variance = np.var(gradient)
        return float(1.0 / (variance + 1e-8))

    def transport_disorder_index(self, concentration: np.ndarray) -> float:
        entropy = self.shannon_entropy(concentration)
        spectral = self.spectral_entropy(concentration)
        return float((entropy + spectral) / 2.0)
