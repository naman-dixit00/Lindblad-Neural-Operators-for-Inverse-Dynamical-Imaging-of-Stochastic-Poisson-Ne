"""
Membrane & Domain Physics Boundary Condition Handlers.
Guarantees mathematical boundary validity across transport loops.
"""

import numpy as np

def apply_dirichlet_boundary(u: np.ndarray, left_val: float = 1.0, right_val: float = 0.1) -> np.ndarray:
    """Forces fixed concentrations or potentials at the system edges."""
    u[0] = left_val
    u[-1] = right_val
    return u

def apply_neumann_boundary(u: np.ndarray) -> np.ndarray:
    """Implements zero-flux / zero-gradient insulating physical barriers."""
    u[0] = u[1]
    u[-1] = u[-2]
    return u

def apply_periodic_boundary(u: np.ndarray) -> np.ndarray:
    """Wraps field states into continuous torus layouts."""
    u[0] = u[-2]
    u[-1] = u[1]
    return u

def apply_reflective_boundary(u: np.ndarray) -> np.ndarray:
    """Perfect particle mirroring behavior matching physical system boxes."""
    u[0] = u[1]
    u[-1] = u[-2]
    return u
