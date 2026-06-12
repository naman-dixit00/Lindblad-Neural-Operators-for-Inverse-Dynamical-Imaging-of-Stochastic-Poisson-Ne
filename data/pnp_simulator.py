"""
Stochastic Poisson-Nernst-Planck Coupled PDE Engine.
Solves nonlinear electrodiffusive profiles with custom open system spatial dissipative operators.
"""

import numpy as np
from .stochastic_forcing import ornstein_uhlenbeck_noise
from .boundary_conditions import apply_dirichlet_boundary, apply_reflective_boundary
from .transport_metrics import compute_entropy, compute_instability

class PhysicsOperatorSimulator:
    def __init__(self, config: dict):
        self.nx = config["physics"]["nx"]
        self.nt = config["physics"]["nt"]
        self.dx = config["physics"]["dx"]
        self.dt = config["physics"]["dt"]
        self.D = config["physics"]["D"]
        self.mu = config["physics"]["mu"]
        self.epsilon = config["physics"]["epsilon"]
        
    def _solve_poisson(self, c: np.ndarray) -> np.ndarray:
        """Solves electric potentials via Finite Difference matrix structures (-eps * d2Phi/dx2 = z * e * c)"""
        A = np.zeros((self.nx, self.nx))
        b = np.zeros(self.nx)
        
        # Build tridiagonal finite-difference matrix layouts
        for i in range(1, self.nx - 1):
            A[i, i - 1] = 1.0 / (self.dx ** 2)
            A[i, i] = -2.0 / (self.dx ** 2)
            A[i, i + 1] = 1.0 / (self.dx ** 2)
            b[i] = -c[i] / self.epsilon
            
        # Dirichlet boundary limits tracking potential drops
        A[0, 0] = 1.0; b[0] = 0.5
        A[-1, -1] = 1.0; b[-1] = 0.0
        
        return np.linalg.solve(A, b)

    def compute_trajectory(self, noise_sigma: float, gamma: float) -> tuple:
        """Propagates numerical state transformations recording specialized thermodynamic signals."""
        states_t = []
        states_t1 = []
        phis_t = []
        fluxes_t = []
        noises_t = []
        dissipations_t = []
        gammas_t = []
        entropies_t = []
        instabilities_t = []

        # Initialize smooth concentration profile across spatial grid
        c = 1.0 + 0.5 * np.sin(2 * np.pi * np.linspace(0, 1, self.nx))
        c = apply_dirichlet_boundary(c, left_val=1.2, right_val=0.2)
        
        for t in range(self.nt):
            phi = self._solve_poisson(c)
            grad_c = np.gradient(c, self.dx)
            grad_phi = np.gradient(phi, self.dx)
            
            # Nernst-Planck ionic flux calculation
            flux = -self.D * grad_c - self.mu * c * grad_phi
            div_flux = np.gradient(flux, self.dx)
            
            # High-fidelity Ornstein-Uhlenbeck noise injection 
            eta = ornstein_uhlenbeck_noise(self.nx, theta=0.2, sigma=noise_sigma, dt=self.dt)
            
            # Dissipation mapping: open system relaxation contraction
            local_dissipation = self.D * (grad_c ** 2)
            
            # Compute system updates
            c_next = c + self.dt * (-div_flux - gamma * local_dissipation + eta)
            c_next = np.clip(c_next, 1e-8, None) # Enforce positivity bound
            c_next = apply_dirichlet_boundary(c_next, left_val=1.2, right_val=0.2)
            
            # Log thermodynamics & tracking functions
            ent_val = compute_entropy(c, self.dx)
            inst_map = compute_instability(flux)
            
            # Store configurations
            states_t.append(c.copy())
            states_t1.append(c_next.copy())
            phis_t.append(phi.copy())
            fluxes_t.append(flux.copy())
            noises_t.append(eta.copy())
            dissipations_t.append(local_dissipation.copy())
            gammas_t.append(np.full(self.nx, gamma))
            entropies_t.append(np.array([ent_val]))
            instabilities_t.append(inst_map.copy())
            
            c = c_next.copy()

        return (
            np.array(states_t, dtype=np.float32),
            np.array(states_t1, dtype=np.float32),
            np.array(phis_t, dtype=np.float32),
            np.array(fluxes_t, dtype=np.float32),
            np.array(noises_t, dtype=np.float32),
            np.array(dissipations_t, dtype=np.float32),
            np.array(gammas_t, dtype=np.float32),
            np.array(entropies_t, dtype=np.float32),
            np.array(instabilities_t, dtype=np.float32)
        )
