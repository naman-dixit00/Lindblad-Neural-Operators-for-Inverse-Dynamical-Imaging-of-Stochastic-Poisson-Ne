"""
LNO-IonTransport Production Pipeline: Physics-Informed & Dissipative Loss Functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsInformedResidualLoss(nn.Module):
    def __init__(self, dx: float = 1.0, dt: float = 1.0):
        super(PhysicsInformedResidualLoss, self).__init__()
        self.dx = dx
        self.dt = dt

    def forward(self, inputs, targets_pred, **kwargs):
        r"""
        Evaluates conformity to: \partial_t c + \nabla \cdot J = \eta - \gamma c
        Supports both packed multi-channel tensors and individual keyword fields.
        """
        # --- Multi-Format Field Resolver ---
        if inputs is not None:
            # Slicing channels matching the packaged dataset layout
            c_t = inputs[:, 0, :]
            J = inputs[:, 2, :]
            noise = inputs[:, 3, :]
            gamma = inputs[:, 5, :]
        else:
            # Extract explicit fields sent individually by the trainer loop
            c_t = kwargs.get('state_t')
            J = kwargs.get('flux')
            noise = kwargs.get('noise')
            gamma = kwargs.get('gamma')

        # --- Dynamic Shape Alignment Layer ---
        if noise.ndim == 1: noise = noise.unsqueeze(-1)
        if gamma.ndim == 1: gamma = gamma.unsqueeze(-1)

        # Finite difference approximation for spatial gradient of flux (\nabla \cdot J)
        dJ_dx = torch.zeros_like(J)
        if J.shape == targets_pred.shape:
            dJ_dx[:, 1:-1] = (J[:, 2:] - J[:, :-2]) / (2.0 * self.dx)

        # \partial_t c predicted by the model operator scaled by real dt (\Delta t)
        dc_dt_pred = (targets_pred - c_t) / self.dt

        # Continuous Continuity Equation Residual: PDE = \partial_t c + \nabla \cdot J - \eta + \gamma * c
        pde_residual = dc_dt_pred + dJ_dx - noise + (gamma * c_t)

        return F.mse_loss(pde_residual, torch.zeros_like(pde_residual))

class TotalLNOLoss(nn.Module):
    def __init__(self, lambda_recon: float = 1.0, lambda_diss: float = 0.2, lambda_entropy: float = 0.05, dx: float = 1.0, dt: float = 1.0, **kwargs):
        """
        Combines structural reconstruction data loss with non-equilibrium thermodynamic constraints.
        """
        super(TotalLNOLoss, self).__init__()
        self.lambda_recon = lambda_recon
        self.lambda_diss = lambda_diss
        self.lambda_entropy = lambda_entropy

        self.mse_loss = nn.MSELoss()
        self.pde_loss = PhysicsInformedResidualLoss(dx=dx, dt=dt)

    def forward(self, *args, **kwargs):
        """
        Universal Keyword & Positional Argument Resolver for seamless trainer integration.
        Maps names dynamically to prevent pipeline interface breakage.
        """
        # 1. Resolve Multi-Channel Input Tensor if available
        inputs = kwargs.get('inputs', kwargs.get('x', kwargs.get('batch_x', kwargs.get('input', None))))

        # 2. Resolve Model Predictions
        targets_pred = kwargs.get('pred', kwargs.get('targets_pred', kwargs.get('outputs', kwargs.get('out', None))))

        # 3. Resolve Ground Truth Targets
        targets_true = kwargs.get('target', kwargs.get('targets_true', kwargs.get('true', kwargs.get('y', None))))

        # 4. Fallback for Positional Arguments
        args_list = list(args)
        if inputs is None and len(args_list) > 0:
            inputs = args_list.pop(0)
        if targets_pred is None and len(args_list) > 0:
            targets_pred = args_list.pop(0)
        if targets_true is None and len(args_list) > 0:
            targets_true = args_list.pop(0)

        # 5. Check if individual explicit fields are passed instead of a packaged input tensor
        has_explicit_fields = all(k in kwargs for k in ['state_t', 'flux', 'noise', 'gamma'])

        # Check alignment integrity
        if (inputs is None and not has_explicit_fields) or targets_pred is None or targets_true is None:
            raise ValueError(
                f"[-] TotalLNOLoss argument parsing failed.\n"
                f"Resolved fields -> inputs: {type(inputs)}, pred: {type(targets_pred)}, target: {type(targets_true)}\n"
                f"Available kwargs: {list(kwargs.keys())} | Positionals: {len(args)}"
            )

        # 6. Ground Truth State Reconstruction Loss (L2 norm)
        recon_loss = self.mse_loss(targets_pred, targets_true)

        # 7. Physics-Informed Conservation & Open-System Dissipation Residual
        physics_loss = self.pde_loss(inputs, targets_pred, **kwargs)

        # 8. Total Multi-Objective Formulation
        total_loss = (self.lambda_recon * recon_loss) + (self.lambda_diss * physics_loss)

        loss_metrics = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "physics_loss": physics_loss.item()
        }

        return total_loss, loss_metrics
