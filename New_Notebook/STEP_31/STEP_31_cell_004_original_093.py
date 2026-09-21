# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 93
# Step            : STEP_31
# Step Heading    : # STEP 31 — CORRECTED FINAL PUBLICATION PACKAGE AUDIT
# Step Cell No.   : 4
# ============================================================

# ======================================================================================
# LOAD EXACT VALIDATED C-LNO ARCHITECTURE
# ======================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================================
# 1. SPECTRAL CONV 1D
# ======================================================================================

class SpectralConv1D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        modes
    ):
        super().__init__()

        self.modes = modes

        scale = (
            1.0
            /
            (in_channels * out_channels) ** 0.5
        )

        self.weight_real = nn.Parameter(
            scale
            *
            torch.randn(
                in_channels,
                out_channels,
                modes
            )
        )

        self.weight_imag = nn.Parameter(
            scale
            *
            torch.randn(
                in_channels,
                out_channels,
                modes
            )
        )

    def forward(self, x):

        B, C, NX = x.shape

        x_ft = torch.fft.rfft(
            x,
            dim=-1
        )

        out_ft = torch.zeros(
            B,
            self.weight_real.shape[1],
            x_ft.shape[-1],
            dtype=torch.complex64,
            device=x.device
        )

        n_modes = min(
            self.modes,
            x_ft.shape[-1]
        )

        weight = torch.complex(
            self.weight_real[..., :n_modes],
            self.weight_imag[..., :n_modes]
        )

        out_ft[..., :n_modes] = torch.einsum(
            "bim,iom->bom",
            x_ft[..., :n_modes],
            weight
        )

        return torch.fft.irfft(
            out_ft,
            n=NX,
            dim=-1
        )


# ======================================================================================
# 2. LINDBLAD KERNEL
# ======================================================================================

class LindbladKernel(nn.Module):

    def __init__(
        self,
        dim=6,
        channels=4
    ):
        super().__init__()

        self.jump_operators = nn.Parameter(
            0.02
            *
            torch.randn(
                channels,
                dim,
                dim
            )
        )

    def forward(self, R):

        total = torch.zeros_like(R)

        for k in range(
            self.jump_operators.shape[0]
        ):

            L = self.jump_operators[k]

            LT = L.transpose(
                -1,
                -2
            )

            A = LT @ L

            total = (
                total
                +
                (L @ R @ LT)
                -
                0.5 * (
                    A @ R
                    +
                    R @ A
                )
            )

        return total


# ======================================================================================
# 3. EXACT FINAL REPAIRED LNO
# ======================================================================================

class FinalRepairedLNO(nn.Module):

    def __init__(
        self,
        dim=6,
        width=64,
        modes=16,
        depth=4,
        lindblad_channels=4
    ):
        super().__init__()

        self.input_projection = nn.Conv1d(
            dim * dim + 3,
            width,
            1
        )

        self.spectral_layers = nn.ModuleList(
            [
                SpectralConv1D(
                    width,
                    width,
                    modes
                )
                for _ in range(depth)
            ]
        )

        self.pointwise_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    width,
                    width,
                    1
                )
                for _ in range(depth)
            ]
        )

        self.R_head = nn.Conv1d(
            width,
            dim * dim,
            1
        )

        self.A_head = nn.Conv1d(
            width,
            1,
            1
        )

        self.lindblad = LindbladKernel(
            dim,
            lindblad_channels
        )

        self.raw_kappa_lindblad = nn.Parameter(
            torch.tensor(0.0)
        )

        self.raw_kappa_neural = nn.Parameter(
            torch.tensor(0.0)
        )


    # ------------------------------------------------------------------
    # Physical-state projection
    # ------------------------------------------------------------------

    def project_R(self, M):

        M = 0.5 * (
            M
            +
            M.transpose(-1, -2)
        )

        if not torch.isfinite(M).all():

            raise RuntimeError(
                "project_R received non-finite matrix "
                "before eigendecomposition."
            )

        original_dtype = M.dtype

        M64 = M.to(torch.float64)

        eye = torch.eye(
            M64.shape[-1],
            device=M64.device,
            dtype=torch.float64
        )

        M64 = (
            M64
            +
            1e-10 * eye
        )

        try:

            eigvals, eigvecs = torch.linalg.eigh(
                M64
            )

        except RuntimeError:

            M64 = (
                M64
                +
                1e-8 * eye
            )

            try:

                eigvals, eigvecs = torch.linalg.eigh(
                    M64
                )

            except RuntimeError as e:

                raise RuntimeError(
                    "project_R eigendecomposition failed "
                    "even after numerical jitter."
                ) from e

        eigvals = torch.clamp(
            eigvals,
            min=1e-8
        )

        M_psd = (
            eigvecs
            @
            torch.diag_embed(
                eigvals
            )
            @
            eigvecs.transpose(
                -1,
                -2
            )
        )

        trace = torch.diagonal(
            M_psd,
            dim1=-2,
            dim2=-1
        ).sum(
            dim=-1
        )

        trace = torch.clamp(
            trace,
            min=1e-12
        )

        M_psd = (
            M_psd
            /
            trace.unsqueeze(-1).unsqueeze(-1)
        )

        M_psd = 0.5 * (
            M_psd
            +
            M_psd.transpose(
                -1,
                -2
            )
        )

        return M_psd.to(
            original_dtype
        )


    # ------------------------------------------------------------------
    # Exact validated forward path
    # ------------------------------------------------------------------

    def forward(
        self,
        R,
        amplitude,
        gamma,
        sigma
    ):

        B, NX, D, D2 = R.shape

        assert D == 6
        assert D2 == 6

        R_flat = R.reshape(
            B,
            NX,
            D * D
        )

        env = torch.stack(
            [
                amplitude,
                gamma,
                sigma
            ],
            dim=-1
        )

        x = torch.cat(
            [
                R_flat,
                env
            ],
            dim=-1
        ).permute(
            0,
            2,
            1
        )

        x = self.input_projection(
            x
        )

        for k in range(
            len(self.spectral_layers)
        ):

            x = (
                self.spectral_layers[k](x)
                +
                self.pointwise_layers[k](x)
            )

            if k < (
                len(self.spectral_layers)
                - 1
            ):

                x = F.gelu(
                    x
                )

        G_N = torch.tanh(
            self.R_head(x)
            .permute(
                0,
                2,
                1
            )
            .reshape(
                B,
                NX,
                D,
                D
            )
        )

        G_L = self.lindblad(
            R
        )

        kappa_L = F.softplus(
            self.raw_kappa_lindblad
        )

        kappa_N = F.softplus(
            self.raw_kappa_neural
        )

        delta_R_lindblad = (
            kappa_L
            *
            gamma[..., None, None]
            *
            G_L
        )

        delta_R_neural = (
            kappa_N
            *
            G_N
        )

        delta_total = (
            delta_R_lindblad
            +
            delta_R_neural
        )

        R_next = self.project_R(
            R
            +
            delta_total
        )

        A_next = self.A_head(
            x
        ).squeeze(
            1
        )

        return {
            "R_next":
                R_next,

            "amplitude_next":
                A_next,

            "delta_R_lindblad":
                delta_R_lindblad,

            "delta_R_neural":
                delta_R_neural,

            "delta_R_total":
                delta_total
        }


# ======================================================================================
# 4. ARCHITECTURE SANITY CHECK
# ======================================================================================

model_check = FinalRepairedLNO(
    dim=6,
    width=64,
    modes=16,
    depth=4,
    lindblad_channels=4
)

param_count = sum(
    p.numel()
    for p in model_check.parameters()
)

print("=" * 110)
print("FINAL C-LNO ARCHITECTURE LOADED")
print("=" * 110)

print(
    "[PASS] Class:",
    "FinalRepairedLNO"
)

print(
    "[PASS] Configuration:",
    "dim=6, width=64, modes=16, depth=4, lindblad_channels=4"
)

print(
    "[INFO] Parameter count:",
    f"{param_count:,}"
)

assert param_count == 546039, (
    f"Unexpected parameter count: {param_count}"
)

print(
    "[PASS] Parameter count = 546,039"
)

print("=" * 110)
