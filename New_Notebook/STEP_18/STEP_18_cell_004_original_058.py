# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 58
# Step            : STEP_18
# Step Heading    : # STEP 18 — FINAL REPAIRED LNO TRAINING
# Step Cell No.   : 4
# ============================================================

# ============================================================
# 13. SPECTRAL CONVOLUTION
# ============================================================

class SpectralConv1D(
    nn.Module
):

    def __init__(
        self,
        in_channels,
        out_channels,
        modes
    ):

        super().__init__()

        self.in_channels = (
            in_channels
        )

        self.out_channels = (
            out_channels
        )

        self.modes = modes

        scale = (
            1.0
            /
            math.sqrt(
                in_channels
                *
                out_channels
            )
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

    def forward(
        self,
        x
    ):

        B, C, NX_local = x.shape

        x_ft = torch.fft.rfft(
            x,
            dim=-1
        )

        n_modes = min(
            self.modes,
            x_ft.shape[-1]
        )

        weight = torch.complex(
            self.weight_real[
                ...,
                :n_modes
            ],
            self.weight_imag[
                ...,
                :n_modes
            ]
        )

        out_ft = torch.zeros(
            B,
            self.out_channels,
            x_ft.shape[-1],
            dtype=torch.complex64,
            device=x.device
        )

        out_ft[
            ...,
            :n_modes
        ] = torch.einsum(
            "bim,iom->bom",
            x_ft[
                ...,
                :n_modes
            ],
            weight
        )

        return torch.fft.irfft(
            out_ft,
            n=NX_local,
            dim=-1
        )


# ============================================================
# 14. LINDBLAD KERNEL
# ============================================================

class LindbladKernel(
    nn.Module
):

    def __init__(
        self,
        dim=6,
        channels=4
    ):

        super().__init__()

        self.dim = dim
        self.channels = channels

        self.jump_operators = nn.Parameter(
            0.02
            *
            torch.randn(
                channels,
                dim,
                dim
            )
        )

    def forward(
        self,
        R
    ):

        total = torch.zeros_like(
            R
        )

        for k in range(
            self.channels
        ):

            L = self.jump_operators[
                k
            ]

            LT = L.transpose(
                -1,
                -2
            )

            A = (
                LT
                @
                L
            )

            jump_term = (
                L
                @
                R
                @
                LT
            )

            anti_term = (
                A
                @
                R
                +
                R
                @
                A
            )

            total = (
                total
                +
                jump_term
                -
                0.5
                *
                anti_term
            )

        return total


# ============================================================
# 15. FINAL REPAIRED LNO
# ============================================================

class FinalRepairedLNO(
    nn.Module
):

    def __init__(
        self,
        dim=6,
        width=64,
        modes=16,
        depth=4,
        lindblad_channels=4
    ):

        super().__init__()

        self.dim = dim
        self.width = width
        self.depth = depth

        # ----------------------------------------------------
        # R = 36 channels
        # amplitude = 1
        # gamma = 1
        # sigma = 1
        #
        # total = 39
        # ----------------------------------------------------

        input_channels = (
            dim * dim
            +
            3
        )

        self.input_projection = nn.Conv1d(
            input_channels,
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
                for _ in range(
                    depth
                )
            ]
        )

        self.pointwise_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    width,
                    width,
                    1
                )
                for _ in range(
                    depth
                )
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

        # ----------------------------------------------------
        # Lindblad forward-path generator
        # ----------------------------------------------------

        self.lindblad = LindbladKernel(
            dim=dim,
            channels=lindblad_channels
        )

        # Positive trainable dynamical couplings
        self.raw_kappa_lindblad = nn.Parameter(
            torch.tensor(
                0.0
            )
        )

        self.raw_kappa_neural = nn.Parameter(
            torch.tensor(
                0.0
            )
        )


    # --------------------------------------------------------
    # Positive couplings
    # --------------------------------------------------------

    def kappa_lindblad(
        self
    ):

        return F.softplus(
            self.raw_kappa_lindblad
        )


    def kappa_neural(
        self
    ):

        return F.softplus(
            self.raw_kappa_neural
        )


    # --------------------------------------------------------
    # PSD + trace-one projection
    # --------------------------------------------------------

    def project_R(
        self,
        M
    ):

        M = (
            0.5
            *
            (
                M
                +
                M.transpose(
                    -1,
                    -2
                )
            )
        )

        eigvals, eigvecs = torch.linalg.eigh(
            M
        )

        eigvals = torch.clamp(
            eigvals,
            min=1e-8
        )

        M_psd = (
            eigvecs
            *
            eigvals.unsqueeze(
                -2
            )
        ) @ eigvecs.transpose(
            -1,
            -2
        )

        trace = torch.diagonal(
            M_psd,
            dim1=-2,
            dim2=-1
        ).sum(
            dim=-1,
            keepdim=True
        )

        M_psd = (
            M_psd
            /
            (
                trace.unsqueeze(
                    -1
                )
                +
                1e-12
            )
        )

        return M_psd


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        R,
        amplitude,
        gamma,
        sigma
    ):

        B, NX_local, D, D2 = R.shape

        assert D == self.dim
        assert D2 == self.dim

        # ----------------------------------------------------
        # Flatten information matrix
        # ----------------------------------------------------

        R_flat = R.reshape(
            B,
            NX_local,
            D * D
        )

        # ----------------------------------------------------
        # Environment conditioning
        # ----------------------------------------------------

        env = torch.stack(
            [
                amplitude,
                gamma,
                sigma
            ],
            dim=-1
        )

        inp = torch.cat(
            [
                R_flat,
                env
            ],
            dim=-1
        )

        x = inp.permute(
            0,
            2,
            1
        )

        # ----------------------------------------------------
        # Input projection
        # ----------------------------------------------------

        x = self.input_projection(
            x
        )

        # ----------------------------------------------------
        # Spatial Fourier operator
        # ----------------------------------------------------

        for k in range(
            self.depth
        ):

            spectral = (
                self.spectral_layers[k](
                    x
                )
            )

            pointwise = (
                self.pointwise_layers[k](
                    x
                )
            )

            x = (
                spectral
                +
                pointwise
            )

            if k < (
                self.depth - 1
            ):

                x = F.gelu(
                    x
                )

        # ----------------------------------------------------
        # Neural dynamical generator
        # ----------------------------------------------------

        G_N = self.R_head(
            x
        )

        G_N = (
            G_N
            .permute(
                0,
                2,
                1
            )
            .reshape(
                B,
                NX_local,
                D,
                D
            )
        )

        G_N = (
            0.5
            *
            (
                G_N
                +
                G_N.transpose(
                    -1,
                    -2
                )
            )
        )

        G_N = torch.tanh(
            G_N
        )

        # ----------------------------------------------------
        # Lindblad dynamical generator
        # ----------------------------------------------------

        G_L = self.lindblad(
            R
        )

        kL = self.kappa_lindblad()
        kN = self.kappa_neural()

        gamma_field = gamma[
            ...,
            None,
            None
        ]

        delta_R_lindblad = (
            kL
            *
            gamma_field
            *
            G_L
        )

        delta_R_neural = (
            kN
            *
            G_N
        )

        delta_R_total = (
            delta_R_lindblad
            +
            delta_R_neural
        )

        # ----------------------------------------------------
        # Candidate next information state
        # ----------------------------------------------------

        R_candidate = (
            R
            +
            delta_R_total
        )

        # ----------------------------------------------------
        # Structural information-state enforcement
        # ----------------------------------------------------

        R_next = self.project_R(
            R_candidate
        )

        # ----------------------------------------------------
        # Amplitude output
        # ----------------------------------------------------

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

            "G_lindblad":
                G_L,

            "G_neural":
                G_N,

            "delta_R_lindblad":
                delta_R_lindblad,

            "delta_R_neural":
                delta_R_neural,

            "delta_R_total":
                delta_R_total,

            "kappa_lindblad":
                kL,

            "kappa_neural":
                kN,
        }
