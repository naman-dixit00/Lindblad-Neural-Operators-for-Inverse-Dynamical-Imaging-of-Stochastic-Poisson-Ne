# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 72
# Step            : STEP_23
# Step Heading    : # STEP 23 — REGIME-WISE LONG-HORIZON ANALYSIS
# Step Cell No.   : 1
# ============================================================

# ============================================================
# STEP 23 — REGIME-WISE LONG-HORIZON ANALYSIS
# ============================================================
#
# Purpose:
#   Analyze Step-22 long-horizon rollout stability separately
#   across all five environmental regimes.
#
# Important:
#   - No retraining
#   - No checkpoint modification
#   - No new ground-truth accuracy benchmark
#   - Reuses Step-22 representative states
# ============================================================

import os
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


# ============================================================
# 1. ROOT / DEVICE
# ============================================================

ROOT = Path(
    "/content/"
    "Lindblad-Neural-Operators-for-Inverse-Dynamical-Imaging-of-Stochastic-Poisson-Ne"
)

os.chdir(ROOT)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 110)
print("STEP 23 — REGIME-WISE LONG-HORIZON ANALYSIS")
print("=" * 110)
print("[+] Device:", device)


# ============================================================
# 2. ARTIFACT LOCATOR
# ============================================================

def locate_file(filename, preferred_paths=None):

    preferred_paths = preferred_paths or []

    for p in preferred_paths:
        p = Path(p)

        if p.is_file():
            return p

    repo_matches = list(
        ROOT.rglob(filename)
    )

    if repo_matches:
        return repo_matches[0]

    matches = []

    for p in Path("/content").rglob(filename):

        try:

            if p.is_file():
                matches.append(p)

        except Exception:
            pass

    if matches:

        matches.sort(
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        return matches[0]

    raise FileNotFoundError(
        f"Could not locate required file: {filename}"
    )


# ============================================================
# 3. ARTIFACT PATHS
# ============================================================

DATASET_PATH = locate_file(
    "environment_conditioned_lno_dataset_v3_expanded.npz",
    preferred_paths=[
        ROOT
        / "results"
        / "dataset_v3_expanded"
        / "environment_conditioned_lno_dataset_v3_expanded.npz"
    ]
)

SPLIT_PATH = locate_file(
    "final_trajectory_level_split.npz",
    preferred_paths=[
        ROOT
        / "results"
        / "step16_final_splits"
        / "final_trajectory_level_split.npz"
    ]
)

FNO_CHECKPOINT = locate_file(
    "fno_final_best.pt",
    preferred_paths=[
        ROOT
        / "results"
        / "step17_final_fno"
        / "fno_final_best.pt"
    ]
)

LNO_CHECKPOINT = locate_file(
    "lno_final_best.pt",
    preferred_paths=[
        ROOT
        / "results"
        / "step18_final_lno"
        / "lno_final_best.pt"
    ]
)

FNO_SUMMARY = locate_file(
    "fno_final_summary.json",
    preferred_paths=[
        ROOT
        / "results"
        / "step17_final_fno"
        / "fno_final_summary.json"
    ]
)

OUT_DIR = (
    ROOT
    / "results"
    / "step23_regime_analysis"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. MODEL DEFINITIONS
#    SAME INFERENCE ARCHITECTURES AS STEP 22
# ============================================================

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


class FinalFNO(nn.Module):

    def __init__(
        self,
        dim=6,
        width=64,
        modes=16,
        depth=4
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

        x = self.input_projection(
            x
        )

        for k in range(
            len(self.spectral_layers)
        ):

            spectral = self.spectral_layers[k](
                x
            )

            pointwise = self.pointwise_layers[k](
                x
            )

            x = (
                spectral
                +
                pointwise
            )

            if k < len(self.spectral_layers) - 1:
                x = F.gelu(x)

        R_next = self.R_head(x)

        R_next = (
            R_next
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

        R_next = (
            0.5
            *
            (
                R_next
                +
                R_next.transpose(
                    -1,
                    -2
                )
            )
        )

        A_next = (
            self.A_head(x)
            .squeeze(1)
        )

        return {
            "R_next": R_next,
            "amplitude_next": A_next
        }


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
                (
                    L
                    @
                    R
                    @
                    LT
                )
                -
                0.5
                *
                (
                    A @ R
                    +
                    R @ A
                )
            )

        return total


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

        if not torch.isfinite(M).all():
            raise RuntimeError(
                "project_R received non-finite matrix."
            )

        original_dtype = M.dtype

        M64 = M.to(
            torch.float64
        )

        eye = torch.eye(
            M64.shape[-1],
            device=M64.device,
            dtype=torch.float64
        )

        M64 = (
            M64
            +
            1e-10
            *
            eye
        )

        try:

            eigvals, eigvecs = torch.linalg.eigh(
                M64
            )

        except RuntimeError:

            M64 = (
                M64
                +
                1e-8
                *
                eye
            )

            eigvals, eigvecs = torch.linalg.eigh(
                M64
            )

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

        M_psd = (
            0.5
            *
            (
                M_psd
                +
                M_psd.transpose(
                    -1,
                    -2
                )
            )
        )

        return M_psd.to(
            original_dtype
        )

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

            if k < len(self.spectral_layers) - 1:
                x = F.gelu(x)

        G_N = self.R_head(x)

        G_N = (
            G_N
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

        G_L = self.lindblad(R)

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
            "R_next": R_next,
            "amplitude_next": A_next
        }


print(
    "\n[PASS] Inference architecture definitions loaded."
)


# ============================================================
# 5. LOAD DATASET / SPLIT
# ============================================================

data = np.load(
    DATASET_PATH
)

split = np.load(
    SPLIT_PATH
)

X_R_all = data[
    "X_R"
].astype(
    np.float32
)

X_A_all = data[
    "X_amplitude"
].astype(
    np.float32
)

ENV_all = data[
    "environment"
].astype(
    np.float32
)

test_indices = split[
    "test_indices"
].astype(
    np.int64
)

print("\n" + "=" * 110)
print("A. DATASET")
print("=" * 110)

print(
    "X_R:",
    X_R_all.shape
)

print(
    "X_amplitude:",
    X_A_all.shape
)

print(
    "Environment:",
    ENV_all.shape
)

print(
    "Test transitions:",
    len(test_indices)
)

assert len(test_indices) == 1485


# ============================================================
# 6. LOAD NORMALIZATION
# ============================================================

with open(
    FNO_SUMMARY,
    "r",
    encoding="utf-8"
) as f:

    fno_summary = json.load(f)

amp_mean = float(
    fno_summary[
        "amplitude_normalization"
    ][
        "mean"
    ]
)

amp_std = float(
    fno_summary[
        "amplitude_normalization"
    ][
        "std"
    ]
)

assert amp_std > 0.0

print(
    "\n[+] Amplitude normalization"
)

print(
    "Mean:",
    amp_mean
)

print(
    "Std :",
    amp_std
)


# ============================================================
# 7. REGIME DEFINITIONS
# ============================================================

regimes = {
    "low_noise": (0.01, 0.02),
    "stochastic": (0.05, 0.45),
    "heavy_dissipation": (0.35, 0.15),
    "collapse": (0.50, 0.65),
    "metastable": (0.15, 0.30),
}

test_env = ENV_all[
    test_indices
]

gamma_test = test_env[
    :,
    0,
    0
]

sigma_test = test_env[
    :,
    0,
    1
]


# ============================================================
# 8. SELECT SAME 25 REPRESENTATIVE STATES AS STEP 22
# ============================================================

REP_PER_REGIME = 5

representative_indices = []
representative_labels = []

for regime_name, (
    g_target,
    s_target
) in regimes.items():

    mask = (
        np.isclose(
            gamma_test,
            g_target,
            atol=1e-5
        )
        &
        np.isclose(
            sigma_test,
            s_target,
            atol=1e-5
        )
    )

    local_positions = np.where(
        mask
    )[0]

    if len(local_positions) < REP_PER_REGIME:

        raise RuntimeError(
            f"Not enough test samples for regime {regime_name}"
        )

    chosen = np.linspace(
        0,
        len(local_positions) - 1,
        REP_PER_REGIME,
        dtype=int
    )

    for pos in chosen:

        representative_indices.append(
            test_indices[
                local_positions[pos]
            ]
        )

        representative_labels.append(
            regime_name
        )

representative_indices = np.asarray(
    representative_indices,
    dtype=np.int64
)

representative_labels = np.asarray(
    representative_labels
)

print(
    "\n[+] Representative states:",
    len(representative_indices)
)

assert len(
    representative_indices
) == 25


# ============================================================
# 9. TORCH INPUTS
# ============================================================

X_R = X_R_all[
    representative_indices
]

X_A = (
    X_A_all[
        representative_indices
    ]
    -
    amp_mean
) / amp_std

ENV = ENV_all[
    representative_indices
]

R0 = torch.from_numpy(
    X_R
).to(
    device
)

A0 = torch.from_numpy(
    X_A.astype(
        np.float32
    )
).to(
    device
)

ENV0 = torch.from_numpy(
    ENV
).to(
    device
)

gamma0 = ENV0[
    ...,
    0
]

sigma0 = ENV0[
    ...,
    1
]


# ============================================================
# 10. LOAD FINAL CHECKPOINTS
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "B. LOAD FINAL CHECKPOINTS"
)

print(
    "=" * 110
)

fno_model = FinalFNO(
    dim=6,
    width=64,
    modes=16,
    depth=4
).to(
    device
)

fno_ckpt = torch.load(
    FNO_CHECKPOINT,
    map_location=device
)

fno_model.load_state_dict(
    fno_ckpt[
        "model_state_dict"
    ]
)

fno_model.eval()

print(
    "[PASS] FNO checkpoint loaded."
)


lno_model = FinalRepairedLNO(
    dim=6,
    width=64,
    modes=16,
    depth=4,
    lindblad_channels=4
).to(
    device
)

lno_ckpt = torch.load(
    LNO_CHECKPOINT,
    map_location=device
)

lno_model.load_state_dict(
    lno_ckpt[
        "model_state_dict"
    ]
)

lno_model.eval()

print(
    "[PASS] LNO checkpoint loaded."
)


# ============================================================
# 11. SETTINGS
# ============================================================

ROLLOUT_STEPS = 100

print(
    "\n" + "=" * 110
)

print(
    "C. LONG-HORIZON SETTINGS"
)

print(
    "=" * 110
)

print(
    "Representative states:",
    len(R0)
)

print(
    "Rollout steps:",
    ROLLOUT_STEPS
)

print(
    "Regimes:",
    list(regimes.keys())
)


# ============================================================
# 12. STRUCTURAL METRICS
# ============================================================

def structural_metrics(R):

    finite_R = bool(
        torch.isfinite(R).all().item()
    )

    if not finite_R:

        return {
            "R_norm": np.nan,
            "mean_trace_error": np.nan,
            "max_trace_error": np.nan,
            "minimum_eigenvalue": np.nan,
            "negative_eigenvalue_fraction": np.nan,
            "finite": False,
        }

    trace = torch.diagonal(
        R,
        dim1=-2,
        dim2=-1
    ).sum(
        dim=-1
    )

    trace_error = torch.abs(
        trace
        -
        1.0
    )

    try:

        eig = torch.linalg.eigvalsh(
            R
        )

        min_eig = float(
            eig.min().item()
        )

        neg_fraction = float(
            (
                eig
                <
                -1e-6
            )
            .float()
            .mean()
            .item()
        )

    except RuntimeError:

        min_eig = np.nan
        neg_fraction = np.nan

    return {
        "R_norm": float(
            torch.linalg.vector_norm(
                R
            ).item()
        ),

        "mean_trace_error": float(
            trace_error.mean().item()
        ),

        "max_trace_error": float(
            trace_error.max().item()
        ),

        "minimum_eigenvalue": min_eig,

        "negative_eigenvalue_fraction": neg_fraction,

        "finite": True,
    }


# ============================================================
# 13. RUN REGIME-WISE ROLLOUTS
# ============================================================

detail_rows = []

summary_rows = []

print(
    "\n" + "=" * 110
)

print(
    "D. REGIME-WISE LONG-HORIZON ROLLOUT"
)

print(
    "=" * 110
)


with torch.no_grad():

    for regime_index, regime_name in enumerate(
        regimes.keys()
    ):

        start = (
            regime_index
            *
            REP_PER_REGIME
        )

        end = (
            start
            +
            REP_PER_REGIME
        )

        R_init = R0[
            start:end
        ].clone()

        A_init = A0[
            start:end
        ].clone()

        gamma_regime = gamma0[
            start:end
        ]

        sigma_regime = sigma0[
            start:end
        ]


        print(
            "\n" + "-" * 100
        )

        print(
            "REGIME:",
            regime_name
        )

        print(
            "-" * 100
        )


        # --------------------------------------------------------
        # FNO
        # --------------------------------------------------------

        R_fno = R_init.clone()
        A_fno = A_init.clone()

        fno_failed = False

        for step in range(
            ROLLOUT_STEPS
        ):

            if (
                not torch.isfinite(R_fno).all()
                or
                not torch.isfinite(A_fno).all()
            ):

                fno_failed = True

                break

            result_fno = fno_model(
                R_fno,
                A_fno,
                gamma_regime,
                sigma_regime
            )

            R_next = result_fno[
                "R_next"
            ]

            A_next = result_fno[
                "amplitude_next"
            ]

            if (
                not torch.isfinite(R_next).all()
                or
                not torch.isfinite(A_next).all()
            ):

                fno_failed = True

                break

            R_fno = R_next
            A_fno = A_next

            metrics = structural_metrics(
                R_fno
            )

            metrics[
                "amplitude_norm"
            ] = float(
                torch.linalg.vector_norm(
                    A_fno
                ).item()
            )

            detail_rows.append(
                {
                    "regime": regime_name,
                    "model": "FNO",
                    "step": step + 1,
                    **metrics,
                }
            )


        # --------------------------------------------------------
        # LNO
        # --------------------------------------------------------

        R_lno = R_init.clone()
        A_lno = A_init.clone()

        lno_failed = False

        for step in range(
            ROLLOUT_STEPS
        ):

            if (
                not torch.isfinite(R_lno).all()
                or
                not torch.isfinite(A_lno).all()
            ):

                lno_failed = True

                break

            try:

                result_lno = lno_model(
                    R_lno,
                    A_lno,
                    gamma_regime,
                    sigma_regime
                )

            except RuntimeError:

                lno_failed = True

                break

            R_next = result_lno[
                "R_next"
            ]

            A_next = result_lno[
                "amplitude_next"
            ]

            if (
                not torch.isfinite(R_next).all()
                or
                not torch.isfinite(A_next).all()
            ):

                lno_failed = True

                break

            R_lno = R_next
            A_lno = A_next

            metrics = structural_metrics(
                R_lno
            )

            metrics[
                "amplitude_norm"
            ] = float(
                torch.linalg.vector_norm(
                    A_lno
                ).item()
            )

            detail_rows.append(
                {
                    "regime": regime_name,
                    "model": "LNO",
                    "step": step + 1,
                    **metrics,
                }
            )


        print(
            "FNO:",
            "FAILED" if fno_failed else "100/100"
        )

        print(
            "LNO:",
            "FAILED" if lno_failed else "100/100"
        )


# ============================================================
# 14. DETAIL DATAFRAME
# ============================================================

detail_df = pd.DataFrame(
    detail_rows
)

DETAIL_CSV = (
    OUT_DIR
    /
    "regime_rollout_details.csv"
)

detail_df.to_csv(
    DETAIL_CSV,
    index=False
)


# ============================================================
# 15. GLOBAL REGIME-WISE SUMMARY
# ============================================================

for regime_name in regimes.keys():

    for model_name in [
        "FNO",
        "LNO"
    ]:

        df = detail_df[
            (
                detail_df[
                    "regime"
                ]
                ==
                regime_name
            )
            &
            (
                detail_df[
                    "model"
                ]
                ==
                model_name
            )
        ]

        if len(df) == 0:

            summary_rows.append(
                {
                    "regime": regime_name,
                    "model": model_name,
                    "completed_steps": 0,
                    "worst_minimum_eigenvalue": np.nan,
                    "worst_max_trace_error": np.nan,
                    "worst_psd_violation_fraction": np.nan,
                    "max_R_norm": np.nan,
                    "min_R_norm": np.nan,
                    "max_amplitude_norm": np.nan,
                    "min_amplitude_norm": np.nan,
                    "all_finite": False,
                }
            )

            continue

        completed_steps = int(
            df[
                "step"
            ].max()
        )

        summary_rows.append(
            {
                "regime": regime_name,
                "model": model_name,

                "completed_steps":
                    completed_steps,

                "worst_minimum_eigenvalue":
                    float(
                        df[
                            "minimum_eigenvalue"
                        ].min()
                    ),

                "worst_max_trace_error":
                    float(
                        df[
                            "max_trace_error"
                        ].max()
                    ),

                "worst_psd_violation_fraction":
                    float(
                        df[
                            "negative_eigenvalue_fraction"
                        ].max()
                    ),

                "max_R_norm":
                    float(
                        df[
                            "R_norm"
                        ].max()
                    ),

                "min_R_norm":
                    float(
                        df[
                            "R_norm"
                        ].min()
                    ),

                "max_amplitude_norm":
                    float(
                        df[
                            "amplitude_norm"
                        ].max()
                    ),

                "min_amplitude_norm":
                    float(
                        df[
                            "amplitude_norm"
                        ].min()
                    ),

                "all_finite":
                    bool(
                        df[
                            "finite"
                        ].all()
                    ),
            }
        )


summary_df = pd.DataFrame(
    summary_rows
)

SUMMARY_CSV = (
    OUT_DIR
    /
    "regime_long_horizon_summary.csv"
)

summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)


# ============================================================
# 16. PRINT REGIME SUMMARY
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "E. REGIME-WISE SUMMARY"
)

print(
    "=" * 110
)

display(
    summary_df.round(8)
)


# ============================================================
# 17. STRUCTURAL ADVANTAGE PER REGIME
# ============================================================

advantage_rows = []

for regime_name in regimes.keys():

    fno = summary_df[
        (
            summary_df[
                "regime"
            ]
            ==
            regime_name
        )
        &
        (
            summary_df[
                "model"
            ]
            ==
            "FNO"
        )
    ].iloc[0]

    lno = summary_df[
        (
            summary_df[
                "regime"
            ]
            ==
            regime_name
        )
        &
        (
            summary_df[
                "model"
            ]
            ==
            "LNO"
        )
    ].iloc[0]

    lno_better = bool(
        lno[
            "all_finite"
        ]
        and
        lno[
            "completed_steps"
        ]
        ==
        ROLLOUT_STEPS
        and
        (
            lno[
                "worst_max_trace_error"
            ]
            <
            fno[
                "worst_max_trace_error"
            ]
        )
        and
        (
            lno[
                "worst_minimum_eigenvalue"
            ]
            >
            fno[
                "worst_minimum_eigenvalue"
            ]
        )
        and
        (
            lno[
                "worst_psd_violation_fraction"
            ]
            <=
            fno[
                "worst_psd_violation_fraction"
            ]
        )
    )

    advantage_rows.append(
        {
            "regime": regime_name,
            "LNO_structurally_better": lno_better
        }
    )

advantage_df = pd.DataFrame(
    advantage_rows
)


# ============================================================
# 18. PLOTS
# ============================================================

plot_files = []


# ------------------------------------------------------------
# R-NORM
# ------------------------------------------------------------

for regime_name in regimes.keys():

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    for model_name in [
        "FNO",
        "LNO"
    ]:

        df = detail_df[
            (
                detail_df[
                    "regime"
                ]
                ==
                regime_name
            )
            &
            (
                detail_df[
                    "model"
                ]
                ==
                model_name
            )
        ]

        if len(df) > 0:

            ax.plot(
                df[
                    "step"
                ],
                df[
                    "R_norm"
                ],
                linewidth=2,
                label=model_name
            )

    ax.set_xlabel(
        "Rollout step"
    )

    ax.set_ylabel(
        "R norm"
    )

    ax.set_title(
        f"{regime_name}: R-Norm Stability"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )

    plt.tight_layout()

    path = (
        OUT_DIR
        /
        f"regime_{regime_name}_R_norm.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    plot_files.append(
        path
    )


# ------------------------------------------------------------
# TRACE ERROR
# ------------------------------------------------------------

for regime_name in regimes.keys():

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    for model_name in [
        "FNO",
        "LNO"
    ]:

        df = detail_df[
            (
                detail_df[
                    "regime"
                ]
                ==
                regime_name
            )
            &
            (
                detail_df[
                    "model"
                ]
                ==
                model_name
            )
        ]

        if len(df) > 0:

            ax.plot(
                df[
                    "step"
                ],
                df[
                    "max_trace_error"
                ],
                linewidth=2,
                label=model_name
            )

    ax.set_xlabel(
        "Rollout step"
    )

    ax.set_ylabel(
        "Maximum trace error"
    )

    ax.set_title(
        f"{regime_name}: Trace Preservation"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )

    plt.tight_layout()

    path = (
        OUT_DIR
        /
        f"regime_{regime_name}_trace_error.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    plot_files.append(
        path
    )


# ------------------------------------------------------------
# MINIMUM EIGENVALUE
# ------------------------------------------------------------

for regime_name in regimes.keys():

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    for model_name in [
        "FNO",
        "LNO"
    ]:

        df = detail_df[
            (
                detail_df[
                    "regime"
                ]
                ==
                regime_name
            )
            &
            (
                detail_df[
                    "model"
                ]
                ==
                model_name
            )
        ]

        if len(df) > 0:

            ax.plot(
                df[
                    "step"
                ],
                df[
                    "minimum_eigenvalue"
                ],
                linewidth=2,
                label=model_name
            )

    ax.axhline(
        0.0,
        linewidth=1
    )

    ax.set_xlabel(
        "Rollout step"
    )

    ax.set_ylabel(
        "Minimum eigenvalue"
    )

    ax.set_title(
        f"{regime_name}: PSD Stability"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )

    plt.tight_layout()

    path = (
        OUT_DIR
        /
        f"regime_{regime_name}_minimum_eigenvalue.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    plot_files.append(
        path
    )


# ------------------------------------------------------------
# AMPLITUDE NORM
# ------------------------------------------------------------

for regime_name in regimes.keys():

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    for model_name in [
        "FNO",
        "LNO"
    ]:

        df = detail_df[
            (
                detail_df[
                    "regime"
                ]
                ==
                regime_name
            )
            &
            (
                detail_df[
                    "model"
                ]
                ==
                model_name
            )
        ]

        if len(df) > 0:

            ax.plot(
                df[
                    "step"
                ],
                df[
                    "amplitude_norm"
                ],
                linewidth=2,
                label=model_name
            )

    ax.set_xlabel(
        "Rollout step"
    )

    ax.set_ylabel(
        "Amplitude norm"
    )

    ax.set_title(
        f"{regime_name}: Amplitude Stability"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )

    plt.tight_layout()

    path = (
        OUT_DIR
        /
        f"regime_{regime_name}_amplitude_norm.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    plot_files.append(
        path
    )


# ============================================================
# 19. JSON SUMMARY
# ============================================================

SUMMARY_JSON = (
    OUT_DIR
    /
    "regime_analysis_summary.json"
)

json_summary = {
    "step": 23,
    "title":
        "Regime-Wise Long-Horizon Analysis",

    "diagnostic_type":
        (
            "Regime-wise repeated-forward stability "
            "and structural-preservation diagnostic."
        ),

    "ground_truth_accuracy_claim":
        False,

    "representative_states":
        int(
            len(
                representative_indices
            )
        ),

    "representatives_per_regime":
        REP_PER_REGIME,

    "rollout_steps":
        ROLLOUT_STEPS,

    "regimes":
        list(
            regimes.keys()
        ),

    "regime_advantage":
        advantage_df.to_dict(
            orient="records"
        ),

    "FNO_vs_LNO_summary":
        summary_df.to_dict(
            orient="records"
        ),

    "outputs": {
        "detail_csv":
            str(DETAIL_CSV),

        "summary_csv":
            str(SUMMARY_CSV),

        "advantage_csv":
            str(
                OUT_DIR
                /
                "regime_structural_advantage.csv"
            ),

        "summary_json":
            str(SUMMARY_JSON),

        "plots":
            [
                str(p)
                for p in plot_files
            ],
    },
}

advantage_df.to_csv(
    OUT_DIR
    /
    "regime_structural_advantage.csv",
    index=False
)

with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_summary,
        f,
        indent=2,
        allow_nan=True
    )


# ============================================================
# 20. FINAL STATUS
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "STEP 23 COMPLETE — REGIME-WISE LONG-HORIZON ANALYSIS"
)

print(
    "=" * 110
)

display(
    advantage_df
)

print(
    "\nResults:",
    OUT_DIR
)

print(
    "\n[INFO] No model retraining performed."
)

print(
    "[INFO] No checkpoint modification performed."
)

print(
    "[INFO] This is a regime-wise long-horizon stability diagnostic."
)

print(
    "[INFO] It is not a new ground-truth accuracy benchmark."
)

print(
    "=" * 110
)
