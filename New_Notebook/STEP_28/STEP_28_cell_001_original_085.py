# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 85
# Step            : STEP_28
# Step Heading    : # STEP 28 — C-LNO LONG-HORIZON + REGIME STABILITY VALIDATION
# Step Cell No.   : 1
# ============================================================

# ======================================================================================
# STEP 28 — C-LNO LONG-HORIZON + REGIME STABILITY VALIDATION
# ======================================================================================
#
# Scientific goal:
#   Verify that the Step-26 C-LNO accuracy improvement is retained during
#   long-horizon rollout and across all five environment regimes.
#
# Models:
#   1. FNO               -> existing Step-22/23 stability evidence
#   2. Original Final LNO -> existing Step-22/23 stability evidence
#   3. Step-26 C-LNO     -> fresh 100-step rollout from saved checkpoint
#
# IMPORTANT:
#   - NO TRAINING
#   - NO CHECKPOINT MODIFICATION
#   - NO NEW GROUND-TRUTH ACCURACY CLAIM
#   - C-LNO is only evaluated using its saved Step-26 checkpoint
# ======================================================================================

import os
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path


# ======================================================================================
# 1. ROOT
# ======================================================================================

CONTENT_ROOT = Path("/content")

repo_candidates = []

for p in CONTENT_ROOT.iterdir():

    if (
        p.is_dir()
        and (p / "results").is_dir()
        and (p / "physics").is_dir()
        and (p / "training").is_dir()
    ):
        repo_candidates.append(p)

assert repo_candidates, (
    "Repository root not found."
)

ROOT = repo_candidates[0]

os.chdir(ROOT)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 110)
print("STEP 28 — C-LNO LONG-HORIZON + REGIME STABILITY VALIDATION")
print("=" * 110)

print("[+] Root  :", ROOT)
print("[+] Device:", device)


# ======================================================================================
# 2. AUTO-DETECT REQUIRED FILES
# ======================================================================================

def find_file(filename):

    direct = CONTENT_ROOT / filename

    if direct.is_file():
        return direct

    repo_matches = list(
        ROOT.rglob(filename)
    )

    if repo_matches:
        return repo_matches[0]

    content_matches = [
        p for p in CONTENT_ROOT.rglob(filename)
        if p.is_file()
    ]

    if content_matches:
        return content_matches[0]

    return None


DATASET_PATH = find_file(
    "environment_conditioned_lno_dataset_v3_expanded.npz"
)

SPLIT_PATH = find_file(
    "final_trajectory_level_split.npz"
)

FNO_PRED_PATH = find_file(
    "fno_final_test_predictions.npz"
)

LNO_PRED_PATH = find_file(
    "lno_final_test_predictions.npz"
)

C_CHECKPOINT = (
    ROOT
    / "results"
    / "step26_r_accuracy_ablation"
    / "C_R_focused_soft_physics_best.pt"
)

assert DATASET_PATH is not None, (
    "Expanded dataset not found."
)

assert SPLIT_PATH is not None, (
    "Final split not found."
)

assert FNO_PRED_PATH is not None, (
    "FNO prediction file not found."
)

assert LNO_PRED_PATH is not None, (
    "LNO prediction file not found."
)

assert C_CHECKPOINT.is_file(), (
    f"C-LNO checkpoint not found: {C_CHECKPOINT}"
)

print("\nFILE LOCATIONS")
print("-" * 110)

print("Dataset     :", DATASET_PATH)
print("Split       :", SPLIT_PATH)
print("FNO         :", FNO_PRED_PATH)
print("Final LNO   :", LNO_PRED_PATH)
print("C-LNO       :", C_CHECKPOINT)


# ======================================================================================
# 3. LOAD DATA
# ======================================================================================

data = np.load(
    DATASET_PATH,
    allow_pickle=False
)

split = np.load(
    SPLIT_PATH,
    allow_pickle=False
)

fno_pred = np.load(
    FNO_PRED_PATH,
    allow_pickle=False
)

lno_pred = np.load(
    LNO_PRED_PATH,
    allow_pickle=False
)

X_R_all = data["X_R"].astype(
    np.float32
)

X_A_all = data["X_amplitude"].astype(
    np.float32
)

ENV_all = data["environment"].astype(
    np.float32
)

test_idx = split[
    "test_indices"
].astype(
    np.int64
)

assert X_R_all.shape == (
    9900,
    128,
    6,
    6
)

assert X_A_all.shape == (
    9900,
    128
)

assert ENV_all.shape == (
    9900,
    128,
    2
)

assert len(test_idx) == 1485

print("\nDATASET")
print("-" * 110)

print("X_R          :", X_R_all.shape)
print("X_amplitude  :", X_A_all.shape)
print("Environment  :", ENV_all.shape)
print("Test         :", len(test_idx))

print(
    "[PASS] Expanded dataset and final split loaded."
)


# ======================================================================================
# 4. AMPLITUDE NORMALIZATION
# ======================================================================================

checkpoint_meta = torch.load(
    C_CHECKPOINT,
    map_location="cpu"
)

amplitude_mean = float(
    checkpoint_meta.get(
        "amplitude_mean",
        X_A_all[
            split["train_indices"]
        ].mean()
    )
)

amplitude_std = float(
    checkpoint_meta.get(
        "amplitude_std",
        X_A_all[
            split["train_indices"]
        ].std()
    )
)

print("\nAMPLITUDE NORMALIZATION")
print("-" * 110)

print("Mean:", amplitude_mean)
print("Std :", amplitude_std)


X_A_norm_all = (
    X_A_all - amplitude_mean
) / max(
    amplitude_std,
    1e-8
)


# ======================================================================================
# 5. REPRESENTATIVE STATES
# ======================================================================================

regimes = {

    "low_noise":
        (0.01, 0.02),

    "stochastic":
        (0.05, 0.45),

    "heavy_dissipation":
        (0.35, 0.15),

    "collapse":
        (0.50, 0.65),

    "metastable":
        (0.15, 0.30),
}


ENV_test = ENV_all[test_idx]

representative_indices = []

print("\nREPRESENTATIVE STATES")
print("-" * 110)

for regime_name, (
    gamma_value,
    sigma_value
) in regimes.items():

    mask = np.all(
        np.isclose(
            ENV_test,
            np.array(
                [
                    gamma_value,
                    sigma_value
                ],
                dtype=np.float32
            )[None, None, :],
            atol=1e-6
        ),
        axis=(1, 2)
    )

    regime_positions = np.where(
        mask
    )[0]

    assert len(regime_positions) == 297, (
        f"{regime_name}: expected 297 test transitions, "
        f"got {len(regime_positions)}"
    )

    # Exactly 5 representative states per regime.
    chosen = np.linspace(
        0,
        len(regime_positions) - 1,
        5,
        dtype=np.int64
    )

    chosen_positions = regime_positions[
        chosen
    ]

    chosen_global_indices = test_idx[
        chosen_positions
    ]

    representative_indices.extend(
        chosen_global_indices.tolist()
    )

    print(
        f"{regime_name:20s}: "
        f"{len(regime_positions)} transitions -> "
        f"5 representatives"
    )


assert len(
    representative_indices
) == 25

representative_indices = np.array(
    representative_indices,
    dtype=np.int64
)

print(
    "\n[PASS] 25 representative states selected "
    "(5 per regime)."
)


# ======================================================================================
# 6. SPECTRAL CONVOLUTION
# ======================================================================================

class SpectralConv1D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        modes
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = (
            1.0 /
            math.sqrt(
                in_channels *
                out_channels
            )
        )

        self.weight_real = nn.Parameter(
            scale *
            torch.randn(
                in_channels,
                out_channels,
                modes
            )
        )

        self.weight_imag = nn.Parameter(
            scale *
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

        n_modes = min(
            self.modes,
            x_ft.shape[-1]
        )

        out_ft = torch.zeros(
            B,
            self.out_channels,
            x_ft.shape[-1],
            dtype=torch.complex64,
            device=x.device
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
            n=NX,
            dim=-1
        )


# ======================================================================================
# 7. LINDBLAD KERNEL
# ======================================================================================

class LindbladKernel(nn.Module):

    def __init__(
        self,
        dim=6,
        channels=4
    ):

        super().__init__()

        self.jump_operators = nn.Parameter(
            0.02 *
            torch.randn(
                channels,
                dim,
                dim
            )
        )

    def forward(self, R):

        total = torch.zeros_like(
            R
        )

        for k in range(4):

            L = self.jump_operators[k]

            LT = L.transpose(
                -1,
                -2
            )

            A = (
                LT @ L
            )

            jump_term = (
                L @ R @ LT
            )

            anti_term = (
                A @ R +
                R @ A
            )

            total = (
                total
                +
                jump_term
                -
                0.5 * anti_term
            )

        return total


# ======================================================================================
# 8. FINAL REPAIRED LNO
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

        self.dim = dim

        self.input_projection = nn.Conv1d(
            dim * dim + 3,
            width,
            kernel_size=1
        )

        self.spectral_layers = nn.ModuleList([
            SpectralConv1D(
                width,
                width,
                modes
            )
            for _ in range(depth)
        ])

        self.pointwise_layers = nn.ModuleList([
            nn.Conv1d(
                width,
                width,
                kernel_size=1
            )
            for _ in range(depth)
        ])

        self.R_head = nn.Conv1d(
            width,
            dim * dim,
            kernel_size=1
        )

        self.A_head = nn.Conv1d(
            width,
            1,
            kernel_size=1
        )

        self.lindblad = LindbladKernel(
            dim=dim,
            channels=lindblad_channels
        )

        self.raw_kappa_lindblad = nn.Parameter(
            torch.tensor(-0.5)
        )

        self.raw_kappa_neural = nn.Parameter(
            torch.tensor(0.5)
        )


    def project_R(self, R):

        original_dtype = R.dtype

        M = R.double()

        M = 0.5 * (
            M +
            M.transpose(-1, -2)
        )

        eigvals, eigvecs = torch.linalg.eigh(
            M
        )

        eigvals = torch.clamp(
            eigvals,
            min=1e-8
        )

        M_psd = (
            eigvecs *
            eigvals.unsqueeze(-2)
        ) @ eigvecs.transpose(
            -1,
            -2
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
            M_psd /
            trace.unsqueeze(-1).unsqueeze(-1)
        )

        M_psd = 0.5 * (
            M_psd +
            M_psd.transpose(-1, -2)
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
            len(
                self.spectral_layers
            )
        ):

            x = (
                self.spectral_layers[k](x)
                +
                self.pointwise_layers[k](x)
            )

            if k < len(
                self.spectral_layers
            ) - 1:

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
            R + delta_total
        )

        # IMPORTANT:
        # Same direct normalized amplitude head used by Step 26.
        A_next = self.A_head(
            x
        ).squeeze(
            1
        )

        return {
            "R_next":
                R_next,

            "amplitude_next":
                A_next
        }


# ======================================================================================
# 9. LOAD C-LNO
# ======================================================================================

c_model = FinalRepairedLNO(
    dim=6,
    width=64,
    modes=16,
    depth=4,
    lindblad_channels=4
).to(device)

c_model.load_state_dict(
    checkpoint_meta[
        "model_state_dict"
    ]
)

c_model.eval()

print("\n[PASS] Step-26 C-LNO checkpoint loaded.")


# ======================================================================================
# 10. STRUCTURAL METRICS
# ======================================================================================

def structural_metrics(
    R,
    A
):

    trace = torch.diagonal(
        R,
        dim1=-2,
        dim2=-1
    ).sum(
        dim=-1
    )

    trace_error = torch.abs(
        trace - 1.0
    )

    eig = torch.linalg.eigvalsh(
        R
    )

    return {

        "R_norm":
            float(
                torch.linalg
                .vector_norm(R)
                .item()
            ),

        "amplitude_norm":
            float(
                torch.linalg
                .vector_norm(A)
                .item()
            ),

        "mean_trace_error":
            float(
                trace_error.mean()
                .item()
            ),

        "max_trace_error":
            float(
                trace_error.max()
                .item()
            ),

        "minimum_eigenvalue":
            float(
                eig.min()
                .item()
            ),

        "negative_eigenvalue_fraction":
            float(
                (
                    eig < -1e-6
                )
                .float()
                .mean()
                .item()
            ),

        "finite":
            bool(
                torch.isfinite(
                    R
                ).all().item()
                and
                torch.isfinite(
                    A
                ).all().item()
            )
    }


# ======================================================================================
# 11. C-LNO ROLLOUT
# ======================================================================================

ROLLOUT_STEPS = 100

rollout_rows = []

print("\n")
print("=" * 110)
print("C-LNO — 100-STEP REGIME-WISE ROLLOUT")
print("=" * 110)


for regime_name, (
    gamma_value,
    sigma_value
) in regimes.items():

    regime_test_positions = np.where(
        np.all(
            np.isclose(
                ENV_test,
                np.array(
                    [
                        gamma_value,
                        sigma_value
                    ],
                    dtype=np.float32
                )[None, None, :],
                atol=1e-6
            ),
            axis=(1, 2)
        )
    )[0]

    representative_positions = np.linspace(
        0,
        len(regime_test_positions) - 1,
        5,
        dtype=np.int64
    )

    representative_test_positions = (
        regime_test_positions[
            representative_positions
        ]
    )

    for rep_number, pos in enumerate(
        representative_test_positions,
        start=1
    ):

        global_index = int(
            test_idx[pos]
        )

        R0 = torch.from_numpy(
            X_R_all[
                global_index:global_index + 1
            ]
        ).to(
            device
        )

        A0 = torch.from_numpy(
            X_A_norm_all[
                global_index:global_index + 1
            ]
        ).to(
            device
        )

        gamma0 = torch.full(
            (
                1,
                128
            ),
            float(gamma_value),
            dtype=torch.float32,
            device=device
        )

        sigma0 = torch.full(
            (
                1,
                128
            ),
            float(sigma_value),
            dtype=torch.float32,
            device=device
        )

        R_current = R0.clone()
        A_current = A0.clone()

        rollout_success = True

        with torch.no_grad():

            for step in range(
                1,
                ROLLOUT_STEPS + 1
            ):

                if not torch.isfinite(
                    R_current
                ).all():

                    rollout_success = False
                    break

                if not torch.isfinite(
                    A_current
                ).all():

                    rollout_success = False
                    break

                output = c_model(
                    R_current,
                    A_current,
                    gamma0,
                    sigma0
                )

                R_current = output[
                    "R_next"
                ]

                A_current = output[
                    "amplitude_next"
                ]

                if not torch.isfinite(
                    R_current
                ).all():

                    rollout_success = False
                    break

                if not torch.isfinite(
                    A_current
                ).all():

                    rollout_success = False
                    break

                s = structural_metrics(
                    R_current,
                    A_current
                )

                rollout_rows.append({

                    "regime":
                        regime_name,

                    "representative":
                        rep_number,

                    "global_test_index":
                        global_index,

                    "step":
                        step,

                    "R_norm":
                        s["R_norm"],

                    "amplitude_norm":
                        s["amplitude_norm"],

                    "mean_trace_error":
                        s["mean_trace_error"],

                    "max_trace_error":
                        s["max_trace_error"],

                    "minimum_eigenvalue":
                        s["minimum_eigenvalue"],

                    "negative_eigenvalue_fraction":
                        s[
                            "negative_eigenvalue_fraction"
                        ],

                    "finite":
                        s["finite"]
                })


        print(
            f"{regime_name:20s} | "
            f"Representative {rep_number} | "
            f"100 steps | "
            f"finite={rollout_success}"
        )

        assert rollout_success, (
            f"C-LNO failed on "
            f"{regime_name}, representative {rep_number}"
        )


print(
    "\n[PASS] C-LNO completed "
    "25/25 representative rollouts."
)


# ======================================================================================
# 12. DATAFRAME
# ======================================================================================

rollout_df = pd.DataFrame(
    rollout_rows
)

assert len(
    rollout_df
) == 25 * 100

assert rollout_df[
    "finite"
].all()

print(
    "[PASS] Expected 2,500 rollout rows generated."
)


# ======================================================================================
# 13. GLOBAL SUMMARY
# ======================================================================================

c_global = {

    "model":
        "Step26_C_LNO",

    "rollout_steps":
        ROLLOUT_STEPS,

    "completed_states":
        25,

    "worst_minimum_eigenvalue":
        float(
            rollout_df[
                "minimum_eigenvalue"
            ].min()
        ),

    "worst_max_trace_error":
        float(
            rollout_df[
                "max_trace_error"
            ].max()
        ),

    "worst_negative_eigenvalue_fraction":
        float(
            rollout_df[
                "negative_eigenvalue_fraction"
            ].max()
        ),

    "max_R_norm":
        float(
            rollout_df[
                "R_norm"
            ].max()
        ),

    "min_R_norm":
        float(
            rollout_df[
                "R_norm"
            ].min()
        ),

    "max_amplitude_norm":
        float(
            rollout_df[
                "amplitude_norm"
            ].max()
        ),

    "min_amplitude_norm":
        float(
            rollout_df[
                "amplitude_norm"
            ].min()
        ),

    "all_finite":
        bool(
            rollout_df[
                "finite"
            ].all()
        )
}


# ======================================================================================
# 14. REGIME-WISE SUMMARY
# ======================================================================================

regime_summary_rows = []

for regime_name in regimes.keys():

    df = rollout_df[
        rollout_df[
            "regime"
        ] == regime_name
    ]

    regime_summary_rows.append({

        "regime":
            regime_name,

        "model":
            "Step26_C_LNO",

        "completed_steps":
            int(
                df[
                    "step"
                ].nunique()
            ),

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
            )
    })


c_regime_df = pd.DataFrame(
    regime_summary_rows
)


# ======================================================================================
# 15. LOAD PREVIOUS FNO / LNO LONG-HORIZON SUMMARIES
# ======================================================================================

def auto_find_summary(filename):

    direct = CONTENT_ROOT / filename

    if direct.is_file():
        return direct

    matches = list(
        ROOT.rglob(filename)
    )

    if matches:
        return matches[0]

    return None


FNO_LONG_SUMMARY = auto_find_summary(
    "long_horizon_summary.json"
)

FNO_STEP23_REGIME = (
    ROOT
    / "results"
    / "step23_regime_analysis"
    / "regime_rollout_summary.csv"
)

FNO_LONGHORIZON_DIR = (
    ROOT
    / "results"
    / "step22_long_horizon"
)

print("\n")
print("=" * 110)
print("PREVIOUS FNO / FINAL-LNO STABILITY EVIDENCE")
print("=" * 110)

print(
    "[INFO] Previous Step22/23 evidence is used only "
    "for contextual comparison."
)

if FNO_LONG_SUMMARY is not None:

    print(
        "Long-horizon summary found:",
        FNO_LONG_SUMMARY
    )

else:

    print(
        "[WARN] Previous long-horizon summary not found."
    )


# ======================================================================================
# 16. SAVE C-LNO RESULTS
# ======================================================================================

OUT_DIR = (
    ROOT
    / "results"
    / "step28_c_lno_long_horizon"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ROLL_OUT_PATH = (
    OUT_DIR
    / "c_lno_rollout_details.csv"
)

GLOBAL_PATH = (
    OUT_DIR
    / "c_lno_global_summary.csv"
)

REGIME_PATH = (
    OUT_DIR
    / "c_lno_regime_summary.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "step28_summary.json"
)

rollout_df.to_csv(
    ROLL_OUT_PATH,
    index=False
)

pd.DataFrame(
    [c_global]
).to_csv(
    GLOBAL_PATH,
    index=False
)

c_regime_df.to_csv(
    REGIME_PATH,
    index=False
)


# ======================================================================================
# 17. FINAL JSON
# ======================================================================================

summary_json = {

    "step":
        28,

    "title":
        "C-LNO Long-Horizon and Regime Stability Validation",

    "diagnostic_type":
        "100-step structural stability diagnostic using the saved Step-26 C-LNO checkpoint.",

    "ground_truth_accuracy_claim":
        False,

    "training_performed":
        False,

    "checkpoint_modified":
        False,

    "representative_states":
        25,

    "representatives_per_regime":
        5,

    "rollout_steps":
        100,

    "global_summary":
        c_global,

    "regime_summary":
        c_regime_df.to_dict(
            orient="records"
        ),

    "outputs":
        {
            "rollout_details":
                str(ROLL_OUT_PATH),

            "global_summary":
                str(GLOBAL_PATH),

            "regime_summary":
                str(REGIME_PATH),

            "summary_json":
                str(SUMMARY_PATH)
        }
}

with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary_json,
        f,
        indent=2
    )


# ======================================================================================
# 18. FINAL PRINT
# ======================================================================================

print("\n")
print("=" * 110)
print("STEP 28 COMPLETE — C-LNO LONG-HORIZON STABILITY")
print("=" * 110)

print(
    "\nGLOBAL C-LNO"
)

print(
    "100-step completion:",
    c_global["completed_states"],
    "/ 25 representative states"
)

print(
    "Worst minimum eigenvalue:",
    f"{c_global['worst_minimum_eigenvalue']:.8e}"
)

print(
    "Worst max trace error:",
    f"{c_global['worst_max_trace_error']:.8e}"
)

print(
    "Worst PSD-violation fraction:",
    f"{c_global['worst_negative_eigenvalue_fraction']:.8f}"
)

print(
    "Max R norm:",
    f"{c_global['max_R_norm']:.8e}"
)

print(
    "Max amplitude norm:",
    f"{c_global['max_amplitude_norm']:.8e}"
)

print(
    "All finite:",
    c_global["all_finite"]
)


print("\nREGIME-WISE C-LNO")
print("-" * 110)

print(
    c_regime_df[
        [
            "regime",
            "completed_steps",
            "worst_minimum_eigenvalue",
            "worst_max_trace_error",
            "worst_psd_violation_fraction",
            "max_R_norm",
            "max_amplitude_norm",
            "all_finite"
        ]
    ].to_string(
        index=False
    )
)


structural_pass = bool(
    c_global["all_finite"]
    and
    c_global[
        "worst_negative_eigenvalue_fraction"
    ] == 0.0
    and
    c_global[
        "worst_max_trace_error"
    ] < 1e-5
)

print(
    "\nSTRUCTURAL STABILITY PASS:",
    structural_pass
)

print(
    "\nResults:",
    OUT_DIR
)

print(
    "\n[INFO] No model retraining performed."
)

print(
    "[INFO] No checkpoint modified."
)

print(
    "[INFO] No new ground-truth accuracy benchmark."
)

print(
    "[PASS] STEP 28 COMPLETE."
)

print("=" * 110)
