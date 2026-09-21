# ============================================================
# AUTO-EXPORTED NOTEBOOK CODE CELL
# ============================================================
# Source Notebook : Project_Notebook.ipynb
# Original Cell   : 70
# Step            : STEP_22
# Step Heading    : # STEP 22 — LONG-HORIZON ROLLOUT STABILITY
# Step Cell No.   : 3
# ============================================================

# ============================================================
# STEP 22 — LONG-HORIZON ROLLOUT STABILITY
# FINAL DIAGNOSTIC VERSION
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
print("STEP 22 — LONG-HORIZON ROLLOUT STABILITY")
print("=" * 110)
print("[+] Device:", device)


# ============================================================
# 2. FLEXIBLE ARTIFACT LOCATOR
# ============================================================

def locate_file(filename, preferred_paths=None):
    preferred_paths = preferred_paths or []

    for p in preferred_paths:
        p = Path(p)
        if p.is_file():
            return p

    repo_matches = list(ROOT.rglob(filename))
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
    / "step22_long_horizon"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("\n" + "=" * 110)
print("A. ARTIFACT LOCATIONS")
print("=" * 110)

print("Dataset:", DATASET_PATH)
print("Split:", SPLIT_PATH)
print("FNO checkpoint:", FNO_CHECKPOINT)
print("LNO checkpoint:", LNO_CHECKPOINT)


# ============================================================
# 4. MODEL DEFINITIONS
#    EXACT INFERENCE ARCHITECTURE USED FOR STEP 21
# ============================================================

class SpectralConv1D(nn.Module):

    def __init__(self,
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

    def __init__(self,
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

        x = inp.permute(0, 2, 1)

        x = self.input_projection(x)

        for k in range(len(self.spectral_layers)):
            spectral = self.spectral_layers[k](x)
            pointwise = self.pointwise_layers[k](x)
            x = spectral + pointwise
            if k < len(self.spectral_layers) - 1:
                x = F.gelu(x)

        R_next = self.R_head(x)
        R_next = (R_next.permute(0, 2, 1).reshape(B, NX, D, D))
        R_next = 0.5 * (R_next + R_next.transpose(-1, -2))
        A_next = self.A_head(x).squeeze(1)

        return {"R_next": R_next, "amplitude_next": A_next}


class LindbladKernel(nn.Module):

    def __init__(self,
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

        for k in range(self.jump_operators.shape[0]):
            L = self.jump_operators[k]
            LT = L.transpose(-1, -2)
            A = LT @ L
            total = (total + (L @ R @ LT) - 0.5 * (A @ R + R @ A))

        return total


class FinalRepairedLNO(nn.Module):

    def __init__(self,
        dim=6,
        width=64,
        modes=16,
        depth=4,
        lindblad_channels=4
    ):
        super().__init__()

        self.input_projection = nn.Conv1d(dim * dim + 3, width, 1)

        self.spectral_layers = nn.ModuleList([
            SpectralConv1D(width, width, modes) for _ in range(depth)
        ])

        self.pointwise_layers = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(depth)
        ])

        self.R_head = nn.Conv1d(width, dim * dim, 1)
        self.A_head = nn.Conv1d(width, 1, 1)

        self.lindblad = LindbladKernel(dim, lindblad_channels)

        self.raw_kappa_lindblad = nn.Parameter(torch.tensor(0.0))
        self.raw_kappa_neural = nn.Parameter(torch.tensor(0.0))

    def project_R(self, M):
        M = 0.5 * (M + M.transpose(-1, -2))

        if not torch.isfinite(M).all():
            raise RuntimeError(
                "project_R received non-finite matrix before eigendecomposition."
            )

        original_dtype = M.dtype
        M64 = M.to(torch.float64)

        eye = torch.eye(
            M64.shape[-1],
            device=M64.device,
            dtype=torch.float64
        )

        M64 = M64 + 1e-10 * eye

        try:
            eigvals, eigvecs = torch.linalg.eigh(M64)
        except RuntimeError:
            M64 = M64 + 1e-8 * eye

            try:
                eigvals, eigvecs = torch.linalg.eigh(M64)
            except RuntimeError as e:
                raise RuntimeError(
                    "project_R eigendecomposition failed even after numerical jitter."
                ) from e

        eigvals = torch.clamp(eigvals, min=1e-8)

        M_psd = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)

        trace = torch.diagonal(M_psd, dim1=-2, dim2=-1).sum(dim=-1)
        trace = torch.clamp(trace, min=1e-12)

        M_psd = M_psd / trace.unsqueeze(-1).unsqueeze(-1)

        M_psd = 0.5 * (M_psd + M_psd.transpose(-1, -2))

        return M_psd.to(original_dtype)

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

        R_flat = R.reshape(B, NX, D * D)

        env = torch.stack([amplitude, gamma, sigma], dim=-1)

        x = torch.cat([R_flat, env], dim=-1).permute(0, 2, 1)

        x = self.input_projection(x)

        for k in range(len(self.spectral_layers)):
            x = (self.spectral_layers[k](x) + self.pointwise_layers[k](x))
            if k < len(self.spectral_layers) - 1:
                x = F.gelu(x)

        G_N = torch.tanh(
            self.R_head(x)
            .permute(0, 2, 1)
            .reshape(B, NX, D, D)
        )

        G_L = self.lindblad(R)

        kappa_L = F.softplus(self.raw_kappa_lindblad)
        kappa_N = F.softplus(self.raw_kappa_neural)

        delta_R_lindblad = (kappa_L * gamma[..., None, None] * G_L)
        delta_R_neural = (kappa_N * G_N)
        delta_total = (delta_R_lindblad + delta_R_neural)

        R_next = self.project_R(R + delta_total)

        # ----------------------------------------------------
        # Amplitude
        #
        # Final Step-18 model outputs the
        # normalized amplitude target directly.
        # ----------------------------------------------------

        A_next = self.A_head(
            x
        ).squeeze(
            1
        )

        return {
            "R_next": R_next,
            "amplitude_next": A_next,
            "delta_R_lindblad": delta_R_lindblad,
            "delta_R_neural": delta_R_neural,
            "delta_R_total": delta_total
        }


print("\n[PASS] Inference architecture definitions loaded.")


# ============================================================
# 5. LOAD DATA
# ============================================================

data = np.load(DATASET_PATH)
split = np.load(SPLIT_PATH)

X_R_all = data["X_R"].astype(np.float32)
X_A_all = data["X_amplitude"].astype(np.float32)
ENV_all = data["environment"].astype(np.float32)

test_indices = split["test_indices"].astype(np.int64)

print("\n" + "=" * 110)
print("B. DATASET")
print("=" * 110)

print("X_R:", X_R_all.shape)
print("X_amplitude:", X_A_all.shape)
print("Environment:", ENV_all.shape)
print("Test transitions:", len(test_indices))

assert len(test_indices) == 1485


# ============================================================
# 6. NORMALIZATION
# ============================================================

with open(FNO_SUMMARY, "r", encoding="utf-8") as f:
    fno_summary = json.load(f)

amp_mean = float(fno_summary["amplitude_normalization"]["mean"])
amp_std = float(fno_summary["amplitude_normalization"]["std"])

print("\n[+] Amplitude normalization")
print("Mean:", amp_mean)
print("Std :", amp_std)

# ============================================================
# 7. IDENTIFY REGIMES
# ============================================================

test_env = ENV_all[test_indices]
gamma_test = test_env[:, 0, 0]
sigma_test = test_env[:, 0, 1]

regimes = {
    "low_noise": (0.01, 0.02),
    "stochastic": (0.05, 0.45),
    "heavy_dissipation": (0.35, 0.15),
    "collapse": (0.50, 0.65),
    "metastable": (0.15, 0.30),
}

print("\n" + "=" * 110)
print("C. REGIME COUNTS")
print("=" * 110)

for regime_name, (g_target, s_target) in regimes.items():
    mask = (np.isclose(gamma_test, g_target, atol=1e-5) & np.isclose(sigma_test, s_target, atol=1e-5))
    print(f"{regime_name:22s}:", int(mask.sum()))


# ============================================================
# 8. SELECT REPRESENTATIVE STATES
# ============================================================

REP_PER_REGIME = 5

representative_indices = []
representative_labels = []

for regime_name, (g_target, s_target) in regimes.items():
    mask = (np.isclose(gamma_test, g_target, atol=1e-5) & np.isclose(sigma_test, s_target, atol=1e-5))
    local_positions = np.where(mask)[0]

    if len(local_positions) < REP_PER_REGIME:
        raise RuntimeError(f"Not enough test samples for regime {regime_name}")

    chosen = np.linspace(0, len(local_positions) - 1, REP_PER_REGIME, dtype=int)

    for pos in chosen:
        representative_indices.append(test_indices[local_positions[pos]])
        representative_labels.append(regime_name)

representative_indices = np.asarray(representative_indices, dtype=np.int64)

print("\n[+] Representative rollout states:", len(representative_indices))


# ============================================================
# 9. TORCH INPUTS
# ============================================================

X_R = X_R_all[representative_indices]
X_A = (X_A_all[representative_indices] - amp_mean) / amp_std
ENV = ENV_all[representative_indices]

R0 = torch.from_numpy(X_R).to(device)
A0 = torch.from_numpy(X_A.astype(np.float32)).to(device)
ENV0 = torch.from_numpy(ENV).to(device)

gamma0 = ENV0[..., 0]
sigma0 = ENV0[..., 1]


# ============================================================
# 10. LOAD CHECKPOINTS
# ============================================================

print("\n" + "=" * 110)
print("D. LOAD FINAL CHECKPOINTS")
print("=" * 110)

fno_model = FinalFNO(dim=6, width=64, modes=16, depth=4).to(device)
fno_ckpt = torch.load(FNO_CHECKPOINT, map_location=device)
fno_model.load_state_dict(fno_ckpt["model_state_dict"])
fno_model.eval()

print("[PASS] FNO checkpoint loaded.")


lno_model = FinalRepairedLNO(dim=6, width=64, modes=16, depth=4, lindblad_channels=4).to(device)
lno_ckpt = torch.load(LNO_CHECKPOINT, map_location=device)
lno_model.load_state_dict(lno_ckpt["model_state_dict"])
lno_model.eval()

print("[PASS] LNO checkpoint loaded.")


# ============================================================
# 11. LONG-HORIZON ROLLOUT SETTINGS
# ============================================================

ROLLOUT_STEPS = 100

print("\n" + "=" * 110)
print("E. LONG-HORIZON SETTINGS")
print("=" * 110)

print("Representative states:", len(R0))

print("Rollout steps:", ROLLOUT_STEPS)
print("Models:", "FNO vs LNO")


# ============================================================
# 12. METRIC FUNCTION
# ============================================================

def structural_metrics(R):
    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1)
    trace_error = torch.abs(trace - 1.0)
    eig = torch.linalg.eigvalsh(R)

    return {
        "R_norm": float(torch.linalg.vector_norm(R).item()),
        "amplitude_norm": None,
        "mean_trace_error": float(trace_error.mean().item()),
        "max_trace_error": float(trace_error.max().item()),
        "minimum_eigenvalue": float(eig.min().item()),
        "negative_eigenvalue_fraction": float((eig < -1e-6).float().mean().item()),
        "finite": bool(torch.isfinite(R).all().item())
    }


# ============================================================
# 13. ROLLOUT LOOP
# ============================================================

rollout_rows = []

print("\n" + "=" * 110)
print("F. RUNNING LONG-HORIZON ROLLOUT")
print("=" * 110)

with torch.no_grad():
    # FNO
    R_fno = R0.clone()
    A_fno = A0.clone()
    for step in range(ROLLOUT_STEPS):
        result_fno = fno_model(R_fno, A_fno, gamma0, sigma0)
        R_fno = result_fno["R_next"]
        A_fno = result_fno["amplitude_next"]
        fno_struct = structural_metrics(R_fno)
        fno_struct["amplitude_norm"] = float(torch.linalg.vector_norm(A_fno).item())
        fno_struct["step"] = step + 1
        fno_struct["model"] = "FNO"
        rollout_rows.append(fno_struct)
    print("[PASS] FNO rollout complete.")
    # LNO
    R_lno = R0.clone()
    A_lno = A0.clone()

    lno_rollout_success = True

    for step in range(ROLLOUT_STEPS):

        if not torch.isfinite(R_lno).all():
            print(
                f"[FAIL] LNO R became non-finite "
                f"before model call at step {step}"
            )
            lno_rollout_success = False
            break

        if not torch.isfinite(A_lno).all():
            print(
                f"[FAIL] LNO amplitude became non-finite "
                f"before model call at step {step}"
            )
            lno_rollout_success = False
            break

        result_lno = lno_model(
            R_lno,
            A_lno,
            gamma0,
            sigma0
        )

        R_next = result_lno["R_next"]
        A_next = result_lno["amplitude_next"]

        if not torch.isfinite(R_next).all():
            print(
                f"[FAIL] LNO R_next became non-finite "
                f"at step {step}"
            )
            lno_rollout_success = False
            break

        if not torch.isfinite(A_next).all():
            print(
                f"[FAIL] LNO amplitude_next became non-finite "
                f"at step {step}"
            )
            print(
                "nonfinite amplitude entries:",
                int((~torch.isfinite(A_next)).sum().item()),
                "/",
                A_next.numel()
            )
            lno_rollout_success = False
            break

        # --------------------------------------------------------
        # LNO structural + amplitude metrics
        # --------------------------------------------------------
        lno_struct = structural_metrics(R_next)

        lno_struct["amplitude_norm"] = float(
            torch.linalg.vector_norm(A_next).item()
        )

        lno_struct["step"] = step + 1
        lno_struct["model"] = "LNO"

        rollout_rows.append(lno_struct)

        # Advance autoregressive state
        R_lno = R_next
        A_lno = A_next

    if lno_rollout_success:
        print(
            f"[PASS] LNO rollout complete: "
            f"{ROLLOUT_STEPS}/{ROLLOUT_STEPS} steps."
        )
    else:
        print("[FAIL] LNO rollout did not complete.")

# ============================================================
# 14. ADD REGIME LABELS
# ============================================================

rollout_df = pd.DataFrame(rollout_rows)
ROLLOUT_CSV = (OUT_DIR / "long_horizon_rollout.csv")
rollout_df.to_csv(ROLLOUT_CSV, index=False)

print("\n" + "[PASS] Rollout trajectory saved:")
print(ROLLOUT_CSV)


# ============================================================
# 15. GLOBAL WORST-CASE SUMMARY
# ============================================================

summary_rows = []

for model_name in ["FNO", "LNO"]:
    df = rollout_df[rollout_df["model"] == model_name]
    summary_rows.append({
        "model": model_name,
        "rollout_steps": ROLLOUT_STEPS,
        "worst_minimum_eigenvalue": float(df["minimum_eigenvalue"].min()),
        "worst_max_trace_error": float(df["max_trace_error"].max()),
        "worst_negative_eigenvalue_fraction": float(df["negative_eigenvalue_fraction"].max()),
        "max_R_norm": float(df["R_norm"].max()),
        "min_R_norm": float(df["R_norm"].min()),
        "max_amplitude_norm": float(df["amplitude_norm"].max()),
        "min_amplitude_norm": float(df["amplitude_norm"].min()),
        "all_finite": bool(df["finite"].all())
    })

summary_df = pd.DataFrame(summary_rows)
SUMMARY_CSV = (OUT_DIR / "long_horizon_summary.csv")
summary_df.to_csv(SUMMARY_CSV, index=False)


# ============================================================
# 16. PER-REGIME STABILITY SUMMARY
# ============================================================

regime_assignment = []
for regime_name in regimes.keys():
    regime_assignment.extend([regime_name] * REP_PER_REGIME)

REGIME_LABELS = np.asarray(regime_assignment)
REPRESENTATIVE_MAP = pd.DataFrame({
    "sample_index": np.arange(len(representative_indices)),
    "dataset_index": representative_indices,
    "regime": REGIME_LABELS
})
REPRESENTATIVE_MAP.to_csv(OUT_DIR / "representative_rollout_states.csv", index=False)


# ============================================================
# 17. STRUCTURAL ADVANTAGE
# ============================================================

fno_summary = summary_df[summary_df["model"] == "FNO"].iloc[0]
lno_summary = summary_df[summary_df["model"] == "LNO"].iloc[0]

lno_structurally_better = bool(
    (lno_summary["worst_max_trace_error"] < fno_summary["worst_max_trace_error"]) &
    (lno_summary["worst_minimum_eigenvalue"] > fno_summary["worst_minimum_eigenvalue"]) &
    (lno_summary["worst_negative_eigenvalue_fraction"] <= fno_summary["worst_negative_eigenvalue_fraction"]) &
    lno_summary["all_finite"]
)


# ============================================================
# 18. LONG-HORIZON PLOTS
# ============================================================

# ---- R norm ----
fig, ax = plt.subplots(figsize=(7, 7))
for model_name in ["FNO", "LNO"]:
    df = rollout_df[rollout_df["model"] == model_name]
    ax.plot(df["step"], df["R_norm"], linewidth=2, label=model_name)
ax.set_xlabel("Rollout step")
ax.set_ylabel("R norm")
ax.set_title("Long-Horizon R-Norm Stability")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False)
plt.tight_layout()
RNORM_PLOT = (OUT_DIR / "long_horizon_R_norm.png")
plt.savefig(RNORM_PLOT, dpi=300)
plt.close()

# ---- Trace error ----
fig, ax = plt.subplots(figsize=(7, 7))
for model_name in ["FNO", "LNO"]:
    df = rollout_df[rollout_df["model"] == model_name]
    ax.plot(df["step"], df["max_trace_error"], linewidth=2, label=model_name)
ax.set_xlabel("Rollout step")
ax.set_ylabel("Maximum trace error")
ax.set_title("Long-Horizon Trace Preservation")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False)
plt.tight_layout()
TRACE_PLOT = (OUT_DIR / "long_horizon_trace_error.png")
plt.savefig(TRACE_PLOT, dpi=300)
plt.close()

# ---- Minimum eigenvalue ----
fig, ax = plt.subplots(figsize=(7, 7))
for model_name in ["FNO", "LNO"]:
    df = rollout_df[rollout_df["model"] == model_name]
    ax.plot(df["step"], df["minimum_eigenvalue"], linewidth=2, label=model_name)
ax.axhline(0.0, linewidth=1)
ax.set_xlabel("Rollout step")
ax.set_ylabel("Minimum eigenvalue")
ax.set_title("Long-Horizon PSD Stability")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False)
plt.tight_layout()
EIG_PLOT = (OUT_DIR / "long_horizon_minimum_eigenvalue.png")
plt.savefig(EIG_PLOT, dpi=300)
plt.close()


# ============================================================
# 19. FINAL JSON
# ============================================================

SUMMARY_JSON = (OUT_DIR / "long_horizon_summary.json")
summary_json = {
    "step": 22,
    "title": "Long-Horizon Rollout Stability",
    "diagnostic_type": "Repeated-forward stability and structural-preservation diagnostic.",
    "ground_truth_accuracy_claim": False,
    "representative_states": int(len(representative_indices)),
    "representatives_per_regime": REP_PER_REGIME,
    "rollout_steps": ROLLOUT_STEPS,
    "regimes": list(regimes.keys()),
    "FNO": fno_summary.to_dict(),
    "LNO": lno_summary.to_dict(),
    "structural_comparison": {"LNO_structurally_better": lno_structurally_better},
    "outputs": {
        "rollout_csv": str(ROLLOUT_CSV), "summary_csv": str(SUMMARY_CSV),
        "representative_states": str(OUT_DIR / "representative_rollout_states.csv"),
        "R_norm_plot": str(RNORM_PLOT), "trace_plot": str(TRACE_PLOT),
        "eigenvalue_plot": str(EIG_PLOT), "summary_json": str(SUMMARY_JSON)
    }
}
with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary_json, f, indent=2, allow_nan=True)


# ============================================================
# 20. FINAL STATUS
# ============================================================

print("\n" + "=" * 110)
print("STEP 22 COMPLETE — LONG-HORIZON ROLLOUT STABILITY")
print("=" * 110)

print("\nFNO:")
print("  Worst minimum eigenvalue:", f"{fno_summary['worst_minimum_eigenvalue']:.8e}")
print("  Worst max trace error:", f"{fno_summary['worst_max_trace_error']:.8e}")
print("  Worst PSD-violation fraction:", f"{fno_summary['worst_negative_eigenvalue_fraction']:.8f}")
print("  All finite:", fno_summary["all_finite"])

print("\nLNO:")
print("  Worst minimum eigenvalue:", f"{lno_summary['worst_minimum_eigenvalue']:.8e}")
print("  Worst max trace error:", f"{lno_summary['worst_max_trace_error']:.8e}")
print("  Worst PSD-violation fraction:", f"{lno_summary['worst_negative_eigenvalue_fraction']:.8f}")
print("  All finite:", lno_summary["all_finite"])

print("\nSTRUCTURAL ADVANTAGE FLAG:", lno_structurally_better)
print("\nResults:", OUT_DIR)

print("\n[PASS] STEP 22 COMPLETE.")
print("[INFO] No model retraining performed.")
print("[INFO] This is a long-horizon stability diagnostic.")
print("[INFO] It is not a new ground-truth accuracy benchmark.")

print("=" * 110)
