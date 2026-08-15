import sys
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

device = torch.device("cpu")
print("[+] Execution Device:", device)

from models.lno_model import LindbladNeuralOperator

# Initialize Model & Load Checkpoint
model = LindbladNeuralOperator()
model.to(device)

ckpt_path = os.path.join(project_root, "results", "check_points", "best_lno.pt")
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    print("[+] Model loaded successfully from checkpoint!")

model.eval()

print("\n" + "="*65)
print("PHASE 1 ZERO-SHOT ABLATION: QUANTITATIVE NOISE SCORE COMPUTATION")
print("="*65)

batch_size, channels, nx = 4, 64, 128

# All inputs must have identical channel dimension [batch, channels, nx] for torch.stack in lno_model.py
dummy_state = torch.randn(batch_size, channels, nx, device=device)
dummy_phi = torch.randn(batch_size, channels, nx, device=device)
dummy_flux = torch.randn(batch_size, channels, nx, device=device)
dummy_noise = torch.randn(batch_size, channels, nx, device=device)
dummy_dissipation = torch.ones(batch_size, channels, nx, device=device) * 0.5
dummy_gamma = torch.ones(batch_size, channels, nx, device=device) * 0.5

zero_dissipation = torch.zeros_like(dummy_dissipation)
zero_gamma = torch.zeros_like(dummy_gamma)

with torch.no_grad():
    # 1. Stochastic Run (Noise ON)
    res_on = model(dummy_state, dummy_phi, dummy_flux, dummy_noise, dummy_dissipation, dummy_gamma)

    # 2. Deterministic / Ablated Run (Noise OFF)
    res_off = model(dummy_state, dummy_phi, dummy_flux, dummy_noise, zero_dissipation, zero_gamma)

    # Extract tensor safely from dictionary output
    out_noise_on = res_on["pred"] if isinstance(res_on, dict) and "pred" in res_on else (list(res_on.values())[0] if isinstance(res_on, dict) else res_on)
    out_noise_off = res_off["pred"] if isinstance(res_off, dict) and "pred" in res_off else (list(res_off.values())[0] if isinstance(res_off, dict) else res_off)

    # 3. Compute Quantitative Metrics
    l2_diff = torch.norm(out_noise_on - out_noise_off).item()
    l2_on = torch.norm(out_noise_on).item()
    relative_noise_score = (l2_diff / (l2_on + 1e-8)) * 100.0

    print("\n[+] --- Gating Parameter Diagnostics (Freshly Initialized) ---")
    for idx, layer in enumerate(model.dissipative_layers):
        gated_gamma = layer.get_gated_gamma()
        print(f"    Layer {idx} Gated Gamma Mean: {gated_gamma.mean().item():.6f} | Min: {gated_gamma.min().item():.6f} | Max: {gated_gamma.max().item():.6f}")

print("\n" + "-"*65)
print(f"[*] Noise ON Output L2 Norm:            {l2_on:.4f}")
print(f"[*] Noise OFF (Ablated) Output L2 Norm:   {torch.norm(out_noise_off).item():.4f}")
print(f"[*] Absolute Difference (L2):            {l2_diff:.4f}")
print(f"[*] Phase 1 Relative Noise Score:        {relative_noise_score:.2f}%")
print("="*65)
print("[+] Phase 1 Zero-Shot Ablation evaluation successfully completed!")
