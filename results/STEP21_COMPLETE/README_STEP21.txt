
STEP 21 — COLLAPSE STRESS TEST
================================

Project:
Lindblad Neural Operators for Inverse Dynamical Imaging
of Stochastic Poisson–Nernst–Planck Ion Transport

Step:
21 — Collapse Stress Test

Date packaged:
2026-09-11T13:46:25

Test regime:
Collapse

Collapse baseline:
gamma = 0.50
sigma = 0.65

Progressive stress:
Level 0: gamma=0.50, sigma=0.65
Level 1: gamma=0.60, sigma=0.75
Level 2: gamma=0.70, sigma=0.85
Level 3: gamma=0.80, sigma=1.00
Level 4: gamma=0.90, sigma=1.15
Level 5: gamma=1.00, sigma=1.30

Collapse test transitions:
297

Diagnostics:
- Prediction response under progressively harder collapse
- Trace preservation
- Minimum eigenvalue
- PSD violation
- Rollout stability
- State/amplitude response
- Lindblad vs neural dynamical contribution

Important scientific note:
This is a counterfactual environment-response and
structural-stability stress test. It is NOT a new
ground-truth accuracy benchmark because no independently
simulated ground-truth trajectories were generated for
the stressed parameter values.

No model retraining was performed.
No checkpoint was modified.

Artifacts:
- collapse_stress_results.csv
- collapse_stress_rollout.csv
- collapse_stress_regime_summary.csv
- collapse_stress_response.png
- collapse_stress_summary.json
