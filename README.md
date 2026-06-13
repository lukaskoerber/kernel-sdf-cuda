# kernel-sdf

CUDA implementation of the CK-PCA Ω (Omega) computation from *"A Kernel Trick for
the Cross-Section"* — the `T×T` double-centered, return-weighted kernel matrix
`Ω[t,s] = Rₜ′ K̃(Zₜ,Zₛ) Rₛ` used to build a characteristics-kernel SDF.

## Layout
- `src/omega_linear.cu` — linear kernel via the `F·F'` shortcut.
- `src/omega_rbf.cu` — baseline per-pair RBF (reference).
- `src/omega_rbf_opt.cu` — optimized RBF (per-month batched GEMM + SM-wide reduction).
- `src/omega_grid.cu` — rbf/poly2 over a c-grid (monthly).
- `src/omega_grid_tiled.cu` — partner-day-tiled rbf/poly2 for daily / large `T`.
- `python/prep.py`, `python/prep_daily.py` — monthly / daily data prep.
- `python/run_grid.py`, `python/bin_to_parquet.py` — grid driver + parquet output.
- `python/diff_omega.py`, `python/check_poly2.py` — verification.

Build: `cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=80 && cmake --build build -j`
(A100 = sm_80). Python deps live in `./.venv` (needs `pyarrow`).

## Status
Monthly Ω (linear/rbf/poly2) is verified against the reference to ~1e-12 and fast
(optimized RBF ≈ 5 s). A daily-frequency pipeline (`daily-omega` branch) is built and
validated but not yet paper-faithful or production-fast — see Limitations.

## Limitations / known gaps
- **Within-month weight drift is NOT implemented.** `prep_daily.py` holds the
  characteristic weights `Z` **constant within each month** (forward-filled from the
  monthly rebalance). The paper instead lets the daily weights **drift intra-month as a
  monthly-rebalanced buy-and-hold strategy** (weights ∝ `zᵢ × cumulative return since the
  rebalance`). Until this drift is added, daily Ω is close but not paper-faithful, and the
  kernel is *not* constant within a month (so no month-block computational shortcut).
- **Daily RBF is fp64 → ~15× slower than the paper.** One daily c-point ≈ 28 min;
  a 32-point sweep ≈ 15 h. The paper runs the full sweep in <1 h via half-precision
  (FP16/TF32) tensor cores. A mixed-precision daily path (low-precision GEMM + FP32 `exp`
  + FP64 accumulation + FP64 eigendecomposition) is planned, to be validated against the
  fp64 oracle.
- **Characteristic set is 54 anomalies; the paper uses 40** — the subset still needs reconciling.
- **Daily sample ends 2019-12** (the anomaly file cutoff; CRSP daily extends to 2024).
- **The SDF / cross-validation stage is not built** — eigendecomposition → dominant
  CK-PCs → Kozak (2019) shrinkage → OOS-Sharpe CV plots remain to be implemented.
