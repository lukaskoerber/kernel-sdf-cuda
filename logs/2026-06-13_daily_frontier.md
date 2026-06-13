# Handoff — daily CK-PCA Omega frontier (2026-06-13)

Branch: `daily-omega`. Full context in memory: `daily-omega-pipeline.md`.

## Where we are
Monthly Omega (linear/rbf/poly2) is done, verified, fast. Daily pipeline built:
- `python/prep_daily.py` → `data/prep_daily/` (T=11,645 days, K=54 anomalies, 12.8M rows, 1973–2019).
- `src/omega_grid_tiled.cu` (`omega_grid_tiled`): partner-day-tiled rbf/poly2, auto-sized to GPU mem. Validated vs monthly oracle + numpy.
- Linear daily works as-is (`omega_linear`, F·F').

## The frontier problem: ~15× too slow
Measured daily RBF = **28 min / c-point in fp64** → ~15 h for a 32-point CV sweep.
The paper does the whole 32-point daily sweep in **<1 h on a weaker Titan V**. Gap is
**precision, not algorithm**: his ~10^18 ops are GEMM-dominated and only fit <1 h via
**FP16/TF32 tensor cores**; we run fp64. Structure is otherwise correct (T≈12k days,
full N×N kernel per day-pair — NO month-block shortcut, because weights drift).

## Two things to fix next
1. **prep_daily correctness:** it currently holds Z constant within a month. Paper (p.18)
   uses **intra-month buy-and-hold weight drift** (weights ∝ z_i × cumret since the monthly
   rebalance). Must apply this drift before results are paper-faithful.
2. **Mixed-precision daily engine (speed):** TF32/FP16 tensor GEMM + FP32 exp +
   **FP64 accumulation** (centering/reduction is cancellation-prone) + FP64 eigendecomp.
   Then **validate OOS-Sharpe/CV curve vs the fp64 oracle** — precision drop must sit far
   below the ~±0.1 statistical noise on a Sharpe.

## Open questions
- Char set is **54** anomalies; paper uses **40** — reconcile the subset before final CV plots.
- Sample ends **2019** (anomaly file cutoff; CRSP daily runs to 2024).
- Goal: reproduce the paper's **cross-validation plots** (eigendecomp → dominant CK-PCs →
  Kozak-2019 shrinkage → OOS Sharpe vs c). That SDF/CV stage is **not built yet**.
