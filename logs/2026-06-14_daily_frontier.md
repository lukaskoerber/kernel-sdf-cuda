# Frontier — daily CK-PCA Omega → SDF replication (2026-06-14)

Branch: `daily-omega`. Supersedes `logs/2026-06-13_daily_frontier.md`.
Goal: replicate Kozak's daily flow — buy-and-hold drifted weights + Gaussian
(RBF) kernel, Omega over a small c-grid, then eigendecomp → CK-PCs → Kozak-2019
shrinkage → OOS-Sharpe cross-validation plots.

## Pipeline status (one line each)
1. Data prep (CPU) ......... DONE — paper-faithful (40 chars, buy-and-hold drift)
2. Omega engine (GPU) ...... DONE — fp64 oracle + validated mixed-precision (TF32)
3. Correctness validation .. DONE — numpy oracle + eigenpair-level check
4. SDF / shrinkage / CV .... NOT BUILT — the remaining replication work
5. Result plots ............ NOT BUILT

---

## What IS implemented

### Prep — `python/prep_daily.py` → `data/prep_daily_bh/`
- T = 11,645 trading days, K = **40** anomalies (paper set), 12,839,504 rows,
  1973-10-31 .. 2019-12-31.
- `--char-set paper40` selects the 40 Kozak chars (CSV has 54; 14 extras dropped:
  dur, valprof, divp, gltnoa, divg, invaci, valmom, valmomprof, shortint, sue,
  roa, indrrevlv, indmomrev, ipo). `--char-list` allows a custom subset.
- `--weight-drift buyhold` (Kozak p.19): monthly-rebalanced buy-and-hold.
  * Membership FROZEN at each month's rebalance (first trading) day; mid-month
    entrants dropped (only 41 across the whole sample), delistings drop naturally.
  * Weight drift: `w_{i,t} = z_{i,M} * cumprod(1+r).shift(1)` within the month,
    factor 1.0 on the rebalance day. NO daily renormalization (true buy-and-hold).
  * Verified: rebalance-day weights == old constant-Z prep to 0.0; within-month
    scale stable; finite. (`--weight-drift none` keeps the old constant-Z path.)

### Omega engine (GPU, A100 sm_80, CUDA 12.8)
- `src/omega_grid_tiled.cu` (`omega_grid_tiled`): fp64 reference engine.
  Partner-day tiled, batched over a c-grid. Omega[t,s] = r_t' K̃(Z_t,Z_s) r_s,
  RBF K=exp(-||z_i-z_j||^2/(2c^2)), double-centered, return-weighted.
- `src/omega_grid_tiled_mp.cu` (`omega_grid_tiled_mp`): MIXED-PRECISION variant.
  TF32 tensor-core GEMM (cublasGemmEx) + fp32 znorms/distance/expf + fp64
  returns/accumulation/centering/Omega. Same binary output format. fp64 engine
  left untouched as the oracle (additive, reversible).

### Validation
- `python/daily_oracle.py`: carves a subsample, computes RBF Omega in numpy fp64
  (exact engine arithmetic), runs an engine (`--engine`), diffs.
  * fp64 engine vs numpy: max abs ~1e-12 (at full N ~1080/day). PASS.
  * MP engine vs numpy: max abs 2e-6, max rel 6e-4. PASS.
- Trusted fp64 reference: `data/out_daily/omega_rbf_0.1.f64.bin` (c=0.1, 1.08 GB).
- MP vs fp64 full-matrix: Frobenius rel err **1.3e-5**, RMSE 2.2e-7.
- MP vs fp64 EIGENPAIRS (scipy eigsh, use `external/.../.venv` for scipy):
  top-30 eigenvalues match to max rel **8.1e-6**, eigenvectors |cos|=1.0, top-30
  subspace identical. => TF32 is amply precise for the downstream SDF.

### Performance (full sample, c=0.1)
- fp64: 1642 s/c. MP: 818 s/c (2x). MP 3-c grid: 571 s/c-point.
- Decomp: GEMM ~371 s (amortized once per batch), reduction ~447 s/c (NOT
  amortized). Bottleneck = HBM traffic on the M tile (~330 TB): GEMM writes once,
  reduction RE-READS once per c. Memory-bound, not compute-bound (why TF32 only
  gave 2x). The chosen 3-point grid => ~28 min total: acceptable.

---

## What is MISSING for full Kozak replication

### A. SDF / shrinkage / cross-validation (the core remaining work)
Not built. Design agreed (see memory `daily-buyhold-drift`):
1. `eigh(Omega)` → top-k eigenpairs (λ_i, v_i).
2. PC return panel `F = V_k · diag(√λ_k)` (T×k) — the dominant
   characteristics-managed PCs (Omega is the Gram matrix; no panel reconstruction).
3. Kozak-2019 ridge SDF: PCs orthogonal ⇒ closed form `b_i = μ_i/(λ_i + γ)`.
4. 5-fold CV over γ maximizing OOS R² (paper eq. 26); pick kernel c by OOS Sharpe.
5. Plots: OOS Sharpe / R² vs c (and vs γ).
- Proposed files: `python/sdf.py` (module) + `python/run_cv.py` (driver) + a thin
  plot script/notebook. eigendecomposition stays fp64.
- UPSA (`external/Universal_Portfolio_Shrinkage`) is feasible but a SUPERSET
  (Kelly–Malamud 2024 ensemble, different CV objective); use its single-ridge as a
  cross-check / ensemble as an extension, NOT the primary replication.

### B. c-grid selection
The 3-point grid must be centered on the actual optimum. No optimum yet (needs the
CV stage, or a coarse scan first). Decision pending: coarse c-scan vs build CV
harness first and let it drive c-selection.

### C. Sample coverage
Daily sample ends **2019-12-31** (anomaly char file cutoff). CRSP daily extends to
2024; the paper's window differs. Char file would need extending for 2020–2024.

### D. Char-set reconciliation (status: likely resolved)
Now using the exact 40-name list the user supplied; all 40 present in the CSV.
Confirm this list matches the paper's Table-1 set if a discrepancy surfaces.

### E. Speed for a LARGE sweep (only if needed)
A full 32-point sweep ≈ 4 h on the MP engine (reduction re-reads M per c). To
match the paper's <1 h, fuse the c-loop INSIDE the reduce kernel (read M once for
all c). NOT needed for the 3-point grid; deferred.

### F. Possible faithfulness checks still open
- Small-cap screen: paper drops stocks < 0.01% of aggregate market cap; current
  prep uses shrcd/exchcd common-stock + NYSE/AMEX/NASDAQ filters only (no mktcap
  screen). Reconcile if results diverge.
- Risk-free / return convention: Omega is invariant to a common additive return
  shift (zero row/col sums), so rf handling is a non-issue for Omega; revisit for
  the SDF mean returns.

---

## Immediate next step
Build the downstream CV harness (`sdf.py` + `run_cv.py`), then either (a) run a
coarse c-scan to locate the optimum and refine to the 3-point grid, or (b) let the
CV harness pick c. Engine + prep are ready and validated.
