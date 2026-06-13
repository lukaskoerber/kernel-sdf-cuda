"""Spot-check the CUDA poly2 Omega against an exact numpy reference.

There is no committed poly2 ground-truth parquet (data/test has only linear and
rbf), so this recomputes the reference's exact poly2 Omega for a random sample
of (t, s) month-pairs straight from the data/prep flat binaries and compares to
the CUDA output. The reference math (reference/ck_pca_cuda_v_aug25.py):

    K          = (X1 @ X2.T + c) ** 2            # degree 2, coef0 = c
    K_tilde    = K - rowmean - colmean + allmean # double-centering
    Omega[t,s] = r1 @ K_tilde @ r2

Usage:
  check_poly2.py --cuda data/out/omega_poly2_<c>.f64.bin --c <c>
                 [--prep data/prep] [--pairs 60] [--seed 0]
                 [--rtol 1e-6] [--atol 1e-8]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def centered_poly2_omega(X1, X2, r1, r2, c: float) -> float:
    K = (X1 @ X2.T + c) ** 2
    row = K.mean(axis=1, keepdims=True)
    col = K.mean(axis=0, keepdims=True)
    allm = K.mean()
    Kt = K - row - col + allm
    return float(r1 @ Kt @ r2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", type=Path, required=True)
    ap.add_argument("--c", type=float, required=True)
    ap.add_argument("--prep", type=Path, default=Path("data/prep"))
    ap.add_argument("--pairs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-8)
    args = ap.parse_args()

    meta = json.loads((args.prep / "meta.json").read_text())
    T, K = meta["T"], meta["K"]
    R = meta["total_rows"]

    Z = np.fromfile(args.prep / "Z.f64.bin", dtype=np.float64).reshape(R, K)
    r = np.fromfile(args.prep / "r.f64.bin", dtype=np.float64)
    off = np.fromfile(args.prep / "offsets.i64.bin", dtype=np.int64)
    assert off.shape == (T + 1,), off.shape

    cuda = np.fromfile(args.cuda, dtype=np.float64).reshape(T, T)

    rng = np.random.default_rng(args.seed)
    # A mix: always include a few diagonal pairs plus random upper-triangle ones.
    pairs = [(t, t) for t in rng.integers(0, T, size=max(1, args.pairs // 6))]
    while len(pairs) < args.pairs:
        t = int(rng.integers(0, T))
        s = int(rng.integers(t, T))
        pairs.append((t, s))

    max_abs = 0.0
    max_rel = 0.0
    worst = None
    for (t, s) in pairs:
        X1 = Z[off[t]:off[t + 1]]
        X2 = Z[off[s]:off[s + 1]]
        r1 = r[off[t]:off[t + 1]]
        r2 = r[off[s]:off[s + 1]]
        ref = centered_poly2_omega(X1, X2, r1, r2, args.c)
        got = cuda[t, s]
        ad = abs(got - ref)
        rd = ad / (max(abs(ref), abs(got)) + np.finfo(np.float64).tiny)
        if ad > max_abs:
            max_abs, worst = ad, (t, s, ref, got)
        max_rel = max(max_rel, rd)

    ok = np.allclose(
        [cuda[t, s] for (t, s) in pairs],
        [centered_poly2_omega(Z[off[t]:off[t + 1]], Z[off[s]:off[s + 1]],
                              r[off[t]:off[t + 1]], r[off[s]:off[s + 1]], args.c)
         for (t, s) in pairs],
        rtol=args.rtol, atol=args.atol,
    )

    print(f"cuda output : {args.cuda}")
    print(f"c           : {args.c:.17g}")
    print(f"pairs tested: {len(pairs)} (numpy exact reference)")
    print(f"max |diff|  : {max_abs:.3e}  at (t,s,ref,got)={worst}")
    print(f"max rel diff: {max_rel:.3e}")
    print(f"symmetry    : |cuda-cuda.T|max={np.abs(cuda - cuda.T).max():.3e}")
    print(f"allclose(rtol={args.rtol:.0e}, atol={args.atol:.0e}) : {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
