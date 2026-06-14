"""fp64 numpy oracle for the daily CK-PCA RBF Omega — step-2 correctness gate.

Takes a (drift-corrected) prep dir, carves out a small subsample (first --days
days, each capped to --maxn stocks), writes it as a fresh prep dir, then:
  1. computes Omega[c] for the Gaussian (RBF) kernel directly in numpy fp64, and
  2. runs the CUDA `omega_grid_tiled` engine on the same subsample,
and reports max abs/rel error + a pass/fail under tolerance.

The numpy path mirrors the engine's exact arithmetic (kernel + the 4-term
double-centered, return-weighted reduction in omega_grid_tiled.cu), so a match
to ~1e-10 validates the engine on drifted weights before any low-precision work.

RBF convention (matches the engine): param = 1/(2 c^2),
  K[i,j] = exp(-||z_i - z_j||^2 * param),
  Omega[t,s] = r_t' K_tilde r_s   with K double-centered over both cross-sections.

Usage:
  daily_oracle.py [--prep data/prep_daily_bh] [--work data/oracle_sub]
                  [--engine build/omega_grid_tiled] [--c 0.1]
                  [--days 40] [--maxn 200] [--rtol 1e-8] [--atol 1e-10]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def load_prep(d: Path):
    meta = json.loads((d / "meta.json").read_text())
    K = meta["K"]
    Z = np.fromfile(d / "Z.f64.bin", dtype=np.float64).reshape(-1, K)
    r = np.fromfile(d / "r.f64.bin", dtype=np.float64)
    off = np.fromfile(d / "offsets.i64.bin", dtype=np.int64)
    dates = np.fromfile(d / "dates.i64.bin", dtype=np.int64)
    return meta, Z, r, off, dates


def subsample(meta, Z, r, off, dates, days: int, maxn: int):
    days = min(days, len(dates))
    Zs, rs, new_off = [], [], [0]
    for t in range(days):
        s, e = int(off[t]), int(off[t + 1])
        n = min(maxn, e - s) if maxn else e - s
        Zs.append(Z[s:s + n])
        rs.append(r[s:s + n])
        new_off.append(new_off[-1] + n)
    return (np.vstack(Zs), np.concatenate(rs),
            np.asarray(new_off, dtype=np.int64), dates[:days].copy())


def write_prep(d: Path, meta: dict, Z, r, off, dates):
    d.mkdir(parents=True, exist_ok=True)
    (d / "Z.f64.bin").write_bytes(np.ascontiguousarray(Z, np.float64).tobytes())
    (d / "r.f64.bin").write_bytes(np.ascontiguousarray(r, np.float64).tobytes())
    (d / "offsets.i64.bin").write_bytes(
        np.ascontiguousarray(off, np.int64).tobytes())
    (d / "dates.i64.bin").write_bytes(
        np.ascontiguousarray(dates, np.int64).tobytes())
    m = dict(meta)
    m["T"] = int(len(dates))
    m["total_rows"] = int(off[-1])
    m["n_per_date_max"] = int(np.diff(off).max())
    m["n_per_date_min"] = int(np.diff(off).min())
    (d / "meta.json").write_text(json.dumps(m, indent=2))


def omega_rbf_numpy(Z, r, off, c: float) -> np.ndarray:
    """Double-centered, return-weighted RBF Omega in fp64, 4-term form."""
    T = len(off) - 1
    param = 1.0 / (2.0 * c * c)
    blocks = [(Z[off[t]:off[t + 1]], r[off[t]:off[t + 1]]) for t in range(T)]
    Om = np.empty((T, T), dtype=np.float64)
    for t in range(T):
        Zt, rt = blocks[t]
        n1 = len(rt)
        Rt = rt.sum()
        zt2 = (Zt * Zt).sum(axis=1)
        for s in range(t, T):
            Zs, rs = blocks[s]
            n2 = len(rs)
            Rs = rs.sum()
            zs2 = (Zs * Zs).sum(axis=1)
            sq = zt2[:, None] + zs2[None, :] - 2.0 * (Zt @ Zs.T)
            K = np.exp(-sq * param)
            s_ab = rt @ K @ rs
            s_k1 = rt @ K.sum(axis=1)          # sum_ij ri K_ij
            s_1k = K.sum(axis=0) @ rs          # sum_ij K_ij rj
            s_k = K.sum()
            val = (s_ab - s_k1 * Rs / n2 - Rt * s_1k / n1
                   + s_k * Rt * Rs / (n1 * n2))
            Om[t, s] = val
            Om[s, t] = val
    return Om


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--prep", type=Path, default=Path("data/prep_daily_bh"))
    p.add_argument("--work", type=Path, default=Path("data/oracle_sub"))
    p.add_argument("--engine", type=Path, default=Path("build/omega_grid_tiled"))
    p.add_argument("--c", type=float, default=0.1)
    p.add_argument("--days", type=int, default=40)
    p.add_argument("--maxn", type=int, default=200)
    p.add_argument("--rtol", type=float, default=1e-8)
    p.add_argument("--atol", type=float, default=1e-10)
    args = p.parse_args()

    meta, Z, r, off, dates = load_prep(args.prep)
    Zs, rs, offs, ds = subsample(meta, Z, r, off, dates, args.days, args.maxn)
    write_prep(args.work, meta, Zs, rs, offs, ds)
    T = len(offs) - 1
    print(f"[oracle] subsample: T={T} days, N/day in "
          f"[{np.diff(offs).min()}, {np.diff(offs).max()}], "
          f"rows={offs[-1]}, c={args.c}")

    cstr = repr(float(args.c))
    cmd = [str(args.engine), "rbf", str(args.work), str(args.work), cstr]
    print("[oracle] running engine:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    eng = np.fromfile(args.work / f"omega_rbf_{cstr}.f64.bin",
                      dtype=np.float64).reshape(T, T)

    print("[oracle] computing numpy fp64 reference ...")
    ref = omega_rbf_numpy(Zs, rs, offs, args.c)

    diff = np.abs(eng - ref)
    denom = np.maximum(np.abs(ref), 1e-300)
    rel = diff / denom
    max_abs = float(diff.max())
    max_rel = float(rel.max())
    ok = np.allclose(eng, ref, rtol=args.rtol, atol=args.atol)
    print(f"[oracle] max abs err = {max_abs:.3e}")
    print(f"[oracle] max rel err = {max_rel:.3e}")
    print(f"[oracle] Omega scale: |ref| in "
          f"[{np.abs(ref).min():.3e}, {np.abs(ref).max():.3e}]")
    print(f"[oracle] symmetry (engine): {np.abs(eng - eng.T).max():.3e}")
    print(f"[oracle] {'PASS' if ok else 'FAIL'} "
          f"(rtol={args.rtol}, atol={args.atol})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
