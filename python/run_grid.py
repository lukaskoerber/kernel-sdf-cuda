"""Main workflow: compute CK-PCA Omega over a grid of c values per kernel.

Defines the per-kernel hyperparameter grids (mirroring the reference notebook),
then invokes the `omega_grid` CUDA engine once per kernel. The engine loads the
data and computes each month's GEMM a single time, reusing it across the whole
c-grid, so a 20-point grid is far cheaper than 20 separate runs.

Reference grids (reference/ck_pca_cuda_v_aug25.py):
    rbf   : np.logspace(-4,  0, 20)
    poly2 : np.logspace(-9, -1, 20)

Usage:
  run_grid.py [--kernels rbf poly2] [--build-dir build] [--prep data/prep]
              [--out data/out] [--points 20] [--dry-run]
              [--rbf-lo -4] [--rbf-hi 0] [--poly2-lo -9] [--poly2-hi -1]

Each Omega is written by the engine to data/out/omega_<kernel>_<c>.f64.bin,
where <c> is the exact string this script passes (repr of the float), so the
filename and the value used in the computation always agree.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from bin_to_parquet import (
    DEFAULT_DATA_SOURCE,
    DEFAULT_FREQ,
    bin_to_parquet,
    canonical_omega_name,
    load_meta,
)

GRID_DEFAULTS = {
    "rbf":   (-4.0, 0.0),
    "poly2": (-9.0, -1.0),
}


def make_grid(lo: float, hi: float, points: int) -> list[str]:
    # repr() round-trips exactly, so the filename string and the parsed double
    # the engine computes with are identical.
    return [repr(float(c)) for c in np.logspace(lo, hi, points)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", nargs="+", default=["rbf", "poly2"],
                    choices=["rbf", "poly2"])
    ap.add_argument("--build-dir", type=Path, default=Path("build"))
    ap.add_argument("--prep", type=Path, default=Path("data/prep"))
    ap.add_argument("--out", type=Path, default=Path("data/out"))
    ap.add_argument("--points", type=int, default=20)
    ap.add_argument("--rbf-lo", type=float, default=GRID_DEFAULTS["rbf"][0])
    ap.add_argument("--rbf-hi", type=float, default=GRID_DEFAULTS["rbf"][1])
    ap.add_argument("--poly2-lo", type=float, default=GRID_DEFAULTS["poly2"][0])
    ap.add_argument("--poly2-hi", type=float, default=GRID_DEFAULTS["poly2"][1])
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running them")
    ap.add_argument("--no-parquet", action="store_true",
                    help="keep only the raw .bin; skip parquet conversion")
    ap.add_argument("--data-source", default=DEFAULT_DATA_SOURCE,
                    help="data_source field in the omega filename")
    ap.add_argument("--freq", default=DEFAULT_FREQ,
                    help="freq field in the omega filename")
    args = ap.parse_args()

    engine = args.build_dir / "omega_grid"
    if not engine.exists():
        print(f"error: engine not found at {engine}; build it first "
              f"(cmake --build {args.build_dir} --target omega_grid)",
              file=sys.stderr)
        return 1

    bounds = {
        "rbf":   (args.rbf_lo, args.rbf_hi),
        "poly2": (args.poly2_lo, args.poly2_hi),
    }

    for kernel in args.kernels:
        lo, hi = bounds[kernel]
        grid = make_grid(lo, hi, args.points)
        cmd = [str(engine), kernel, str(args.prep), str(args.out), *grid]
        print(f"\n=== {kernel}: {args.points} points in "
              f"logspace({lo}, {hi}) ===")
        print(f"  c-grid: {grid[0]} ... {grid[-1]}")
        if args.dry_run:
            print("  (dry-run) " + " ".join(cmd[:4]) + " <c1..cN>")
            continue
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"error: engine failed for kernel {kernel} (rc={rc})",
                  file=sys.stderr)
            return rc

        if not args.no_parquet:
            meta = load_meta(args.prep)
            for c in grid:
                bin_path = args.out / f"omega_{kernel}_{c}.f64.bin"
                pq_name = canonical_omega_name(kernel, c, meta,
                                               args.data_source, args.freq)
                bin_to_parquet(bin_path, args.out / pq_name, args.prep)
            print(f"  converted {len(grid)} file(s) to parquet in {args.out}")
            print(f"  e.g. {canonical_omega_name(kernel, grid[0], meta, args.data_source, args.freq)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
