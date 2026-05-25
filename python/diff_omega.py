"""Compare a C++/CUDA-produced Omega bin file against a reference parquet.

Loads data/out/omega_<...>.f64.bin (row-major fp64, T x T) and the test
parquet, aligns by the data/prep date index (already validated to match in
step 2), and reports max abs/rel error plus a pass/fail under np.allclose.

Usage:
  diff_omega.py [--cuda data/out/omega_linear.f64.bin]
                [--reference data/test/omega_linear_..._cz82_OSAP_M.parquet]
                [--rtol 1e-6] [--atol 1e-8]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cuda", type=Path, default=Path("data/out/omega_linear.f64.bin"))
    p.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "data/test/omega_linear_1973-10-31_2024-11-30_cz82_OSAP_M.parquet"
        ),
    )
    p.add_argument("--prep", type=Path, default=Path("data/prep"))
    p.add_argument("--rtol", type=float, default=1e-6)
    p.add_argument("--atol", type=float, default=1e-8)
    args = p.parse_args()

    meta = json.loads((args.prep / "meta.json").read_text())
    T = meta["T"]

    ref_df = pd.read_parquet(args.reference)
    if ref_df.shape != (T, T):
        raise SystemExit(
            f"reference shape {ref_df.shape} != (T={T}, T={T})"
        )
    ref = ref_df.to_numpy(dtype=np.float64)

    dates = np.fromfile(args.prep / "dates.i64.bin", dtype=np.int64).astype(
        "datetime64[D]"
    )
    ref_dates = np.array(
        [pd.Timestamp(d).to_datetime64() for d in ref_df.index],
        dtype="datetime64[D]",
    )
    if not np.array_equal(dates, ref_dates):
        raise SystemExit("data/prep/dates.i64.bin does not match reference index")

    cuda = np.fromfile(args.cuda, dtype=np.float64).reshape(T, T)

    diff = np.abs(cuda - ref)
    max_abs = float(diff.max())
    denom = np.maximum(np.abs(ref), np.abs(cuda)) + np.finfo(np.float64).tiny
    max_rel = float((diff / denom).max())
    sym = float(np.abs(cuda - cuda.T).max())
    ok = np.allclose(cuda, ref, rtol=args.rtol, atol=args.atol)

    argmax = np.unravel_index(int(diff.argmax()), diff.shape)
    print(f"reference     : {args.reference}")
    print(f"cuda output   : {args.cuda}")
    print(f"shape         : {ref.shape}")
    print(
        f"ref:    min={ref.min(): .3e} max={ref.max(): .3e} "
        f"|.|max={np.abs(ref).max():.3e}"
    )
    print(
        f"cuda:   min={cuda.min(): .3e} max={cuda.max(): .3e} "
        f"|.|max={np.abs(cuda).max():.3e}"
    )
    print(f"max |diff|    : {max_abs:.3e} at index {argmax}")
    print(f"max rel diff  : {max_rel:.3e}")
    print(f"|cuda-cuda.T| : {sym:.3e}  (symmetry check)")
    print(f"allclose(rtol={args.rtol:.0e}, atol={args.atol:.0e}) : {ok}")

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
