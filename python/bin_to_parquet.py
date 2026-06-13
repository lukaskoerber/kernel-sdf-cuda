"""Convert a raw Omega .bin (row-major fp64, T x T) into a parquet DataFrame,
using the same filename convention as the reference Python flow.

The CUDA engines write Omega as a flat little-endian fp64 binary for speed (no
Arrow/Parquet dependency in C++). This wraps one into the same parquet layout as
the committed ground truth in data/test/: a (T x T) float64 DataFrame whose index
and columns are both the month-end dates (datetime64[ns], named "date").

Filename convention (reference/ck_pca_cuda_v_aug25.py :: init_omega):
    linear     : omega_linear_{start}_{end}_{char_model}_{data_source}_{freq}.parquet
    rbf / poly2: omega_{kernel}_{c}_{start}_{end}_{char_model}_{data_source}_{freq}.parquet
where start/end are the actual sample date range and char_model is the
characteristics model (data/prep/meta.json: actual_date_min/max, kernel_set).

Usage:
  # canonical name (writes into --out-dir):
  bin_to_parquet.py --bin data/out/omega_poly2_0.001.f64.bin \
      --kernel poly2 --c 0.001 [--out-dir data/out] \
      [--data-source OSAP] [--freq M]
  # explicit output path (no convention):
  bin_to_parquet.py --bin <file.bin> --out <file.parquet>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_SOURCE = "OSAP"
DEFAULT_FREQ = "M"


def load_meta(prep_dir: Path) -> dict:
    return json.loads((prep_dir / "meta.json").read_text())


def load_dates(prep_dir: Path) -> pd.DatetimeIndex:
    days = np.fromfile(prep_dir / "dates.i64.bin", dtype=np.int64)
    idx = pd.DatetimeIndex(days.astype("datetime64[D]").astype("datetime64[ns]"))
    idx.name = "date"
    return idx


def canonical_omega_name(kernel: str, c, meta: dict,
                         data_source: str = DEFAULT_DATA_SOURCE,
                         freq: str = DEFAULT_FREQ) -> str:
    """Build the reference-style omega filename (without directory)."""
    start = meta["actual_date_min"]
    end = meta["actual_date_max"]
    char_model = meta["kernel_set"]
    parts = ["omega", kernel]
    if kernel != "linear":
        if c is None:
            raise ValueError(f"kernel '{kernel}' requires a c value for naming")
        parts.append(str(c))
    parts += [start, end, char_model, data_source, freq]
    return "_".join(parts) + ".parquet"


def bin_to_parquet(bin_path: Path, parquet_path: Path, prep_dir: Path) -> Path:
    meta = load_meta(prep_dir)
    T = meta["T"]
    dates = load_dates(prep_dir)
    if len(dates) != T:
        raise SystemExit(f"dates ({len(dates)}) != T ({T})")

    omega = np.fromfile(bin_path, dtype=np.float64)
    if omega.size != T * T:
        raise SystemExit(
            f"{bin_path}: {omega.size} values != T*T={T*T} "
            f"(wrong file or T mismatch)"
        )
    omega = omega.reshape(T, T)

    df = pd.DataFrame(omega, index=dates, columns=dates)
    df.to_parquet(parquet_path)
    return parquet_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="explicit output path (overrides the naming convention)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="directory for the canonically-named parquet "
                         "(default: same dir as --bin)")
    ap.add_argument("--kernel", choices=["linear", "rbf", "poly2"], default=None,
                    help="build a reference-convention filename for this kernel")
    ap.add_argument("--c", default=None,
                    help="c value string (required for rbf/poly2 canonical name)")
    ap.add_argument("--prep", type=Path, default=Path("data/prep"))
    ap.add_argument("--data-source", default=DEFAULT_DATA_SOURCE)
    ap.add_argument("--freq", default=DEFAULT_FREQ)
    args = ap.parse_args()

    if args.out is not None:
        out = args.out
    elif args.kernel is not None:
        meta = load_meta(args.prep)
        name = canonical_omega_name(args.kernel, args.c, meta,
                                    args.data_source, args.freq)
        out_dir = args.out_dir or args.bin.parent
        out = out_dir / name
    else:
        out = args.bin.with_suffix("").with_suffix(".parquet")

    bin_to_parquet(args.bin, out, args.prep)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
