"""CK-PCA data preparation.

Replicates the colab pipeline in reference/ck_pca_cuda_v_aug25.py and writes
flat-binary files in data/prep/ for the C++/CUDA Omega computation to consume.

Pipeline (in order):
  1. Read parquet (MultiIndex permno x date).
  2. Pop the 're' column as fwd-1m returns; drop 're','retadj' from chars.
  3. Subset characteristic columns to cz82.
  4. Date filter on raw chars: [sample_start, sample_end].
  5. drop_low_coverage_securities(max_missing=0.3): drop (permno,date) rows
     with more than 30% missing characteristics.
  6. compute_characteristic_signals: per-date rank/(count+1), demean, normalize
     by mean-abs, leverage-adjust (divide by sum of positives), divide by 2.
  7. asset_rets = fwd_rets_1m.reindex(chars.index)   -- NaNs preserved.
  8. chars.fillna(0)   -- characteristics only; rets untouched.

Output files in data/prep/:
  Z.f64.bin       row-major (total_rows x K) float64, characteristic signals
  r.f64.bin       (total_rows,) float64, 1m forward returns (may contain NaN)
  offsets.i64.bin (T+1,) int64, indptr: month t spans rows offsets[t]:offsets[t+1]
  dates.i64.bin   (T,) int64, datetime64[D] (days since 1970-01-01)
  meta.json       schema_version, T, K, total_rows, column names, parameters
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cz82 import CZ82

SCHEMA_VERSION = 1


def drop_low_coverage_securities(
    characteristics: pd.DataFrame, max_missing: float
) -> pd.DataFrame:
    if not (0 <= max_missing <= 1):
        raise ValueError(f"max_missing must be in [0,1], got {max_missing}")
    na_ratio = characteristics.isna().mean(axis=1)
    keep = na_ratio <= max_missing
    n_drop = int((~keep).sum())
    n_total = len(keep)
    print(
        f"[missing-filter] drop {n_drop}/{n_total} "
        f"({n_drop/n_total*100:.2f}%) rows with >{max_missing:.0%} missing"
    )
    return characteristics[keep].copy()


def _leverage_adjust(signal: pd.Series) -> pd.Series:
    return signal * (1.0 / signal[signal > 0].sum())


def compute_characteristic_signals(characteristics: pd.DataFrame) -> pd.DataFrame:
    """Per-date: rank/(count+1) -> demean -> /mean(|.|) -> /sum(positives) -> /2."""
    by_date = characteristics.groupby(level="date")
    ranked = by_date.rank(method="average") / (by_date.transform("count") + 1)
    cs_mean = ranked.groupby(level="date").transform("mean")
    signal = ranked - cs_mean
    signal = signal.groupby(level="date").transform(lambda x: x / x.abs().mean())
    signal = signal.groupby(level="date").transform(_leverage_adjust)
    return signal / 2.0


def prep(
    parquet_path: Path,
    out_dir: Path,
    sample_start: str = "1973-10-31",
    sample_end: str = "2024-12-31",
    max_missing: float = 0.3,
) -> dict:
    print(f"[prep] reading {parquet_path}")
    df = pd.read_parquet(parquet_path, engine="pyarrow")

    if df.index.nlevels != 2:
        raise ValueError(
            f"expected MultiIndex with 2 levels (permno,date), got "
            f"{df.index.nlevels} levels {df.index.names}"
        )
    # Reference uses level=1 / 'date' interchangeably; verify it matches.
    if df.index.names[1] != "date":
        raise ValueError(f"expected index.names[1]='date', got {df.index.names}")

    fwd_rets_1m = df["re"].copy()
    chars = df.drop(columns=["re", "retadj"])

    missing = [c for c in CZ82 if c not in chars.columns]
    if missing:
        raise ValueError(f"parquet is missing cz82 columns: {missing[:5]}...")
    chars = chars[CZ82]

    dates_lvl = chars.index.get_level_values("date")
    mask = (dates_lvl >= np.datetime64(sample_start)) & (
        dates_lvl <= np.datetime64(sample_end)
    )
    chars = chars.loc[mask]
    print(f"[prep] after date filter: {len(chars):,} rows")

    chars = drop_low_coverage_securities(chars, max_missing=max_missing)

    chars = compute_characteristic_signals(chars)

    asset_rets = fwd_rets_1m.reindex(chars.index)  # NaNs preserved

    chars = chars.fillna(0.0)

    # Sort by (date, permno) so per-month slices are contiguous and well-ordered.
    chars = chars.sort_index(level=["date", chars.index.names[0]])
    asset_rets = asset_rets.reindex(chars.index)

    # Per-date slicing: build offsets / dates / concatenated Z, r in date order.
    sample_dates = chars.index.get_level_values("date").unique().sort_values()
    T = len(sample_dates)
    K = len(CZ82)
    total_rows = len(chars)

    z_all = chars.to_numpy(dtype=np.float64, copy=False)
    r_all = asset_rets.to_numpy(dtype=np.float64, copy=False)
    if z_all.shape != (total_rows, K):
        raise AssertionError(f"Z shape {z_all.shape} != ({total_rows},{K})")
    if r_all.shape != (total_rows,):
        raise AssertionError(f"r shape {r_all.shape} != ({total_rows},)")

    # Group sizes per date (preserves the sorted-by-date layout).
    n_per_date = chars.groupby(level="date", sort=True).size().reindex(sample_dates)
    offsets = np.zeros(T + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(n_per_date.to_numpy(dtype=np.int64))
    if int(offsets[-1]) != total_rows:
        raise AssertionError(f"offsets[-1]={offsets[-1]} != total_rows={total_rows}")

    dates_i64 = sample_dates.to_numpy(dtype="datetime64[D]").astype(np.int64)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "Z.f64.bin").write_bytes(
        np.ascontiguousarray(z_all, dtype=np.float64).tobytes()
    )
    (out_dir / "r.f64.bin").write_bytes(
        np.ascontiguousarray(r_all, dtype=np.float64).tobytes()
    )
    (out_dir / "offsets.i64.bin").write_bytes(offsets.tobytes())
    (out_dir / "dates.i64.bin").write_bytes(dates_i64.tobytes())

    n_t = np.diff(offsets)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "kernel_set": "cz82",
        "T": int(T),
        "K": int(K),
        "total_rows": int(total_rows),
        "n_per_date_min": int(n_t.min()),
        "n_per_date_max": int(n_t.max()),
        "n_per_date_mean": float(n_t.mean()),
        "n_per_date_p50": int(np.median(n_t)),
        "dates_dtype": "datetime64[D]",
        "z_dtype": "float64",
        "r_dtype": "float64",
        "offsets_dtype": "int64",
        "endianness": "little",
        "kernel_cols": list(chars.columns),
        "n_rets_nan": int(np.isnan(r_all).sum()),
        "params": {
            "sample_start_date": sample_start,
            "sample_end_date": sample_end,
            "max_missing": max_missing,
        },
        "source_parquet": str(parquet_path),
        "actual_date_min": str(pd.Timestamp(sample_dates.min()).date()),
        "actual_date_max": str(pd.Timestamp(sample_dates.max()).date()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(
        f"[prep] wrote T={T}, K={K}, total_rows={total_rows:,}, "
        f"N_t in [{meta['n_per_date_min']}, {meta['n_per_date_max']}], "
        f"mean={meta['n_per_date_mean']:.0f}"
    )
    print(
        f"[prep] dates: {meta['actual_date_min']} .. {meta['actual_date_max']}, "
        f"r NaN count={meta['n_rets_nan']:,}"
    )
    print(f"[prep] outputs in {out_dir}")
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--parquet",
        type=Path,
        default=Path(
            "data/raw/osap_characteristics_with_fwd_1m_rets_20260109.parquet"
        ),
    )
    p.add_argument("--out", type=Path, default=Path("data/prep"))
    p.add_argument("--start", default="1973-10-31")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--max-missing", type=float, default=0.3)
    args = p.parse_args()
    prep(
        parquet_path=args.parquet,
        out_dir=args.out,
        sample_start=args.start,
        sample_end=args.end,
        max_missing=args.max_missing,
    )


if __name__ == "__main__":
    main()
