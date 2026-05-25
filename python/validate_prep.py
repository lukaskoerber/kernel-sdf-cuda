"""Validate data/prep/ binaries against a fresh in-memory run of the prep pipeline.

For each of N sampled months we:
  1. Decode (Z_t, r_t) directly from the flat-binary files using meta.json offsets.
  2. Materialize the same arrays in memory via the reference-style API
     (`characteristics.xs(date, level='date').to_numpy()`), starting from the
     parquet and re-running prep.py's pipeline.
  3. Assert array_equal (bit-identical) and report any deviation.

Also cross-checks the date index against the test omega's index so we know the
binaries line up with the validation gate that comes in step 4.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prep import compute_characteristic_signals
from cz82 import CZ82


def _load_bins(prep_dir: Path):
    meta = json.loads((prep_dir / "meta.json").read_text())
    T = meta["T"]
    K = meta["K"]
    total_rows = meta["total_rows"]

    Z = np.fromfile(prep_dir / "Z.f64.bin", dtype=np.float64).reshape(total_rows, K)
    r = np.fromfile(prep_dir / "r.f64.bin", dtype=np.float64)
    offsets = np.fromfile(prep_dir / "offsets.i64.bin", dtype=np.int64)
    dates_i64 = np.fromfile(prep_dir / "dates.i64.bin", dtype=np.int64)

    if r.shape != (total_rows,):
        raise AssertionError(f"r shape {r.shape} != ({total_rows},)")
    if offsets.shape != (T + 1,):
        raise AssertionError(f"offsets shape {offsets.shape} != ({T + 1},)")
    if dates_i64.shape != (T,):
        raise AssertionError(f"dates shape {dates_i64.shape} != ({T},)")
    if int(offsets[-1]) != total_rows:
        raise AssertionError(
            f"offsets[-1]={offsets[-1]} != total_rows={total_rows}"
        )
    if list(meta["kernel_cols"]) != list(CZ82):
        raise AssertionError("meta.kernel_cols != CZ82")
    dates = dates_i64.astype("datetime64[D]")
    return meta, Z, r, offsets, dates


def _run_prep_inmemory(parquet: Path, sample_start: str, sample_end: str):
    df = pd.read_parquet(parquet, engine="pyarrow")
    fwd = df["re"].copy()
    chars = df.drop(columns=["re", "retadj"])[CZ82]
    d = chars.index.get_level_values("date")
    chars = chars.loc[
        (d >= np.datetime64(sample_start)) & (d <= np.datetime64(sample_end))
    ]
    chars = compute_characteristic_signals(chars)
    rets = fwd.reindex(chars.index)
    chars = chars.fillna(0.0)
    chars = chars.sort_index(level=["date", chars.index.names[0]])
    rets = rets.reindex(chars.index)
    return chars, rets


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prep", type=Path, default=Path("data/prep"))
    p.add_argument(
        "--parquet",
        type=Path,
        default=Path(
            "data/raw/osap_characteristics_with_fwd_1m_rets_20260109.parquet"
        ),
    )
    p.add_argument(
        "--omega-ref",
        type=Path,
        default=Path(
            "data/test/omega_linear_1973-10-31_2024-11-30_cz82_OSAP_M.parquet"
        ),
    )
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    meta, Z, r, offsets, dates = _load_bins(args.prep)
    print(
        f"[validate] bins: T={meta['T']}, K={meta['K']}, "
        f"total_rows={meta['total_rows']:,}, "
        f"dates [{meta['actual_date_min']} .. {meta['actual_date_max']}]"
    )

    # 1. Cross-check dates against the linear test omega's index.
    omega_ref = pd.read_parquet(args.omega_ref)
    ref_dates = np.array(
        [pd.Timestamp(d).to_datetime64() for d in omega_ref.index],
        dtype="datetime64[D]",
    )
    if dates.shape != ref_dates.shape or not np.array_equal(dates, ref_dates):
        n_diff = int((dates != ref_dates).sum()) if dates.shape == ref_dates.shape else -1
        raise AssertionError(
            f"prep dates do not match test omega index "
            f"(T_prep={len(dates)}, T_ref={len(ref_dates)}, n_diff={n_diff})"
        )
    print(f"[validate] dates match test omega index ({len(dates)} months) ✓")

    # 2. Per-month round-trip vs in-memory pandas.
    print("[validate] re-running prep in memory to compare a few months...")
    chars, rets = _run_prep_inmemory(
        args.parquet,
        sample_start=meta["params"]["sample_start_date"],
        sample_end=meta["params"]["sample_end_date"],
    )

    if len(chars) != meta["total_rows"]:
        raise AssertionError(
            f"in-memory total_rows={len(chars)} != bin total_rows={meta['total_rows']}"
        )

    rng = np.random.default_rng(args.seed)
    n = min(args.n_samples, len(dates))
    sample_idx = sorted(set([0, len(dates) - 1] + rng.choice(len(dates), n, replace=False).tolist()))
    print(f"[validate] sampling t indices: {sample_idx}")

    all_ok = True
    for t in sample_idx:
        date = pd.Timestamp(dates[t])
        s, e = int(offsets[t]), int(offsets[t + 1])
        Z_bin = Z[s:e]
        r_bin = r[s:e]

        chars_t = chars.xs(date, level="date")
        rets_t = rets.xs(date, level="date")
        Z_mem = chars_t.to_numpy(dtype=np.float64)
        r_mem = rets_t.to_numpy(dtype=np.float64)

        z_eq = np.array_equal(Z_bin, Z_mem, equal_nan=True)
        r_eq = np.array_equal(r_bin, r_mem, equal_nan=True)
        z_max_err = float(np.abs(Z_bin - Z_mem).max()) if Z_bin.shape == Z_mem.shape else float("nan")
        r_max_err = float(np.abs(r_bin - r_mem).max()) if r_bin.shape == r_mem.shape else float("nan")
        tag = "OK " if (z_eq and r_eq) else "FAIL"
        print(
            f"  [{tag}] t={t:>4} date={date.date()} N_t={e - s:>5} "
            f"Z eq={z_eq} max|d|={z_max_err:.3e} | r eq={r_eq} max|d|={r_max_err:.3e}"
        )
        if not (z_eq and r_eq):
            all_ok = False

    print()
    print("[validate] PASS" if all_ok else "[validate] FAIL")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
