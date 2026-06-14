"""CK-PCA data preparation — Kozak DAILY setup.

Builds the same Z / r / offsets / dates / meta binaries as python/prep.py, but at
DAILY frequency: each period is a trading day, the cross-section is the stocks
trading that day, Z is that month's (pre-normalized) characteristic weights, and
r is the daily stock return.

The within-month weight convention is selected with --weight-drift:
  none    (default) hold the monthly weights constant through the month.
  buyhold monthly-rebalanced buy-and-hold (Kozak, p.19): the cross-section is
          frozen to the stocks present on each month's rebalance (first trading)
          day, and weights drift intra-month by each stock's cumulative gross
          return since the rebalance (no daily re-normalization).

Inputs (data/raw/):
  characteristics_anom.csv   monthly (permno, date='MM/YYYY', re, <54 anomaly
                             columns>). The anomaly columns are ALREADY rank-
                             transformed/normalized portfolio weights (Kozak),
                             so they are used directly -- no re-normalization.
  crsp_daily_*.parquet       daily (permno, date, ret, retadj, shrcd, exchcd, ...)

Alignment / conventions (all configurable):
  * Lag: characteristic row dated month M governs the daily returns of the SAME
    calendar month M (Kozak's char file is already lagged/tradeable). Override
    with --char-lag-months if your file uses a different convention.
  * Return: retadj (delisting-adjusted) preferred, falling back to ret.
  * Universe: common stocks (shrcd in 10,11) on NYSE/AMEX/NASDAQ (exchcd 1,2,3).
  * No risk-free needed: Omega is invariant to a common additive return shift
    because the double-centered kernel has zero row/column sums.
  * Missing characteristics -> 0 (a normalized, demeaned weight of 0 = no position).

Output (default data/prep_daily/): identical binary layout to python/prep.py, so
the existing loader and omega engines consume it unchanged (modulo the daily
memory footprint -- see README/notes on partner-tiling for omega_grid).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
META_COLS = ("permno", "date", "re")  # non-characteristic columns in the CSV

# The 40 anomaly characteristics used in the paper (Kozak et al. 2019 set). The
# characteristics_anom.csv carries 54 columns; selecting these 40 makes the run
# paper-consistent. Order is immaterial to Omega (kernels use dot products /
# pairwise distances over the whole set) but is kept as in the paper for clarity.
PAPER40 = [
    "size", "value", "prof", "fscore", "debtiss", "repurch", "nissa",
    "accruals", "growth", "aturnover", "gmargins", "ep", "cfp", "noa", "inv",
    "invcap", "igrowth", "sgrowth", "lev", "roaa", "roea", "sp", "mom",
    "indmom", "mom12", "momrev", "lrrev", "valuem", "nissm", "roe", "rome",
    "strev", "ivol", "betaarb", "season", "indrrev", "ciss", "price", "age",
    "shvol",
]


def load_characteristics(csv_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load monthly anomaly weights; return (df indexed by (permno, monthkey), cols)."""
    df = pd.read_csv(csv_path)
    char_cols = [c for c in df.columns if c not in META_COLS]
    # 'MM/YYYY' -> month key (year*12 + month-1), a single integer for joining.
    dt = pd.to_datetime(df["date"], format="%m/%Y")
    df["permno"] = df["permno"].astype(np.int64)
    df["monthkey"] = (dt.dt.year * 12 + (dt.dt.month - 1)).astype(np.int64)
    df[char_cols] = df[char_cols].fillna(0.0).astype(np.float64)
    df = df[["permno", "monthkey", *char_cols]].set_index(["permno", "monthkey"])
    # Guard against duplicate (permno, month) rows.
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df, char_cols


def load_daily_returns(parquet_path: Path, start: str, end: str,
                       return_col: str, universe_filter: bool) -> pd.DataFrame:
    cols = ["permno", "date", "ret", "retadj", "shrcd", "exchcd"]
    filters = [("date", ">=", pd.Timestamp(start)),
               ("date", "<=", pd.Timestamp(end))]
    df = pd.read_parquet(parquet_path, columns=cols, filters=filters,
                         engine="pyarrow")
    if universe_filter:
        df = df[df["shrcd"].isin((10, 11)) & df["exchcd"].isin((1, 2, 3))]
    # delisting-adjusted return preferred, fall back to raw ret
    ret = df["retadj"]
    if return_col == "retadj":
        ret = ret.where(ret.notna(), df["ret"])
    else:
        ret = df[return_col]
    out = pd.DataFrame({
        "permno": df["permno"].astype(np.int64).to_numpy(),
        "date": df["date"].to_numpy(),
        "r": ret.astype(np.float64).to_numpy(),
    })
    out = out[np.isfinite(out["r"].to_numpy())]
    return out


def apply_weight_drift(merged: pd.DataFrame, char_cols: list[str]) -> pd.DataFrame:
    """Convert constant within-month weights into a monthly-rebalanced
    buy-and-hold strategy (Kozak, p.19).

    Two effects, both keyed to each month's *rebalance day* (its first trading
    day in the sample):

      1. Membership freeze. A position is only established at a rebalance, so a
         month's cross-section is fixed to the stocks present on that month's
         rebalance day. Mid-month entrants are dropped (they wait for the next
         month-end rebalance to be assigned a weight); mid-month delistings just
         end their buy-and-hold path naturally.

      2. Weight drift. Holding fixed shares from the rebalance, the effective
         weight on stock i entering day t is its rebalance weight scaled by its
         own cumulative gross return since the rebalance:
             w_{i,t} = z_{i,M} * prod_{s=rebalance..t-1} (1 + r_{i,s}),
         with the product = 1 on the rebalance day. No daily re-normalization
         (re-normalizing each day would be daily rebalancing, not buy-and-hold).

    `merged` must carry permno, date, monthkey, r and the `char_cols`. Returns a
    new frame, restricted to month-start membership and with `char_cols` scaled
    by the drift factor, sorted (date, permno) for contiguous daily slices.
    """
    # 1. Membership: keep only (permno, monthkey) pairs present on the month's
    #    first trading day.
    first_dates = merged.groupby("monthkey")["date"].transform("min")
    is_first_day = merged["date"].to_numpy() == first_dates.to_numpy()
    member_idx = pd.MultiIndex.from_frame(
        merged.loc[is_first_day, ["permno", "monthkey"]])
    pair_idx = pd.MultiIndex.from_frame(merged[["permno", "monthkey"]])
    merged = merged[pair_idx.isin(member_idx)]

    # 2. Drift factor: per (permno, monthkey), cumprod(1+r) lagged one day so the
    #    rebalance day itself carries factor 1.0.
    merged = merged.sort_values(["permno", "monthkey", "date"], kind="stable")
    cum = (1.0 + merged["r"]).groupby(
        [merged["permno"], merged["monthkey"]], sort=False).cumprod()
    factor = cum.groupby([merged["permno"], merged["monthkey"]], sort=False) \
                .shift(1).fillna(1.0).to_numpy()
    merged[char_cols] = merged[char_cols].to_numpy() * factor[:, None]

    # Restore day-contiguous, permno-sorted order for the binary output.
    return merged.sort_values(["date", "permno"], kind="stable")


def prep_daily(chars_csv: Path, crsp_parquet: Path, out_dir: Path,
               sample_start: str, sample_end: str, char_lag_months: int,
               return_col: str, universe_filter: bool,
               weight_drift: str = "none",
               char_subset: list[str] | None = None) -> dict:
    print(f"[prep-daily] characteristics: {chars_csv}")
    chars, char_cols = load_characteristics(chars_csv)
    if char_subset is not None:
        missing = [c for c in char_subset if c not in char_cols]
        if missing:
            raise SystemExit(f"requested characteristics not in CSV: {missing}")
        char_cols = list(char_subset)
        chars = chars[char_cols]
        print(f"[prep-daily]   subset to {len(char_cols)} characteristics")
    K = len(char_cols)
    print(f"[prep-daily]   {len(chars):,} (permno,month) rows, K={K} anomalies")

    print(f"[prep-daily] daily returns: {crsp_parquet}")
    daily = load_daily_returns(crsp_parquet, sample_start, sample_end,
                               return_col, universe_filter)
    print(f"[prep-daily]   {len(daily):,} daily (permno,day) rows after filters")

    # Map each trading day to the characteristics month it should use.
    dt = pd.to_datetime(daily["date"])
    daily["monthkey"] = (dt.dt.year * 12 + (dt.dt.month - 1)).astype(np.int64) \
        - char_lag_months

    # Inner-join: keep (permno, day) rows that have characteristics that month.
    merged = daily.join(chars, on=["permno", "monthkey"], how="inner")
    print(f"[prep-daily]   {len(merged):,} rows after joining characteristics")
    if merged.empty:
        raise SystemExit("no overlap between characteristics and daily returns "
                         "-- check date ranges / lag convention")

    if weight_drift == "buyhold":
        rows_before = len(merged)
        merged = apply_weight_drift(merged, char_cols)
        print(f"[prep-daily]   buy-and-hold drift: {len(merged):,} rows after "
              f"month-start membership freeze (dropped {rows_before - len(merged):,} "
              f"mid-month entrants)")
    else:
        # Order so each day's cross-section is contiguous and permno-sorted.
        merged = merged.sort_values(["date", "permno"], kind="stable")

    sample_days = np.sort(merged["date"].unique())
    T = len(sample_days)
    total_rows = len(merged)

    z_all = np.ascontiguousarray(merged[char_cols].to_numpy(dtype=np.float64))
    r_all = np.ascontiguousarray(merged["r"].to_numpy(dtype=np.float64))

    n_per_day = merged.groupby("date", sort=True).size().to_numpy(dtype=np.int64)
    offsets = np.zeros(T + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(n_per_day)
    assert int(offsets[-1]) == total_rows, (offsets[-1], total_rows)

    dates_i64 = sample_days.astype("datetime64[D]").astype(np.int64)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "Z.f64.bin").write_bytes(z_all.tobytes())
    (out_dir / "r.f64.bin").write_bytes(r_all.tobytes())
    (out_dir / "offsets.i64.bin").write_bytes(offsets.tobytes())
    (out_dir / "dates.i64.bin").write_bytes(dates_i64.tobytes())

    nan_r = int(np.isnan(r_all).sum())
    meta = {
        "schema_version": SCHEMA_VERSION,
        "kernel_set": "anom",
        "freq": "D",
        "T": int(T),
        "K": int(K),
        "total_rows": int(total_rows),
        "n_per_date_min": int(n_per_day.min()),
        "n_per_date_max": int(n_per_day.max()),
        "n_per_date_mean": float(n_per_day.mean()),
        "n_per_date_p50": int(np.median(n_per_day)),
        "dates_dtype": "datetime64[D]",
        "z_dtype": "float64",
        "r_dtype": "float64",
        "offsets_dtype": "int64",
        "endianness": "little",
        "kernel_cols": char_cols,
        "n_rets_nan": nan_r,
        "params": {
            "sample_start_date": sample_start,
            "sample_end_date": sample_end,
            "char_lag_months": char_lag_months,
            "return_col": return_col,
            "universe_filter": universe_filter,
            "weight_drift": weight_drift,
            "char_subset": list(char_subset) if char_subset is not None else "all",
        },
        "data_source": "CRSP",
        "char_source": str(chars_csv),
        "crsp_source": str(crsp_parquet),
        "actual_date_min": str(pd.Timestamp(sample_days.min()).date()),
        "actual_date_max": str(pd.Timestamp(sample_days.max()).date()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[prep-daily] wrote T={T:,} days, K={K}, total_rows={total_rows:,}, "
          f"N/day in [{meta['n_per_date_min']}, {meta['n_per_date_max']}], "
          f"mean={meta['n_per_date_mean']:.0f}")
    print(f"[prep-daily] dates: {meta['actual_date_min']} .. {meta['actual_date_max']}")
    print(f"[prep-daily] outputs in {out_dir}")
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chars", type=Path,
                   default=Path("data/raw/characteristics_anom.csv"))
    p.add_argument("--crsp", type=Path,
                   default=Path("data/raw/crsp_daily_19691231_20241231.parquet"))
    p.add_argument("--out", type=Path, default=Path("data/prep_daily"))
    p.add_argument("--start", default="1973-10-31")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--char-lag-months", type=int, default=0,
                   help="months to lag characteristics relative to the return "
                        "month (0 = same month; Kozak's file is already lagged)")
    p.add_argument("--return-col", default="retadj", choices=["retadj", "ret"])
    p.add_argument("--no-universe-filter", action="store_true",
                   help="skip the shrcd/exchcd common-stock filter")
    p.add_argument("--weight-drift", default="none", choices=["none", "buyhold"],
                   help="none = hold monthly weights constant within the month; "
                        "buyhold = monthly-rebalanced buy-and-hold drift with "
                        "month-start membership freeze (Kozak, p.19)")
    p.add_argument("--char-set", default="all", choices=["all", "paper40"],
                   help="all = every characteristic column in the CSV; "
                        "paper40 = the 40-characteristic Kozak et al. (2019) set")
    p.add_argument("--char-list", default=None,
                   help="comma-separated custom characteristic subset "
                        "(overrides --char-set)")
    args = p.parse_args()

    if args.char_list:
        char_subset = [c.strip() for c in args.char_list.split(",") if c.strip()]
    elif args.char_set == "paper40":
        char_subset = PAPER40
    else:
        char_subset = None
    prep_daily(
        chars_csv=args.chars,
        crsp_parquet=args.crsp,
        out_dir=args.out,
        sample_start=args.start,
        sample_end=args.end,
        char_lag_months=args.char_lag_months,
        return_col=args.return_col,
        universe_filter=not args.no_universe_filter,
        weight_drift=args.weight_drift,
        char_subset=char_subset,
    )


if __name__ == "__main__":
    main()
