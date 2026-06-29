#!/usr/bin/env python3
"""Grid-search dataset filters and score each dataset by return-variance metrics.

This script does not use pipeline dataset builders. For each parameter combo it:
1) Filters rows from `really_good_tickers/*.csv`.
2) Writes the filtered ticker CSVs into a temporary dataset directory.
3) Computes close-to-close return variance metrics.
4) Appends one result row to the output CSV.
5) Deletes the temporary dataset directory.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd

RETURNS_CLIP = 2.5
# If True, compute bucket medians from 3m_rolling_median_dollar_volume and
# keep/drop by bucket-level threshold checks. If False, apply row-level
# threshold filtering.
FILTER_VIA_BUCKETS = True
# When FILTER_VIA_BUCKETS is True:
# - True: one bucket per ticker (entire date-filtered ticker).
# - False: split each ticker into sequential 3-month windows (anchored at the
#   first date-filtered row), and keep/drop each window independently.
ENTIRE_TICKER_BUCKET = False
DEFAULT_MAX_WORKERS = 45
MAX_ALLOWED_WORKERS = 45
SCORE_ALPHAS = (0.3, 0.5, 0.7)
SCORE_BETA = 1.0
REQUIRED_PARAM_KEYS = ("upper_threshold", "lower_threshold", "year")
RESULT_FIELDNAMES = [
    "combo_id",
    "upper_threshold",
    "lower_threshold",
    "year",
    "dataset_dir",
    "row_count_n",
    "per_row_returns_variance_s2",
    "mean_per_date_returns_variance",
    "variance_of_per_date_returns_variance",
    "mean_per_ticker_returns_variance",
    "variance_of_per_ticker_returns_variance",
    "num_tickers_with_variance",
    "num_dates_with_variance",
    "dataset_score_alpha_0.3",
    "dataset_score_alpha_0.5",
    "dataset_score_alpha_0.7",
    "error",
]


@dataclass(frozen=True)
class ParamCombo:
    combo_id: int
    upper_threshold: float
    lower_threshold: float
    year: int


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        if not np.isfinite(value):
            return
        self.count += 1
        delta = value - self.mean
        self.mean += delta / float(self.count)
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def variance(self) -> float:
        if self.count <= 0:
            return float("nan")
        return self.m2 / float(self.count)


def assign_three_month_bucket_ids(date_series: pd.Series) -> pd.Series:
    """Assign sequential 3-month window bucket IDs from the first valid row."""
    parsed = pd.to_datetime(date_series, errors="coerce")
    out = np.full(shape=(len(parsed),), fill_value=-1, dtype=np.int64)

    valid_positions = np.where(pd.notna(parsed.to_numpy()))[0]
    if valid_positions.size == 0:
        return pd.Series(out, index=date_series.index, dtype=np.int64)

    first_pos = int(valid_positions[0])
    first_date = pd.Timestamp(parsed.iloc[first_pos])
    bucket_end = first_date + pd.DateOffset(months=3)
    bucket_id = 0

    for pos in valid_positions:
        ts = pd.Timestamp(parsed.iloc[int(pos)])
        while ts >= bucket_end:
            bucket_end = bucket_end + pd.DateOffset(months=3)
            bucket_id += 1
        out[int(pos)] = int(bucket_id)

    return pd.Series(out, index=date_series.index, dtype=np.int64)


def parse_threshold_value(raw_value: str) -> float:
    token = str(raw_value).strip().lower().replace(",", "").replace("_", "")
    if not token:
        raise ValueError("empty threshold token")
    suffix_mult = 1.0
    if token.endswith("k"):
        suffix_mult = 1_000.0
        token = token[:-1]
    elif token.endswith("m"):
        suffix_mult = 1_000_000.0
        token = token[:-1]
    elif token.endswith("b"):
        suffix_mult = 1_000_000_000.0
        token = token[:-1]

    if not token:
        raise ValueError(f"invalid threshold value: {raw_value!r}")
    value = float(token) * suffix_mult
    if not np.isfinite(value):
        raise ValueError(f"non-finite threshold value: {raw_value!r}")
    return float(value)


def parse_year_value(raw_value: str) -> int:
    token = str(raw_value).strip()
    if not token:
        raise ValueError("empty year token")
    year = int(token)
    if year < 1900 or year > 2100:
        raise ValueError(f"year out of expected range: {year}")
    return year


def parse_param_list(raw_values: str) -> List[str]:
    value = str(raw_values).strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1].strip()
    if not value:
        return []
    out: List[str] = []
    for token in value.split(","):
        cleaned = token.strip()
        if not cleaned:
            continue
        # Ignore common placeholder ellipsis tokens.
        if cleaned in {"...", "…", ".."}:
            continue
        out.append(cleaned)
    return out


def parse_params_file(path: Path) -> Dict[str, List[str]]:
    params: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"invalid line {line_no} in {path}: missing ':'")
            key, raw_values = line.split(":", 1)
            key_clean = key.strip()
            if key_clean not in REQUIRED_PARAM_KEYS:
                raise ValueError(
                    f"invalid key {key_clean!r} on line {line_no}; "
                    f"expected one of {REQUIRED_PARAM_KEYS}"
                )
            params[key_clean] = parse_param_list(raw_values)

    missing = [k for k in REQUIRED_PARAM_KEYS if k not in params]
    if missing:
        raise ValueError(f"missing required keys in {path}: {missing}")
    for key in REQUIRED_PARAM_KEYS:
        if not params[key]:
            raise ValueError(f"no values provided for key {key!r} in {path}")
    return params


def build_combos(raw_params: Dict[str, List[str]]) -> List[ParamCombo]:
    uppers = [parse_threshold_value(v) for v in raw_params["upper_threshold"]]
    lowers = [parse_threshold_value(v) for v in raw_params["lower_threshold"]]
    years = [parse_year_value(v) for v in raw_params["year"]]

    combos: List[ParamCombo] = []
    combo_id = 0
    for upper, lower, year in itertools.product(uppers, lowers, years):
        combos.append(
            ParamCombo(
                combo_id=combo_id,
                upper_threshold=float(upper),
                lower_threshold=float(lower),
                year=int(year),
            )
        )
        combo_id += 1
    return combos


def safe_combo_name(combo: ParamCombo) -> str:
    upper = re.sub(r"[^0-9A-Za-z_.-]+", "_", f"{combo.upper_threshold:.6g}")
    lower = re.sub(r"[^0-9A-Za-z_.-]+", "_", f"{combo.lower_threshold:.6g}")
    return f"dataset_combo_{combo.combo_id:06d}_u{upper}_l{lower}_y{combo.year}"


def compute_scores(s2: float, n_rows: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if (not np.isfinite(s2)) or n_rows <= 0:
        for alpha in SCORE_ALPHAS:
            out[f"dataset_score_alpha_{alpha}"] = float("nan")
        return out
    s2_term = float(s2) ** float(SCORE_BETA)
    for alpha in SCORE_ALPHAS:
        out[f"dataset_score_alpha_{alpha}"] = s2_term * (float(n_rows) ** float(alpha))
    return out


def _nan_result_row(combo: ParamCombo, dataset_dir: Path, error: str = "") -> Dict[str, object]:
    base = {
        "combo_id": combo.combo_id,
        "upper_threshold": combo.upper_threshold,
        "lower_threshold": combo.lower_threshold,
        "year": combo.year,
        "dataset_dir": str(dataset_dir),
        "row_count_n": 0,
        "per_row_returns_variance_s2": float("nan"),
        "mean_per_date_returns_variance": float("nan"),
        "variance_of_per_date_returns_variance": float("nan"),
        "mean_per_ticker_returns_variance": float("nan"),
        "variance_of_per_ticker_returns_variance": float("nan"),
        "num_tickers_with_variance": 0,
        "num_dates_with_variance": 0,
        "error": error,
    }
    base.update(compute_scores(s2=float("nan"), n_rows=0))
    return base


def process_combo(
    combo: ParamCombo,
    ticker_csv_paths: Sequence[Path],
    base_dataset_dir: Path,
    progress_every_tickers: int = 0,
    progress_logger: Callable[[str], None] | None = None,
) -> Dict[str, object]:
    def _log(msg: str) -> None:
        if progress_logger is not None:
            progress_logger(msg)

    combo_dir = base_dataset_dir / safe_combo_name(combo)
    result_row: Dict[str, object] = _nan_result_row(combo, combo_dir)
    start_date = f"{combo.year:04d}-01-01"

    if combo.lower_threshold > combo.upper_threshold:
        result_row["error"] = "lower_threshold is greater than upper_threshold"
        return result_row

    overall_return_stats = RunningStats()
    date_stats: Dict[str, RunningStats] = {}
    ticker_variance_stats = RunningStats()
    tickers_with_variance = 0
    tickers_written = 0
    buckets_kept = 0
    buckets_dropped = 0

    try:
        combo_dir = Path(
            tempfile.mkdtemp(
                prefix=f"{safe_combo_name(combo)}_",
                dir=str(base_dataset_dir),
            )
        )
        result_row["dataset_dir"] = str(combo_dir)
        _log(
            f"dataset_dir={combo_dir} scan_start tickers_total={len(ticker_csv_paths)} "
            f"date_gte={start_date} filter_via_buckets={FILTER_VIA_BUCKETS} "
            f"entire_ticker_bucket={ENTIRE_TICKER_BUCKET}"
        )

        for ticker_idx, ticker_path in enumerate(ticker_csv_paths, start=1):
            df = pd.read_csv(ticker_path)
            if df.empty:
                if progress_every_tickers > 0 and ticker_idx % progress_every_tickers == 0:
                    _log(
                        f"scan_progress {ticker_idx}/{len(ticker_csv_paths)} "
                        f"kept_tickers={tickers_written} returns={overall_return_stats.count}"
                    )
                continue

            # Keep the original schema in written dataset files.
            if "date" not in df.columns or "close" not in df.columns:
                if progress_every_tickers > 0 and ticker_idx % progress_every_tickers == 0:
                    _log(
                        f"scan_progress {ticker_idx}/{len(ticker_csv_paths)} "
                        f"kept_tickers={tickers_written} returns={overall_return_stats.count}"
                    )
                continue
            if "3m_rolling_median_dollar_volume" not in df.columns:
                if progress_every_tickers > 0 and ticker_idx % progress_every_tickers == 0:
                    _log(
                        f"scan_progress {ticker_idx}/{len(ticker_csv_paths)} "
                        f"kept_tickers={tickers_written} returns={overall_return_stats.count}"
                    )
                continue

            volume_med = pd.to_numeric(
                df["3m_rolling_median_dollar_volume"], errors="coerce"
            )
            date_str = df["date"].astype(str)
            date_mask = date_str.ge(start_date)
            date_filtered_df = df.loc[date_mask].copy()

            if FILTER_VIA_BUCKETS:
                if date_filtered_df.empty:
                    if progress_every_tickers > 0 and ticker_idx % progress_every_tickers == 0:
                        _log(
                            f"scan_progress {ticker_idx}/{len(ticker_csv_paths)} "
                            f"kept_tickers={tickers_written} dropped_buckets={buckets_dropped} "
                            f"returns={overall_return_stats.count}"
                        )
                    continue

                if ENTIRE_TICKER_BUCKET:
                    bucket_series = pd.to_numeric(
                        date_filtered_df["3m_rolling_median_dollar_volume"], errors="coerce"
                    )
                    bucket_median = float(bucket_series.median(skipna=True))
                    if (
                        (not np.isfinite(bucket_median))
                        or bucket_median < combo.lower_threshold
                        or bucket_median > combo.upper_threshold
                    ):
                        buckets_dropped += 1
                        if progress_every_tickers > 0 and ticker_idx % progress_every_tickers == 0:
                            _log(
                                f"scan_progress {ticker_idx}/{len(ticker_csv_paths)} "
                                f"kept_tickers={tickers_written} dropped_buckets={buckets_dropped} "
                                f"returns={overall_return_stats.count}"
                            )
                        continue
                    filtered_df = date_filtered_df
                    buckets_kept += 1
                else:
                    bucket_ids = assign_three_month_bucket_ids(date_filtered_df["date"])
                    staged = date_filtered_df.copy()
                    staged["_bucket_id"] = bucket_ids

                    kept_parts: List[pd.DataFrame] = []
                    for bucket_id, bucket_df in staged.groupby("_bucket_id", sort=True):
                        if int(bucket_id) < 0:
                            continue
                        bucket_series = pd.to_numeric(
                            bucket_df["3m_rolling_median_dollar_volume"], errors="coerce"
                        )
                        bucket_median = float(bucket_series.median(skipna=True))
                        if (
                            np.isfinite(bucket_median)
                            and combo.lower_threshold <= bucket_median <= combo.upper_threshold
                        ):
                            kept_parts.append(bucket_df.drop(columns=["_bucket_id"]))
                            buckets_kept += 1
                        else:
                            buckets_dropped += 1

                    if kept_parts:
                        filtered_df = pd.concat(kept_parts, axis=0).sort_index()
                    else:
                        filtered_df = date_filtered_df.iloc[0:0]
            else:
                mask = (
                    volume_med.ge(combo.lower_threshold)
                    & volume_med.le(combo.upper_threshold)
                    & date_mask
                )
                filtered_df = df.loc[mask].copy()
            if filtered_df.empty:
                if progress_every_tickers > 0 and ticker_idx % progress_every_tickers == 0:
                    _log(
                        f"scan_progress {ticker_idx}/{len(ticker_csv_paths)} "
                        f"kept_tickers={tickers_written} dropped_buckets={buckets_dropped} "
                        f"returns={overall_return_stats.count}"
                    )
                continue

            # Persist filtered ticker rows for this temporary dataset.
            output_csv_path = combo_dir / ticker_path.name
            filtered_df.to_csv(output_csv_path, index=False)
            tickers_written += 1

            close_series = pd.to_numeric(filtered_df["close"], errors="coerce")
            close_values = close_series.to_numpy(dtype=np.float64)
            if close_values.shape[0] < 2:
                if progress_every_tickers > 0 and ticker_idx % progress_every_tickers == 0:
                    _log(
                        f"scan_progress {ticker_idx}/{len(ticker_csv_paths)} "
                        f"kept_tickers={tickers_written} dropped_buckets={buckets_dropped} "
                        f"returns={overall_return_stats.count}"
                    )
                continue

            prev_close = close_values[:-1]
            curr_close = close_values[1:]
            with np.errstate(divide="ignore", invalid="ignore"):
                returns = (curr_close / prev_close) - 1.0

            # Return date is the current row's date in close-to-close return.
            ret_dates = filtered_df["date"].astype(str).to_numpy()[1:]
            valid_mask = np.isfinite(returns)
            valid_mask &= returns <= float(RETURNS_CLIP)  # Exactly as requested.
            if not np.any(valid_mask):
                if progress_every_tickers > 0 and ticker_idx % progress_every_tickers == 0:
                    _log(
                        f"scan_progress {ticker_idx}/{len(ticker_csv_paths)} "
                        f"kept_tickers={tickers_written} dropped_buckets={buckets_dropped} "
                        f"returns={overall_return_stats.count}"
                    )
                continue

            returns = returns[valid_mask]
            ret_dates = ret_dates[valid_mask]

            # Update global row-return variance.
            for ret_val in returns:
                overall_return_stats.update(float(ret_val))

            # Update per-date running stats.
            for d, ret_val in zip(ret_dates, returns):
                stats = date_stats.get(d)
                if stats is None:
                    stats = RunningStats()
                    date_stats[d] = stats
                stats.update(float(ret_val))

            # Update per-ticker variance summary.
            if returns.shape[0] >= 2:
                ticker_var = float(np.var(returns, ddof=0))
                if np.isfinite(ticker_var):
                    ticker_variance_stats.update(ticker_var)
                    tickers_with_variance += 1

            if progress_every_tickers > 0 and ticker_idx % progress_every_tickers == 0:
                _log(
                    f"scan_progress {ticker_idx}/{len(ticker_csv_paths)} "
                    f"kept_tickers={tickers_written} dropped_buckets={buckets_dropped} "
                    f"returns={overall_return_stats.count}"
                )

        per_date_variance_stats = RunningStats()
        for stats in date_stats.values():
            if stats.count >= 2:
                date_var = stats.variance()
                if np.isfinite(date_var):
                    per_date_variance_stats.update(date_var)

        s2 = overall_return_stats.variance()
        row_count_n = int(overall_return_stats.count)
        date_var_mean = per_date_variance_stats.mean if per_date_variance_stats.count > 0 else float("nan")
        date_var_var = per_date_variance_stats.variance()
        ticker_var_mean = ticker_variance_stats.mean if ticker_variance_stats.count > 0 else float("nan")
        ticker_var_var = ticker_variance_stats.variance()
        scores = compute_scores(s2=s2, n_rows=row_count_n)

        result_row = {
            "combo_id": combo.combo_id,
            "upper_threshold": combo.upper_threshold,
            "lower_threshold": combo.lower_threshold,
            "year": combo.year,
            "dataset_dir": str(combo_dir),
            "row_count_n": row_count_n,
            "per_row_returns_variance_s2": s2,
            "mean_per_date_returns_variance": date_var_mean,
            "variance_of_per_date_returns_variance": date_var_var,
            "mean_per_ticker_returns_variance": ticker_var_mean,
            "variance_of_per_ticker_returns_variance": ticker_var_var,
            "num_tickers_with_variance": int(tickers_with_variance),
            "num_dates_with_variance": int(per_date_variance_stats.count),
            "error": "",
            **scores,
        }
        if row_count_n == 0:
            # Keep explicit 0-row behavior even without runtime errors.
            result_row["error"] = "no valid returns after filtering and clipping"
        _log(
            f"scan_done kept_tickers={tickers_written} "
            f"kept_buckets={buckets_kept} dropped_buckets={buckets_dropped} "
            f"returns={row_count_n} per_row_var={result_row['per_row_returns_variance_s2']}"
        )
        return result_row

    except Exception as exc:  # noqa: BLE001
        result_row["error"] = f"{type(exc).__name__}: {exc}"
        _log(f"scan_error {result_row['error']}")
        return result_row
    finally:
        shutil.rmtree(combo_dir, ignore_errors=True)


def run_grid_search(
    params_file: Path,
    source_dir: Path,
    output_csv: Path,
    base_dataset_dir: Path,
    max_workers: int,
) -> None:
    raw_params = parse_params_file(params_file)
    combos = build_combos(raw_params)
    if not combos:
        raise ValueError("no parameter combinations generated")

    ticker_csv_paths = sorted(
        p for p in Path(source_dir).glob("*.csv") if p.is_file() and not p.name.startswith("_")
    )
    if not ticker_csv_paths:
        raise ValueError(f"no ticker CSV files found in source_dir: {source_dir}")

    base_dataset_dir.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    worker_count = max(1, min(int(max_workers), MAX_ALLOWED_WORKERS, len(combos)))

    total = len(combos)
    completed = 0
    print(
        f"Starting grid search: combos={total}, ticker_files={len(ticker_csv_paths)}, "
        f"workers={worker_count}, returns_clip={RETURNS_CLIP}, "
        f"filter_via_buckets={FILTER_VIA_BUCKETS}, "
        f"entire_ticker_bucket={ENTIRE_TICKER_BUCKET}"
    )

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    process_combo,
                    combo,
                    ticker_csv_paths,
                    base_dataset_dir,
                ): combo
                for combo in combos
            }

            for future in as_completed(future_map):
                combo = future_map[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = _nan_result_row(
                        combo=combo,
                        dataset_dir=base_dataset_dir / safe_combo_name(combo),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                writer.writerow(row)
                f.flush()

                completed += 1
                print(
                    f"[{completed}/{total}] combo_id={combo.combo_id} "
                    f"year={combo.year} lower={combo.lower_threshold:.6g} "
                    f"upper={combo.upper_threshold:.6g} error={bool(row.get('error'))}"
                )

    print(f"Done. Results written to: {output_csv}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Grid-search dataset filters over really_good_tickers, compute variance "
            "metrics, and write results to CSV."
        )
    )
    parser.add_argument(
        "--params-file",
        type=Path,
        default=Path("dataset_params.txt"),
        help="Path to dataset_params.txt",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("really_good_tickers"),
        help="Directory containing source ticker CSVs",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("dataset_grid_search_results.csv"),
        help="Output CSV for grid-search results",
    )
    parser.add_argument(
        "--base-dataset-dir",
        type=Path,
        default=Path("/ephemeral/dirs"),
        help="Temporary root for per-combo dataset directories",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Max concurrent workers (capped at {MAX_ALLOWED_WORKERS})",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_grid_search(
        params_file=Path(args.params_file),
        source_dir=Path(args.source_dir),
        output_csv=Path(args.output_csv),
        base_dataset_dir=Path(args.base_dataset_dir),
        max_workers=int(args.max_workers),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
