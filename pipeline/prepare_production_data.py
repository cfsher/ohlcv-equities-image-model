#!/usr/bin/env python3
"""Prepare production ticker data, datasets, and OHLC image shards."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

import fetch_tickers as fetch_daily
import prepare_daily_data as prepare_daily

RUSSELL_3000_HOLDINGS_URLS = (
    "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund",
    "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings",
)

DEFAULT_TICKERS_DIR = Path("production_tickers")
DEFAULT_DATASETS_DIR = Path("production_datasets")
DEFAULT_DATASET_NAME = "production_dataset"
DEFAULT_UNIVERSE_DIR = Path("really_good_tickers")
DEFAULT_PRED_TICKERS_FILE = Path("tickers/_union_nasdaq_nyse_russell_3000.txt")
DEFAULT_HISTORY_DAYS = 70
DEFAULT_RATE_LIMIT_RETRIES = 6
DEFAULT_RATE_LIMIT_RETRY_SLEEP_SECONDS = 30
DEFAULT_HORIZON = int(prepare_daily.DEFAULT_HORIZON)
DEFAULT_HISTORY_BUFFER_PCT = 0.20
DEFAULT_HISTORY_BUFFER_MIN = 15


def normalize_ticker(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def dedupe_keep_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        key = normalize_ticker(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def parse_ishares_holdings_csv(text: str) -> List[str]:
    lines = text.splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        token = line.split(",", 1)[0].strip().strip('"').lstrip("\ufeff")
        if token.lower() == "ticker":
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError("could not find 'Ticker' header in holdings CSV")

    reader = csv.DictReader(lines[header_idx:])
    if not reader.fieldnames or "Ticker" not in reader.fieldnames:
        raise ValueError("holdings CSV does not include a Ticker column")

    tickers: List[str] = []
    for row in reader:
        ticker_raw = str(row.get("Ticker", "")).strip()
        if not ticker_raw:
            continue
        asset_class = str(row.get("Asset Class", "")).strip().lower()
        if asset_class and asset_class != "equity":
            continue
        if not all(ch.isalnum() or ch in ".-" for ch in ticker_raw):
            continue
        ticker = normalize_ticker(ticker_raw)
        if ticker in {"-", "N/A", "USD", "CASH"}:
            continue
        tickers.append(ticker)

    tickers = dedupe_keep_order(tickers)
    if not tickers:
        raise ValueError("no equity tickers were parsed from holdings CSV")
    return tickers


def download_text(url: str, timeout_seconds: int) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=int(timeout_seconds)) as response:
        content = response.read()
        encoding = response.headers.get_content_charset() or "utf-8"
    return content.decode(encoding, errors="replace")


def resolve_russell_3000_tickers(
    tickers_file: Optional[Path],
    timeout_seconds: int,
) -> Tuple[List[str], str]:
    if tickers_file is not None:
        tickers = fetch_daily.read_local_sp500_tickers(Path(tickers_file))
        tickers = dedupe_keep_order(tickers)
        if not tickers:
            raise ValueError(f"no tickers found in {tickers_file}")
        return tickers, f"file:{tickers_file}"

    failures: List[str] = []
    for url in RUSSELL_3000_HOLDINGS_URLS:
        try:
            text = download_text(url=url, timeout_seconds=timeout_seconds)
            tickers = parse_ishares_holdings_csv(text)
            return tickers, url
        except Exception as exc:
            failures.append(f"{url} -> {exc}")

    joined = "\n".join(failures)
    raise RuntimeError(
        "failed to resolve Russell 3000 tickers from iShares URLs.\n"
        "Provide --tickers-file to run with a local list.\n"
        f"Details:\n{joined}"
    )


def resolve_ticker_universe_from_csv_dir(universe_dir: Path) -> Tuple[List[str], str]:
    source_dir = Path(universe_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"universe_dir not found: {source_dir}")
    if not source_dir.is_dir():
        raise ValueError(f"universe_dir must be a directory: {source_dir}")

    csv_paths = sorted(
        p for p in source_dir.glob("*.csv") if p.is_file() and not p.name.startswith("_")
    )
    tickers = dedupe_keep_order([normalize_ticker(p.stem) for p in csv_paths])
    if not tickers:
        raise ValueError(f"no ticker CSV files found in universe_dir: {source_dir}")
    return tickers, f"csv_dir:{source_dir}"


def resolve_ticker_universe_from_file(tickers_file: Path) -> Tuple[List[str], str]:
    source_file = resolve_ticker_file_path(Path(tickers_file))
    tickers = fetch_daily.read_local_sp500_tickers(source_file)
    tickers = dedupe_keep_order(tickers)
    if not tickers:
        raise ValueError(f"no tickers found in ticker file: {source_file}")
    return tickers, f"file:{source_file}"


def _candidate_roots() -> List[Path]:
    script_dir = Path(__file__).resolve().parent
    roots = [Path.cwd(), script_dir, script_dir.parent]
    out: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def resolve_ticker_file_path(tickers_file: Path) -> Path:
    source = Path(tickers_file)
    if source.is_absolute():
        if source.is_file():
            return source
        raise FileNotFoundError(f"ticker file not found: {source}")

    candidates: List[Path] = []
    for root in _candidate_roots():
        candidates.append(root / source)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"ticker file not found: {source} (searched: {searched})"
    )


def find_auto_pred_tickers_file() -> Optional[Path]:
    # Prefer direct pred_tickers.txt in common roots.
    for root in _candidate_roots():
        candidate = root / DEFAULT_PRED_TICKERS_FILE
        if candidate.is_file():
            return candidate

    # Fallback: production_predictions/<run_id>/pred_tickers.txt (latest numeric run first).
    for root in _candidate_roots():
        prod_dir = root / "production_predictions"
        if not prod_dir.is_dir():
            continue
        run_dirs = sorted(
            (
                p
                for p in prod_dir.iterdir()
                if p.is_dir() and str(p.name).isdigit()
            ),
            key=lambda p: int(str(p.name)),
            reverse=True,
        )
        for run_dir in run_dirs:
            candidate = run_dir / DEFAULT_PRED_TICKERS_FILE.name
            if candidate.is_file():
                return candidate

    return None


def resolve_ticker_universe(
    tickers_file: Optional[Path],
    universe_dir: Path,
) -> Tuple[List[str], str]:
    if tickers_file is not None:
        return resolve_ticker_universe_from_file(tickers_file)

    auto_tickers_file = find_auto_pred_tickers_file()
    if auto_tickers_file is not None:
        return resolve_ticker_universe_from_file(auto_tickers_file)

    return resolve_ticker_universe_from_csv_dir(universe_dir)


def build_universe_ticker_view_dir(
    source_dir: Path,
    universe_tickers: Sequence[str],
    view_dir: Path,
) -> Tuple[int, List[str]]:
    src_dir = Path(source_dir)
    dst_dir = Path(view_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for old_csv in dst_dir.glob("*.csv"):
        if old_csv.is_file() or old_csv.is_symlink():
            old_csv.unlink()

    missing: List[str] = []
    linked = 0
    for ticker in dedupe_keep_order(universe_tickers):
        src_csv = src_dir / f"{ticker}.csv"
        if not src_csv.exists():
            missing.append(ticker)
            continue
        dst_csv = dst_dir / f"{ticker}.csv"
        try:
            dst_csv.symlink_to(src_csv.resolve())
        except OSError:
            shutil.copy2(src_csv, dst_csv)
        linked += 1

    return int(linked), list(missing)


def compute_default_fetch_start(
    end_date_exclusive: date,
    history_days: int,
) -> date:
    if history_days < 1:
        raise ValueError("history_days must be >= 1")
    anchor_date = end_date_exclusive - timedelta(days=1)
    start_ts = pd.Timestamp(anchor_date) - pd.offsets.BDay(int(history_days) - 1)
    return start_ts.date()


def minimum_recommended_history_days(
    lookback: int,
    decomp_windows: int = int(prepare_daily.DEFAULT_DECOMP_WINDOWS),
    decomp_scales: Optional[int] = None,
    decomp_include_ma: bool = False,
) -> int:
    """Return a robust default to survive feature warmup and market holidays."""
    lb = int(lookback)
    if lb < 1:
        raise ValueError("lookback must be >= 1")
    ma_window = int(prepare_daily.compute_ma_n_window(lb))
    if bool(decomp_include_ma) and bool(prepare_daily.DECOMP_SCALE_AWARE_MA_FEATURE_ENABLED):
        window_sizes = prepare_daily.compute_decomposition_window_sizes(
            lookback=lb,
            windows=int(decomp_windows),
            scales=decomp_scales,
        )
        scale_ma_periods = prepare_daily.compute_scale_aware_ma_periods(
            window_sizes=window_sizes,
            windows=int(decomp_windows),
        )
        if scale_ma_periods:
            ma_window = max(ma_window, int(max(scale_ma_periods)))
    base_needed = int(lb + ma_window - 1)
    extra = max(
        int(DEFAULT_HISTORY_BUFFER_MIN),
        int(np.ceil(float(base_needed) * float(DEFAULT_HISTORY_BUFFER_PCT))),
    )
    return int(base_needed + extra)


def parse_iso_date(value: str, name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be YYYY-MM-DD; got {value!r}") from exc


def fetch_ticker_history_attempt(
    tickers: Sequence[str],
    output_dir: Path,
    start_date_inclusive: date,
    end_date_inclusive: date,
    chunk_size: int,
) -> Tuple[int, List[str], List[str]]:
    data, rate_limited_tickers = fetch_daily.download_all(
        tickers=list(tickers),
        start_date=start_date_inclusive,
        end_date_inclusive=end_date_inclusive,
        chunk_size=int(chunk_size),
    )
    saved = 0
    missing: List[str] = list(tickers)
    if data is not None and not data.empty:
        saved, missing = fetch_daily.split_and_save(
            data=data,
            tickers=list(tickers),
            out_dir=output_dir,
        )
    return int(saved), dedupe_keep_order(missing), dedupe_keep_order(rate_limited_tickers)


def fetch_russell_ticker_history(
    tickers: Sequence[str],
    output_dir: Path,
    start_date: str,
    end_date: str,
    chunk_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    start_date_inclusive = parse_iso_date(start_date, "start_date")
    end_date_exclusive = parse_iso_date(end_date, "end_date")
    if start_date_inclusive >= end_date_exclusive:
        raise ValueError(
            "start_date must be before end_date; "
            f"got {start_date_inclusive} >= {end_date_exclusive}"
        )
    end_date_inclusive = end_date_exclusive - timedelta(days=1)
    if start_date_inclusive > end_date_inclusive:
        raise ValueError(
            "empty fetch window after end-date conversion; "
            f"start_date={start_date_inclusive} end_date={end_date_exclusive}"
        )

    requested_tickers = [normalize_ticker(ticker) for ticker in tickers]
    resolved_tickers = set()
    failure_reason_by_ticker: Dict[str, str] = {}
    retry_tickers = dedupe_keep_order(requested_tickers)
    seen_rate_limited = set()
    attempts_run = 0
    total_saved = 0
    max_attempts = 1 + int(DEFAULT_RATE_LIMIT_RETRIES)

    for attempt_idx in range(1, max_attempts + 1):
        if not retry_tickers:
            break

        attempts_run = attempt_idx
        print(f"fetch attempt {attempt_idx}/{max_attempts}: tickers={len(retry_tickers)}")
        saved, missing, rate_limited_tickers = fetch_ticker_history_attempt(
            tickers=retry_tickers,
            output_dir=output_dir,
            start_date_inclusive=start_date_inclusive,
            end_date_inclusive=end_date_inclusive,
            chunk_size=int(chunk_size),
        )
        total_saved += int(saved)

        missing_set = {normalize_ticker(ticker) for ticker in missing}
        rate_limited_set = {normalize_ticker(ticker) for ticker in rate_limited_tickers}
        unresolved_rate_limited = [
            ticker for ticker in retry_tickers if ticker in missing_set and ticker in rate_limited_set
        ]

        for ticker in rate_limited_tickers:
            ticker_key = normalize_ticker(ticker)
            if ticker_key:
                seen_rate_limited.add(ticker_key)

        for ticker in retry_tickers:
            if ticker in missing_set:
                failure_reason_by_ticker[ticker] = (
                    "rate_limited" if ticker in rate_limited_set else "missing_ohlc"
                )
                continue
            resolved_tickers.add(ticker)
            failure_reason_by_ticker.pop(ticker, None)

        print(
            "fetch attempt summary: "
            f"saved={int(saved)} missing={int(len(missing_set))} "
            f"rate_limited_missing={int(len(unresolved_rate_limited))}"
        )
        if not unresolved_rate_limited:
            break
        if attempt_idx >= max_attempts:
            break
        print(
            "retrying rate-limited tickers: "
            f"remaining={len(unresolved_rate_limited)} "
            f"next_attempt={attempt_idx + 1}/{max_attempts}"
        )
        if float(DEFAULT_RATE_LIMIT_RETRY_SLEEP_SECONDS) > 0:
            print(
                "sleeping before rate-limit retry: "
                f"{float(DEFAULT_RATE_LIMIT_RETRY_SLEEP_SECONDS):.1f}s"
            )
            time.sleep(float(DEFAULT_RATE_LIMIT_RETRY_SLEEP_SECONDS))
        retry_tickers = unresolved_rate_limited

    request_start = start_date_inclusive.isoformat()
    request_end = end_date_exclusive.isoformat()
    final_rate_limited_tickers = [
        ticker
        for ticker in requested_tickers
        if ticker not in resolved_tickers and failure_reason_by_ticker.get(ticker) == "rate_limited"
    ]

    meta_rows: List[Dict[str, object]] = []
    failed_rows: List[Dict[str, object]] = []
    for ticker in requested_tickers:
        csv_path = output_dir / f"{ticker}.csv"
        ok = ticker in resolved_tickers and csv_path.exists()
        if ok:
            meta_rows.append(
                {
                    "ticker": ticker,
                    "status": "ok",
                    "failure_reason": "",
                    "request_start": request_start,
                    "request_end": request_end,
                }
            )
            continue

        reason = failure_reason_by_ticker.get(ticker, "missing_ohlc")
        meta_rows.append(
            {
                "ticker": ticker,
                "status": "missing",
                "failure_reason": reason,
                "request_start": request_start,
                "request_end": request_end,
            }
        )
        failed_rows.append(
            {
                "ticker": ticker,
                "status": "missing",
                "failure_reason": reason,
                "request_start": request_start,
                "request_end": request_end,
            }
        )

    meta_df = pd.DataFrame(meta_rows)
    failed_df = pd.DataFrame(failed_rows)
    if meta_df.empty:
        meta_df = pd.DataFrame(
            columns=["ticker", "status", "failure_reason", "request_start", "request_end"]
        )
    if failed_df.empty:
        failed_df = pd.DataFrame(
            columns=["ticker", "status", "failure_reason", "request_start", "request_end"]
        )

    meta_df.to_csv(output_dir / "_meta.csv", index=False)
    failed_df.to_csv(output_dir / "_failed.csv", index=False)
    failed_txt = output_dir / "_failed.txt"
    if failed_df.empty:
        failed_txt.write_text("", encoding="utf-8")
    else:
        failed_txt.write_text(
            "\n".join(failed_df["ticker"].astype(str).tolist()) + "\n",
            encoding="utf-8",
        )

    rate_limit_path = output_dir / fetch_daily.RATE_LIMIT_FILENAME
    fetch_daily.write_ticker_file(rate_limit_path, final_rate_limited_tickers)
    if seen_rate_limited:
        print(
            "rate-limit retry summary: "
            f"encountered={len(seen_rate_limited)} "
            f"remaining={len(final_rate_limited_tickers)} "
            f"attempts_run={attempts_run}"
        )
    if final_rate_limited_tickers:
        print(
            "rate-limit ticker file written: "
            f"{rate_limit_path} ({len(final_rate_limited_tickers)} tickers)"
        )
    print(
        "fetch save summary: "
        f"saved={int(total_saved)} missing={int(len(failed_rows))}"
    )
    return meta_df, failed_df


def _canonicalize_column_name(value: object) -> str:
    key = str(value).strip().lower().replace(" ", "_")
    return key.replace("-", "_")


def count_turnover_columns(input_dir: Path) -> Tuple[int, int]:
    csv_paths = sorted(input_dir.glob("*.csv"))
    total = 0
    with_turnover = 0
    for csv_path in csv_paths:
        total += 1
        try:
            cols = pd.read_csv(csv_path, nrows=0).columns
        except Exception:
            continue
        canon = {_canonicalize_column_name(col) for col in cols}
        if "turnover" in canon:
            with_turnover += 1
    return total, with_turnover


def resolve_effective_liquidity_feature(input_dir: Path, requested: str) -> str:
    feature = prepare_daily.resolve_liquidity_feature(requested)
    if feature != "turnover":
        return feature
    total_csv, with_turnover = count_turnover_columns(input_dir)
    if total_csv <= 0:
        return feature
    if with_turnover == total_csv:
        return feature
    print(
        "liquidity feature fallback: requested turnover but no turnover column "
        f"coverage across ticker CSVs is {with_turnover}/{total_csv}; using volume"
    )
    return "volume"


def latest_sample_indices(
    ticker_ids: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    ticker_ids_arr = np.asarray(ticker_ids, dtype=np.int64).reshape(-1)
    if ticker_ids_arr.size == 0:
        return np.empty((0,), dtype=np.int64)

    ts = pd.to_datetime(np.asarray(timestamps).astype(str), errors="coerce")
    ts_order = ts.where(~ts.isna(), pd.Timestamp("1900-01-01"))
    frame = pd.DataFrame(
        {
            "idx": np.arange(ticker_ids_arr.shape[0], dtype=np.int64),
            "ticker_id": ticker_ids_arr,
            "ts_order": ts_order,
        }
    )
    latest = (
        frame.sort_values(["ticker_id", "ts_order", "idx"])
        .groupby("ticker_id", sort=True)["idx"]
        .last()
        .to_numpy(dtype=np.int64)
    )
    return latest


def keep_latest_per_ticker_in_npz(npz_path: Path, sample_keys: Sequence[str]) -> int:
    with np.load(npz_path, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}

    if "ticker_ids" not in arrays or "timestamps" not in arrays:
        raise ValueError(f"{npz_path} missing ticker_ids or timestamps")

    keep_idx = latest_sample_indices(arrays["ticker_ids"], arrays["timestamps"])
    if keep_idx.size == 0:
        raise ValueError(f"{npz_path} has no samples to keep")

    keys_to_slice = list(sample_keys) + ["ticker_ids", "timestamps"]
    seen = set()
    unique_keys: List[str] = []
    for key in keys_to_slice:
        if key in seen:
            continue
        seen.add(key)
        unique_keys.append(key)

    for key in unique_keys:
        if key not in arrays:
            raise ValueError(f"{npz_path} missing expected key: {key}")
        arrays[key] = arrays[key][keep_idx]

    np.savez_compressed(npz_path, **arrays)
    return int(keep_idx.shape[0])


def filter_npz_by_timestamp_range(
    npz_path: Path,
    sample_keys: Sequence[str],
    start_date_inclusive: Optional[date] = None,
    end_date_exclusive: Optional[date] = None,
) -> int:
    with np.load(npz_path, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}

    if "ticker_ids" not in arrays or "timestamps" not in arrays:
        raise ValueError(f"{npz_path} missing ticker_ids or timestamps")

    timestamps = pd.to_datetime(
        np.asarray(arrays["timestamps"]).astype(str), errors="coerce"
    )
    keep_mask = ~pd.isna(timestamps)
    if start_date_inclusive is not None:
        keep_mask &= timestamps >= pd.Timestamp(start_date_inclusive)
    if end_date_exclusive is not None:
        keep_mask &= timestamps < pd.Timestamp(end_date_exclusive)

    keep_idx = np.flatnonzero(np.asarray(keep_mask, dtype=bool))
    if keep_idx.size == 0:
        raise ValueError(
            f"{npz_path} has no samples after timestamp filtering "
            f"(start={start_date_inclusive}, end={end_date_exclusive})"
        )

    keys_to_slice = list(sample_keys) + ["ticker_ids", "timestamps"]
    seen = set()
    unique_keys: List[str] = []
    for key in keys_to_slice:
        if key in seen:
            continue
        seen.add(key)
        unique_keys.append(key)

    for key in unique_keys:
        if key not in arrays:
            raise ValueError(f"{npz_path} missing expected key: {key}")
        arrays[key] = arrays[key][keep_idx]

    np.savez_compressed(npz_path, **arrays)
    return int(keep_idx.shape[0])


def run_image_builder(
    decomp_npz: Path,
    output_dir: Path,
    shard_size: int,
    volume_feature: str,
    enable_weekend_feature: bool,
    preview_count: int,
    vix_image: bool,
    include_moving_average: bool,
) -> None:
    script_path = Path(__file__).resolve().parent / "build_decomp_ohlc_images.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--input-npz",
        str(decomp_npz),
        "--output-dir",
        str(output_dir),
        "--max-samples",
        "-1",
        "--shard-size",
        str(int(shard_size)),
        "--preview-count",
        str(int(preview_count)),
        "--overwrite",
        "--include-volume",
        "--volume-feature",
        str(volume_feature),
    ]
    if enable_weekend_feature:
        cmd.append("--enable-weekend-feature")
    if bool(vix_image):
        cmd.append("--vix-image")
    if bool(include_moving_average):
        cmd.append("--include-moving-average")
    subprocess.run(cmd, check=True)


def latest_sample_timestamp_from_npz(npz_path: Path) -> pd.Timestamp:
    with np.load(npz_path, allow_pickle=True) as data:
        if "timestamps" not in data.files:
            raise ValueError(f"{npz_path} missing required 'timestamps' array")
        raw_timestamps = np.asarray(data["timestamps"])
    parsed = pd.to_datetime(raw_timestamps.astype(str), errors="coerce")
    parsed = parsed[~pd.isna(parsed)]
    if parsed.size < 1:
        raise ValueError(f"{npz_path} has no valid timestamps")
    ts = pd.Timestamp(parsed.max())
    if ts.tz is not None:
        ts = ts.tz_convert(None)
    return ts


def resolve_latest_sample_date(npz_paths: Sequence[Path]) -> date:
    latest_ts: Optional[pd.Timestamp] = None
    for npz_path in npz_paths:
        ts = latest_sample_timestamp_from_npz(Path(npz_path))
        if latest_ts is None or ts > latest_ts:
            latest_ts = ts
    if latest_ts is None:
        raise ValueError("no dataset timestamps found while resolving output date")
    return latest_ts.date()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare production or backtest data for image_model.py: fetch daily "
            "data for a fixed ticker universe, build prepare_daily_data datasets, and optionally "
            "build decomp OHLC image shards."
        )
    )
    parser.add_argument(
        "--tickers-file",
        type=Path,
        default=None,
        help=(
            "Ticker list file (.txt/.lst/.csv) used to define the dataset universe. "
            "If omitted and ./pred_tickers.txt exists, that file is used. "
            "Otherwise the universe is sourced from --universe-dir."
        ),
    )
    parser.add_argument(
        "--universe-dir",
        type=Path,
        default=DEFAULT_UNIVERSE_DIR,
        help=(
            "Directory whose *.csv filenames define the ticker universe "
            f"when no ticker list file is selected (default: {DEFAULT_UNIVERSE_DIR})."
        ),
    )
    parser.add_argument(
        "--include-spy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include SPY in the ticker universe (default: enabled)."
        ),
    )
    parser.add_argument(
        "--extra-ticker",
        action="append",
        default=[],
        help=(
            "Append ticker(s) to the resolved universe after base selection "
            "(may be supplied multiple times)."
        ),
    )
    parser.add_argument(
        "--tickers-dir",
        type=Path,
        default=DEFAULT_TICKERS_DIR,
        help=f"Directory to store fetched ticker CSVs (default: {DEFAULT_TICKERS_DIR}).",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=DEFAULT_DATASETS_DIR,
        help=(
            "Root directory to store dated dataset/image-shard outputs "
            f"(default: {DEFAULT_DATASETS_DIR})."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help=f"Base dataset filename stem in datasets-dir (default: {DEFAULT_DATASET_NAME}).",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help=(
            "Fetch this many business days ending at today "
            f"(default: {DEFAULT_HISTORY_DAYS})."
        ),
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help=(
            "Inclusive sample start date YYYY-MM-DD. When provided, fetch history is "
            "automatically extended backward to satisfy lookback/decomposition warmup."
        ),
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Exclusive fetch end date YYYY-MM-DD. Defaults to tomorrow.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="prepare_daily_data lookback (default: 60).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help=f"prepare_daily_data horizon (default: {DEFAULT_HORIZON}).",
    )
    parser.add_argument(
        "--label-mode",
        type=prepare_daily.resolve_label_mode,
        default=prepare_daily.LABEL_MODE_RANGE_ATR,
        choices=list(prepare_daily.LABEL_MODE_CHOICES),
        help=(
            "prepare_daily_data label mode. "
            "'range_atr' preserves existing horizon-based labels. "
            "'next_day_close_return' uses signal-day close -> next-day close labels "
            "and forces an effective label horizon of 1."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="prepare_daily_data stride (default: 1).",
    )
    parser.add_argument(
        "--decomp-windows",
        type=int,
        default=5,
        help="prepare_daily_data decomposition windows (default: 5).",
    )
    parser.add_argument(
        "--decomp-scales",
        type=int,
        default=None,
        help="prepare_daily_data decomposition scales (default: auto).",
    )
    parser.add_argument(
        "--decomp-normalization",
        type=prepare_daily.resolve_decomp_normalization,
        default=prepare_daily.DECOMP_NORMALIZATION,
        help=(
            "prepare_daily_data decomposition normalization: "
            "'synthetic', 'lookback', or 'none' (default: synthetic)."
        ),
    )
    parser.add_argument(
        "--global-ma-n-window",
        "--global_ma_n_window",
        dest="global_ma_n_window",
        type=int,
        default=int(prepare_daily.GLOBAL_MA_N_WINDOW),
        help=(
            "Global MA window used for ma_n feature engineering "
            f"(default: {int(prepare_daily.GLOBAL_MA_N_WINDOW)})."
        ),
    )
    parser.add_argument(
        "--decomp-scale-aware-ma-feature-enabled",
        "--decomp_scale_aware_ma_feature_enabled",
        dest="decomp_scale_aware_ma_feature_enabled",
        action=argparse.BooleanOptionalAction,
        default=bool(prepare_daily.DECOMP_SCALE_AWARE_MA_FEATURE_ENABLED),
        help=(
            "Enable scale-aware MA values for decomposition ma feature when "
            "--decomp-include-ma is enabled."
        ),
    )
    parser.add_argument(
        "--decomp-include-ma",
        "--decomp_include_ma",
        dest="decomp_include_ma",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include ma_n as an additional decomposition feature/channel "
            "while building decomposition datasets (default: enabled)."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="yfinance chunk size for downloads (default: 200).",
    )
    parser.add_argument(
        "--liquidity-feature",
        default="turnover",
        choices=["turnover", "volume"],
        help="prepare_daily_data liquidity feature (default: turnover).",
    )
    parser.add_argument(
        "--disable-turnover-backfill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Disable split-based shares_outstanding/turnover backfill when building "
            "datasets (default: enabled). Use --no-disable-turnover-backfill to opt in."
        ),
    )
    parser.add_argument(
        "--volume-feature",
        default="turnover",
        help="build_decomp_ohlc_images volume feature (default: turnover).",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=512,
        help="build_decomp_ohlc_images shard size (default: 512).",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=0,
        help="Number of preview PNG strips (default: 0).",
    )
    parser.add_argument(
        "--disable-weekend-feature",
        action="store_true",
        help="Disable weekend gap feature in image builder.",
    )
    parser.add_argument(
        "--vix-image",
        action="store_true",
        help=(
            "Enable 60-bar VIX OHLC image generation as an additional sample image."
        ),
    )
    parser.add_argument(
        "--http-timeout",
        type=int,
        default=30,
        help=(
            "Deprecated/ignored for universe selection. Kept for backward compatibility."
        ),
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip fetching raw CSVs and reuse existing files in --tickers-dir.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip build_decomp_ohlc_images step.",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help=(
            "Backtest mode: keep multiple samples per ticker, require real labels "
            "for all samples, and when --start-date is set fetch warmup history so "
            "samples can start at that date."
        ),
    )
    parser.add_argument(
        "--latest-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Filter output datasets to one latest sample per ticker after any other "
            "timestamp filtering (default: disabled)."
        ),
    )
    parser.add_argument(
        "--ret-pct-include-preceding-gap",
        action=argparse.BooleanOptionalAction,
        default=prepare_daily.RET_PCT_INCLUDE_PRECEDING_GAP_DEFAULT,
        help=(
            "Include the gap immediately before output-horizon start when computing "
            "return labels (ret_pct/ret_atr/avg_ret_atr/log_avg_ret_atr). "
            "For entry_offset=1, this switches the return base from open[t+1] to close[t]. "
            "Ignored when --label-mode=next_day_close_return."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if int(args.global_ma_n_window) < 1:
        raise ValueError("global_ma_n_window must be >= 1")

    prepare_daily.GLOBAL_MA_N_WINDOW = int(args.global_ma_n_window)
    prepare_daily.DECOMP_SCALE_AWARE_MA_FEATURE_ENABLED = bool(
        args.decomp_scale_aware_ma_feature_enabled
    )
    backtest_mode = bool(args.backtest)
    label_mode = prepare_daily.resolve_label_mode(args.label_mode)
    effective_label_horizon = prepare_daily.resolve_effective_label_horizon(
        horizon=int(args.horizon),
        label_mode=label_mode,
    )

    tickers_dir = Path(args.tickers_dir)
    datasets_root = Path(args.datasets_dir)
    datasets_root.mkdir(parents=True, exist_ok=True)
    staging_token = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S_%f")
    datasets_dir = datasets_root / f"_staging_{staging_token}"
    datasets_dir.mkdir(parents=True, exist_ok=False)
    print(f"dataset staging dir: {datasets_dir}")

    base_tickers, ticker_source = resolve_ticker_universe(
        tickers_file=Path(args.tickers_file) if args.tickers_file is not None else None,
        universe_dir=Path(args.universe_dir),
    )
    tickers = list(base_tickers)

    if args.extra_ticker:
        tickers = dedupe_keep_order([*tickers, *args.extra_ticker])

    if bool(args.include_spy):
        tickers = dedupe_keep_order([*tickers, "SPY"])

    if args.tickers_file is None and ticker_source == f"file:{DEFAULT_PRED_TICKERS_FILE}":
        print(
            "auto-detected ticker universe file: "
            f"{DEFAULT_PRED_TICKERS_FILE}"
        )

    tickers_file_out = tickers_dir / "universe_tickers.txt"
    tickers_dir.mkdir(parents=True, exist_ok=True)
    tickers_file_out.write_text("\n".join(tickers) + "\n", encoding="utf-8")
    print(
        "resolved "
        f"{len(tickers)} universe tickers from {ticker_source} "
        f"(include_spy={'yes' if bool(args.include_spy) else 'no'})"
    )
    print(f"ticker list: {tickers_file_out}")

    if args.end_date:
        end_date_exclusive = parse_iso_date(args.end_date, "end_date")
    else:
        end_date_exclusive = date.today() + timedelta(days=1)

    sample_start_date_inclusive: Optional[date] = None
    if args.start_date:
        provided_start_date = parse_iso_date(args.start_date, "start_date")
        sample_start_date_inclusive = provided_start_date
        recommended_history_days = minimum_recommended_history_days(
            lookback=int(args.lookback),
            decomp_windows=int(args.decomp_windows),
            decomp_scales=args.decomp_scales,
            decomp_include_ma=bool(args.decomp_include_ma),
        )
        start_date_inclusive = compute_default_fetch_start(
            end_date_exclusive=provided_start_date + timedelta(days=1),
            history_days=recommended_history_days,
        )
        print(
            f"{'backtest' if backtest_mode else 'production'} start-date warmup enabled: "
            f"sample_start={provided_start_date.isoformat()} "
            f"fetch_start={start_date_inclusive.isoformat()} "
            f"(warmup_history_days={recommended_history_days})"
        )
    else:
        requested_history_days = int(args.history_days)
        recommended_history_days = minimum_recommended_history_days(
            lookback=int(args.lookback),
            decomp_windows=int(args.decomp_windows),
            decomp_scales=args.decomp_scales,
            decomp_include_ma=bool(args.decomp_include_ma),
        )
        effective_history_days = max(requested_history_days, recommended_history_days)
        if effective_history_days > requested_history_days:
            print(
                "history-days auto-adjusted for lookback warmup/holidays: "
                f"{requested_history_days} -> {effective_history_days}"
            )
        start_date_inclusive = compute_default_fetch_start(
            end_date_exclusive=end_date_exclusive,
            history_days=effective_history_days,
        )

    start_date_for_validation = sample_start_date_inclusive or start_date_inclusive
    if start_date_for_validation >= end_date_exclusive:
        raise ValueError(
            "start_date must be before end_date; "
            f"got {start_date_for_validation} >= {end_date_exclusive}"
        )

    start_date_str = start_date_inclusive.isoformat()
    end_date_str = end_date_exclusive.isoformat()
    print(f"fetch window (inclusive/exclusive): {start_date_str} -> {end_date_str}")

    if not args.skip_fetch:
        meta_df, failed_df = fetch_russell_ticker_history(
            tickers=tickers,
            output_dir=tickers_dir,
            start_date=start_date_str,
            end_date=end_date_str,
            chunk_size=int(args.chunk_size),
        )
        ok_count = int(
            (
                meta_df.get("status", pd.Series(dtype=object))
                .astype(str)
                .str.lower()
                .eq("ok")
            ).sum()
        )
        fail_count = int(len(failed_df))
        print(
            f"fetched ticker CSVs: ok={ok_count} failed_or_missing={fail_count} "
            f"output_dir={tickers_dir}"
        )
        print(f"meta: {tickers_dir / '_meta.csv'}")
        print(f"failed detail: {tickers_dir / '_failed.csv'}")
        print(f"failed tickers: {tickers_dir / '_failed.txt'}")
    else:
        print("skip-fetch enabled; using existing CSVs in ticker directory")

    universe_view_dir = datasets_dir / "_ticker_universe_view"
    linked_count, missing_universe_tickers = build_universe_ticker_view_dir(
        source_dir=tickers_dir,
        universe_tickers=tickers,
        view_dir=universe_view_dir,
    )
    missing_universe_path = tickers_dir / "_universe_missing.txt"
    if missing_universe_tickers:
        missing_universe_path.write_text(
            "\n".join(missing_universe_tickers) + "\n", encoding="utf-8"
        )
    else:
        missing_universe_path.write_text("", encoding="utf-8")

    if linked_count <= 0:
        raise RuntimeError(
            "no ticker CSVs available for dataset build after applying universe filter; "
            f"source_dir={tickers_dir} universe_size={len(tickers)} "
            f"missing_file={missing_universe_path}"
        )

    print(
        "ticker universe view prepared: "
        f"{universe_view_dir} ({linked_count} tickers)"
    )
    if missing_universe_tickers:
        preview = ", ".join(missing_universe_tickers[:20])
        more = int(len(missing_universe_tickers) - 20)
        suffix = "" if more <= 0 else f", ... (+{more} more)"
        print(
            "universe tickers missing from fetched CSVs; continuing with available subset: "
            f"missing={len(missing_universe_tickers)} [{preview}{suffix}]"
        )
        print(f"missing universe tickers: {missing_universe_path}")

    effective_liquidity_feature = resolve_effective_liquidity_feature(
        input_dir=universe_view_dir,
        requested=str(args.liquidity_feature),
    )
    if effective_liquidity_feature != str(args.liquidity_feature):
        print(
            "liquidity feature effective value: "
            f"{args.liquidity_feature} -> {effective_liquidity_feature}"
        )

    base_npz_path = datasets_dir / f"{args.dataset_name}.npz"
    include_preceding_gap = bool(args.ret_pct_include_preceding_gap)
    base_npz, decomp_npz, dual_npz = prepare_daily.build_combined_datasets(
        input_dir=universe_view_dir,
        output_path=base_npz_path,
        lookback=int(args.lookback),
        horizon=int(args.horizon),
        stride=int(args.stride),
        liquidity_feature=str(effective_liquidity_feature),
        decomp_windows=int(args.decomp_windows),
        decomp_scales=args.decomp_scales,
        decomp_normalization=str(args.decomp_normalization),
        decomp_include_ma=bool(args.decomp_include_ma),
        include_synthetic_features=False,
        disable_turnover_backfill=bool(args.disable_turnover_backfill),
        production_no_labels=not backtest_mode,
        include_preceding_gap_in_ret_pct=include_preceding_gap,
        label_mode=label_mode,
    )
    if label_mode == prepare_daily.LABEL_MODE_NEXT_DAY_CLOSE_RETURN:
        print(
            "label mode: "
            f"{label_mode} (signal close -> next close; effective_label_horizon={effective_label_horizon})"
        )
    else:
        print(
            "label mode: "
            f"{label_mode} (effective_label_horizon={effective_label_horizon}; "
            "include preceding gap before output horizon start candle "
            f"{'enabled' if include_preceding_gap else 'disabled'})"
        )

    if sample_start_date_inclusive is not None:
        kept_base = filter_npz_by_timestamp_range(
            base_npz,
            sample_keys=["X", "y_raw"],
            start_date_inclusive=sample_start_date_inclusive,
        )
        kept_decomp = filter_npz_by_timestamp_range(
            decomp_npz,
            sample_keys=["X", "y_raw"],
            start_date_inclusive=sample_start_date_inclusive,
        )
        kept_dual = filter_npz_by_timestamp_range(
            dual_npz,
            sample_keys=["X_seq", "X_decomp", "y_raw"],
            start_date_inclusive=sample_start_date_inclusive,
        )
        print(
            f"{'backtest' if backtest_mode else 'production'} timestamp filter applied: "
            f"start={sample_start_date_inclusive.isoformat()} "
            f"base={kept_base} decomp={kept_decomp} dual={kept_dual}"
        )
    elif backtest_mode:
        print("backtest mode enabled: keeping all per-ticker samples with labels")
    else:
        print("production mode enabled: keeping all per-ticker samples without labels")

    if bool(args.latest_only):
        kept_base = keep_latest_per_ticker_in_npz(base_npz, sample_keys=["X", "y_raw"])
        kept_decomp = keep_latest_per_ticker_in_npz(
            decomp_npz, sample_keys=["X", "y_raw"]
        )
        kept_dual = keep_latest_per_ticker_in_npz(
            dual_npz,
            sample_keys=["X_seq", "X_decomp", "y_raw"],
        )
        print(
            "latest-only filter applied (one sample per ticker): "
            f"base={kept_base} decomp={kept_decomp} dual={kept_dual}"
        )

    latest_sample_date = resolve_latest_sample_date([base_npz, decomp_npz, dual_npz])
    final_datasets_dir = datasets_root / latest_sample_date.isoformat()
    if final_datasets_dir.exists():
        print(
            "replacing existing dated dataset folder: "
            f"{final_datasets_dir}"
        )
        shutil.rmtree(final_datasets_dir)
    datasets_dir.rename(final_datasets_dir)
    datasets_dir = final_datasets_dir
    base_npz = datasets_dir / base_npz.name
    decomp_npz = datasets_dir / decomp_npz.name
    dual_npz = datasets_dir / dual_npz.name
    print(
        "dataset date folder resolved from latest sample timestamp: "
        f"{datasets_dir} (latest_sample_date={latest_sample_date.isoformat()})"
    )
    print(f"dataset saved: {base_npz}")
    print(f"dataset saved: {decomp_npz}")
    print(f"dataset saved: {dual_npz}")

    if not args.skip_images:
        run_image_builder(
            decomp_npz=decomp_npz,
            output_dir=datasets_dir,
            shard_size=int(args.shard_size),
            volume_feature=str(args.volume_feature),
            enable_weekend_feature=not bool(args.disable_weekend_feature),
            preview_count=int(args.preview_count),
            vix_image=bool(args.vix_image),
            include_moving_average=bool(args.decomp_include_ma),
        )
        print(f"image shards dir: {datasets_dir / 'shards'}")
        print(f"image manifest: {datasets_dir / 'manifest.json'}")
    else:
        print("skip-images enabled; image shard build skipped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
