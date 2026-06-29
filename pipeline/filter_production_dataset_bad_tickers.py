#!/usr/bin/env python3
"""Filter a production dataset by removing bad tickers from ticker CSV quality stats.

Bad tickers are identified from `tickers/*.csv` using these definitions:
- flat candle: `open == high == low == close`
- zero-volume candle: `volume == 0`

A ticker is removed when either percentage is strictly greater than the
configured threshold. The script writes a filtered copy of the chosen dataset
directory and preserves the original dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_PRODUCTION_DATASETS_DIR = Path("production_datasets")
DEFAULT_TICKERS_DIR = Path("tickers")
DEFAULT_FLAT_THRESHOLD_PCT = 5.0
DEFAULT_ZERO_VOLUME_THRESHOLD_PCT = 5.0
DEFAULT_PROGRESS_EVERY = 250
SUMMARY_JSON_NAME = "bad_ticker_filter_summary.json"
REMOVED_TICKERS_CSV_NAME = "bad_ticker_filter_removed_tickers.csv"


@dataclass(frozen=True)
class TickerScanStats:
    ticker: str
    file_exists: bool
    valid_rows: int
    skipped_rows: int
    flat_rows: int
    flat_pct: float
    zero_volume_rows: int
    zero_volume_pct: float
    removed: bool
    removal_reason: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a filtered copy of a production dataset directory by removing "
            "tickers whose source CSV history has too many flat or zero-volume candles."
        )
    )
    parser.add_argument(
        "dataset",
        help=(
            "Dataset folder name inside production_datasets/ or an explicit path to "
            "the dataset directory."
        ),
    )
    parser.add_argument(
        "--production-datasets-dir",
        default=DEFAULT_PRODUCTION_DATASETS_DIR,
        help="Base directory containing production dataset folders.",
    )
    parser.add_argument(
        "--tickers-dir",
        default=DEFAULT_TICKERS_DIR,
        help="Directory containing per-ticker source CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for the filtered dataset. Defaults to a sibling directory.",
    )
    parser.add_argument(
        "--flat-threshold-pct",
        type=float,
        default=DEFAULT_FLAT_THRESHOLD_PCT,
        help="Remove tickers with flat-candle percentage strictly greater than this value.",
    )
    parser.add_argument(
        "--zero-volume-threshold-pct",
        type=float,
        default=DEFAULT_ZERO_VOLUME_THRESHOLD_PCT,
        help="Remove tickers with zero-volume percentage strictly greater than this value.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="Print scan progress every N ticker CSVs. Use 0 to disable.",
    )
    parser.add_argument(
        "--copy-universe-view",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy a filtered _ticker_universe_view directory when the source dataset has one.",
    )
    return parser.parse_args(argv)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_dataset_dir(dataset_arg: str, production_datasets_dir: Path) -> Path:
    candidate = Path(dataset_arg)
    if candidate.exists():
        return candidate.resolve()
    resolved = (production_datasets_dir / dataset_arg).resolve()
    if resolved.exists():
        return resolved
    raise FileNotFoundError(
        f"dataset not found: {dataset_arg!r} (checked {candidate} and {resolved})"
    )


def resolve_output_dir(input_dir: Path, explicit_output_dir: str | None) -> Path:
    if explicit_output_dir:
        output_dir = Path(explicit_output_dir).resolve()
    else:
        output_dir = input_dir.with_name(f"{input_dir.name}_filtered_flat_zero_volume")
    if output_dir == input_dir:
        raise ValueError("output directory must differ from input directory")
    try:
        output_dir.relative_to(input_dir)
    except ValueError:
        pass
    else:
        raise ValueError("output directory must not be inside the input directory")
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"output directory already exists and is not empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def normalize_ticker(value: Any) -> str:
    return str(value).strip().upper()


def resolve_dataset_universe(input_dir: Path) -> np.ndarray:
    top_level_tickers = input_dir / "tickers.npy"
    if top_level_tickers.is_file():
        arr = np.load(top_level_tickers, allow_pickle=True)
        return np.asarray(arr, dtype=object).reshape(-1)
    for path in sorted(input_dir.glob("*.npz")):
        with np.load(path, allow_pickle=True) as data:
            if "tickers" in data.files:
                return np.asarray(data["tickers"], dtype=object).reshape(-1)
    raise FileNotFoundError(
        f"could not resolve dataset ticker universe from {input_dir / 'tickers.npy'} or top-level npz files"
    )


def scan_ticker_csv(
    ticker: str,
    csv_path: Path,
    flat_threshold: float,
    zero_volume_threshold: float,
) -> TickerScanStats:
    if not csv_path.is_file():
        return TickerScanStats(
            ticker=ticker,
            file_exists=False,
            valid_rows=0,
            skipped_rows=0,
            flat_rows=0,
            flat_pct=0.0,
            zero_volume_rows=0,
            zero_volume_pct=0.0,
            removed=False,
            removal_reason="missing_file",
        )

    valid_rows = 0
    skipped_rows = 0
    flat_rows = 0
    zero_volume_rows = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return TickerScanStats(
                ticker=ticker,
                file_exists=True,
                valid_rows=0,
                skipped_rows=0,
                flat_rows=0,
                flat_pct=0.0,
                zero_volume_rows=0,
                zero_volume_pct=0.0,
                removed=False,
                removal_reason="empty_file",
            )
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset({str(x).strip() for x in reader.fieldnames}):
            return TickerScanStats(
                ticker=ticker,
                file_exists=True,
                valid_rows=0,
                skipped_rows=0,
                flat_rows=0,
                flat_pct=0.0,
                zero_volume_rows=0,
                zero_volume_pct=0.0,
                removed=False,
                removal_reason="missing_required_columns",
            )
        for row in reader:
            try:
                open_v = float(str(row.get("open", "")).strip())
                high_v = float(str(row.get("high", "")).strip())
                low_v = float(str(row.get("low", "")).strip())
                close_v = float(str(row.get("close", "")).strip())
                volume_v = float(str(row.get("volume", "")).strip())
            except (TypeError, ValueError):
                skipped_rows += 1
                continue
            if not all(math.isfinite(v) for v in (open_v, high_v, low_v, close_v, volume_v)):
                skipped_rows += 1
                continue
            valid_rows += 1
            if open_v == high_v == low_v == close_v:
                flat_rows += 1
            if volume_v == 0.0:
                zero_volume_rows += 1

    flat_pct = float(flat_rows / valid_rows) if valid_rows > 0 else 0.0
    zero_volume_pct = float(zero_volume_rows / valid_rows) if valid_rows > 0 else 0.0
    removed_reasons: list[str] = []
    if valid_rows > 0 and flat_pct > flat_threshold:
        removed_reasons.append("flat_pct_gt_threshold")
    if valid_rows > 0 and zero_volume_pct > zero_volume_threshold:
        removed_reasons.append("zero_volume_pct_gt_threshold")
    return TickerScanStats(
        ticker=ticker,
        file_exists=True,
        valid_rows=int(valid_rows),
        skipped_rows=int(skipped_rows),
        flat_rows=int(flat_rows),
        flat_pct=float(flat_pct),
        zero_volume_rows=int(zero_volume_rows),
        zero_volume_pct=float(zero_volume_pct),
        removed=bool(removed_reasons),
        removal_reason="|".join(removed_reasons) if removed_reasons else "",
    )


def scan_dataset_tickers(
    dataset_tickers: Iterable[Any],
    tickers_dir: Path,
    flat_threshold_pct: float,
    zero_volume_threshold_pct: float,
    progress_every: int,
) -> list[TickerScanStats]:
    unique_tickers = sorted({normalize_ticker(x) for x in dataset_tickers if normalize_ticker(x)})
    total = len(unique_tickers)
    flat_threshold = float(flat_threshold_pct) / 100.0
    zero_volume_threshold = float(zero_volume_threshold_pct) / 100.0
    stats: list[TickerScanStats] = []
    for idx, ticker in enumerate(unique_tickers, start=1):
        item = scan_ticker_csv(
            ticker=ticker,
            csv_path=tickers_dir / f"{ticker}.csv",
            flat_threshold=flat_threshold,
            zero_volume_threshold=zero_volume_threshold,
        )
        stats.append(item)
        if progress_every > 0 and (idx % progress_every == 0 or idx == total):
            removed_so_far = sum(1 for x in stats if x.removed)
            print(
                f"scanned {idx}/{total} tickers "
                f"removed={removed_so_far}"
            )
    return stats


def build_keep_map(tickers: np.ndarray, removed_tickers: set[str]) -> tuple[np.ndarray, np.ndarray]:
    tickers_arr = np.asarray(tickers, dtype=object).reshape(-1)
    keep_mask = np.array(
        [normalize_ticker(ticker) not in removed_tickers for ticker in tickers_arr],
        dtype=bool,
    )
    filtered_tickers = tickers_arr[keep_mask]
    remap = np.full(tickers_arr.shape[0], -1, dtype=np.int64)
    remap[np.flatnonzero(keep_mask)] = np.arange(filtered_tickers.shape[0], dtype=np.int64)
    return filtered_tickers, remap


def filter_sampled_payload(
    arrays: dict[str, np.ndarray],
    filtered_tickers: np.ndarray,
    keep_map: np.ndarray,
) -> tuple[dict[str, np.ndarray], int, int]:
    if "ticker_ids" not in arrays:
        raise KeyError("payload is missing ticker_ids")
    ticker_ids = np.asarray(arrays["ticker_ids"])
    if ticker_ids.ndim != 1:
        ticker_ids = ticker_ids.reshape(-1)
    if ticker_ids.size == 0:
        row_mask = np.zeros((0,), dtype=bool)
    else:
        if np.any(ticker_ids < 0) or np.any(ticker_ids >= keep_map.shape[0]):
            raise ValueError(
                "ticker_ids contain values outside the ticker universe: "
                f"min={int(ticker_ids.min())} max={int(ticker_ids.max())} universe={int(keep_map.shape[0])}"
            )
        row_mask = keep_map[ticker_ids.astype(np.int64, copy=False)] >= 0
    rows_before = int(ticker_ids.shape[0])
    rows_after = int(np.count_nonzero(row_mask))
    filtered: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        arr = np.asarray(value)
        if key == "tickers":
            filtered[key] = np.asarray(filtered_tickers, dtype=arr.dtype if arr.dtype != object else object)
            continue
        if key == "ticker_ids":
            kept_ids = ticker_ids[row_mask].astype(np.int64, copy=False)
            filtered[key] = keep_map[kept_ids].astype(arr.dtype, copy=False)
            continue
        if arr.ndim >= 1 and int(arr.shape[0]) == rows_before:
            filtered[key] = arr[row_mask]
        else:
            filtered[key] = arr
    return filtered, rows_before, rows_after


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def write_npz(path: Path, arrays: dict[str, np.ndarray], compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def process_top_level_npzs(
    input_dir: Path,
    output_dir: Path,
    removed_tickers: set[str],
) -> dict[str, dict[str, int]]:
    artifact_stats: dict[str, dict[str, int]] = {}
    for npz_path in sorted(input_dir.glob("*.npz")):
        arrays = read_npz(npz_path)
        if "ticker_ids" not in arrays:
            continue
        local_tickers = (
            np.asarray(arrays["tickers"], dtype=object).reshape(-1)
            if "tickers" in arrays
            else resolve_dataset_universe(input_dir)
        )
        filtered_tickers, keep_map = build_keep_map(local_tickers, removed_tickers)
        filtered_arrays, rows_before, rows_after = filter_sampled_payload(
            arrays=arrays,
            filtered_tickers=filtered_tickers,
            keep_map=keep_map,
        )
        write_npz(output_dir / npz_path.name, filtered_arrays, compressed=False)
        artifact_stats[npz_path.name] = {
            "rows_before": int(rows_before),
            "rows_after": int(rows_after),
            "rows_removed": int(rows_before - rows_after),
        }
    return artifact_stats


def process_shards(
    input_dir: Path,
    output_dir: Path,
    removed_tickers: set[str],
    dataset_universe: np.ndarray,
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    manifest_path = input_dir / "manifest.json"
    shards_dir = input_dir / "shards"
    if not manifest_path.is_file() or not shards_dir.is_dir():
        return None, {
            "rows_before": 0,
            "rows_after": 0,
            "rows_removed": 0,
            "shards_before": 0,
            "shards_after": 0,
        }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    filtered_tickers, keep_map = build_keep_map(dataset_universe, removed_tickers)
    compressed = str(manifest.get("shard_save_mode", "uncompressed")).strip().lower() == "compressed"
    subset_start = int(manifest.get("subset_start", 0))
    new_shards: list[dict[str, int | str]] = []
    running_count = 0
    rows_before = 0
    rows_after = 0

    for shard_index, entry in enumerate(manifest.get("shards", [])):
        shard_file = str(entry["file"])
        shard_path = input_dir / shard_file
        arrays = read_npz(shard_path)
        filtered_arrays, shard_rows_before, shard_rows_after = filter_sampled_payload(
            arrays=arrays,
            filtered_tickers=filtered_tickers,
            keep_map=keep_map,
        )
        rows_before += int(shard_rows_before)
        rows_after += int(shard_rows_after)
        if shard_rows_after <= 0:
            continue
        rel_start = int(running_count)
        rel_end = int(rel_start + shard_rows_after)
        sample_start = int(subset_start + rel_start)
        sample_end = int(subset_start + rel_end)
        out_name = f"shard_{len(new_shards):06d}.npz"
        write_npz(output_dir / "shards" / out_name, filtered_arrays, compressed=compressed)
        new_shards.append(
            {
                "file": str(Path("shards") / out_name),
                "sample_start": int(sample_start),
                "sample_end": int(sample_end),
                "count": int(shard_rows_after),
                "relative_start": int(rel_start),
                "relative_end": int(rel_end),
            }
        )
        print(
            f"wrote shard {shard_index:06d} -> {len(new_shards) - 1:06d} "
            f"rows={shard_rows_after}"
        )
        running_count = rel_end

    new_manifest = dict(manifest)
    new_manifest["created_utc"] = now_utc_iso()
    new_manifest["tickers_count"] = int(filtered_tickers.shape[0])
    new_manifest["subset_count"] = int(rows_after)
    new_manifest["subset_end"] = int(subset_start + rows_after)
    new_manifest["num_shards"] = int(len(new_shards))
    new_manifest["shards"] = new_shards
    if "vix_feature_retained_count" in new_manifest:
        new_manifest["vix_feature_retained_count"] = int(rows_after)
    if "vix_image_retained_count" in new_manifest:
        new_manifest["vix_image_retained_count"] = int(rows_after)

    return new_manifest, {
        "rows_before": int(rows_before),
        "rows_after": int(rows_after),
        "rows_removed": int(rows_before - rows_after),
        "shards_before": int(len(manifest.get("shards", []))),
        "shards_after": int(len(new_shards)),
    }


def copy_filtered_universe_view(
    input_dir: Path,
    output_dir: Path,
    removed_tickers: set[str],
) -> int:
    source_dir = input_dir / "_ticker_universe_view"
    if not source_dir.is_dir():
        return 0
    written = 0
    dest_dir = output_dir / "_ticker_universe_view"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_dir.glob("*.csv")):
        if normalize_ticker(path.stem) in removed_tickers:
            continue
        shutil.copy2(path, dest_dir / path.name)
        written += 1
    return written


def write_removed_tickers_csv(path: Path, stats: Sequence[TickerScanStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [item for item in stats if item.removed]
    fieldnames = [
        "ticker",
        "file_exists",
        "valid_rows",
        "skipped_rows",
        "flat_rows",
        "flat_pct",
        "zero_volume_rows",
        "zero_volume_pct",
        "removed",
        "removal_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            writer.writerow(asdict(item))


def build_summary(
    *,
    input_dir: Path,
    output_dir: Path,
    tickers_dir: Path,
    scan_stats: Sequence[TickerScanStats],
    dataset_universe: np.ndarray,
    filtered_universe: np.ndarray,
    top_level_artifacts: dict[str, dict[str, int]],
    shard_stats: dict[str, int],
    universe_view_written: int,
    flat_threshold_pct: float,
    zero_volume_threshold_pct: float,
) -> dict[str, Any]:
    removed_stats = [item for item in scan_stats if item.removed]
    missing_stats = [item for item in scan_stats if not item.file_exists]
    return {
        "created_utc": now_utc_iso(),
        "source_dataset_dir": str(input_dir),
        "output_dataset_dir": str(output_dir),
        "tickers_dir": str(tickers_dir),
        "criteria": {
            "flat_pct_gt": float(flat_threshold_pct) / 100.0,
            "zero_volume_pct_gt": float(zero_volume_threshold_pct) / 100.0,
            "flat_definition": "open == high == low == close",
            "zero_volume_definition": "volume == 0",
        },
        "dataset_tickers_before": int(np.asarray(dataset_universe).shape[0]),
        "dataset_tickers_after": int(np.asarray(filtered_universe).shape[0]),
        "dataset_tickers_removed": int(np.asarray(dataset_universe).shape[0] - np.asarray(filtered_universe).shape[0]),
        "scanned_tickers": int(len(scan_stats)),
        "removed_tickers_count": int(len(removed_stats)),
        "removed_tickers_sample": [item.ticker for item in removed_stats[:100]],
        "missing_ticker_files_count": int(len(missing_stats)),
        "missing_ticker_files_sample": [item.ticker for item in missing_stats[:100]],
        "valid_rows_scanned_total": int(sum(item.valid_rows for item in scan_stats)),
        "flat_rows_scanned_total": int(sum(item.flat_rows for item in scan_stats)),
        "zero_volume_rows_scanned_total": int(sum(item.zero_volume_rows for item in scan_stats)),
        "top_level_npz_artifacts": top_level_artifacts,
        "shards": shard_stats,
        "ticker_universe_view_files_written": int(universe_view_written),
        "removed_tickers_csv": str(output_dir / REMOVED_TICKERS_CSV_NAME),
    }


def run_filter(
    *,
    input_dir: Path,
    output_dir: Path,
    tickers_dir: Path,
    flat_threshold_pct: float,
    zero_volume_threshold_pct: float,
    progress_every: int,
    copy_universe_view: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_universe = resolve_dataset_universe(input_dir)
    scan_stats = scan_dataset_tickers(
        dataset_tickers=dataset_universe,
        tickers_dir=tickers_dir,
        flat_threshold_pct=flat_threshold_pct,
        zero_volume_threshold_pct=zero_volume_threshold_pct,
        progress_every=progress_every,
    )
    removed_tickers = {item.ticker for item in scan_stats if item.removed}
    filtered_universe, _ = build_keep_map(dataset_universe, removed_tickers)

    np.save(output_dir / "tickers.npy", filtered_universe)

    top_level_artifacts = process_top_level_npzs(
        input_dir=input_dir,
        output_dir=output_dir,
        removed_tickers=removed_tickers,
    )
    new_manifest, shard_stats = process_shards(
        input_dir=input_dir,
        output_dir=output_dir,
        removed_tickers=removed_tickers,
        dataset_universe=dataset_universe,
    )
    if new_manifest is not None:
        new_manifest["bad_ticker_filter"] = {
            "created_utc": now_utc_iso(),
            "flat_pct_gt": float(flat_threshold_pct) / 100.0,
            "zero_volume_pct_gt": float(zero_volume_threshold_pct) / 100.0,
            "removed_tickers_count": int(len(removed_tickers)),
            "removed_tickers_csv": str(output_dir / REMOVED_TICKERS_CSV_NAME),
            "source_dataset_dir": str(input_dir),
            "tickers_dir": str(tickers_dir),
        }
        write_json(output_dir / "manifest.json", new_manifest)

    universe_view_written = 0
    if copy_universe_view:
        universe_view_written = copy_filtered_universe_view(
            input_dir=input_dir,
            output_dir=output_dir,
            removed_tickers=removed_tickers,
        )

    write_removed_tickers_csv(output_dir / REMOVED_TICKERS_CSV_NAME, scan_stats)
    summary = build_summary(
        input_dir=input_dir,
        output_dir=output_dir,
        tickers_dir=tickers_dir,
        scan_stats=scan_stats,
        dataset_universe=dataset_universe,
        filtered_universe=filtered_universe,
        top_level_artifacts=top_level_artifacts,
        shard_stats=shard_stats,
        universe_view_written=universe_view_written,
        flat_threshold_pct=flat_threshold_pct,
        zero_volume_threshold_pct=zero_volume_threshold_pct,
    )
    write_json(output_dir / SUMMARY_JSON_NAME, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    production_datasets_dir = Path(args.production_datasets_dir).resolve()
    tickers_dir = Path(args.tickers_dir).resolve()
    if not tickers_dir.is_dir():
        raise FileNotFoundError(f"tickers dir not found: {tickers_dir}")
    input_dir = resolve_dataset_dir(
        dataset_arg=str(args.dataset),
        production_datasets_dir=production_datasets_dir,
    )
    output_dir = resolve_output_dir(input_dir=input_dir, explicit_output_dir=args.output_dir)

    summary = run_filter(
        input_dir=input_dir,
        output_dir=output_dir,
        tickers_dir=tickers_dir,
        flat_threshold_pct=float(args.flat_threshold_pct),
        zero_volume_threshold_pct=float(args.zero_volume_threshold_pct),
        progress_every=max(0, int(args.progress_every)),
        copy_universe_view=bool(args.copy_universe_view),
    )
    print(
        "done: "
        f"removed_tickers={summary['removed_tickers_count']} "
        f"dataset_tickers_before={summary['dataset_tickers_before']} "
        f"dataset_tickers_after={summary['dataset_tickers_after']}"
    )
    print(f"output_dir: {output_dir}")
    print(f"summary: {output_dir / SUMMARY_JSON_NAME}")


if __name__ == "__main__":
    main()
