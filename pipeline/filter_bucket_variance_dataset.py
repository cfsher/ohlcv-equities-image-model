#!/usr/bin/env python3
"""Filter ticker CSVs by 6-month bucket return variance."""

from __future__ import annotations

import argparse
import calendar
import csv
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence


DEFAULT_CLIP_ABS = 2.5
DEFAULT_TARGET_ROWS = 6_000_000
OUTPUT_EXTRA_FIELDS = (
    "bucket_6m_id",
    "ret_cc_clip_2p5",
    "returns_variance_clip_2p5",
    "source_id",
    "_source_row_id_orig",
)


@dataclass(frozen=True)
class BucketSummary:
    ticker: str
    bucket_id: int
    row_count: int
    variance: float


class RunningVariance:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / float(self.n)
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.n < 1:
            return 0.0
        return self.m2 / float(self.n)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split each ticker into exact 6-month buckets anchored to the first row, "
            "compute bucket variance over clipped close-to-close returns, and keep the "
            "highest-variance buckets above the maximal threshold that still leaves "
            "more than the target number of rows."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="yes",
        help="Directory containing source ticker CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default="imgay2",
        help="Directory to write the filtered ticker CSVs and summaries.",
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=DEFAULT_TARGET_ROWS,
        help="Keep the largest variance threshold whose retained rows stay strictly above this count.",
    )
    parser.add_argument(
        "--return-clip-abs",
        type=float,
        default=DEFAULT_CLIP_ABS,
        help="Absolute clip applied to close-to-close returns before variance is computed.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Compute the threshold and summaries without writing filtered per-ticker CSVs.",
    )
    return parser.parse_args(argv)


def iter_ticker_paths(input_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv" and not path.name.startswith("_")
    )


def parse_iso_date(raw: str) -> date:
    text = str(raw).strip()
    if not text:
        raise ValueError("empty date")
    return date.fromisoformat(text[:10])


def add_months(anchor: date, months: int) -> date:
    total_month = anchor.year * 12 + (anchor.month - 1) + int(months)
    year = total_month // 12
    month = total_month % 12 + 1
    day = min(anchor.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def clip_value(value: float, clip_abs: float) -> float:
    if value > clip_abs:
        return clip_abs
    if value < -clip_abs:
        return -clip_abs
    return value


def resolve_source_id(row: Dict[str, str], fallback: int) -> int:
    for col in ("source_id", "_source_row_id_orig", "_source_row_id"):
        raw = str(row.get(col, "")).strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError:
            try:
                value = int(float(raw))
            except ValueError:
                continue
        return value
    return int(fallback)


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    if not math.isfinite(value):
        return ""
    return repr(float(value))


def scan_ticker(
    path: Path,
    clip_abs: float,
) -> tuple[List[BucketSummary], int, int, int]:
    ticker = path.stem
    bucket_summaries: List[BucketSummary] = []
    rows_scanned = 0
    skipped_rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return bucket_summaries, rows_scanned, skipped_rows, 1
        if "date" not in reader.fieldnames or "close" not in reader.fieldnames:
            return bucket_summaries, rows_scanned, skipped_rows, 1

        first_date: date | None = None
        prev_close: float | None = None
        bucket_id = 0
        next_boundary: date | None = None
        bucket_row_count = 0
        bucket_var = RunningVariance()
        saw_any_valid_row = False

        for row_idx, row in enumerate(reader):
            rows_scanned += 1
            raw_date = str(row.get("date", "")).strip()
            raw_close = str(row.get("close", "")).strip()
            if not raw_date or not raw_close:
                skipped_rows += 1
                continue
            try:
                current_date = parse_iso_date(raw_date)
                current_close = float(raw_close)
            except (ValueError, TypeError):
                skipped_rows += 1
                continue
            if not math.isfinite(current_close):
                skipped_rows += 1
                continue

            if first_date is None:
                first_date = current_date
                next_boundary = add_months(first_date, 6)
            else:
                assert next_boundary is not None
                while current_date >= next_boundary:
                    bucket_summaries.append(
                        BucketSummary(
                            ticker=ticker,
                            bucket_id=int(bucket_id),
                            row_count=int(bucket_row_count),
                            variance=float(bucket_var.variance),
                        )
                    )
                    bucket_id += 1
                    bucket_row_count = 0
                    bucket_var = RunningVariance()
                    next_boundary = add_months(next_boundary, 6)

            ret_clipped: float | None = None
            if prev_close is not None and prev_close != 0.0 and math.isfinite(prev_close):
                raw_ret = current_close / prev_close - 1.0
                if math.isfinite(raw_ret):
                    ret_clipped = clip_value(raw_ret, clip_abs=clip_abs)
            if ret_clipped is not None:
                bucket_var.add(ret_clipped)
            bucket_row_count += 1
            prev_close = current_close
            saw_any_valid_row = True

        if saw_any_valid_row:
            bucket_summaries.append(
                BucketSummary(
                    ticker=ticker,
                    bucket_id=int(bucket_id),
                    row_count=int(bucket_row_count),
                    variance=float(bucket_var.variance),
                )
            )

    return bucket_summaries, rows_scanned, skipped_rows, 0


def resolve_threshold(
    bucket_summaries: Sequence[BucketSummary],
    target_rows: int,
) -> tuple[float, int, int]:
    if target_rows < 0:
        raise ValueError("target_rows must be >= 0")
    if not bucket_summaries:
        raise ValueError("no bucket summaries were computed")

    rows_by_variance: Dict[float, int] = {}
    buckets_by_variance: Dict[float, int] = {}
    for summary in bucket_summaries:
        rows_by_variance[summary.variance] = (
            rows_by_variance.get(summary.variance, 0) + int(summary.row_count)
        )
        buckets_by_variance[summary.variance] = (
            buckets_by_variance.get(summary.variance, 0) + 1
        )

    cumulative_rows = 0
    cumulative_buckets = 0
    threshold: float | None = None
    for variance in sorted(rows_by_variance.keys(), reverse=True):
        cumulative_rows += int(rows_by_variance[variance])
        cumulative_buckets += int(buckets_by_variance[variance])
        if cumulative_rows > int(target_rows):
            threshold = float(variance)
            break
    if threshold is None:
        raise ValueError(
            f"no threshold retains more than target_rows={int(target_rows)}; "
            f"total_rows={sum(int(item.row_count) for item in bucket_summaries)}"
        )
    return float(threshold), int(cumulative_rows), int(cumulative_buckets)


def build_kept_bucket_map(
    bucket_summaries: Iterable[BucketSummary],
    threshold: float,
) -> Dict[str, Dict[int, float]]:
    kept: Dict[str, Dict[int, float]] = {}
    for summary in bucket_summaries:
        if float(summary.variance) < float(threshold):
            continue
        kept.setdefault(summary.ticker, {})[int(summary.bucket_id)] = float(summary.variance)
    return kept


def ensure_writable_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        existing = [
            path.name
            for path in output_dir.iterdir()
            if not path.name.startswith(".")
        ]
        if existing:
            raise FileExistsError(
                f"output directory already exists and is not empty: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def write_filtered_outputs(
    ticker_paths: Sequence[Path],
    output_dir: Path,
    kept_bucket_map: Dict[str, Dict[int, float]],
    clip_abs: float,
) -> tuple[int, int]:
    written_tickers = 0
    written_rows = 0
    for path in ticker_paths:
        ticker = path.stem
        kept_for_ticker = kept_bucket_map.get(ticker)
        if not kept_for_ticker:
            continue

        out_path = output_dir / path.name
        out_handle = None
        writer = None
        wrote_any_rows = False
        with path.open("r", encoding="utf-8-sig", newline="") as in_handle:
            reader = csv.DictReader(in_handle)
            if reader.fieldnames is None:
                continue
            base_fieldnames = [
                name for name in reader.fieldnames if name not in OUTPUT_EXTRA_FIELDS
            ]
            if "date" not in reader.fieldnames or "close" not in reader.fieldnames:
                continue

            first_date: date | None = None
            prev_close: float | None = None
            bucket_id = 0
            next_boundary: date | None = None
            for row_idx, row in enumerate(reader):
                raw_date = str(row.get("date", "")).strip()
                raw_close = str(row.get("close", "")).strip()
                if not raw_date or not raw_close:
                    continue
                try:
                    current_date = parse_iso_date(raw_date)
                    current_close = float(raw_close)
                except (ValueError, TypeError):
                    continue
                if not math.isfinite(current_close):
                    continue

                if first_date is None:
                    first_date = current_date
                    next_boundary = add_months(first_date, 6)
                else:
                    assert next_boundary is not None
                    while current_date >= next_boundary:
                        bucket_id += 1
                        next_boundary = add_months(next_boundary, 6)

                ret_clipped: float | None = None
                if prev_close is not None and prev_close != 0.0 and math.isfinite(prev_close):
                    raw_ret = current_close / prev_close - 1.0
                    if math.isfinite(raw_ret):
                        ret_clipped = clip_value(raw_ret, clip_abs=clip_abs)

                source_id = resolve_source_id(row, fallback=row_idx)
                bucket_variance = kept_for_ticker.get(int(bucket_id))
                if bucket_variance is not None:
                    if writer is None:
                        out_handle = out_path.open("w", encoding="utf-8", newline="")
                        writer = csv.DictWriter(
                            out_handle,
                            fieldnames=[*base_fieldnames, *OUTPUT_EXTRA_FIELDS],
                        )
                        writer.writeheader()
                    out_row = {name: row.get(name, "") for name in base_fieldnames}
                    out_row["bucket_6m_id"] = int(bucket_id)
                    out_row["ret_cc_clip_2p5"] = format_float(ret_clipped)
                    out_row["returns_variance_clip_2p5"] = format_float(bucket_variance)
                    out_row["source_id"] = int(source_id)
                    out_row["_source_row_id_orig"] = int(source_id)
                    writer.writerow(out_row)
                    wrote_any_rows = True
                    written_rows += 1

                prev_close = current_close

        if out_handle is not None:
            out_handle.close()
        if wrote_any_rows:
            written_tickers += 1
        elif out_path.exists():
            out_path.unlink()

    return int(written_tickers), int(written_rows)


def write_summary_files(
    output_dir: Path,
    input_dir: Path,
    rows_source_total: int,
    skipped_empty: int,
    skipped_no_close_or_no_date: int,
    bucket_summaries: Sequence[BucketSummary],
    threshold: float,
    rows_kept: int,
    buckets_kept: int,
    written_tickers: int,
    written_rows: int,
    clip_abs: float,
    target_rows: int,
) -> None:
    kept_buckets = sorted(
        (item for item in bucket_summaries if float(item.variance) >= float(threshold)),
        key=lambda item: (-float(item.variance), item.ticker, int(item.bucket_id)),
    )
    kept_tickers = sorted({item.ticker for item in kept_buckets})

    kept_buckets_path = output_dir / "_kept_buckets.csv"
    with kept_buckets_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ticker", "bucket_6m_id", "bucket_ret_var_clip_2p5"],
        )
        writer.writeheader()
        for item in kept_buckets:
            writer.writerow(
                {
                    "ticker": item.ticker,
                    "bucket_6m_id": int(item.bucket_id),
                    "bucket_ret_var_clip_2p5": format_float(item.variance),
                }
            )

    kept_tickers_path = output_dir / "_kept_tickers.txt"
    kept_tickers_path.write_text("\n".join(kept_tickers) + ("\n" if kept_tickers else ""), encoding="utf-8")

    summary_lines = [
        f"source_dir={input_dir.resolve()}",
        f"output_dir={output_dir.resolve()}",
        f"tickers_scanned={len(iter_ticker_paths(input_dir))}",
        f"rows_source_total={int(rows_source_total)}",
        f"skipped_empty={int(skipped_empty)}",
        f"skipped_no_close_or_no_date={int(skipped_no_close_or_no_date)}",
        f"bucket_rows_total={len(bucket_summaries)}",
        f"x_threshold_max_for_rows_gt_{int(target_rows)}={format_float(threshold)}",
        f"rows_kept_at_x={int(rows_kept)}",
        f"buckets_kept_at_x={int(buckets_kept)}",
        f"written_tickers={int(written_tickers)}",
        f"written_rows={int(written_rows)}",
        f"return_clip_abs={format_float(clip_abs)}",
        "files: kept_buckets_csv=_kept_buckets.csv, kept_tickers_txt=_kept_tickers.txt",
    ]
    (output_dir / "_build_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"input dir not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input path is not a directory: {input_dir}")

    ticker_paths = iter_ticker_paths(input_dir)
    rows_source_total = 0
    skipped_empty = 0
    skipped_no_close_or_no_date = 0
    bucket_summaries: List[BucketSummary] = []
    for path in ticker_paths:
        ticker_buckets, rows_scanned, skipped_rows, skipped_file = scan_ticker(
            path,
            clip_abs=float(args.return_clip_abs),
        )
        bucket_summaries.extend(ticker_buckets)
        rows_source_total += int(rows_scanned)
        skipped_no_close_or_no_date += int(skipped_rows)
        skipped_empty += int(skipped_file)

    threshold, rows_kept, buckets_kept = resolve_threshold(
        bucket_summaries,
        target_rows=int(args.target_rows),
    )
    kept_bucket_map = build_kept_bucket_map(bucket_summaries, threshold=threshold)

    ensure_writable_output_dir(output_dir)
    written_tickers = 0
    written_rows = 0
    if not bool(args.summary_only):
        written_tickers, written_rows = write_filtered_outputs(
            ticker_paths=ticker_paths,
            output_dir=output_dir,
            kept_bucket_map=kept_bucket_map,
            clip_abs=float(args.return_clip_abs),
        )

    write_summary_files(
        output_dir=output_dir,
        input_dir=input_dir,
        rows_source_total=rows_source_total,
        skipped_empty=skipped_empty,
        skipped_no_close_or_no_date=skipped_no_close_or_no_date,
        bucket_summaries=bucket_summaries,
        threshold=threshold,
        rows_kept=rows_kept,
        buckets_kept=buckets_kept,
        written_tickers=written_tickers,
        written_rows=written_rows,
        clip_abs=float(args.return_clip_abs),
        target_rows=int(args.target_rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
