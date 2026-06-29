#!/usr/bin/env python3
"""Run one dataset-variance combo and write one CSV result row."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from grid_search_dataset_variance import ParamCombo, RESULT_FIELDNAMES, process_combo


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Worker script: process one (upper_threshold, lower_threshold, year) "
            "combo and write one result CSV row."
        )
    )
    parser.add_argument("--combo-id", type=int, required=True)
    parser.add_argument("--upper-threshold", type=float, required=True)
    parser.add_argument("--lower-threshold", type=float, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--base-dataset-dir", type=Path, required=True)
    parser.add_argument("--row-csv-out", type=Path, required=True)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Emit intra-combo progress every N tickers scanned (default: 250).",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    source_dir = Path(args.source_dir)
    ticker_csv_paths = sorted(
        p for p in source_dir.glob("*.csv") if p.is_file() and not p.name.startswith("_")
    )
    if not ticker_csv_paths:
        raise ValueError(f"no ticker CSV files found in source_dir: {source_dir}")

    combo = ParamCombo(
        combo_id=int(args.combo_id),
        upper_threshold=float(args.upper_threshold),
        lower_threshold=float(args.lower_threshold),
        year=int(args.year),
    )
    base_dataset_dir = Path(args.base_dataset_dir)
    base_dataset_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[worker] start combo_id={combo.combo_id} year={combo.year} "
        f"lower={combo.lower_threshold:.6g} upper={combo.upper_threshold:.6g} "
        f"tickers={len(ticker_csv_paths)}",
        flush=True,
    )

    row = process_combo(
        combo=combo,
        ticker_csv_paths=ticker_csv_paths,
        base_dataset_dir=base_dataset_dir,
        progress_every_tickers=max(0, int(args.progress_every)),
        progress_logger=lambda msg: print(
            f"[worker] combo_id={combo.combo_id} {msg}",
            flush=True,
        ),
    )

    out_path = Path(args.row_csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)

    print(
        f"[worker] done combo_id={combo.combo_id} error={bool(row.get('error'))}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
