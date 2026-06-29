#!/usr/bin/env python3
"""Attach source row id columns to filtered ticker CSVs.

Matches each filtered row back to the original source ticker CSV by `date` and
writes `_source_row_id_orig` plus `_source_row_id` using the 0-based data-row
position in the source file.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ROW_ID_COLS = ("_source_row_id_orig", "_source_row_id")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach _source_row_id_orig/_source_row_id to filtered ticker CSVs by "
            "matching rows back to a source ticker directory on date."
        )
    )
    parser.add_argument(
        "--filtered-dir",
        default="55_percent",
        help="Directory containing filtered ticker CSVs.",
    )
    parser.add_argument(
        "--source-dir",
        default="tickers",
        help="Directory containing original source ticker CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory. If omitted, files are updated in place in "
            "--filtered-dir."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing row id columns if already present.",
    )
    return parser.parse_args(argv)


def iter_csv_paths(directory: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv" and not path.name.startswith("_")
    )


def build_source_row_id_map(source_path: Path) -> Dict[str, int]:
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "date" not in reader.fieldnames:
            raise ValueError(f"missing required date column in source file: {source_path}")
        mapping: Dict[str, int] = {}
        for row_idx, row in enumerate(reader):
            raw_date = str(row.get("date", "")).strip()
            if not raw_date:
                continue
            if raw_date in mapping:
                raise ValueError(
                    f"duplicate date {raw_date!r} in source file {source_path}"
                )
            mapping[raw_date] = int(row_idx)
    return mapping


def resolve_output_fieldnames(
    fieldnames: List[str],
    overwrite: bool,
) -> List[str]:
    if overwrite:
        base = [name for name in fieldnames if name not in ROW_ID_COLS]
    else:
        base = list(fieldnames)
    for name in ROW_ID_COLS:
        if name not in base:
            base.append(name)
    return base


def write_augmented_csv(
    filtered_path: Path,
    source_row_ids: Dict[str, int],
    output_path: Path,
    overwrite: bool,
) -> int:
    with filtered_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "date" not in reader.fieldnames:
            raise ValueError(f"missing required date column in filtered file: {filtered_path}")
        fieldnames = resolve_output_fieldnames(list(reader.fieldnames), overwrite=overwrite)
        rows_written = 0
        with output_path.open("w", encoding="utf-8", newline="") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                raw_date = str(row.get("date", "")).strip()
                if not raw_date:
                    raise ValueError(f"empty date in filtered file: {filtered_path}")
                if raw_date not in source_row_ids:
                    raise ValueError(
                        f"date {raw_date!r} from {filtered_path} not found in source file"
                    )
                out_row = {
                    name: row.get(name, "")
                    for name in fieldnames
                    if name not in ROW_ID_COLS
                }
                row_id = int(source_row_ids[raw_date])
                out_row["_source_row_id_orig"] = row_id
                out_row["_source_row_id"] = row_id
                writer.writerow(out_row)
                rows_written += 1
    return rows_written


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    filtered_dir = Path(args.filtered_dir)
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir) if args.output_dir else filtered_dir

    if not filtered_dir.exists():
        raise FileNotFoundError(f"filtered dir not found: {filtered_dir}")
    if not source_dir.exists():
        raise FileNotFoundError(f"source dir not found: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    files_processed = 0
    rows_written_total = 0
    for filtered_path in iter_csv_paths(filtered_dir):
        source_path = source_dir / filtered_path.name
        if not source_path.exists():
            raise FileNotFoundError(
                f"source file missing for {filtered_path.name}: {source_path}"
            )

        source_row_ids = build_source_row_id_map(source_path)
        if output_dir.resolve() == filtered_dir.resolve():
            tmp_path = filtered_path.with_suffix(filtered_path.suffix + ".tmp")
            rows_written = write_augmented_csv(
                filtered_path=filtered_path,
                source_row_ids=source_row_ids,
                output_path=tmp_path,
                overwrite=bool(args.overwrite),
            )
            tmp_path.replace(filtered_path)
        else:
            output_path = output_dir / filtered_path.name
            rows_written = write_augmented_csv(
                filtered_path=filtered_path,
                source_row_ids=source_row_ids,
                output_path=output_path,
                overwrite=bool(args.overwrite),
            )

        files_processed += 1
        rows_written_total += int(rows_written)

    print(f"files_processed={files_processed}")
    print(f"rows_written={rows_written_total}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
