#!/usr/bin/env python3
"""Run dataset variance grid search in fixed-size parallel batches."""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

from grid_search_dataset_variance import (
    RESULT_FIELDNAMES,
    _nan_result_row,
    build_combos,
    parse_params_file,
    safe_combo_name,
)

DEFAULT_BATCH_SIZE = 45
DEFAULT_MAX_PARALLEL = 45


def chunked(values: List[object], size: int) -> Iterable[List[object]]:
    step = max(1, int(size))
    for i in range(0, len(values), step):
        yield values[i : i + step]


def write_commands_file(path: Path, commands: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cmd in commands:
            f.write(cmd)
            f.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch launcher for dataset variance grid search. Executes fixed-size "
            "command batches via pipeline/run_commands_parallel.py."
        )
    )
    parser.add_argument("--params-file", type=Path, default=Path("dataset_params.txt"))
    parser.add_argument("--source-dir", type=Path, default=Path("really_good_tickers"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("dataset_grid_search_results.csv"),
    )
    parser.add_argument(
        "--base-dataset-dir",
        type=Path,
        default=Path("/ephemeral/dirs"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of combos to run per batch (default: 45).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=DEFAULT_MAX_PARALLEL,
        help="Max concurrent commands per batch (default: 45).",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path("/tmp"),
        help="Scratch directory for per-batch command/row files.",
    )
    parser.add_argument(
        "--run-commands-script",
        type=Path,
        default=Path("pipeline/run_commands_parallel.py"),
        help="Path to run_commands_parallel.py",
    )
    parser.add_argument(
        "--worker-progress-every",
        type=int,
        default=250,
        help="Emit worker progress every N tickers scanned (default: 250).",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    params = parse_params_file(Path(args.params_file))
    combos = build_combos(params)
    if not combos:
        raise ValueError("no parameter combinations generated")

    source_dir = Path(args.source_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    base_dataset_dir = Path(args.base_dataset_dir).resolve()
    run_commands_script = Path(args.run_commands_script).resolve()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"source_dir not found: {source_dir}")
    if not run_commands_script.is_file():
        raise FileNotFoundError(f"run_commands_parallel.py not found: {run_commands_script}")

    batch_size = max(1, int(args.batch_size))
    max_parallel = max(1, int(args.max_parallel))
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    total = len(combos)
    total_batches = int(math.ceil(total / float(batch_size)))
    print(
        f"Starting batched grid search: combos={total}, batches={total_batches}, "
        f"batch_size={batch_size}, max_parallel={max_parallel}",
        flush=True,
    )

    with output_csv.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        out_f.flush()

        completed = 0
        for batch_idx, combo_batch_raw in enumerate(chunked(list(combos), batch_size), start=1):
            combo_batch = [c for c in combo_batch_raw]
            print(
                f"\n[batch {batch_idx}/{total_batches}] launching {len(combo_batch)} combos",
                flush=True,
            )

            with tempfile.TemporaryDirectory(
                prefix="grid_batch_",
                dir=str(Path(args.scratch_dir)),
            ) as tmp_dir_str:
                tmp_dir = Path(tmp_dir_str)
                rows_dir = tmp_dir / "rows"
                commands_file = tmp_dir / "commands.txt"
                rows_dir.mkdir(parents=True, exist_ok=True)

                commands: List[str] = []
                for combo in combo_batch:
                    assert hasattr(combo, "combo_id")
                    row_csv = rows_dir / f"combo_{int(combo.combo_id):06d}.csv"
                    cmd = " ".join(
                        [
                            "python",
                            "-u",
                            shlex.quote(str(Path(__file__).resolve().parent / "dataset_combo_variance_worker.py")),
                            "--combo-id",
                            shlex.quote(str(int(combo.combo_id))),
                            "--upper-threshold",
                            shlex.quote(str(float(combo.upper_threshold))),
                            "--lower-threshold",
                            shlex.quote(str(float(combo.lower_threshold))),
                            "--year",
                            shlex.quote(str(int(combo.year))),
                            "--source-dir",
                            shlex.quote(str(source_dir)),
                            "--base-dataset-dir",
                            shlex.quote(str(base_dataset_dir)),
                            "--row-csv-out",
                            shlex.quote(str(row_csv)),
                            "--progress-every",
                            shlex.quote(str(int(args.worker_progress_every))),
                        ]
                    )
                    commands.append(cmd)

                write_commands_file(commands_file, commands)
                print(
                    f"[batch {batch_idx}/{total_batches}] commands_file={commands_file}",
                    flush=True,
                )

                launcher_cmd = [
                    "python",
                    "-u",
                    str(run_commands_script),
                    "--commands-file",
                    str(commands_file),
                    "--max-parallel",
                    str(max_parallel),
                ]
                proc = subprocess.run(launcher_cmd, check=False)
                print(
                    f"[batch {batch_idx}/{total_batches}] launcher_exit={proc.returncode}",
                    flush=True,
                )

                batch_rows: List[Dict[str, object]] = []
                for combo in combo_batch:
                    row_csv = rows_dir / f"combo_{int(combo.combo_id):06d}.csv"
                    if row_csv.is_file():
                        try:
                            with row_csv.open("r", newline="", encoding="utf-8") as f:
                                reader = csv.DictReader(f)
                                payload = next(reader, None)
                            if payload is None:
                                raise ValueError("worker CSV has no data rows")
                        except Exception as exc:  # noqa: BLE001
                            payload = _nan_result_row(
                                combo=combo,
                                dataset_dir=base_dataset_dir / safe_combo_name(combo),
                                error=f"invalid worker CSV: {type(exc).__name__}: {exc}",
                            )
                    else:
                        payload = _nan_result_row(
                            combo=combo,
                            dataset_dir=base_dataset_dir / safe_combo_name(combo),
                            error="missing worker row output",
                        )
                        if proc.returncode != 0:
                            payload["error"] = (
                                f"missing worker row output (batch launcher rc={proc.returncode})"
                            )
                    batch_rows.append(payload)

                # Write rows in combo-id order for deterministic output.
                batch_rows.sort(key=lambda r: int(r.get("combo_id", -1)))
                for row in batch_rows:
                    writer.writerow(row)
                    completed += 1
                out_f.flush()

                batch_errors = sum(1 for row in batch_rows if str(row.get("error", "")).strip())
                print(
                    f"[batch {batch_idx}/{total_batches}] completed rows={len(batch_rows)} "
                    f"errors={batch_errors} cumulative={completed}/{total}",
                    flush=True,
                )

    print(f"\nDone. Results written to: {output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
