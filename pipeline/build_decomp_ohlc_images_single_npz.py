#!/usr/bin/env python3
"""Build decomp OHLC images into one consolidated .npy bundle directory.

This is a chunked writer that avoids sharded outputs and writes uncompressed
NumPy arrays suitable for memmap-based runtime access:
  X_img.npy, y_raw.npy, timestamps.npy, sample_indices.npy, ticker_ids.npy
  (and X_vix_img.npy when --vix-image is enabled)

It also writes metadata arrays and a manifest JSON in the same directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from build_decomp_ohlc_images import (
    MOVING_AVERAGE_CONNECT_ACROSS_WEEKEND_GAP_DEFAULT,
    MOVING_AVERAGE_MODE_DEFAULT,
    MOVING_AVERAGE_WINDOW_DEFAULT,
    OhlcImageSpec,
    VIX_DAILY_CSV_DEFAULT,
    VIX_FEATURE_LOOKBACK_DAYS,
    VIX_FEATURE_PANEL_HEIGHT,
    VIX_FEATURE_PANEL_GAP_ROWS,
    VIX_FEATURE_SOURCE_COLUMN,
    VIX_FEATURE_WINDOW_AGGREGATION,
    VIX_IMAGE_BARS,
    VIX_IMAGE_ENABLED_DEFAULT,
    VIX_IMAGE_MOVING_AVERAGE_MODE,
    VIX_IMAGE_MOVING_AVERAGE_WINDOW,
    VIX_FEATURE_ENABLED_DEFAULT,
    WEEKEND_FEATURE_ENABLED_DEFAULT,
    build_vix_close_moving_average_by_row,
    build_vix_feature_values_by_scale,
    build_vix_image_spec,
    compute_window_sizes,
    find_required_feature_indices,
    load_vix_feature_series_by_row,
    load_vix_ohlc_history,
    render_batch,
    render_vix_image_batch,
    resolve_moving_average_index,
    resolve_weekend_gap_settings,
    resolve_volume_index,
    select_source_indices_for_vix_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=Path("data/daily/stock_dataset_decomp.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/daily/stock_dataset_decomp_ohlc_images_turnover_npy"),
        help=(
            "Output directory for .npy bundle files "
            "(X_img.npy, y_raw.npy, timestamps.npy, ...)."
        ),
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help=(
            "Deprecated compatibility alias. If provided, output is written to "
            "a directory derived from this path ('.npz' suffix removed)."
        ),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="<=0 means all samples from start-index.",
    )
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument(
        "--vix-image",
        action="store_true",
        default=bool(VIX_IMAGE_ENABLED_DEFAULT),
        help=(
            "Render and store X_vix_img.npy using 60-bar VIX OHLC windows. "
            "Includes an overlaid simple moving-average line "
            f"(window={int(VIX_IMAGE_MOVING_AVERAGE_WINDOW)}), sourced from "
            "daily_vix.csv history. Samples without sufficient VIX history are dropped."
        ),
    )
    parser.add_argument(
        "--vix-daily-csv",
        type=Path,
        default=Path(VIX_DAILY_CSV_DEFAULT),
        help=f"Daily VIX OHLC CSV path (default: {VIX_DAILY_CSV_DEFAULT}).",
    )
    parser.add_argument(
        "--include-vix-feature",
        action="store_true",
        default=bool(VIX_FEATURE_ENABLED_DEFAULT),
        help=(
            "Append a per-scale VIX feature panel to X_img. "
            "Uses 60-day VIX lookback with robust_zscore_close averaged "
            "within each aggregation window. "
            f"Adds {int(VIX_FEATURE_PANEL_GAP_ROWS + VIX_FEATURE_PANEL_HEIGHT)}px height."
        ),
    )
    parser.add_argument("--include-volume", action="store_true")
    parser.add_argument("--include-moving-average", action="store_true")
    parser.add_argument(
        "--moving-average-feature",
        type=str,
        default=None,
        help=(
            "Feature name for moving average values "
            "(default: ma_n_syn, then ma_n if present)."
        ),
    )
    parser.add_argument(
        "--moving-average-window",
        type=int,
        default=None,
        help=(
            "Moving-average window metadata value to record in manifest "
            "(default: source npz value when present, else 5)."
        ),
    )
    parser.add_argument(
        "--moving-average-mode",
        type=str,
        default=None,
        help=(
            "Moving-average mode metadata value to record in manifest "
            "(default: source npz value when present, else trailing_sma_min_periods_1)."
        ),
    )
    parser.add_argument(
        "--moving-average-connect-across-weekend-gap",
        action=argparse.BooleanOptionalAction,
        default=bool(MOVING_AVERAGE_CONNECT_ACROSS_WEEKEND_GAP_DEFAULT),
        help=(
            "Connect moving-average line segments across the inserted weekend gap "
            "(default: disabled)."
        ),
    )
    parser.add_argument("--volume-feature", type=str, default="turnover")
    parser.add_argument(
        "--enable-weekend-feature",
        action="store_true",
        default=bool(WEEKEND_FEATURE_ENABLED_DEFAULT),
        help="Enable weekend gap rendering (disabled by default).",
    )
    parser.add_argument("--price-height", type=int, default=25)
    parser.add_argument("--volume-height", type=int, default=6)
    parser.add_argument("--temp-dir", type=Path, default=None)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted build from output-dir/manifest.json and existing "
            ".npy files."
        ),
    )
    return parser.parse_args()


def resolve_output_dir(output_dir: Path, output_npz_compat: Optional[Path]) -> Path:
    if output_npz_compat is None:
        return Path(output_dir)
    compat = Path(output_npz_compat)
    if compat.suffix == ".npz":
        compat = compat.with_suffix("")
    return compat


def write_manifest_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _remove_path_if_exists(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def publish_file_alias(link_dir: Path, file_name: str, target_dir: Path) -> None:
    if link_dir.resolve() == target_dir.resolve():
        return
    link_dir.mkdir(parents=True, exist_ok=True)
    link_path = link_dir / str(file_name)
    target_path = target_dir / str(file_name)
    if link_path.exists() or link_path.is_symlink():
        _remove_path_if_exists(link_path)
    try:
        link_path.symlink_to(target_path)
    except OSError:
        if not target_path.exists():
            return
        os.link(target_path, link_path)


def main() -> None:
    args = parse_args()
    args.output_dir = resolve_output_dir(args.output_dir, args.output_npz)
    if bool(args.overwrite) and bool(args.resume):
        raise ValueError("--overwrite and --resume are mutually exclusive")
    resume_manifest: dict | None = None
    if bool(args.resume):
        if not args.output_dir.exists():
            raise FileNotFoundError(
                f"resume requested but output dir does not exist: {args.output_dir}"
            )
        resume_manifest_path = args.output_dir / "manifest.json"
        if not resume_manifest_path.exists():
            raise FileNotFoundError(
                f"resume requested but manifest not found: {resume_manifest_path}"
            )
        with resume_manifest_path.open("r", encoding="utf-8") as f:
            resume_manifest = json.load(f)
        if not isinstance(resume_manifest, dict):
            raise ValueError(f"invalid resume manifest: {resume_manifest_path}")
    elif args.output_dir.exists():
        if not bool(args.overwrite):
            raise FileExistsError(
                f"output dir exists: {args.output_dir} "
                "(use --overwrite to replace)"
            )
        shutil.rmtree(args.output_dir, ignore_errors=True)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)

    with np.load(args.input_npz, allow_pickle=True) as data:
        X = data["X"]
        y_raw = data["y_raw"]
        timestamps = data["timestamps"]
        ticker_ids = data["ticker_ids"] if "ticker_ids" in data.files else None
        tickers = data["tickers"] if "tickers" in data.files else None
        feature_cols = [str(x) for x in data["feature_cols"].tolist()]
        label_cols = (
            [str(x) for x in data["label_cols"].tolist()]
            if "label_cols" in data.files
            else []
        )
        lookback = int(data["lookback"][0]) if "lookback" in data.files else None
        scales = (
            int(data["decomposition_scales"][0])
            if "decomposition_scales" in data.files
            else int(X.shape[1])
        )
        windows = (
            int(data["decomposition_windows"][0])
            if "decomposition_windows" in data.files
            else int(X.shape[3])
        )
        normalization = (
            str(data["decomposition_normalization"][0])
            if "decomposition_normalization" in data.files
            else ""
        )
        moving_average_window_src = (
            int(data["moving_average_window"][0])
            if "moving_average_window" in data.files
            else None
        )
        moving_average_mode_src = (
            str(data["moving_average_mode"][0])
            if "moving_average_mode" in data.files
            else None
        )

    total = int(X.shape[0])
    start = max(0, int(args.start_index))
    if int(args.max_samples) <= 0:
        end = total
    else:
        end = min(total, start + int(args.max_samples))
    if start >= end:
        raise ValueError(f"empty slice start={start}, end={end}, total={total}")
    source_indices = np.arange(start, end, dtype=np.int64)
    source_count_before_vix_filter = int(source_indices.shape[0])
    requires_vix_history = bool(args.vix_image) or bool(args.include_vix_feature)
    vix_end_rows: np.ndarray | None = None
    vix_history_ohlc: np.ndarray | None = None
    vix_moving_average_by_row: np.ndarray | None = None
    vix_feature_series_by_row: np.ndarray | None = None
    vix_spec: OhlcImageSpec | None = None
    vix_drop_missing_date_count = 0
    vix_drop_insufficient_history_count = 0
    if bool(requires_vix_history):
        date_to_row, vix_history_ohlc = load_vix_ohlc_history(Path(args.vix_daily_csv))
        vix_required_bars = int(VIX_FEATURE_LOOKBACK_DAYS)
        if bool(args.vix_image):
            vix_required_bars = max(
                int(vix_required_bars),
                int(VIX_IMAGE_BARS + VIX_IMAGE_MOVING_AVERAGE_WINDOW - 1),
            )
        (
            source_indices,
            vix_end_rows,
            vix_drop_missing_date_count,
            vix_drop_insufficient_history_count,
        ) = select_source_indices_for_vix_image(
            source_indices=source_indices,
            timestamps=timestamps,
            date_to_row=date_to_row,
            bars=int(vix_required_bars),
        )
        if int(source_indices.shape[0]) <= 0:
            raise ValueError(
                "vix-source filtering dropped all samples; "
                f"source_count={source_count_before_vix_filter} "
                f"dropped_missing_date={vix_drop_missing_date_count} "
                f"dropped_insufficient_history={vix_drop_insufficient_history_count}"
            )
        if bool(args.include_vix_feature):
            vix_feature_series_by_row = load_vix_feature_series_by_row(
                csv_path=Path(args.vix_daily_csv),
                date_to_row=date_to_row,
                feature_col=str(VIX_FEATURE_SOURCE_COLUMN),
            )
        if bool(args.vix_image):
            vix_moving_average_by_row = build_vix_close_moving_average_by_row(
                vix_ohlc_history=vix_history_ohlc,
                window=int(VIX_IMAGE_MOVING_AVERAGE_WINDOW),
            )
    if bool(args.vix_image):
        vix_spec = build_vix_image_spec()
    subset_count = int(source_indices.shape[0])
    subset_start = int(start)
    subset_end = int(subset_start + subset_count)

    idx = find_required_feature_indices(feature_cols)
    volume_idx, volume_name = resolve_volume_index(
        feature_cols,
        include_volume=bool(args.include_volume),
        preferred=args.volume_feature,
    )
    moving_average_idx, moving_average_feature = resolve_moving_average_index(
        feature_cols,
        include_moving_average=bool(args.include_moving_average),
        preferred=args.moving_average_feature,
    )
    moving_average_window = (
        int(args.moving_average_window)
        if args.moving_average_window is not None
        else (
            int(moving_average_window_src)
            if moving_average_window_src is not None
            else int(MOVING_AVERAGE_WINDOW_DEFAULT)
        )
    )
    if int(moving_average_window) < 1:
        raise ValueError("moving_average_window must be >= 1")
    moving_average_mode = (
        str(args.moving_average_mode)
        if args.moving_average_mode is not None
        else (
            str(moving_average_mode_src)
            if moving_average_mode_src is not None
            else str(MOVING_AVERAGE_MODE_DEFAULT)
        )
    )
    moving_average_mode = moving_average_mode.strip()
    if not moving_average_mode:
        moving_average_mode = str(MOVING_AVERAGE_MODE_DEFAULT)
    weekend_gap_width, weekend_gap_scale_indices = resolve_weekend_gap_settings(
        enabled=bool(args.enable_weekend_feature)
    )
    spec = OhlcImageSpec(
        height=int(args.height),
        day_width=3,
        weekend_gap_width=weekend_gap_width,
        weekend_gap_scale_indices=weekend_gap_scale_indices,
        include_volume=bool(args.include_volume),
        price_height=int(args.price_height) if args.price_height is not None else None,
        volume_height=(
            int(args.volume_height) if args.volume_height is not None else None
        ),
        foreground=1,
        background=0,
    )
    price_h, volume_h = spec.split_heights()
    vix_feature_extra_height = (
        int(VIX_FEATURE_PANEL_GAP_ROWS + VIX_FEATURE_PANEL_HEIGHT)
        if bool(args.include_vix_feature)
        else 0
    )
    image_height_total = int(spec.height + vix_feature_extra_height)
    scale_widths = spec.widths_for_scales(windows, scales)
    image_width = int(max(scale_widths))

    if resume_manifest is not None:
        temp_dir_path = args.output_dir
        publish_dir = args.output_dir
        publish_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_root = args.temp_dir
        if temp_root is None:
            temp_root = args.output_dir.parent
        temp_root = Path(temp_root)
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir_path = Path(
            tempfile.mkdtemp(prefix="ohlc_single_", dir=str(temp_root))
        )
        publish_dir = args.output_dir
        publish_dir.mkdir(parents=True, exist_ok=True)

    try:
        x_shape = (subset_count, int(scales), int(image_height_total), int(image_width))
        y_shape = (subset_count,) + tuple(y_raw.shape[1:])
        vix_shape = None
        if bool(args.vix_image):
            if vix_spec is None:
                raise RuntimeError("vix-image enabled but vix spec is missing")
            vix_shape = (
                int(subset_count),
                int(vix_spec.height),
                int(vix_spec.width_for_scale(int(VIX_IMAGE_BARS), 0)),
            )
        resume_ready_count = 0
        if resume_manifest is not None:
            def _normalize_path_like(value: object) -> str:
                return str(Path(str(value)).resolve())

            expected_pairs: list[tuple[str, object]] = [
                ("subset_start", int(subset_start)),
                ("subset_end_target", int(subset_end)),
                ("subset_count_target", int(subset_count)),
                ("decomposition_scales", int(scales)),
                ("decomposition_windows", int(windows)),
                ("image_height", int(image_height_total)),
                ("image_width", int(image_width)),
                ("include_vix_image", bool(args.vix_image)),
                ("include_vix_feature", bool(args.include_vix_feature)),
            ]
            for key, expected in expected_pairs:
                got = resume_manifest.get(key, None)
                if got is None:
                    continue
                if got != expected:
                    raise ValueError(
                        "resume manifest mismatch for "
                        f"{key}: expected={expected} got={got}"
                    )
            source_npz_prev = resume_manifest.get("source_npz", None)
            if source_npz_prev is not None:
                if _normalize_path_like(source_npz_prev) != _normalize_path_like(args.input_npz):
                    raise ValueError(
                        "resume manifest mismatch for source_npz: "
                        f"expected={args.input_npz} got={source_npz_prev}"
                    )
            resume_ready_count = int(
                resume_manifest.get(
                    "arrays_valid_count",
                    resume_manifest.get("subset_count", 0),
                )
            )
            resume_ready_count = int(
                max(0, min(int(subset_count), int(resume_ready_count)))
            )
            print(
                "resume: "
                f"ready_count={resume_ready_count} "
                f"remaining={int(subset_count - resume_ready_count)} "
                f"dir={args.output_dir}"
            )

        memmap_mode = "r+" if resume_manifest is not None else "w+"
        X_mm = np.lib.format.open_memmap(
            temp_dir_path / "X_img.npy",
            mode=memmap_mode,
            dtype=np.uint8,
            shape=x_shape,
        )
        if tuple(X_mm.shape) != tuple(x_shape) or X_mm.dtype != np.dtype(np.uint8):
            raise ValueError(
                "X_img.npy shape/dtype mismatch for resume: "
                f"expected shape={x_shape} dtype={np.dtype(np.uint8)} "
                f"got shape={X_mm.shape} dtype={X_mm.dtype}"
            )
        publish_file_alias(publish_dir, "X_img.npy", temp_dir_path)
        vix_mm = None
        if vix_shape is not None:
            vix_mm = np.lib.format.open_memmap(
                temp_dir_path / "X_vix_img.npy",
                mode=memmap_mode,
                dtype=np.uint8,
                shape=vix_shape,
            )
            if tuple(vix_mm.shape) != tuple(vix_shape) or vix_mm.dtype != np.dtype(np.uint8):
                raise ValueError(
                    "X_vix_img.npy shape/dtype mismatch for resume: "
                    f"expected shape={vix_shape} dtype={np.dtype(np.uint8)} "
                    f"got shape={vix_mm.shape} dtype={vix_mm.dtype}"
                )
            publish_file_alias(publish_dir, "X_vix_img.npy", temp_dir_path)
        y_mm = np.lib.format.open_memmap(
            temp_dir_path / "y_raw.npy",
            mode=memmap_mode,
            dtype=y_raw.dtype,
            shape=y_shape,
        )
        if tuple(y_mm.shape) != tuple(y_shape) or y_mm.dtype != np.dtype(y_raw.dtype):
            raise ValueError(
                "y_raw.npy shape/dtype mismatch for resume: "
                f"expected shape={y_shape} dtype={np.dtype(y_raw.dtype)} "
                f"got shape={y_mm.shape} dtype={y_mm.dtype}"
            )
        publish_file_alias(publish_dir, "y_raw.npy", temp_dir_path)
        ts_mm = np.lib.format.open_memmap(
            temp_dir_path / "timestamps.npy",
            mode=memmap_mode,
            dtype=np.dtype("<U32"),
            shape=(subset_count,),
        )
        if tuple(ts_mm.shape) != (subset_count,) or ts_mm.dtype != np.dtype("<U32"):
            raise ValueError(
                "timestamps.npy shape/dtype mismatch for resume: "
                f"expected shape={(subset_count,)} dtype={np.dtype('<U32')} "
                f"got shape={ts_mm.shape} dtype={ts_mm.dtype}"
            )
        publish_file_alias(publish_dir, "timestamps.npy", temp_dir_path)
        si_mm = np.lib.format.open_memmap(
            temp_dir_path / "sample_indices.npy",
            mode=memmap_mode,
            dtype=np.int64,
            shape=(subset_count,),
        )
        if tuple(si_mm.shape) != (subset_count,) or si_mm.dtype != np.dtype(np.int64):
            raise ValueError(
                "sample_indices.npy shape/dtype mismatch for resume: "
                f"expected shape={(subset_count,)} dtype={np.dtype(np.int64)} "
                f"got shape={si_mm.shape} dtype={si_mm.dtype}"
            )
        publish_file_alias(publish_dir, "sample_indices.npy", temp_dir_path)
        tid_mm = None
        if ticker_ids is not None:
            tid_mm = np.lib.format.open_memmap(
                temp_dir_path / "ticker_ids.npy",
                mode=memmap_mode,
                dtype=np.int32,
                shape=(subset_count,),
            )
            if tuple(tid_mm.shape) != (subset_count,) or tid_mm.dtype != np.dtype(np.int32):
                raise ValueError(
                    "ticker_ids.npy shape/dtype mismatch for resume: "
                    f"expected shape={(subset_count,)} dtype={np.dtype(np.int32)} "
                    f"got shape={tid_mm.shape} dtype={tid_mm.dtype}"
                )
            publish_file_alias(publish_dir, "ticker_ids.npy", temp_dir_path)

        chunk = max(1, int(args.chunk_size))
        side_meta = {
            "output_dir": str(args.output_dir),
            "source_npz": str(args.input_npz),
            "subset_start": int(subset_start),
            "subset_end": int(subset_start),
            "subset_count": 0,
            "subset_end_target": int(subset_end),
            "subset_count_target": int(subset_count),
            "source_selection_start": int(start),
            "source_selection_end": int(end),
            "source_selection_count": int(source_count_before_vix_filter),
            "arrays_valid_count": 0,
            "in_progress": True,
            "chunk_size": int(chunk),
            "decomposition_scales": int(scales),
            "decomposition_windows": int(windows),
            "image_height": int(image_height_total),
            "image_height_base": int(spec.height),
            "image_width": int(image_width),
            "image_width_per_scale": [int(x) for x in scale_widths],
            "weekend_feature_enabled": bool(args.enable_weekend_feature),
            "weekend_gap_width": int(spec.weekend_gap_width),
            "weekend_gap_scale_indices": spec.normalized_weekend_gap_scale_indices(),
            "price_panel_height": int(price_h),
            "volume_panel_height": int(volume_h),
            "include_volume": bool(spec.include_volume),
            "volume_feature": volume_name,
            "include_moving_average": bool(args.include_moving_average),
            "moving_average_window": int(moving_average_window),
            "moving_average_mode": str(moving_average_mode),
            "moving_average_connect_across_weekend_gap": bool(
                args.moving_average_connect_across_weekend_gap
            ),
            "include_vix_image": bool(args.vix_image),
            "include_vix_feature": bool(args.include_vix_feature),
            "arrays": {
                "X_img": {
                    "file": "X_img.npy",
                    "dtype": str(np.dtype(np.uint8)),
                    "shape": [int(x) for x in x_shape],
                    "ready_count": 0,
                },
                "X_vix_img": (
                    {
                        "file": "X_vix_img.npy",
                        "dtype": str(np.dtype(np.uint8)),
                        "shape": [int(x) for x in vix_shape],
                        "ready_count": 0,
                    }
                    if vix_shape is not None
                    else None
                ),
                "y_raw": {
                    "file": "y_raw.npy",
                    "dtype": str(y_raw.dtype),
                    "shape": [int(x) for x in y_shape],
                    "ready_count": 0,
                },
                "timestamps": {
                    "file": "timestamps.npy",
                    "dtype": str(np.dtype("<U32")),
                    "shape": [int(subset_count)],
                    "ready_count": 0,
                },
                "sample_indices": {
                    "file": "sample_indices.npy",
                    "dtype": str(np.dtype(np.int64)),
                    "shape": [int(subset_count)],
                    "ready_count": 0,
                },
                "ticker_ids": (
                    {
                        "file": "ticker_ids.npy",
                        "dtype": str(np.dtype(np.int32)),
                        "shape": [int(subset_count)],
                        "ready_count": 0,
                    }
                    if ticker_ids is not None
                    else None
                ),
                "feature_cols_source": {"file": "feature_cols_source.npy"},
                "label_cols": {"file": "label_cols.npy"},
                "tickers": {"file": "tickers.npy"} if tickers is not None else None,
            },
        }
        if bool(args.vix_image):
            if vix_spec is None:
                raise RuntimeError("vix-image enabled but vix spec is missing")
            side_meta["vix_image_height"] = int(vix_spec.height)
            side_meta["vix_image_width"] = int(
                vix_spec.width_for_scale(int(VIX_IMAGE_BARS), 0)
            )
            side_meta["vix_image_bars"] = int(VIX_IMAGE_BARS)
            side_meta["vix_image_day_width"] = int(vix_spec.day_width)
            side_meta["vix_image_source_csv"] = str(Path(args.vix_daily_csv))
            side_meta["vix_image_include_moving_average"] = True
            side_meta["vix_image_moving_average_window"] = int(
                VIX_IMAGE_MOVING_AVERAGE_WINDOW
            )
            side_meta["vix_image_moving_average_mode"] = str(
                VIX_IMAGE_MOVING_AVERAGE_MODE
            )
            side_meta["vix_image_dropped_missing_date_count"] = int(
                vix_drop_missing_date_count
            )
            side_meta["vix_image_dropped_insufficient_history_count"] = int(
                vix_drop_insufficient_history_count
            )
            side_meta["vix_image_source_selected_count"] = int(
                source_count_before_vix_filter
            )
            side_meta["vix_image_retained_count"] = int(subset_count)
        if bool(args.include_vix_feature):
            side_meta["vix_feature_lookback_days"] = int(VIX_FEATURE_LOOKBACK_DAYS)
            side_meta["vix_feature_panel_height"] = int(VIX_FEATURE_PANEL_HEIGHT)
            side_meta["vix_feature_gap_rows"] = int(VIX_FEATURE_PANEL_GAP_ROWS)
            side_meta["vix_feature_source_csv"] = str(Path(args.vix_daily_csv))
            side_meta["vix_feature_source_column"] = str(VIX_FEATURE_SOURCE_COLUMN)
            side_meta["vix_feature_window_aggregation"] = str(
                VIX_FEATURE_WINDOW_AGGREGATION
            )
            side_meta["vix_feature_window_sizes"] = [
                int(x)
                for x in compute_window_sizes(
                    lookback=int(VIX_FEATURE_LOOKBACK_DAYS),
                    windows=int(windows),
                    scales=int(scales),
                )
            ]
            side_meta["vix_feature_panel_rows"] = [
                int(spec.height + int(VIX_FEATURE_PANEL_GAP_ROWS)),
                int(image_height_total - 1),
            ]
            side_meta["vix_feature_dropped_missing_date_count"] = int(
                vix_drop_missing_date_count
            )
            side_meta["vix_feature_dropped_insufficient_history_count"] = int(
                vix_drop_insufficient_history_count
            )
            side_meta["vix_feature_source_selected_count"] = int(
                source_count_before_vix_filter
            )
            side_meta["vix_feature_retained_count"] = int(subset_count)
        if moving_average_feature is not None:
            side_meta["moving_average_source"] = "feature_channel"
            side_meta["moving_average_feature"] = str(moving_average_feature)

        def save_side_array(file_name: str, value: np.ndarray) -> None:
            np.save(temp_dir_path / file_name, value)
            publish_file_alias(publish_dir, file_name, temp_dir_path)

        def write_progress_manifest(ready_count: int, in_progress: bool) -> None:
            ready = int(max(0, min(int(subset_count), int(ready_count))))
            side_meta["subset_end"] = int(subset_start + ready)
            side_meta["subset_count"] = int(ready)
            side_meta["arrays_valid_count"] = int(ready)
            side_meta["in_progress"] = bool(in_progress)
            arrays_meta = side_meta.get("arrays")
            if isinstance(arrays_meta, dict):
                for key in (
                    "X_img",
                    "X_vix_img",
                    "y_raw",
                    "timestamps",
                    "sample_indices",
                    "ticker_ids",
                ):
                    entry = arrays_meta.get(key)
                    if isinstance(entry, dict):
                        entry["ready_count"] = int(ready)
            write_manifest_atomic(temp_dir_path / "manifest.json", side_meta)
            publish_file_alias(publish_dir, "manifest.json", temp_dir_path)

        write_progress_manifest(ready_count=resume_ready_count, in_progress=True)
        for rel_start in range(resume_ready_count, subset_count, chunk):
            rel_end = min(subset_count, rel_start + chunk)
            src_idx_chunk = source_indices[rel_start:rel_end]
            if int(src_idx_chunk.shape[0]) <= 0:
                continue

            X_chunk = X[src_idx_chunk].astype(np.float64, copy=False)
            ts_chunk = np.asarray(timestamps[src_idx_chunk]).astype(str)
            vix_feature_chunk = None
            if bool(args.include_vix_feature):
                if vix_feature_series_by_row is None or vix_end_rows is None:
                    raise RuntimeError(
                        "include-vix-feature enabled but vix runtime state is missing"
                    )
                vix_feature_chunk = build_vix_feature_values_by_scale(
                    vix_feature_series_by_row=vix_feature_series_by_row,
                    end_rows=vix_end_rows[rel_start:rel_end],
                    scales=int(scales),
                    windows=int(windows),
                    lookback_days=int(VIX_FEATURE_LOOKBACK_DAYS),
                )
            img_chunk = render_batch(
                X_chunk,
                open_idx=idx["open"],
                high_idx=idx["high"],
                low_idx=idx["low"],
                close_idx=idx["close"],
                volume_idx=volume_idx,
                spec=spec,
                sample_timestamps=ts_chunk,
                moving_average_idx=moving_average_idx,
                moving_average_connect_across_weekend_gap=bool(
                    args.moving_average_connect_across_weekend_gap
                ),
                vix_feature_values=vix_feature_chunk,
                vix_feature_panel_height=int(VIX_FEATURE_PANEL_HEIGHT),
                vix_feature_gap_rows=int(VIX_FEATURE_PANEL_GAP_ROWS),
            )
            X_mm[rel_start:rel_end] = img_chunk
            if bool(args.vix_image):
                if vix_mm is None or vix_history_ohlc is None or vix_end_rows is None or vix_spec is None:
                    raise RuntimeError("vix-image enabled but vix runtime state is missing")
                vix_mm[rel_start:rel_end] = render_vix_image_batch(
                    vix_ohlc_history=vix_history_ohlc,
                    end_rows=vix_end_rows[rel_start:rel_end],
                    spec=vix_spec,
                    bars=int(VIX_IMAGE_BARS),
                    moving_average_by_row=vix_moving_average_by_row,
                )
            y_mm[rel_start:rel_end] = y_raw[src_idx_chunk]
            ts_mm[rel_start:rel_end] = np.asarray(ts_chunk, dtype="<U32")
            si_mm[rel_start:rel_end] = np.asarray(src_idx_chunk, dtype=np.int64)
            if tid_mm is not None:
                tid_mm[rel_start:rel_end] = ticker_ids[src_idx_chunk]
            X_mm.flush()
            if vix_mm is not None:
                vix_mm.flush()
            y_mm.flush()
            ts_mm.flush()
            si_mm.flush()
            if tid_mm is not None:
                tid_mm.flush()
            write_progress_manifest(ready_count=rel_end, in_progress=True)

            print(f"processed rel samples [{rel_start}, {rel_end})")

        # Ensure memmap buffers are written before directory move.
        X_mm.flush()
        if vix_mm is not None:
            vix_mm.flush()
        y_mm.flush()
        ts_mm.flush()
        si_mm.flush()
        if tid_mm is not None:
            tid_mm.flush()

        # Persist smaller metadata arrays as standalone .npy for easy loading.
        save_side_array("feature_cols_source.npy", np.array(feature_cols, dtype=object))
        save_side_array("label_cols.npy", np.array(label_cols, dtype=object))
        save_side_array("decomposition_scales.npy", np.array([int(scales)]))
        save_side_array("decomposition_windows.npy", np.array([int(windows)]))
        save_side_array("decomposition_normalization.npy", np.array([normalization], dtype=object))
        save_side_array("image_height.npy", np.array([int(image_height_total)]))
        save_side_array("image_height_base.npy", np.array([int(spec.height)]))
        save_side_array("image_width.npy", np.array([int(image_width)]))
        save_side_array(
            "image_width_per_scale.npy",
            np.array([int(x) for x in scale_widths], dtype=np.int64),
        )
        save_side_array("day_width.npy", np.array([int(spec.day_width)]))
        save_side_array(
            "weekend_feature_enabled.npy",
            np.array([bool(args.enable_weekend_feature)]),
        )
        save_side_array("weekend_gap_width.npy", np.array([int(spec.weekend_gap_width)]))
        save_side_array(
            "weekend_gap_scale_indices.npy",
            np.array(spec.normalized_weekend_gap_scale_indices(), dtype=np.int64),
        )
        save_side_array("price_panel_height.npy", np.array([int(price_h)]))
        save_side_array("volume_panel_height.npy", np.array([int(volume_h)]))
        save_side_array("include_volume.npy", np.array([bool(spec.include_volume)]))
        save_side_array(
            "volume_feature.npy",
            np.array([volume_name if volume_name else ""], dtype=object),
        )
        save_side_array(
            "include_moving_average.npy",
            np.array([bool(args.include_moving_average)]),
        )
        save_side_array(
            "moving_average_source.npy",
            np.array(
                ["feature_channel" if moving_average_feature is not None else ""],
                dtype=object,
            ),
        )
        save_side_array(
            "moving_average_feature.npy",
            np.array(
                [str(moving_average_feature) if moving_average_feature is not None else ""],
                dtype=object,
            ),
        )
        save_side_array(
            "moving_average_window.npy",
            np.array([int(moving_average_window)], dtype=np.int64),
        )
        save_side_array(
            "moving_average_mode.npy",
            np.array([str(moving_average_mode)], dtype=object),
        )
        save_side_array(
            "moving_average_connect_across_weekend_gap.npy",
            np.array([bool(args.moving_average_connect_across_weekend_gap)]),
        )
        save_side_array("include_vix_image.npy", np.array([bool(args.vix_image)]))
        save_side_array(
            "include_vix_feature.npy",
            np.array([bool(args.include_vix_feature)]),
        )
        if bool(args.vix_image):
            if vix_spec is None:
                raise RuntimeError("vix-image enabled but vix spec is missing")
            save_side_array("vix_image_height.npy", np.array([int(vix_spec.height)]))
            save_side_array(
                "vix_image_width.npy",
                np.array([int(vix_spec.width_for_scale(int(VIX_IMAGE_BARS), 0))]),
            )
            save_side_array("vix_image_bars.npy", np.array([int(VIX_IMAGE_BARS)]))
            save_side_array("vix_image_day_width.npy", np.array([int(vix_spec.day_width)]))
            save_side_array("vix_image_include_moving_average.npy", np.array([True]))
            save_side_array(
                "vix_image_moving_average_window.npy",
                np.array([int(VIX_IMAGE_MOVING_AVERAGE_WINDOW)], dtype=np.int64),
            )
            save_side_array(
                "vix_image_moving_average_mode.npy",
                np.array([str(VIX_IMAGE_MOVING_AVERAGE_MODE)], dtype=object),
            )
            save_side_array(
                "vix_image_source_csv.npy",
                np.array([str(Path(args.vix_daily_csv))], dtype=object),
            )
            save_side_array(
                "vix_image_dropped_missing_date_count.npy",
                np.array([int(vix_drop_missing_date_count)]),
            )
            save_side_array(
                "vix_image_dropped_insufficient_history_count.npy",
                np.array([int(vix_drop_insufficient_history_count)]),
            )
            save_side_array(
                "vix_image_source_selected_count.npy",
                np.array([int(source_count_before_vix_filter)]),
            )
            save_side_array("vix_image_retained_count.npy", np.array([int(subset_count)]))
        if bool(args.include_vix_feature):
            save_side_array(
                "vix_feature_lookback_days.npy",
                np.array([int(VIX_FEATURE_LOOKBACK_DAYS)], dtype=np.int64),
            )
            save_side_array(
                "vix_feature_panel_height.npy",
                np.array([int(VIX_FEATURE_PANEL_HEIGHT)], dtype=np.int64),
            )
            save_side_array(
                "vix_feature_gap_rows.npy",
                np.array([int(VIX_FEATURE_PANEL_GAP_ROWS)], dtype=np.int64),
            )
            save_side_array(
                "vix_feature_source_csv.npy",
                np.array([str(Path(args.vix_daily_csv))], dtype=object),
            )
            save_side_array(
                "vix_feature_source_column.npy",
                np.array([str(VIX_FEATURE_SOURCE_COLUMN)], dtype=object),
            )
            save_side_array(
                "vix_feature_window_aggregation.npy",
                np.array([str(VIX_FEATURE_WINDOW_AGGREGATION)], dtype=object),
            )
            save_side_array(
                "vix_feature_window_sizes.npy",
                np.array(
                    compute_window_sizes(
                        lookback=int(VIX_FEATURE_LOOKBACK_DAYS),
                        windows=int(windows),
                        scales=int(scales),
                    ),
                    dtype=np.int64,
                ),
            )
            save_side_array(
                "vix_feature_panel_rows.npy",
                np.array(
                    [int(spec.height + int(VIX_FEATURE_PANEL_GAP_ROWS)), int(image_height_total - 1)],
                    dtype=np.int64,
                ),
            )
            save_side_array(
                "vix_feature_dropped_missing_date_count.npy",
                np.array([int(vix_drop_missing_date_count)], dtype=np.int64),
            )
            save_side_array(
                "vix_feature_dropped_insufficient_history_count.npy",
                np.array([int(vix_drop_insufficient_history_count)], dtype=np.int64),
            )
            save_side_array(
                "vix_feature_source_selected_count.npy",
                np.array([int(source_count_before_vix_filter)], dtype=np.int64),
            )
            save_side_array(
                "vix_feature_retained_count.npy",
                np.array([int(subset_count)], dtype=np.int64),
            )
        save_side_array("subset_start.npy", np.array([int(subset_start)]))
        save_side_array("subset_end.npy", np.array([int(subset_end)]))
        save_side_array("subset_count.npy", np.array([int(subset_count)]))
        save_side_array("source_npz.npy", np.array([str(args.input_npz)], dtype=object))
        save_side_array(
            "created_utc.npy",
            np.array([datetime.now(timezone.utc).isoformat()], dtype=object),
        )
        if lookback is not None:
            save_side_array("lookback.npy", np.array([int(lookback)]))
            save_side_array(
                "scale_window_sizes.npy",
                np.array(
                    compute_window_sizes(
                        lookback=int(lookback), windows=int(windows), scales=int(scales)
                    ),
                    dtype=np.int64,
                ),
            )
        if tickers is not None:
            save_side_array("tickers.npy", tickers)

        write_progress_manifest(ready_count=subset_count, in_progress=False)
        if temp_dir_path.resolve() != args.output_dir.resolve():
            if args.output_dir.exists():
                shutil.rmtree(args.output_dir, ignore_errors=True)
            shutil.move(str(temp_dir_path), str(args.output_dir))
        if bool(requires_vix_history):
            print(
                "vix-source filter: "
                f"input={source_count_before_vix_filter} "
                f"retained={subset_count} "
                f"dropped_missing_date={vix_drop_missing_date_count} "
                f"dropped_insufficient_history={vix_drop_insufficient_history_count}"
            )
        print(f"done: {args.output_dir}")
        print(f"manifest: {args.output_dir / 'manifest.json'}")
        temp_dir_path = args.output_dir

        if args.keep_temp:
            print(f"temp preserved: {temp_dir_path}")
    finally:
        if not args.keep_temp and temp_dir_path.exists() and temp_dir_path != args.output_dir:
            if not args.output_dir.exists():
                shutil.rmtree(temp_dir_path, ignore_errors=True)


if __name__ == "__main__":
    main()
