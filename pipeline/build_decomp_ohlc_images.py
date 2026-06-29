#!/usr/bin/env python3
"""Build OHLC chart images from decomposition datasets.

This script reads a decomposition dataset with shape:
  X: (samples, scales, features, windows)

It renders one binary OHLC image per scale and stores shard files that are
easy to stream in model code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image

EPS = 1e-12
WEEKEND_FEATURE_ENABLED_DEFAULT = True
WEEKEND_GAP_WIDTH = 3
WEEKEND_GAP_SCALE_INDICES = (0,)
MOVING_AVERAGE_WINDOW_DEFAULT = 5
MOVING_AVERAGE_MODE_DEFAULT = "trailing_sma_min_periods_1"
MOVING_AVERAGE_CONNECT_ACROSS_WEEKEND_GAP_DEFAULT = False
VIX_IMAGE_ENABLED_DEFAULT = False
VIX_IMAGE_BARS = 60
VIX_IMAGE_HEIGHT = 96
VIX_IMAGE_DAY_WIDTH = 3
VIX_IMAGE_WIDTH = int(VIX_IMAGE_BARS * VIX_IMAGE_DAY_WIDTH)
VIX_IMAGE_MOVING_AVERAGE_WINDOW = 60
VIX_IMAGE_MOVING_AVERAGE_MODE = "trailing_sma_min_periods_window"
VIX_DAILY_CSV_DEFAULT = Path(__file__).resolve().parent.parent / "daily_vix.csv"
VIX_FEATURE_ENABLED_DEFAULT = False
VIX_FEATURE_LOOKBACK_DAYS = 60
VIX_FEATURE_PANEL_HEIGHT = 6
VIX_FEATURE_PANEL_GAP_ROWS = 1
VIX_FEATURE_SOURCE_COLUMN = "robust_zscore_close"
VIX_FEATURE_WINDOW_AGGREGATION = "mean_within_window"


def resolve_weekend_gap_settings(
    enabled: bool = WEEKEND_FEATURE_ENABLED_DEFAULT,
) -> tuple[int, tuple[int, ...]]:
    if not bool(enabled):
        return 0, ()
    return int(WEEKEND_GAP_WIDTH), tuple(int(x) for x in WEEKEND_GAP_SCALE_INDICES)


@dataclass(frozen=True)
class OhlcImageSpec:
    height: int = 32
    day_width: int = 3
    weekend_gap_width: int = 3
    weekend_gap_scale_indices: tuple[int, ...] = (0,)
    include_volume: bool = False
    price_height: int | None = None
    volume_height: int | None = None
    volume_fraction: float = 0.2
    foreground: int = 1
    background: int = 0

    def has_weekend_gap(self, scale_index: int) -> bool:
        if int(self.weekend_gap_width) <= 0:
            return False
        return int(scale_index) in self.weekend_gap_scale_indices

    def width_for_scale(self, windows: int, scale_index: int) -> int:
        width = int(self.day_width * windows)
        if self.has_weekend_gap(scale_index):
            width += int(self.weekend_gap_width)
        return int(width)

    def widths_for_scales(self, windows: int, scales: int) -> list[int]:
        return [self.width_for_scale(windows, s) for s in range(int(scales))]

    def width(self, windows: int) -> int:
        # Backward-compatible shorthand for scale index 0.
        return self.width_for_scale(windows, 0)

    def max_width(self, windows: int, scales: int) -> int:
        widths = self.widths_for_scales(windows, scales)
        if not widths:
            raise ValueError("scales must be >= 1")
        return int(max(widths))

    def __post_init__(self) -> None:
        if int(self.day_width) < 1:
            raise ValueError("day_width must be >= 1")
        if int(self.weekend_gap_width) < 0:
            raise ValueError("weekend_gap_width must be >= 0")
        if any(int(x) < 0 for x in self.weekend_gap_scale_indices):
            raise ValueError("weekend_gap_scale_indices must be non-negative")

    def normalized_weekend_gap_scale_indices(self) -> list[int]:
        seen = set()
        out = []
        for idx in self.weekend_gap_scale_indices:
            x = int(idx)
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    def split_heights(self) -> tuple[int, int]:
        if not self.include_volume:
            return self.height, 0
        if self.price_height is not None or self.volume_height is not None:
            price_h = (
                int(self.price_height)
                if self.price_height is not None
                else self.height - int(self.volume_height)
            )
            volume_h = (
                int(self.volume_height)
                if self.volume_height is not None
                else self.height - int(self.price_height)
            )
            if price_h < 1 or volume_h < 1:
                raise ValueError("price_height and volume_height must be >= 1")
            if price_h + volume_h > self.height:
                raise ValueError(
                    "price_height + volume_height must be <= total image height"
                )
            return price_h, volume_h
        volume_h = int(round(float(self.height) * float(self.volume_fraction)))
        volume_h = max(1, min(self.height - 1, volume_h))
        return self.height - volume_h, volume_h


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=Path("data/daily/stock_dataset_decomp.npz"),
        help="Input decomposition npz path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/daily/stock_dataset_decomp_ohlc_images_subset"),
        help="Output directory for manifest, shards, and previews.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting sample index in the input dataset.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2048,
        help="Max samples to process. Use <=0 for all samples from start-index.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=512,
        help="Samples per output shard.",
    )
    parser.add_argument(
        "--shard-save-mode",
        type=str,
        choices=["uncompressed", "compressed"],
        default="uncompressed",
        help=(
            "Shard encoding mode. "
            "'uncompressed' is typically fastest for DDP training reads; "
            "'compressed' saves disk space but adds CPU decompression overhead."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=32,
        help="Output image height in pixels.",
    )
    parser.add_argument(
        "--vix-image",
        action="store_true",
        default=bool(VIX_IMAGE_ENABLED_DEFAULT),
        help=(
            "Render an additional VIX OHLC image per sample "
            f"({VIX_IMAGE_HEIGHT}x{VIX_IMAGE_WIDTH}, {VIX_IMAGE_BARS} bars). "
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
            "Each sample builds a 60-day VIX window and downsamples it per scale "
            "using robust_zscore_close averaged within each aggregation window. "
            f"Adds {int(VIX_FEATURE_PANEL_GAP_ROWS + VIX_FEATURE_PANEL_HEIGHT)}px "
            f"height ({int(VIX_FEATURE_PANEL_HEIGHT)}px bars + "
            f"{int(VIX_FEATURE_PANEL_GAP_ROWS)}px gap)."
        ),
    )
    parser.add_argument(
        "--include-volume",
        action="store_true",
        help="Render bottom volume bars using the liquidity feature.",
    )
    parser.add_argument(
        "--include-moving-average",
        action="store_true",
        help=(
            "Render moving-average points in the candle middle column and connect "
            "adjacent points with line segments."
        ),
    )
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
    parser.add_argument(
        "--volume-feature",
        type=str,
        default=None,
        help="Feature name for volume bars (default: turnover, then volume if present).",
    )
    parser.add_argument(
        "--price-height",
        type=int,
        default=25,
        help="Explicit pixel height for the top OHLC panel.",
    )
    parser.add_argument(
        "--volume-height",
        type=int,
        default=6,
        help="Explicit pixel height for the bottom volume panel.",
    )
    parser.add_argument(
        "--enable-weekend-feature",
        action="store_true",
        default=bool(WEEKEND_FEATURE_ENABLED_DEFAULT),
        help="Enable weekend gap rendering (disabled by default).",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=12,
        help="How many preview strips to write (0 disables previews).",
    )
    parser.add_argument(
        "--preview-scale",
        type=int,
        default=16,
        help="Nearest-neighbor upscaling factor for preview PNGs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing output directory.",
    )
    return parser.parse_args()


def find_required_feature_indices(feature_cols: Sequence[str]) -> Dict[str, int]:
    cols = [str(x) for x in feature_cols]
    idx: Dict[str, int] = {}
    for key, aliases in (
        ("open", ("open_syn", "open_ratio", "open")),
        ("high", ("high_syn", "high_ratio", "high")),
        ("low", ("low_syn", "low_ratio", "low")),
        ("close", ("close_syn", "close_ratio", "close")),
    ):
        pos = next((i for i, name in enumerate(cols) if name in aliases), None)
        if pos is None:
            raise ValueError(f"missing required OHLC feature ({key}) in {cols}")
        idx[key] = int(pos)
    return idx


def resolve_volume_index(
    feature_cols: Sequence[str],
    include_volume: bool,
    preferred: str | None,
) -> tuple[int | None, str | None]:
    if not include_volume:
        return None, None
    cols = [str(x) for x in feature_cols]
    candidates: List[str] = []
    if preferred:
        candidates.append(str(preferred))
    candidates.extend(["turnover", "volume", "liquidity"])
    seen = set()
    candidates = [x for x in candidates if not (x in seen or seen.add(x))]
    for name in candidates:
        if name in cols:
            return cols.index(name), name
    raise ValueError(
        "include_volume requested but no volume-like feature found. "
        f"feature_cols={cols}"
    )


def resolve_moving_average_index(
    feature_cols: Sequence[str],
    include_moving_average: bool,
    preferred: str | None,
) -> tuple[int | None, str | None]:
    if not include_moving_average:
        return None, None
    cols = [str(x) for x in feature_cols]
    candidates: List[str] = []
    if preferred:
        candidates.append(str(preferred))
    candidates.extend(["ma_n_syn", "ma_n", "moving_average"])
    seen = set()
    candidates = [x for x in candidates if not (x in seen or seen.add(x))]
    for name in candidates:
        if name in cols:
            return cols.index(name), name
    raise ValueError(
        "include_moving_average requested but no moving-average feature found. "
        f"feature_cols={cols}"
    )


def compute_window_sizes(lookback: int, windows: int, scales: int) -> List[int]:
    if windows < 1:
        raise ValueError("windows must be >= 1")
    if scales < 1:
        raise ValueError("scales must be >= 1")
    max_window = lookback // windows
    if max_window < 1:
        raise ValueError("lookback must be >= windows")
    out: List[int] = []
    for i in range(scales):
        size = windows**i
        if size > max_window:
            size = max_window
        out.append(int(size))
    return out


def compute_trailing_sma_min_periods_window(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    if int(window) < 1:
        raise ValueError("window must be >= 1")
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    if int(arr.shape[0]) < int(window):
        return out
    csum = np.cumsum(arr, dtype=np.float64)
    prev = np.concatenate(([0.0], csum[: -int(window)]))
    out[int(window) - 1 :] = (csum[int(window) - 1 :] - prev) / float(window)
    return out


def _parse_date_to_iso(value: object) -> str | None:
    text = str(value).strip()
    if not text:
        return None
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    prefix = text[:10]
    for fmt in ("%m/%d/%Y", "%Y/%m/%d", "%m-%d-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(prefix, fmt).date().isoformat()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        return None


def load_vix_ohlc_history(csv_path: Path) -> tuple[dict[str, int], np.ndarray]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"daily VIX CSV not found: {path}")

    date_to_row: dict[str, int] = {}
    rows: list[tuple[float, float, float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"daily VIX CSV has no header row: {path}")
        header_map = {str(x).strip().lower(): str(x) for x in reader.fieldnames}
        date_col = header_map.get("date")
        open_col = header_map.get("open")
        high_col = header_map.get("high")
        low_col = header_map.get("low")
        close_col = header_map.get("close")
        if not all([date_col, open_col, high_col, low_col, close_col]):
            raise ValueError(
                "daily VIX CSV must include DATE/OPEN/HIGH/LOW/CLOSE columns; "
                f"found={reader.fieldnames}"
            )

        for row in reader:
            iso_date = _parse_date_to_iso(row.get(str(date_col), ""))
            if iso_date is None:
                continue
            try:
                o = float(str(row.get(str(open_col), "")).strip())
                h = float(str(row.get(str(high_col), "")).strip())
                l = float(str(row.get(str(low_col), "")).strip())
                c = float(str(row.get(str(close_col), "")).strip())
            except ValueError:
                continue
            if not (
                math.isfinite(o)
                and math.isfinite(h)
                and math.isfinite(l)
                and math.isfinite(c)
            ):
                continue
            date_to_row[iso_date] = int(len(rows))
            rows.append((o, h, l, c))

    if not rows:
        raise ValueError(f"daily VIX CSV has no usable OHLC rows: {path}")
    ohlc = np.asarray(rows, dtype=np.float64)
    if ohlc.ndim != 2 or int(ohlc.shape[1]) != 4:
        raise RuntimeError(f"invalid parsed VIX OHLC shape: {ohlc.shape}")
    return date_to_row, ohlc


def load_vix_feature_series_by_row(
    csv_path: Path,
    date_to_row: dict[str, int],
    feature_col: str = str(VIX_FEATURE_SOURCE_COLUMN),
) -> np.ndarray:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"daily VIX CSV not found: {path}")
    col_key = str(feature_col).strip().lower()
    if not col_key:
        raise ValueError("feature_col must be non-empty")

    value_by_date: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"daily VIX CSV has no header row: {path}")
        header_map = {str(x).strip().lower(): str(x) for x in reader.fieldnames}
        date_col = header_map.get("date")
        value_col = header_map.get(col_key)
        if date_col is None:
            raise ValueError(f"daily VIX CSV missing DATE column: {path}")
        if value_col is None:
            raise ValueError(
                "daily VIX CSV missing requested feature column; "
                f"requested={feature_col!r} available={reader.fieldnames}"
            )
        for row in reader:
            iso_date = _parse_date_to_iso(row.get(str(date_col), ""))
            if iso_date is None:
                continue
            try:
                value = float(str(row.get(str(value_col), "")).strip())
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            value_by_date[iso_date] = float(value)

    out = np.empty((int(len(date_to_row)),), dtype=np.float64)
    missing: list[str] = []
    for iso_date, row_idx in date_to_row.items():
        value = value_by_date.get(str(iso_date))
        if value is None or not math.isfinite(float(value)):
            missing.append(str(iso_date))
            continue
        out[int(row_idx)] = float(value)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(
            "daily VIX feature series has missing/non-finite dates required by "
            f"OHLC history: missing={len(missing)} [{preview}{suffix}] "
            f"column={feature_col!r}"
        )
    return out


def select_source_indices_for_vix_image(
    source_indices: np.ndarray,
    timestamps: Sequence[object],
    date_to_row: dict[str, int],
    bars: int = int(VIX_IMAGE_BARS),
) -> tuple[np.ndarray, np.ndarray, int, int]:
    src = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    if src.size <= 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            0,
            0,
        )
    min_end = int(max(0, int(bars) - 1))
    kept_source: list[int] = []
    kept_end_rows: list[int] = []
    dropped_missing = 0
    dropped_short = 0
    for abs_idx in src.tolist():
        ts_val = timestamps[int(abs_idx)]
        iso_date = _parse_date_to_iso(ts_val)
        if iso_date is None:
            dropped_missing += 1
            continue
        end_row = date_to_row.get(str(iso_date))
        if end_row is None:
            dropped_missing += 1
            continue
        if int(end_row) < int(min_end):
            dropped_short += 1
            continue
        kept_source.append(int(abs_idx))
        kept_end_rows.append(int(end_row))
    return (
        np.asarray(kept_source, dtype=np.int64),
        np.asarray(kept_end_rows, dtype=np.int64),
        int(dropped_missing),
        int(dropped_short),
    )


def build_vix_image_spec() -> OhlcImageSpec:
    return OhlcImageSpec(
        height=int(VIX_IMAGE_HEIGHT),
        day_width=int(VIX_IMAGE_DAY_WIDTH),
        weekend_gap_width=0,
        weekend_gap_scale_indices=(),
        include_volume=False,
        foreground=1,
        background=0,
    )


def build_vix_close_moving_average_by_row(
    vix_ohlc_history: np.ndarray,
    window: int = int(VIX_IMAGE_MOVING_AVERAGE_WINDOW),
) -> np.ndarray:
    history = np.asarray(vix_ohlc_history, dtype=np.float64)
    if history.ndim != 2 or int(history.shape[1]) != 4:
        raise ValueError(
            "vix_ohlc_history must have shape (rows, 4); "
            f"got shape={history.shape}"
        )
    close_values = history[:, 3]
    return compute_trailing_sma_min_periods_window(close_values, window=int(window))


def render_vix_image_batch(
    vix_ohlc_history: np.ndarray,
    end_rows: np.ndarray,
    spec: OhlcImageSpec,
    bars: int = int(VIX_IMAGE_BARS),
    moving_average_by_row: np.ndarray | None = None,
) -> np.ndarray:
    end_idx = np.asarray(end_rows, dtype=np.int64).reshape(-1)
    if end_idx.size <= 0:
        width = int(spec.width_for_scale(int(bars), 0))
        return np.empty((0, int(spec.height), int(width)), dtype=np.uint8)
    bars_i = int(bars)
    width = int(spec.width_for_scale(bars_i, 0))
    out = np.zeros((int(end_idx.shape[0]), int(spec.height), int(width)), dtype=np.uint8)
    history = np.asarray(vix_ohlc_history, dtype=np.float64)
    ma_history: np.ndarray | None = None
    if moving_average_by_row is not None:
        ma_history = np.asarray(moving_average_by_row, dtype=np.float64).reshape(-1)
        if int(ma_history.shape[0]) != int(history.shape[0]):
            raise ValueError(
                "moving_average_by_row length must match vix history rows; "
                f"got ma_len={int(ma_history.shape[0])} rows={int(history.shape[0])}"
            )

    for i, row_idx in enumerate(end_idx.tolist()):
        row = int(row_idx)
        start = int(row - bars_i + 1)
        stop = int(row + 1)
        if start < 0 or stop > int(history.shape[0]):
            raise ValueError(
                "invalid VIX window bounds while rendering image batch: "
                f"row={row} bars={bars_i} history={int(history.shape[0])}"
            )
        window = history[start:stop]
        if int(window.shape[0]) != int(bars_i):
            raise ValueError(
                f"expected VIX window size {bars_i}, got {window.shape}"
            )
        ma_window = None
        if ma_history is not None:
            ma_window = np.asarray(ma_history[start:stop], dtype=np.float64)
            if int(ma_window.shape[0]) != int(bars_i):
                raise ValueError(
                    "expected VIX moving-average window length "
                    f"{bars_i}, got {ma_window.shape}"
                )
            if not bool(np.isfinite(ma_window).all()):
                raise ValueError(
                    "VIX moving-average window contains non-finite values; "
                    "ensure sufficient warmup history is available."
                )
        img = render_ohlc_chart(
            ohlc=np.asarray(window.T, dtype=np.float64),
            volume=None,
            spec=spec,
            scale_index=0,
            weekend_gap_after_bar=None,
            apply_speckle_cleanup=True,
            moving_average=ma_window,
        )
        out[i] = img
    return out


def build_vix_feature_values_by_scale(
    vix_feature_series_by_row: np.ndarray,
    end_rows: np.ndarray,
    scales: int,
    windows: int,
    lookback_days: int = int(VIX_FEATURE_LOOKBACK_DAYS),
) -> np.ndarray:
    end_idx = np.asarray(end_rows, dtype=np.int64).reshape(-1)
    if end_idx.size <= 0:
        return np.empty((0, int(scales), int(windows)), dtype=np.float64)
    bars = int(lookback_days)
    if bars < 1:
        raise ValueError("lookback_days must be >= 1")
    scales_i = int(scales)
    windows_i = int(windows)
    if scales_i < 1:
        raise ValueError("scales must be >= 1")
    if windows_i < 1:
        raise ValueError("windows must be >= 1")

    feature_series = np.asarray(vix_feature_series_by_row, dtype=np.float64).reshape(-1)
    if feature_series.ndim != 1:
        raise ValueError(
            "vix_feature_series_by_row must be 1D; "
            f"got shape={feature_series.shape}"
        )
    if int(feature_series.shape[0]) < bars:
        raise ValueError(
            "vix history is shorter than requested lookback: "
            f"history={int(feature_series.shape[0])} lookback_days={bars}"
        )
    if not bool(np.isfinite(feature_series).all()):
        raise ValueError("vix_feature_series_by_row contains non-finite values")

    window_sizes = compute_window_sizes(
        lookback=int(bars), windows=windows_i, scales=scales_i
    )
    out = np.empty((int(end_idx.shape[0]), scales_i, windows_i), dtype=np.float64)
    window_start_rows = end_idx - int(bars - 1)
    for scale_idx, window_size in enumerate(window_sizes):
        ws = int(window_size)
        take = int(ws * windows_i)
        if take > bars:
            raise ValueError(
                "invalid downsample setup: "
                f"window_size={ws} windows={windows_i} lookback_days={bars}"
            )
        start_base = int(bars - take)
        rel_positions = start_base + np.arange(take, dtype=np.int64)
        abs_positions = window_start_rows[:, None] + rel_positions[None, :]
        if int(np.min(abs_positions)) < 0 or int(np.max(abs_positions)) >= int(
            feature_series.shape[0]
        ):
            raise ValueError(
                "computed VIX feature positions exceed source bounds: "
                f"min={int(np.min(abs_positions))} max={int(np.max(abs_positions))} "
                f"rows={int(feature_series.shape[0])}"
            )
        values = feature_series[abs_positions]
        values = values.reshape(int(end_idx.shape[0]), windows_i, ws)
        out[:, int(scale_idx), :] = values.mean(axis=2)
    return out


def append_volume_like_panel(
    base_img: np.ndarray,
    values: np.ndarray,
    spec: OhlcImageSpec,
    scale_index: int,
    weekend_gap_after_bar: int | None,
    panel_height: int,
    gap_rows: int,
) -> np.ndarray:
    if base_img.ndim != 2:
        raise ValueError(f"base_img must be 2D, got {base_img.shape}")
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    windows = int(vals.shape[0])
    if windows < 1:
        raise ValueError("values must contain at least 1 bar")
    panel_h = int(panel_height)
    gap_h = int(gap_rows)
    if panel_h < 1:
        raise ValueError("panel_height must be >= 1")
    if gap_h < 0:
        raise ValueError("gap_rows must be >= 0")

    extra_h = int(gap_h + panel_h)
    out = np.full(
        (int(base_img.shape[0] + extra_h), int(base_img.shape[1])),
        int(spec.background),
        dtype=np.uint8,
    )
    out[: base_img.shape[0], : base_img.shape[1]] = base_img

    finite_pos = np.isfinite(vals) & (vals > 0.0)
    if not bool(np.any(finite_pos)):
        return out
    v_max = float(np.max(vals[finite_pos]))
    if not math.isfinite(v_max) or v_max <= EPS:
        return out

    bar_x0 = compute_bar_x0_positions(
        windows=windows,
        spec=spec,
        scale_index=int(scale_index),
        weekend_gap_after_bar=weekend_gap_after_bar,
    )
    panel_bottom = int(out.shape[0] - 1)
    panel_pixels = int(panel_h)
    for j in range(windows):
        v = float(vals[j])
        if not math.isfinite(v) or v <= 0.0:
            continue
        x = int(bar_x0[j]) + 1
        if x < 0 or x >= int(out.shape[1]):
            continue
        pixels = int(round((v / v_max) * panel_pixels))
        pixels = max(1, min(panel_pixels, pixels))
        y_top = int(panel_bottom - pixels + 1)
        out[y_top : panel_bottom + 1, x] = np.uint8(spec.foreground)
    return out


def scale_to_rows(
    values: np.ndarray,
    top: int,
    bottom: int,
    lo: float | None = None,
    hi: float | None = None,
) -> np.ndarray:
    if lo is None:
        lo = float(np.min(values))
    if hi is None:
        hi = float(np.max(values))
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError("non-finite values found while mapping to rows")
    if abs(hi - lo) < EPS:
        return np.full(values.shape, (top + bottom) // 2, dtype=np.int32)
    ratio = (hi - values) / (hi - lo)
    rows = np.rint(top + ratio * float(bottom - top)).astype(np.int32)
    return np.clip(rows, top, bottom)


def parse_timestamp_weekday(value: object) -> int:
    text = str(value)
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        try:
            y = int(text[0:4])
            m = int(text[5:7])
            d = int(text[8:10])
            return int(date(y, m, d).weekday())
        except Exception:
            pass
    try:
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).date().weekday())
    except Exception as exc:
        raise ValueError(
            f"could not parse timestamp '{value}' for weekend placement"
        ) from exc


def infer_weekend_gap_after_bar(sample_timestamp: object, windows: int) -> int:
    if windows < 1:
        raise ValueError("windows must be >= 1")
    weekday = parse_timestamp_weekday(sample_timestamp)
    if weekday < 0 or weekday > 4:
        return int(windows - 1)
    weekdays = [int((weekday - (windows - 1 - j)) % 5) for j in range(windows)]
    for j in range(windows - 1):
        if weekdays[j] == 4 and weekdays[j + 1] == 0:
            return int(j)
    return int(windows - 1)


def compute_bar_x0_positions(
    windows: int,
    spec: OhlcImageSpec,
    scale_index: int,
    weekend_gap_after_bar: int | None,
) -> np.ndarray:
    x0 = np.arange(windows, dtype=np.int32) * int(spec.day_width)
    if not spec.has_weekend_gap(scale_index):
        return x0
    gap_w = int(spec.weekend_gap_width)
    after = int(windows - 1) if weekend_gap_after_bar is None else int(weekend_gap_after_bar)
    after = max(-1, min(windows - 1, after))
    shift = (np.arange(windows, dtype=np.int32) > after).astype(np.int32)
    return x0 + shift * gap_w


def draw_line_bresenham(
    arr: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    value: int,
) -> None:
    h, w = arr.shape
    x = int(x0)
    y = int(y0)
    x1 = int(x1)
    y1 = int(y1)
    dx = abs(x1 - x)
    sx = 1 if x < x1 else -1
    dy = -abs(y1 - y)
    sy = 1 if y < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x < w and 0 <= y < h:
            arr[y, x] = np.uint8(value)
        if x == x1 and y == y1:
            break
        e2 = err * 2
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def render_ohlc_chart(
    ohlc: np.ndarray,
    volume: np.ndarray | None,
    spec: OhlcImageSpec,
    scale_index: int = 0,
    weekend_gap_after_bar: int | None = None,
    apply_speckle_cleanup: bool = True,
    *,
    moving_average: np.ndarray | None = None,
    moving_average_connect_across_weekend_gap: bool = bool(
        MOVING_AVERAGE_CONNECT_ACROSS_WEEKEND_GAP_DEFAULT
    ),
) -> np.ndarray:
    if ohlc.ndim != 2 or ohlc.shape[0] != 4:
        raise ValueError(f"ohlc must be (4, windows), got {ohlc.shape}")
    windows = int(ohlc.shape[1])
    width = spec.width_for_scale(windows, scale_index)
    price_h, volume_h = spec.split_heights()
    arr = np.full((spec.height, width), spec.background, dtype=np.uint8)
    bar_x0 = compute_bar_x0_positions(
        windows=windows,
        spec=spec,
        scale_index=scale_index,
        weekend_gap_after_bar=weekend_gap_after_bar,
    )
    ma_vals: np.ndarray | None = None
    ma_valid: np.ndarray | None = None
    if moving_average is not None:
        ma_vals = np.asarray(moving_average, dtype=np.float64)
        if ma_vals.ndim != 1 or int(ma_vals.shape[0]) != windows:
            raise ValueError(
                "moving_average must be 1D with matching windows; "
                f"got shape={ma_vals.shape}, windows={windows}"
            )
        ma_valid = np.isfinite(ma_vals)

    open_vals = ohlc[0]
    high_vals = np.maximum.reduce([ohlc[1], ohlc[0], ohlc[3]])
    low_vals = np.minimum.reduce([ohlc[2], ohlc[0], ohlc[3]])
    close_vals = ohlc[3]
    price_all = np.concatenate([open_vals, high_vals, low_vals, close_vals], axis=0)
    lo = float(np.min(price_all))
    hi = float(np.max(price_all))
    if ma_vals is not None and ma_valid is not None and bool(np.any(ma_valid)):
        # Let MA extrema expand the y-range so MA can set chart high/low bounds.
        ma_finite = ma_vals[ma_valid]
        lo = min(lo, float(np.min(ma_finite)))
        hi = max(hi, float(np.max(ma_finite)))
    high_rows = scale_to_rows(high_vals, 0, price_h - 1, lo=lo, hi=hi)
    low_rows = scale_to_rows(low_vals, 0, price_h - 1, lo=lo, hi=hi)
    open_rows = scale_to_rows(open_vals, 0, price_h - 1, lo=lo, hi=hi)
    close_rows = scale_to_rows(close_vals, 0, price_h - 1, lo=lo, hi=hi)

    for j in range(windows):
        x0 = int(bar_x0[j])
        xm = x0 + 1
        x2 = x0 + 2
        y_open = int(open_rows[j])
        y_close = int(close_rows[j])
        y_hi = int(min(high_rows[j], low_rows[j], y_open, y_close))
        y_lo = int(max(high_rows[j], low_rows[j], y_open, y_close))
        arr[y_hi : y_lo + 1, xm] = spec.foreground
        arr[y_open, x0] = spec.foreground
        arr[y_close, x2] = spec.foreground

    if volume_h > 0 and volume is not None:
        vol = np.asarray(volume, dtype=np.float64)
        v_max = float(np.max(vol))
        v_max = max(v_max, EPS)
        vol_top = spec.height - volume_h
        vol_bottom = spec.height - 1
        vol_pixels = volume_h
        for j in range(windows):
            x = int(bar_x0[j]) + 1
            v = float(vol[j])
            if not math.isfinite(v) or v <= 0.0:
                continue
            pixels = int(round((v / v_max) * vol_pixels))
            pixels = max(1, min(vol_pixels, pixels))
            y_top = vol_bottom - pixels + 1
            arr[y_top : vol_bottom + 1, x] = spec.foreground

    if bool(apply_speckle_cleanup):
        # Guard against accidental isolated speckles in the candle panel only.
        # Use 8-neighbor connectivity so diagonal line pixels are preserved.
        fg = arr > 0
        n = np.zeros_like(fg, dtype=bool)
        n[1:, :] |= fg[:-1, :]
        n[:-1, :] |= fg[1:, :]
        n[:, 1:] |= fg[:, :-1]
        n[:, :-1] |= fg[:, 1:]
        n[1:, 1:] |= fg[:-1, :-1]
        n[1:, :-1] |= fg[:-1, 1:]
        n[:-1, 1:] |= fg[1:, :-1]
        n[:-1, :-1] |= fg[1:, 1:]
        candle_mask = np.zeros_like(fg, dtype=bool)
        candle_mask[:price_h, :] = True
        arr[fg & (~n) & candle_mask] = 0

    if ma_vals is not None and ma_valid is not None:
        ma_rows = np.zeros(windows, dtype=np.int32)
        if bool(np.any(ma_valid)):
            ma_rows[ma_valid] = scale_to_rows(
                ma_vals[ma_valid], 0, price_h - 1, lo=lo, hi=hi
            )
        ma_x = bar_x0 + 1
        for j in range(windows):
            if not bool(ma_valid[j]):
                continue
            arr[int(ma_rows[j]), int(ma_x[j])] = spec.foreground

        weekend_after = int(windows - 1) if weekend_gap_after_bar is None else int(
            weekend_gap_after_bar
        )
        weekend_after = max(-1, min(windows - 1, weekend_after))
        skip_across_weekend_gap = bool(spec.has_weekend_gap(scale_index)) and (
            not bool(moving_average_connect_across_weekend_gap)
        )
        left_stop_col = None
        right_start_col = None
        if skip_across_weekend_gap and 0 <= int(weekend_after) < int(windows - 1):
            gap_w = int(spec.weekend_gap_width)
            gap_start_col = int(bar_x0[int(weekend_after) + 1]) - gap_w
            # Keep the weekend gap empty at exact width:
            # - left segment stops before first gap column
            # - right segment starts at first non-gap column after the gap
            left_stop_col = int(gap_start_col - 1)
            right_start_col = int(gap_start_col + gap_w)
        for j in range(windows - 1):
            if not (bool(ma_valid[j]) and bool(ma_valid[j + 1])):
                continue
            x0_seg = int(ma_x[j])
            y0_seg = int(ma_rows[j])
            x1_seg = int(ma_x[j + 1])
            y1_seg = int(ma_rows[j + 1])
            if (
                skip_across_weekend_gap
                and left_stop_col is not None
                and int(j) == int(weekend_after)
            ):
                # Left side of weekend gap: do not enter the first gap column.
                x_stop = int(left_stop_col)
                if x_stop <= x0_seg:
                    continue
                if x1_seg == x0_seg:
                    y_stop = y0_seg
                else:
                    t = float(x_stop - x0_seg) / float(x1_seg - x0_seg)
                    y_stop = int(np.rint(float(y0_seg) + t * float(y1_seg - y0_seg)))
                y_stop = max(0, min(int(price_h - 1), int(y_stop)))
                draw_line_bresenham(
                    arr,
                    x0=x0_seg,
                    y0=y0_seg,
                    x1=x_stop,
                    y1=y_stop,
                    value=int(spec.foreground),
                )
                # Also draw the right boundary piece from the first post-gap
                # column to the right endpoint, so the visible blank gap width
                # is exactly weekend_gap_width columns even at window end.
                if right_start_col is not None:
                    x_begin = int(right_start_col)
                    if x_begin < x1_seg:
                        if x1_seg == x0_seg:
                            y_begin = y1_seg
                        else:
                            t_begin = float(x_begin - x0_seg) / float(x1_seg - x0_seg)
                            y_begin = int(
                                np.rint(float(y0_seg) + t_begin * float(y1_seg - y0_seg))
                            )
                        y_begin = max(0, min(int(price_h - 1), int(y_begin)))
                        draw_line_bresenham(
                            arr,
                            x0=x_begin,
                            y0=y_begin,
                            x1=x1_seg,
                            y1=y1_seg,
                            value=int(spec.foreground),
                        )
                continue
            draw_line_bresenham(
                arr,
                x0=x0_seg,
                y0=y0_seg,
                x1=x1_seg,
                y1=y1_seg,
                value=int(spec.foreground),
            )

    return arr


def render_batch(
    X: np.ndarray,
    open_idx: int,
    high_idx: int,
    low_idx: int,
    close_idx: int,
    volume_idx: int | None,
    spec: OhlcImageSpec,
    sample_timestamps: Sequence[object] | None = None,
    apply_speckle_cleanup: bool = True,
    *,
    moving_average_idx: int | None = None,
    moving_average_connect_across_weekend_gap: bool = bool(
        MOVING_AVERAGE_CONNECT_ACROSS_WEEKEND_GAP_DEFAULT
    ),
    vix_feature_values: np.ndarray | None = None,
    vix_feature_panel_height: int = int(VIX_FEATURE_PANEL_HEIGHT),
    vix_feature_gap_rows: int = int(VIX_FEATURE_PANEL_GAP_ROWS),
) -> np.ndarray:
    if X.ndim != 4:
        raise ValueError(f"X must be 4D, got {X.shape}")
    n, scales, _, windows = X.shape
    if sample_timestamps is not None and len(sample_timestamps) != n:
        raise ValueError(
            "sample_timestamps length must match batch size; "
            f"got len={len(sample_timestamps)} n={n}"
        )
    vix_vals = None
    if vix_feature_values is not None:
        vix_vals = np.asarray(vix_feature_values, dtype=np.float64)
        expected_shape = (int(n), int(scales), int(windows))
        if tuple(vix_vals.shape) != tuple(expected_shape):
            raise ValueError(
                "vix_feature_values must have shape (n, scales, windows); "
                f"got {vix_vals.shape} expected {expected_shape}"
            )
        if int(vix_feature_panel_height) < 1:
            raise ValueError("vix_feature_panel_height must be >= 1")
        if int(vix_feature_gap_rows) < 0:
            raise ValueError("vix_feature_gap_rows must be >= 0")
    scale_widths = spec.widths_for_scales(windows, scales)
    extra_height = (
        int(vix_feature_panel_height + vix_feature_gap_rows)
        if vix_vals is not None
        else 0
    )
    out = np.full(
        (n, scales, int(spec.height + extra_height), int(max(scale_widths))),
        spec.background,
        dtype=np.uint8,
    )
    weekend_scales = [spec.has_weekend_gap(s) for s in range(scales)]
    need_weekend_ts = any(weekend_scales)
    for i in range(n):
        weekend_after = None
        if need_weekend_ts:
            ts_value = sample_timestamps[i] if sample_timestamps is not None else None
            if ts_value is None:
                weekend_after = int(windows - 1)
            else:
                weekend_after = infer_weekend_gap_after_bar(ts_value, windows)
        for s in range(scales):
            xs = X[i, s]
            ohlc = np.stack(
                [
                    xs[open_idx],
                    xs[high_idx],
                    xs[low_idx],
                    xs[close_idx],
                ],
                axis=0,
            )
            vol = xs[volume_idx] if volume_idx is not None else None
            ma_vals = xs[moving_average_idx] if moving_average_idx is not None else None
            img = render_ohlc_chart(
                ohlc=ohlc,
                volume=vol,
                moving_average=ma_vals,
                spec=spec,
                scale_index=s,
                weekend_gap_after_bar=weekend_after if weekend_scales[s] else None,
                moving_average_connect_across_weekend_gap=bool(
                    moving_average_connect_across_weekend_gap
                ),
                apply_speckle_cleanup=apply_speckle_cleanup,
            )
            if vix_vals is not None:
                img = append_volume_like_panel(
                    base_img=img,
                    values=vix_vals[i, s],
                    spec=spec,
                    scale_index=s,
                    weekend_gap_after_bar=(
                        weekend_after if weekend_scales[s] else None
                    ),
                    panel_height=int(vix_feature_panel_height),
                    gap_rows=int(vix_feature_gap_rows),
                )
            out[i, s, :, : img.shape[1]] = img
    return out


def save_preview_strip(
    images_by_scale: np.ndarray,
    out_path: Path,
    scale: int,
    widths_by_scale: Sequence[int] | None = None,
) -> None:
    if images_by_scale.ndim != 3:
        raise ValueError("images_by_scale must be (scales, height, width)")
    scales, h, w = images_by_scale.shape
    if widths_by_scale is not None and len(widths_by_scale) != scales:
        raise ValueError(
            "widths_by_scale length must match number of scales; "
            f"got len={len(widths_by_scale)} scales={scales}"
        )
    panels = []
    for i in range(scales):
        panel_arr = images_by_scale[i]
        if widths_by_scale is not None:
            panel_w = max(1, min(int(widths_by_scale[i]), int(w)))
            panel_arr = panel_arr[:, :panel_w]
        panel = Image.fromarray((panel_arr * 255).astype(np.uint8), mode="L")
        if scale > 1:
            panel = panel.resize((panel.width * scale, h * scale), Image.NEAREST)
        panels.append(panel)
    gap = scale
    canvas_w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    canvas_h = panels[0].height if panels else 0
    canvas = Image.new("L", (canvas_w, canvas_h), color=0)
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite and any(path.iterdir()):
            raise FileExistsError(
                f"output dir is not empty: {path} (use --overwrite to reuse)"
            )
    path.mkdir(parents=True, exist_ok=True)


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir, overwrite=args.overwrite)

    with np.load(args.input_npz, allow_pickle=True) as data:
        X = data["X"]
        y_raw = data["y_raw"]
        timestamps = data["timestamps"]
        ticker_ids = data["ticker_ids"] if "ticker_ids" in data.files else None
        tickers = data["tickers"] if "tickers" in data.files else None
        label_cols = data["label_cols"] if "label_cols" in data.files else None
        feature_cols = data["feature_cols"]
        lookback = int(data["lookback"][0]) if "lookback" in data.files else None
        scales = int(data["decomposition_scales"][0]) if "decomposition_scales" in data.files else int(X.shape[1])
        windows = int(data["decomposition_windows"][0]) if "decomposition_windows" in data.files else int(X.shape[3])
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
    if args.max_samples is None or int(args.max_samples) <= 0:
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

    feature_cols_list = [str(x) for x in feature_cols.tolist()]
    idx = find_required_feature_indices(feature_cols_list)
    volume_idx, volume_name = resolve_volume_index(
        feature_cols_list, include_volume=args.include_volume, preferred=args.volume_feature
    )
    moving_average_idx, moving_average_feature = resolve_moving_average_index(
        feature_cols_list,
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
        price_height=(
            int(args.price_height) if args.price_height is not None else None
        ),
        volume_height=(
            int(args.volume_height) if args.volume_height is not None else None
        ),
        volume_fraction=0.2,
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

    window_sizes = (
        compute_window_sizes(lookback=lookback, windows=windows, scales=scales)
        if lookback is not None
        else []
    )

    shard_dir = args.output_dir / "shards"
    preview_dir = args.output_dir / "previews"
    shard_dir.mkdir(parents=True, exist_ok=True)
    if args.preview_count > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)

    shard_size = max(1, int(args.shard_size))
    shard_save_mode = str(args.shard_save_mode).strip().lower()
    shard_manifest = []
    written_previews = 0
    shard_index = 0

    for rel_start in range(0, subset_count, shard_size):
        rel_end = min(subset_count, rel_start + shard_size)
        src_idx_chunk = source_indices[rel_start:rel_end]
        chunk_n = int(src_idx_chunk.shape[0])
        if chunk_n <= 0:
            continue
        sample_start = int(subset_start + rel_start)
        sample_end = int(subset_start + rel_end)

        X_chunk = X[src_idx_chunk].astype(np.float64, copy=False)
        ts_chunk = np.asarray(timestamps[src_idx_chunk]).astype(str)
        vix_feature_chunk = None
        if bool(args.include_vix_feature):
            if vix_feature_series_by_row is None or vix_end_rows is None:
                raise RuntimeError(
                    "include-vix-feature enabled but vix lookup state is missing"
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
            moving_average_idx=moving_average_idx,
            spec=spec,
            sample_timestamps=ts_chunk,
            moving_average_connect_across_weekend_gap=bool(
                args.moving_average_connect_across_weekend_gap
            ),
            vix_feature_values=vix_feature_chunk,
            vix_feature_panel_height=int(VIX_FEATURE_PANEL_HEIGHT),
            vix_feature_gap_rows=int(VIX_FEATURE_PANEL_GAP_ROWS),
        )
        shard_path = shard_dir / f"shard_{shard_index:06d}.npz"
        payload = {
            "X_img": img_chunk,
            "y_raw": y_raw[src_idx_chunk],
            "timestamps": np.asarray(ts_chunk, dtype="<U32"),
            "sample_indices": np.asarray(src_idx_chunk, dtype=np.int64),
        }
        if bool(args.vix_image):
            if vix_end_rows is None or vix_history_ohlc is None or vix_spec is None:
                raise RuntimeError("vix-image enabled but vix lookup state is missing")
            payload["X_vix_img"] = render_vix_image_batch(
                vix_ohlc_history=vix_history_ohlc,
                end_rows=vix_end_rows[rel_start:rel_end],
                spec=vix_spec,
                bars=int(VIX_IMAGE_BARS),
                moving_average_by_row=vix_moving_average_by_row,
            )
        if ticker_ids is not None:
            payload["ticker_ids"] = ticker_ids[src_idx_chunk]
        if shard_save_mode == "compressed":
            np.savez_compressed(shard_path, **payload)
        else:
            np.savez(shard_path, **payload)
        shard_manifest.append(
            {
                "file": str(shard_path.relative_to(args.output_dir)),
                "sample_start": int(sample_start),
                "sample_end": int(sample_end),
                "count": int(chunk_n),
                "relative_start": int(rel_start),
                "relative_end": int(rel_end),
            }
        )

        if args.preview_count > 0 and written_previews < args.preview_count:
            remaining = int(args.preview_count) - written_previews
            to_take = min(remaining, img_chunk.shape[0])
            for i in range(to_take):
                sample_idx = int(src_idx_chunk[i])
                out_name = f"sample_{sample_idx:07d}.png"
                save_preview_strip(
                    images_by_scale=img_chunk[i],
                    out_path=preview_dir / out_name,
                    scale=max(1, int(args.preview_scale)),
                    widths_by_scale=scale_widths,
                )
                written_previews += 1

        print(
            f"wrote shard {shard_index:06d}: rel_samples [{rel_start}, {rel_end}) "
            f"shape={img_chunk.shape}"
        )
        shard_index += 1

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_npz": str(args.input_npz),
        "normalization": normalization,
        "feature_cols_source": feature_cols_list,
        "label_cols": [str(x) for x in (label_cols.tolist() if label_cols is not None else [])],
        "tickers_count": int(len(tickers)) if tickers is not None else None,
        "subset_start": int(subset_start),
        "subset_end": int(subset_end),
        "subset_count": int(subset_count),
        "source_selection_start": int(start),
        "source_selection_end": int(end),
        "source_selection_count": int(source_count_before_vix_filter),
        "total_samples_source": int(total),
        "decomposition_scales": int(scales),
        "decomposition_windows": int(windows),
        "scale_window_sizes": [int(x) for x in window_sizes],
        "image_height": int(image_height_total),
        "image_height_base": int(spec.height),
        "image_width": int(image_width),
        "image_width_per_scale": [int(x) for x in scale_widths],
        "day_width": int(spec.day_width),
        "weekend_feature_enabled": bool(args.enable_weekend_feature),
        "weekend_gap_width": int(spec.weekend_gap_width),
        "weekend_gap_scale_indices": spec.normalized_weekend_gap_scale_indices(),
        "price_panel_height": int(price_h),
        "volume_panel_height": int(volume_h),
        "price_panel_rows": [0, int(price_h - 1)] if price_h > 0 else None,
        "volume_panel_rows": (
            [int(spec.height - volume_h), int(spec.height - 1)]
            if volume_h > 0
            else None
        ),
        "include_volume": bool(spec.include_volume),
        "volume_feature": volume_name,
        "include_moving_average": bool(args.include_moving_average),
        "moving_average_window": int(moving_average_window),
        "moving_average_mode": str(moving_average_mode),
        "moving_average_connect_across_weekend_gap": bool(
            args.moving_average_connect_across_weekend_gap
        ),
        "value_mapping": {
            "background": int(spec.background),
            "foreground": int(spec.foreground),
        },
        "shard_size": int(shard_size),
        "shard_save_mode": str(shard_save_mode),
        "num_shards": int(len(shard_manifest)),
        "shards": shard_manifest,
        "include_vix_image": bool(args.vix_image),
        "include_vix_feature": bool(args.include_vix_feature),
    }
    if bool(args.include_vix_feature):
        manifest["vix_feature_lookback_days"] = int(VIX_FEATURE_LOOKBACK_DAYS)
        manifest["vix_feature_panel_height"] = int(VIX_FEATURE_PANEL_HEIGHT)
        manifest["vix_feature_gap_rows"] = int(VIX_FEATURE_PANEL_GAP_ROWS)
        manifest["vix_feature_source_csv"] = str(Path(args.vix_daily_csv))
        manifest["vix_feature_source_column"] = str(VIX_FEATURE_SOURCE_COLUMN)
        manifest["vix_feature_window_aggregation"] = str(
            VIX_FEATURE_WINDOW_AGGREGATION
        )
        manifest["vix_feature_window_sizes"] = [
            int(x)
            for x in compute_window_sizes(
                lookback=int(VIX_FEATURE_LOOKBACK_DAYS),
                windows=int(windows),
                scales=int(scales),
            )
        ]
        vix_panel_top = int(spec.height + int(VIX_FEATURE_PANEL_GAP_ROWS))
        vix_panel_bottom = int(image_height_total - 1)
        manifest["vix_feature_panel_rows"] = [vix_panel_top, vix_panel_bottom]
        manifest["vix_feature_dropped_missing_date_count"] = int(
            vix_drop_missing_date_count
        )
        manifest["vix_feature_dropped_insufficient_history_count"] = int(
            vix_drop_insufficient_history_count
        )
        manifest["vix_feature_source_selected_count"] = int(
            source_count_before_vix_filter
        )
        manifest["vix_feature_retained_count"] = int(subset_count)
    if bool(args.vix_image):
        if vix_spec is None:
            raise RuntimeError("vix-image enabled but vix spec is missing")
        manifest["vix_image_height"] = int(vix_spec.height)
        manifest["vix_image_width"] = int(vix_spec.width_for_scale(int(VIX_IMAGE_BARS), 0))
        manifest["vix_image_bars"] = int(VIX_IMAGE_BARS)
        manifest["vix_image_day_width"] = int(vix_spec.day_width)
        manifest["vix_image_source_csv"] = str(Path(args.vix_daily_csv))
        manifest["vix_image_include_moving_average"] = True
        manifest["vix_image_moving_average_window"] = int(
            VIX_IMAGE_MOVING_AVERAGE_WINDOW
        )
        manifest["vix_image_moving_average_mode"] = str(VIX_IMAGE_MOVING_AVERAGE_MODE)
        manifest["vix_image_dropped_missing_date_count"] = int(vix_drop_missing_date_count)
        manifest["vix_image_dropped_insufficient_history_count"] = int(
            vix_drop_insufficient_history_count
        )
        manifest["vix_image_source_selected_count"] = int(source_count_before_vix_filter)
        manifest["vix_image_retained_count"] = int(subset_count)
    if moving_average_feature is not None:
        manifest["moving_average_source"] = "feature_channel"
        manifest["moving_average_feature"] = str(moving_average_feature)
    write_manifest(args.output_dir / "manifest.json", manifest)
    if tickers is not None:
        np.save(args.output_dir / "tickers.npy", tickers)

    if bool(requires_vix_history):
        print(
            "vix-source filter: "
            f"input={source_count_before_vix_filter} "
            f"retained={subset_count} "
            f"dropped_missing_date={vix_drop_missing_date_count} "
            f"dropped_insufficient_history={vix_drop_insufficient_history_count}"
        )
    print(f"done: wrote {len(shard_manifest)} shard(s) to {args.output_dir}")
    print(f"manifest: {args.output_dir / 'manifest.json'}")
    if written_previews:
        print(f"previews: {written_previews} image strip(s) in {preview_dir}")


if __name__ == "__main__":
    main()
