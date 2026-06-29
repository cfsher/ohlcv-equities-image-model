#!/usr/bin/env python3
"""Utilities for loading sharded OHLC image datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from jagged_width_config import JAGGED_IMAGE_WIDTHS_ENABLED


@dataclass(frozen=True)
class ShardInfo:
    file: str
    sample_start: int
    sample_end: int
    count: int

    @property
    def span(self) -> tuple[int, int]:
        return self.sample_start, self.sample_end


class ShardedOhlcImageDataset:
    """Random-access loader for `build_decomp_ohlc_images.py` outputs.

    Each global index maps to:
      X_img: (scales, height, width), uint8 in {0,1}
      y_raw: (...), float64
      timestamp: str
      ticker_id: int (if available)
      sample_index: int (global source index)
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        manifest_path = self.root_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.shards: List[ShardInfo] = [
            ShardInfo(
                file=str(entry["file"]),
                sample_start=int(entry["sample_start"]),
                sample_end=int(entry["sample_end"]),
                count=int(entry["count"]),
            )
            for entry in self.manifest["shards"]
        ]
        self.subset_start = int(self.manifest["subset_start"])
        self.subset_end = int(self.manifest["subset_end"])
        self.subset_count = int(self.manifest["subset_count"])
        self.scales = int(self.manifest["decomposition_scales"])
        self.image_height = int(self.manifest["image_height"])
        self.image_width = int(self.manifest["image_width"])
        self.image_width_per_scale = self._resolve_image_width_per_scale()
        self.max_image_width = int(max(self.image_width_per_scale))
        if self.max_image_width > self.image_width:
            # Be permissive if manifest.image_width is stale or missing.
            self.image_width = self.max_image_width
        self.jagged_image_widths_enabled = bool(JAGGED_IMAGE_WIDTHS_ENABLED)

        self._cache_shard_index: int | None = None
        self._cache_data: Dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return self.subset_count

    def _find_shard(self, global_sample_index: int) -> tuple[int, ShardInfo]:
        lo, hi = 0, len(self.shards) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            shard = self.shards[mid]
            if global_sample_index < shard.sample_start:
                hi = mid - 1
            elif global_sample_index >= shard.sample_end:
                lo = mid + 1
            else:
                return mid, shard
        raise IndexError(
            f"global sample index {global_sample_index} outside shard ranges"
        )

    def _load_shard_arrays(self, shard_index: int) -> Dict[str, np.ndarray]:
        if self._cache_shard_index == shard_index and self._cache_data is not None:
            return self._cache_data
        shard = self.shards[shard_index]
        shard_path = self.root_dir / shard.file
        with np.load(shard_path, allow_pickle=True) as data:
            arrays = {key: data[key] for key in data.files}
        self._cache_shard_index = shard_index
        self._cache_data = arrays
        return arrays

    def _resolve_image_width_per_scale(self) -> tuple[int, ...]:
        if not bool(JAGGED_IMAGE_WIDTHS_ENABLED):
            day_width = int(self.manifest.get("day_width", 0) or 0)
            windows = int(self.manifest.get("decomposition_windows", 0) or 0)
            width = int(day_width * windows) if day_width > 0 and windows > 0 else 0
            if width < 1:
                width = int(self.manifest.get("image_width", 0) or 0)
            if width < 1:
                raw = self.manifest.get("image_width_per_scale")
                if isinstance(raw, list) and raw:
                    width = int(max(int(x) for x in raw))
            if width < 1:
                width = 1
            return tuple(int(width) for _ in range(self.scales))

        raw = self.manifest.get("image_width_per_scale")
        if isinstance(raw, list) and len(raw) == self.scales:
            widths = [int(x) for x in raw]
        else:
            widths = self._infer_widths_from_weekend_spec()
        if len(widths) != self.scales:
            raise ValueError(
                "image width spec does not match decomposition scales: "
                f"len(widths)={len(widths)} scales={self.scales}"
            )
        if any(int(w) < 1 for w in widths):
            raise ValueError(f"image widths must be >= 1, got {widths}")
        return tuple(int(w) for w in widths)

    def _infer_widths_from_weekend_spec(self) -> List[int]:
        # Fallback to the exact build_decomp_ohlc_images formula:
        # width = day_width * windows (+ weekend_gap_width for selected scales).
        day_width = int(self.manifest.get("day_width", 0) or 0)
        windows = int(self.manifest.get("decomposition_windows", 0) or 0)
        image_width = int(self.manifest.get("image_width", 0) or 0)
        if day_width > 0 and windows > 0:
            base_width = int(day_width * windows)
            widths = [base_width for _ in range(self.scales)]
            if bool(self.manifest.get("weekend_feature_enabled", False)):
                gap_width = int(self.manifest.get("weekend_gap_width", 0) or 0)
                weekend_scales = self.manifest.get("weekend_gap_scale_indices", [])
                weekend_set = {int(x) for x in weekend_scales}
                if gap_width > 0:
                    for s in range(self.scales):
                        if s in weekend_set:
                            widths[s] += int(gap_width)
            return widths
        fallback = int(image_width) if int(image_width) > 0 else 1
        return [fallback for _ in range(self.scales)]

    def trim_scale_padding(self, x_img: np.ndarray) -> List[np.ndarray]:
        if x_img.ndim != 3:
            raise ValueError(f"expected X_img shape (scales, height, width), got {x_img.shape}")
        if int(x_img.shape[0]) != self.scales:
            raise ValueError(f"expected scales={self.scales}, got {int(x_img.shape[0])}")
        width_max = int(x_img.shape[2])
        out: List[np.ndarray] = []
        for s, width in enumerate(self.image_width_per_scale):
            panel_w = max(1, min(int(width), width_max))
            out.append(np.ascontiguousarray(x_img[s, :, :panel_w]))
        return out

    def get(self, index: int, trim_to_scale_widths: bool = False) -> Dict[str, Any]:
        if index < 0:
            index = self.subset_count + index
        if index < 0 or index >= self.subset_count:
            raise IndexError(f"index {index} out of range for len={self.subset_count}")
        global_idx = self.subset_start + index
        shard_idx, shard = self._find_shard(global_idx)
        arrays = self._load_shard_arrays(shard_idx)
        local = global_idx - shard.sample_start
        x_img = arrays["X_img"][local]
        item = {
            "X_img": x_img,
            "y_raw": arrays["y_raw"][local],
            "timestamp": str(arrays["timestamps"][local]),
            "sample_index": int(arrays["sample_indices"][local]),
            "image_width_per_scale": self.image_width_per_scale,
            "jagged_image_widths_enabled": self.jagged_image_widths_enabled,
        }
        if trim_to_scale_widths:
            item["X_img_by_scale"] = self.trim_scale_padding(x_img)
        if "ticker_ids" in arrays:
            item["ticker_id"] = int(arrays["ticker_ids"][local])
        return item

    def get_batch(
        self,
        start: int,
        stop: int,
        trim_to_scale_widths: bool = False,
    ) -> Dict[str, np.ndarray | List[np.ndarray]]:
        if start < 0 or stop < 0 or stop < start:
            raise ValueError(f"invalid range [{start}, {stop})")
        start = min(start, self.subset_count)
        stop = min(stop, self.subset_count)
        if start >= stop:
            out_empty: Dict[str, np.ndarray | List[np.ndarray]] = {
                "X_img": np.empty(
                    (0, int(self.scales), int(self.image_height), int(self.image_width)),
                    dtype=np.uint8,
                ),
                "y_raw": np.empty((0,), dtype=np.float64),
                "timestamps": np.empty((0,), dtype=object),
                "sample_indices": np.empty((0,), dtype=np.int64),
                "ticker_ids": np.empty((0,), dtype=np.int32),
                "image_width_per_scale": np.asarray(
                    self.image_width_per_scale, dtype=np.int64
                ),
                "jagged_image_widths_enabled": np.asarray(
                    [int(self.jagged_image_widths_enabled)], dtype=np.int64
                ),
            }
            if trim_to_scale_widths:
                out_empty["X_img_by_scale"] = [
                    np.empty((0, int(self.image_height), int(w)), dtype=np.uint8)
                    for w in self.image_width_per_scale
                ]
            return out_empty

        xs = []
        ys = []
        ts = []
        si = []
        ti = []
        for idx in range(start, stop):
            item = self.get(idx)
            xs.append(item["X_img"])
            ys.append(item["y_raw"])
            ts.append(item["timestamp"])
            si.append(item["sample_index"])
            ti.append(item.get("ticker_id", -1))
        x_batch = np.stack(xs, axis=0)
        out: Dict[str, np.ndarray | List[np.ndarray]] = {
            "X_img": x_batch,
            "y_raw": np.stack(ys, axis=0),
            "timestamps": np.array(ts),
            "sample_indices": np.array(si, dtype=np.int64),
            "ticker_ids": np.array(ti, dtype=np.int32),
            "image_width_per_scale": np.asarray(self.image_width_per_scale, dtype=np.int64),
            "jagged_image_widths_enabled": np.asarray(
                [int(self.jagged_image_widths_enabled)], dtype=np.int64
            ),
        }
        if trim_to_scale_widths:
            out["X_img_by_scale"] = [
                np.ascontiguousarray(x_batch[:, s, :, : int(width)])
                for s, width in enumerate(self.image_width_per_scale)
            ]
        return out
