#!/usr/bin/env python3
"""Combine daily dataset shards into single .npz files.

Reads a shard manifest produced by prepare_daily_data_chunked.py and concatenates
seq, decomp, and dual datasets into the standard single-file outputs.
"""

from __future__ import annotations

import argparse
import heapq
import json
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import prepare_daily_data as pdd

ASSEMBLY_REORDER_CHUNK = 16384


def _coerce_str(value) -> str:
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine daily shard .npz files into single dataset files."
    )
    parser.add_argument(
        "--manifest",
        default="data/daily_shards/stock_dataset_manifest.json",
        help="Path to shard manifest JSON.",
    )
    parser.add_argument(
        "--output-path",
        default="data/daily/stock_dataset.npz",
        help="Base output .npz path (sequence).",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Use np.savez_compressed (smaller, slower).",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Optional temp dir for memmaps (default: alongside output).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary memmap files instead of deleting them.",
    )
    parser.add_argument(
        "--skip-decomp",
        action="store_true",
        help="Skip combining the decomposition dataset.",
    )
    parser.add_argument(
        "--skip-dual",
        action="store_true",
        help="Skip combining the dual dataset.",
    )
    parser.set_defaults(chronological_assembly=None)
    parser.add_argument(
        "--chronological-assembly",
        dest="chronological_assembly",
        action="store_true",
        help=(
            "Assemble combined samples chronologically across shards/tickers. "
            "If omitted, inherit from manifest args when present (default: enabled)."
        ),
    )
    parser.add_argument(
        "--disable-chronological-assembly",
        dest="chronological_assembly",
        action="store_false",
        help="Disable chronological assembly and keep shard/ticker concatenation order.",
    )
    return parser.parse_args(argv)


def _save_npz(path: Path, payload: Dict[str, np.ndarray], compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def _load_manifest(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text())
    shards = data.get("shards", [])
    if not shards:
        raise ValueError("manifest contains no shards")
    data["shards"] = list(shards)
    return data


def _build_global_tickers(manifest: Dict[str, object]) -> List[str]:
    tickers = manifest.get("global_tickers")
    if tickers:
        return list(tickers)
    seen = set()
    ordered = []
    for shard in manifest["shards"]:
        for t in shard.get("tickers", []):
            if t in seen:
                continue
            seen.add(t)
            ordered.append(t)
    return ordered


def _int_or_zero(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _select_dataset_shards(
    shards: Sequence[Dict[str, object]],
    dataset_name: str,
    path_key: str,
    samples_key: str,
) -> Tuple[List[Dict[str, object]], int]:
    selected: List[Dict[str, object]] = []
    total_samples = 0
    missing_paths: List[Tuple[int, int]] = []
    for idx, shard in enumerate(shards):
        n = _int_or_zero(shard.get(samples_key, 0))
        if n <= 0:
            continue
        path_value = shard.get(path_key)
        path_text = _coerce_str(path_value).strip()
        if not path_text:
            missing_paths.append((idx, n))
            continue
        selected.append(shard)
        total_samples += n

    if missing_paths:
        preview = ", ".join(f"{idx}:{n}" for idx, n in missing_paths[:5])
        suffix = ""
        if len(missing_paths) > 5:
            suffix = f", ... +{len(missing_paths) - 5} more"
        raise ValueError(
            f"manifest has {dataset_name} shards with {samples_key}>0 but missing "
            f"'{path_key}': {preview}{suffix}"
        )
    return selected, total_samples


def _timestamp_dtype() -> np.dtype:
    return np.dtype("<U32")


def _alloc_memmap(
    path: Path,
    shape: Tuple[int, ...],
    dtype: np.dtype,
) -> np.memmap:
    return np.memmap(str(path), mode="w+", dtype=dtype, shape=shape)


def _build_chronological_assembly_order(
    timestamps: np.memmap,
    ticker_ids: np.memmap,
    part_bounds: Sequence[Tuple[int, int]],
) -> Optional[np.ndarray]:
    total = int(timestamps.shape[0])
    if total <= 1:
        return None

    ts_dt = np.asarray(timestamps).astype("datetime64[ns]")
    if np.isnat(ts_dt).any():
        raise ValueError(
            "cannot assemble chronologically: timestamps contain invalid values"
        )
    ts_ns = ts_dt.astype(np.int64, copy=False)
    tid_all = np.asarray(ticker_ids).astype(np.int64, copy=False)
    if int(tid_all.shape[0]) != total:
        raise ValueError(
            "chronological assembly shape mismatch: "
            f"timestamps={total} ticker_ids={int(tid_all.shape[0])}"
        )

    cursor = 0
    for part, (start, length) in enumerate(part_bounds):
        start_i = int(start)
        length_i = int(length)
        if length_i < 0:
            raise ValueError(f"invalid shard part length: part={part} length={length_i}")
        if start_i != cursor:
            raise ValueError(
                "invalid shard part bounds: "
                f"part={part} start={start_i} expected={cursor}"
            )
        end_i = start_i + length_i
        if end_i > total:
            raise ValueError(
                "invalid shard part bounds: "
                f"part={part} end={end_i} total={total}"
            )
        if length_i > 1:
            ts_part = ts_ns[start_i:end_i]
            if bool(np.any(ts_part[1:] < ts_part[:-1])):
                raise ValueError(
                    "cannot assemble chronologically: part timestamps are not monotonic "
                    f"increasing (part={part})"
                )
            tid_part = tid_all[start_i:end_i]
            same_ts = ts_part[1:] == ts_part[:-1]
            if bool(same_ts.any()) and bool(
                np.any(tid_part[1:][same_ts] < tid_part[:-1][same_ts])
            ):
                raise ValueError(
                    "cannot assemble chronologically: part ticker ids are not "
                    f"monotonic for equal timestamps (part={part})"
                )
        cursor = end_i

    if cursor != total:
        raise ValueError(
            "invalid shard part bounds: "
            f"covered={cursor} total={total}"
        )

    heap: list[tuple[int, int, int, int]] = []
    for part, (start, length) in enumerate(part_bounds):
        if int(length) <= 0:
            continue
        pos = int(start)
        heapq.heappush(
            heap,
            (
                int(ts_ns[pos]),
                int(tid_all[pos]),
                int(part),
                0,
            ),
        )

    order = np.empty((total,), dtype=np.int64)
    w = 0
    while heap:
        _, _, part, local = heapq.heappop(heap)
        start, length = part_bounds[int(part)]
        position = int(start) + int(local)
        order[w] = position
        w += 1
        nxt = int(local) + 1
        if nxt < int(length):
            nxt_pos = int(start) + nxt
            heapq.heappush(
                heap,
                (
                    int(ts_ns[nxt_pos]),
                    int(tid_all[nxt_pos]),
                    int(part),
                    nxt,
                ),
            )

    if w != total:
        raise RuntimeError(
            "chronological assembly order construction failed: "
            f"expected={total} built={w}"
        )
    if bool(np.all(order[1:] > order[:-1])):
        return None
    return order


def _reorder_memmap_axis0(
    src: np.memmap,
    order: np.ndarray,
    out_path: Path,
    chunk_size: int = ASSEMBLY_REORDER_CHUNK,
) -> np.memmap:
    if src.shape[0] != order.shape[0]:
        raise ValueError("reorder shape mismatch")
    dst = _alloc_memmap(out_path, tuple(src.shape), src.dtype)
    n = int(order.shape[0])
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        idx = order[start:end]
        dst[start:end] = src[idx]
    dst.flush()
    return dst


def _apply_chronological_assembly(
    dataset_tag: str,
    ts_mm: np.memmap,
    arrays: Dict[str, np.memmap],
    part_bounds: Sequence[Tuple[int, int]],
    temp_dir: Path,
) -> Tuple[np.memmap, Dict[str, np.memmap]]:
    ticker_ids = arrays.get("ticker_ids")
    if ticker_ids is None:
        raise ValueError("ticker_ids array is required for chronological assembly")
    order = _build_chronological_assembly_order(ts_mm, ticker_ids, part_bounds)
    if order is None:
        return ts_mm, arrays

    print(
        f"[assembly] {dataset_tag}: applying chronological assembly "
        f"({order.size} samples)"
    )
    assembled_ts = _reorder_memmap_axis0(
        ts_mm, order, temp_dir / f"{dataset_tag}_timestamps_assembled.npy"
    )
    assembled_arrays: Dict[str, np.memmap] = {}
    for name, arr in arrays.items():
        assembled_arrays[name] = _reorder_memmap_axis0(
            arr, order, temp_dir / f"{dataset_tag}_{name}_assembled.npy"
        )
    return assembled_ts, assembled_arrays


def _map_ticker_ids(
    data: Dict[str, np.ndarray],
    global_index: Dict[str, int],
) -> np.ndarray:
    local_ids = data["ticker_ids"].astype(np.int64)
    if "global_ticker_ids" in data:
        global_map = data["global_ticker_ids"].astype(np.int64)
    else:
        local_tickers = [_coerce_str(v) for v in data["tickers"].tolist()]
        global_map = np.array(
            [global_index[t] for t in local_tickers], dtype=np.int64
        )
    return global_map[local_ids].astype(np.int32)


def _load_first_meta(path: Path) -> Dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        meta = {
            "X_shape": data["X"].shape,
            "X_dtype": data["X"].dtype,
            "y_shape": data["y_raw"].shape,
            "y_dtype": data["y_raw"].dtype,
            "feature_cols": data.get("feature_cols", None),
            "label_cols": data.get("label_cols", None),
            "lookback": int(data.get("lookback", [0])[0]),
            "horizon": int(data.get("horizon", [0])[0]),
            "stride": int(data.get("stride", [0])[0]),
            "entry_offset": int(data.get("entry_offset", [0])[0]),
        }
    return meta


def _load_first_meta_dual(path: Path) -> Dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        meta = {
            "X_seq_shape": data["X_seq"].shape,
            "X_seq_dtype": data["X_seq"].dtype,
            "X_decomp_shape": data["X_decomp"].shape,
            "X_decomp_dtype": data["X_decomp"].dtype,
            "y_shape": data["y_raw"].shape,
            "y_dtype": data["y_raw"].dtype,
            "feature_cols_seq": data.get("feature_cols_seq", None),
            "feature_cols_decomp": data.get("feature_cols_decomp", None),
            "label_cols": data.get("label_cols", None),
            "lookback": int(data.get("lookback", [0])[0]),
            "horizon": int(data.get("horizon", [0])[0]),
            "stride": int(data.get("stride", [0])[0]),
            "entry_offset": int(data.get("entry_offset", [0])[0]),
            "decomposition_scales": int(data.get("decomposition_scales", [0])[0]),
            "decomposition_windows": int(data.get("decomposition_windows", [0])[0]),
            "decomposition_normalization": _coerce_str(
                data.get("decomposition_normalization", [""])[0]
            ),
        }
    return meta


def _extract_decomp_extra(path: Path) -> Dict[str, np.ndarray]:
    extra = {}
    with np.load(path, allow_pickle=True) as data:
        for key in ("decomposition_scales", "decomposition_windows", "decomposition_normalization"):
            if key in data:
                extra[key] = data[key]
    return extra


def combine_base(
    shards: List[Dict[str, object]],
    total_samples: int,
    output_path: Path,
    global_tickers: List[str],
    compress: bool,
    temp_dir: Path,
    chronological_assembly: bool,
) -> None:
    if not shards:
        raise ValueError("no sequence shards selected")
    first_path = Path(_coerce_str(shards[0].get("base")))
    meta = _load_first_meta(first_path)
    if total_samples <= 0:
        raise ValueError("no sequence samples to combine")

    feature_cols = meta["feature_cols"]
    if isinstance(feature_cols, np.ndarray):
        feature_cols = [_coerce_str(v) for v in feature_cols.tolist()]
    label_cols = meta["label_cols"]
    if isinstance(label_cols, np.ndarray):
        label_cols = [_coerce_str(v) for v in label_cols.tolist()]

    x_shape = (total_samples,) + tuple(meta["X_shape"][1:])
    y_cols = meta["y_shape"][1:] if len(meta["y_shape"]) > 1 else (1,)
    y_shape = (total_samples,) + tuple(y_cols)

    X_mm = _alloc_memmap(temp_dir / "X_seq.npy", x_shape, meta["X_dtype"])
    y_mm = _alloc_memmap(temp_dir / "y_seq.npy", y_shape, meta["y_dtype"])
    ts_mm = _alloc_memmap(temp_dir / "ts_seq.npy", (total_samples,), _timestamp_dtype())
    tid_mm = _alloc_memmap(temp_dir / "ticker_seq.npy", (total_samples,), np.int32)

    global_index = {t: i for i, t in enumerate(global_tickers)}
    offset = 0
    part_bounds: List[Tuple[int, int]] = []
    for shard in shards:
        path = Path(_coerce_str(shard.get("base")))
        with np.load(path, allow_pickle=True) as data:
            X = data["X"]
            y = data["y_raw"]
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            n = X.shape[0]
            start = offset
            X_mm[offset : offset + n] = X
            y_mm[offset : offset + n] = y
            ts_mm[offset : offset + n] = data["timestamps"].astype(str)
            tid_mm[offset : offset + n] = _map_ticker_ids(data, global_index)
        offset += n
        part_bounds.append((start, int(n)))

    if bool(chronological_assembly):
        ts_mm, assembled_arrays = _apply_chronological_assembly(
            "seq",
            ts_mm,
            {"X": X_mm, "y": y_mm, "ticker_ids": tid_mm},
            part_bounds,
            temp_dir,
        )
        X_mm = assembled_arrays["X"]
        y_mm = assembled_arrays["y"]
        tid_mm = assembled_arrays["ticker_ids"]

    payload = {
        "X": np.asarray(X_mm),
        "y_raw": np.asarray(y_mm),
        "timestamps": np.asarray(ts_mm),
        "feature_cols": np.array(feature_cols, dtype=object),
        "label_cols": np.array(label_cols, dtype=object),
        "lookback": np.array([meta["lookback"]]),
        "horizon": np.array([meta["horizon"]]),
        "stride": np.array([meta["stride"]]),
        "entry_offset": np.array([meta["entry_offset"]]),
        "ticker_ids": np.asarray(tid_mm),
        "tickers": np.array(list(global_tickers), dtype=object),
    }
    _save_npz(output_path, payload, compress=compress)


def combine_decomp(
    shards: List[Dict[str, object]],
    total_samples: int,
    output_path: Path,
    global_tickers: List[str],
    compress: bool,
    temp_dir: Path,
    chronological_assembly: bool,
) -> None:
    if not shards:
        raise ValueError("no decomposition shards selected")
    first_path = Path(_coerce_str(shards[0].get("decomp")))
    meta = _load_first_meta(first_path)
    if total_samples <= 0:
        raise ValueError("no decomposition samples to combine")

    feature_cols = meta["feature_cols"]
    if isinstance(feature_cols, np.ndarray):
        feature_cols = [_coerce_str(v) for v in feature_cols.tolist()]
    label_cols = meta["label_cols"]
    if isinstance(label_cols, np.ndarray):
        label_cols = [_coerce_str(v) for v in label_cols.tolist()]

    x_shape = (total_samples,) + tuple(meta["X_shape"][1:])
    y_cols = meta["y_shape"][1:] if len(meta["y_shape"]) > 1 else (1,)
    y_shape = (total_samples,) + tuple(y_cols)

    X_mm = _alloc_memmap(temp_dir / "X_decomp.npy", x_shape, meta["X_dtype"])
    y_mm = _alloc_memmap(temp_dir / "y_decomp.npy", y_shape, meta["y_dtype"])
    ts_mm = _alloc_memmap(temp_dir / "ts_decomp.npy", (total_samples,), _timestamp_dtype())
    tid_mm = _alloc_memmap(temp_dir / "ticker_decomp.npy", (total_samples,), np.int32)

    global_index = {t: i for i, t in enumerate(global_tickers)}
    offset = 0
    part_bounds: List[Tuple[int, int]] = []
    for shard in shards:
        path = Path(_coerce_str(shard.get("decomp")))
        with np.load(path, allow_pickle=True) as data:
            X = data["X"]
            y = data["y_raw"]
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            n = X.shape[0]
            start = offset
            X_mm[offset : offset + n] = X
            y_mm[offset : offset + n] = y
            ts_mm[offset : offset + n] = data["timestamps"].astype(str)
            tid_mm[offset : offset + n] = _map_ticker_ids(data, global_index)
        offset += n
        part_bounds.append((start, int(n)))

    if bool(chronological_assembly):
        ts_mm, assembled_arrays = _apply_chronological_assembly(
            "decomp",
            ts_mm,
            {"X": X_mm, "y": y_mm, "ticker_ids": tid_mm},
            part_bounds,
            temp_dir,
        )
        X_mm = assembled_arrays["X"]
        y_mm = assembled_arrays["y"]
        tid_mm = assembled_arrays["ticker_ids"]

    extra = _extract_decomp_extra(first_path)
    payload = {
        "X": np.asarray(X_mm),
        "y_raw": np.asarray(y_mm),
        "timestamps": np.asarray(ts_mm),
        "feature_cols": np.array(feature_cols, dtype=object),
        "label_cols": np.array(label_cols, dtype=object),
        "lookback": np.array([meta["lookback"]]),
        "horizon": np.array([meta["horizon"]]),
        "stride": np.array([meta["stride"]]),
        "entry_offset": np.array([meta["entry_offset"]]),
        "ticker_ids": np.asarray(tid_mm),
        "tickers": np.array(list(global_tickers), dtype=object),
    }
    payload.update(extra)
    _save_npz(output_path, payload, compress=compress)


def combine_dual(
    shards: List[Dict[str, object]],
    total_samples: int,
    output_path: Path,
    global_tickers: List[str],
    compress: bool,
    temp_dir: Path,
    chronological_assembly: bool,
) -> None:
    if not shards:
        raise ValueError("no dual shards selected")
    first_path = Path(_coerce_str(shards[0].get("dual")))
    meta = _load_first_meta_dual(first_path)
    if total_samples <= 0:
        raise ValueError("no dual samples to combine")

    feature_cols_seq = meta["feature_cols_seq"]
    if isinstance(feature_cols_seq, np.ndarray):
        feature_cols_seq = [_coerce_str(v) for v in feature_cols_seq.tolist()]
    feature_cols_decomp = meta["feature_cols_decomp"]
    if isinstance(feature_cols_decomp, np.ndarray):
        feature_cols_decomp = [
            _coerce_str(v) for v in feature_cols_decomp.tolist()
        ]
    label_cols = meta["label_cols"]
    if isinstance(label_cols, np.ndarray):
        label_cols = [_coerce_str(v) for v in label_cols.tolist()]

    x_seq_shape = (total_samples,) + tuple(meta["X_seq_shape"][1:])
    x_decomp_shape = (total_samples,) + tuple(meta["X_decomp_shape"][1:])
    y_cols = meta["y_shape"][1:] if len(meta["y_shape"]) > 1 else (1,)
    y_shape = (total_samples,) + tuple(y_cols)

    X_seq_mm = _alloc_memmap(temp_dir / "X_seq_dual.npy", x_seq_shape, meta["X_seq_dtype"])
    X_decomp_mm = _alloc_memmap(
        temp_dir / "X_decomp_dual.npy", x_decomp_shape, meta["X_decomp_dtype"]
    )
    y_mm = _alloc_memmap(temp_dir / "y_dual.npy", y_shape, meta["y_dtype"])
    ts_mm = _alloc_memmap(temp_dir / "ts_dual.npy", (total_samples,), _timestamp_dtype())
    tid_mm = _alloc_memmap(temp_dir / "ticker_dual.npy", (total_samples,), np.int32)

    global_index = {t: i for i, t in enumerate(global_tickers)}
    offset = 0
    part_bounds: List[Tuple[int, int]] = []
    for shard in shards:
        path = Path(_coerce_str(shard.get("dual")))
        with np.load(path, allow_pickle=True) as data:
            X_seq = data["X_seq"]
            X_decomp = data["X_decomp"]
            y = data["y_raw"]
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            n = X_seq.shape[0]
            start = offset
            X_seq_mm[offset : offset + n] = X_seq
            X_decomp_mm[offset : offset + n] = X_decomp
            y_mm[offset : offset + n] = y
            ts_mm[offset : offset + n] = data["timestamps"].astype(str)
            tid_mm[offset : offset + n] = _map_ticker_ids(data, global_index)
        offset += n
        part_bounds.append((start, int(n)))

    if bool(chronological_assembly):
        ts_mm, assembled_arrays = _apply_chronological_assembly(
            "dual",
            ts_mm,
            {"X_seq": X_seq_mm, "X_decomp": X_decomp_mm, "y": y_mm, "ticker_ids": tid_mm},
            part_bounds,
            temp_dir,
        )
        X_seq_mm = assembled_arrays["X_seq"]
        X_decomp_mm = assembled_arrays["X_decomp"]
        y_mm = assembled_arrays["y"]
        tid_mm = assembled_arrays["ticker_ids"]

    payload = {
        "X_seq": np.asarray(X_seq_mm),
        "X_decomp": np.asarray(X_decomp_mm),
        "y_raw": np.asarray(y_mm),
        "timestamps": np.asarray(ts_mm),
        "feature_cols_seq": np.array(feature_cols_seq, dtype=object),
        "feature_cols_decomp": np.array(feature_cols_decomp, dtype=object),
        "label_cols": np.array(label_cols, dtype=object),
        "lookback": np.array([meta["lookback"]]),
        "horizon": np.array([meta["horizon"]]),
        "stride": np.array([meta["stride"]]),
        "entry_offset": np.array([meta["entry_offset"]]),
        "ticker_ids": np.asarray(tid_mm),
        "tickers": np.array(list(global_tickers), dtype=object),
        "decomposition_scales": np.array([meta["decomposition_scales"]]),
        "decomposition_windows": np.array([meta["decomposition_windows"]]),
        "decomposition_normalization": np.array([meta["decomposition_normalization"]]),
    }
    _save_npz(output_path, payload, compress=compress)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest)
    manifest = _load_manifest(manifest_path)
    shards = manifest["shards"]
    seq_shards, seq_samples = _select_dataset_shards(
        shards, "sequence", "base", "samples_seq"
    )
    decomp_shards, decomp_samples = _select_dataset_shards(
        shards, "decomposition", "decomp", "samples_decomp"
    )
    dual_shards, dual_samples = _select_dataset_shards(
        shards, "dual", "dual", "samples_dual"
    )

    output_path = Path(args.output_path)
    decomp_path = pdd.derive_decomposition_npz_path(output_path)
    dual_path = pdd.derive_dual_npz_path(output_path)

    global_tickers = _build_global_tickers(manifest)
    if not global_tickers:
        raise ValueError("no tickers found in manifest")
    manifest_args = manifest.get("args")
    if not isinstance(manifest_args, dict):
        manifest_args = {}
    if args.chronological_assembly is None:
        chronological_assembly = bool(
            manifest_args.get("chronological_assembly", True)
        )
    else:
        chronological_assembly = bool(args.chronological_assembly)
    print(f"chronological_assembly={chronological_assembly}")

    if args.temp_dir:
        temp_root = Path(args.temp_dir)
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
        temp_base = temp_root / (output_path.stem + "_tmp")
        temp_base.mkdir(parents=True, exist_ok=True)
    else:
        try:
            temp_ctx = tempfile.TemporaryDirectory(dir=str(output_path.parent))
        except OSError:
            print(
                f"warning: cannot create temp directory under {output_path.parent}; "
                "falling back to system temp directory"
            )
            temp_ctx = tempfile.TemporaryDirectory()
        temp_base = Path(temp_ctx.name)

    try:
        combined_any = False
        if seq_samples > 0:
            combine_base(
                seq_shards,
                seq_samples,
                output_path,
                global_tickers,
                args.compress,
                temp_base,
                chronological_assembly,
            )
            print(f"saved: {output_path}")
            combined_any = True
        else:
            print("skipping sequence combine: no sequence shards with samples")
        if not args.skip_decomp:
            if decomp_samples > 0:
                combine_decomp(
                    decomp_shards,
                    decomp_samples,
                    decomp_path,
                    global_tickers,
                    args.compress,
                    temp_base,
                    chronological_assembly,
                )
                print(f"saved: {decomp_path}")
                combined_any = True
            else:
                print(
                    "skipping decomposition combine: no decomposition shards with samples"
                )
        if not args.skip_dual:
            if dual_samples > 0:
                combine_dual(
                    dual_shards,
                    dual_samples,
                    dual_path,
                    global_tickers,
                    args.compress,
                    temp_base,
                    chronological_assembly,
                )
                print(f"saved: {dual_path}")
                combined_any = True
            else:
                print("skipping dual combine: no dual shards with samples")

        if not combined_any:
            raise ValueError(
                "nothing to combine: no datasets have samples after applying --skip flags"
            )
    finally:
        if args.keep_temp:
            print(f"temp kept at: {temp_base}")
        else:
            if temp_ctx is not None:
                temp_ctx.cleanup()
            else:
                shutil.rmtree(temp_base, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
