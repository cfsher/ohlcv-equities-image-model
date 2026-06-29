#!/usr/bin/env python3
"""Prepare daily candle datasets in low-memory shards.

Builds sequence, decomposition, and dual datasets from ticker CSVs while
writing shards incrementally to avoid large RAM spikes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import prepare_daily_data as pdd


def _save_npz(
    output_path: Path,
    payload: Dict[str, np.ndarray],
    compress: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(output_path, **payload)
    else:
        np.savez(output_path, **payload)


def save_dataset_npz(
    output_path: Path,
    X: np.ndarray,
    y_raw: np.ndarray,
    sample_index: pd.DatetimeIndex,
    feature_cols: Sequence[str],
    label_cols: Sequence[str],
    lookback: int,
    horizon: int,
    stride: int,
    ticker_ids: np.ndarray,
    tickers: Sequence[str],
    compress: bool,
    extra_payload: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    timestamps = sample_index.astype(str).to_numpy()
    payload = {
        "X": X,
        "y_raw": np.asarray(y_raw),
        "timestamps": timestamps,
        "feature_cols": np.array(list(feature_cols), dtype=object),
        "label_cols": np.array(list(label_cols), dtype=object),
        "lookback": np.array([lookback]),
        "horizon": np.array([horizon]),
        "stride": np.array([pdd.normalize_stride(stride)]),
        "entry_offset": np.array([pdd.ENTRY_OFFSET]),
        "ticker_ids": ticker_ids.astype(np.int32),
        "tickers": np.array(list(tickers), dtype=object),
    }
    if extra_payload:
        payload.update(extra_payload)
    _save_npz(output_path, payload, compress=compress)


def save_dual_dataset_npz(
    output_path: Path,
    X_seq: np.ndarray,
    X_decomp: np.ndarray,
    y_raw: np.ndarray,
    sample_index: pd.DatetimeIndex,
    feature_cols_seq: Sequence[str],
    feature_cols_decomp: Sequence[str],
    label_cols: Sequence[str],
    lookback: int,
    horizon: int,
    stride: int,
    ticker_ids: np.ndarray,
    tickers: Sequence[str],
    compress: bool,
    extra_payload: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    timestamps = sample_index.astype(str).to_numpy()
    payload = {
        "X_seq": X_seq,
        "X_decomp": X_decomp,
        "y_raw": np.asarray(y_raw),
        "timestamps": timestamps,
        "feature_cols_seq": np.array(list(feature_cols_seq), dtype=object),
        "feature_cols_decomp": np.array(list(feature_cols_decomp), dtype=object),
        "label_cols": np.array(list(label_cols), dtype=object),
        "lookback": np.array([lookback]),
        "horizon": np.array([horizon]),
        "stride": np.array([pdd.normalize_stride(stride)]),
        "entry_offset": np.array([pdd.ENTRY_OFFSET]),
        "ticker_ids": ticker_ids.astype(np.int32),
        "tickers": np.array(list(tickers), dtype=object),
    }
    if extra_payload:
        payload.update(extra_payload)
    _save_npz(output_path, payload, compress=compress)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare daily candle datasets in shards (low memory)."
    )
    parser.add_argument(
        "--input-dir",
        default=pdd.DEFAULT_INPUT_DIR,
        help=f"Directory with ticker CSVs (default: {pdd.DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/daily_shards",
        help="Directory to write shard .npz files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="stock_dataset",
        help="Base filename prefix for shard outputs.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        help=(
            "Unified lookback for both seq/decomp in chunked mode. "
            "If omitted, uses prepare_daily_data.DEFAULT_LOOKBACK_SEQ and "
            "prepare_daily_data.DEFAULT_LOOKBACK_DECOMP."
        ),
    )
    parser.add_argument(
        "--lookback-seq",
        type=int,
        default=None,
        help=(
            "Sequence lookback override. If omitted, falls back to --lookback "
            "when provided, otherwise prepare_daily_data.DEFAULT_LOOKBACK_SEQ."
        ),
    )
    parser.add_argument(
        "--lookback-decomp",
        type=int,
        default=None,
        help=(
            "Decomposition lookback override. If omitted, falls back to --lookback "
            "when provided, otherwise prepare_daily_data.DEFAULT_LOOKBACK_DECOMP."
        ),
    )
    parser.add_argument("--horizon", type=int, default=pdd.DEFAULT_HORIZON)
    parser.add_argument(
        "--label-mode",
        type=pdd.resolve_label_mode,
        default=pdd.LABEL_MODE_RANGE_ATR,
        choices=list(pdd.LABEL_MODE_CHOICES),
        help=(
            "Label construction mode. "
            "'range_atr' preserves the existing horizon-based labels. "
            "'next_day_close_return' uses signal-day close -> next-day close returns "
            "and forces an effective label horizon of 1."
        ),
    )
    parser.add_argument("--stride", type=int, default=pdd.DEFAULT_STRIDE)
    parser.add_argument(
        "--liquidity-feature",
        default=None,
        choices=["turnover", "volume"],
        help=(
            "Liquidity feature for decomposition branch. If omitted, uses "
            "prepare_daily_data.LIQUIDITY_FEATURE_DEFAULT."
        ),
    )
    parser.add_argument("--decomp-windows", type=int, default=pdd.DEFAULT_DECOMP_WINDOWS)
    parser.add_argument("--decomp-scales", type=int, default=None)
    parser.add_argument(
        "--decomp-normalization",
        type=pdd.resolve_decomp_normalization,
        default=None,
        help=(
            "Decomposition normalization: 'synthetic', 'lookback', or 'none' "
            "(raw decomposition values without normalization)."
        ),
    )
    parser.add_argument(
        "--final-candle-move-filter-enabled",
        "--final_candle_move_filter_enabled",
        dest="final_candle_move_filter_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable sample filtering based on large final-candle move thresholds "
            "across decomposition scales."
        ),
    )
    parser.add_argument(
        "--final-candle-move-threshold",
        "--final_candle_move_threshold",
        dest="final_candle_move_threshold",
        type=float,
        default=None,
        help=(
            "Absolute return threshold used by final-candle move filtering "
            "across decomposition scales."
        ),
    )
    parser.add_argument(
        "--final-candle-move-filter-scales",
        "--final_candle_move_filter_scales",
        dest="final_candle_move_filter_scales",
        choices=["0", "all"],
        default=None,
        help=(
            "Which decomposition scales to inspect for final-candle move filtering: "
            "'0' (highest-resolution only) or 'all'."
        ),
    )
    parser.add_argument(
        "--global-ma-n-window",
        "--global_ma_n_window",
        dest="global_ma_n_window",
        type=int,
        default=None,
        help=(
            "Global MA window used for ma_n feature engineering."
        ),
    )
    parser.add_argument(
        "--decomp-scale-aware-ma-feature-enabled",
        "--decomp_scale_aware_ma_feature_enabled",
        dest="decomp_scale_aware_ma_feature_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable scale-aware MA values for decomposition ma feature when "
            "--decomp-include-ma is enabled."
        ),
    )
    parser.add_argument(
        "--decomp-include-ma",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Include ma_n as an additional decomposition feature/channel. "
            "When enabled, decomposition X shape becomes "
            "(samples, scales, 6, windows)."
        ),
    )
    parser.add_argument(
        "--include-synthetic-features",
        action="store_true",
        help="Append synthetic price series features to the sequence dataset.",
    )
    parser.add_argument(
        "--disable-turnover-backfill",
        action="store_true",
        help="Disable split-based backfill for shares_outstanding/turnover.",
    )
    parser.add_argument(
        "--disable-no-split-history-fallback",
        action="store_true",
        help=(
            "Disable flat shares_outstanding backfill when split history is unavailable."
        ),
    )
    parser.add_argument(
        "--disable-turnover-fallback-backfill",
        action="store_true",
        help=(
            "Disable fallback fill of missing turnover using "
            "volume / shares_outstanding."
        ),
    )
    parser.set_defaults(chronological_assembly=True)
    parser.add_argument(
        "--chronological-assembly",
        dest="chronological_assembly",
        action="store_true",
        help=(
            "Assemble shard samples in ascending timestamp order across tickers "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--disable-chronological-assembly",
        dest="chronological_assembly",
        action="store_false",
        help=(
            "Disable chronological assembly and keep ticker-grouped shard order."
        ),
    )
    parser.add_argument(
        "--max-samples-per-shard",
        type=int,
        default=50_000,
        help="Max samples per shard (applies to seq/decomp/dual).",
    )
    parser.add_argument(
        "--max-tickers-per-shard",
        type=int,
        default=0,
        help="Optional cap on tickers per shard (0 = no cap).",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Use np.savez_compressed (smaller, slower).",
    )
    parser.add_argument(
        "--decomp-only",
        action="store_true",
        help="Build and save only decomposition shard outputs.",
    )
    parser.add_argument(
        "--start-date",
        "--start_date",
        dest="start_date",
        default=None,
        help=(
            "Optional inclusive start date filter for input rows "
            "(e.g. 2012-01-01)."
        ),
    )
    parser.add_argument(
        "--min-valid-volume",
        type=float,
        default=None,
        help=(
            "Minimum allowed daily volume during data-quality filtering "
            "(if omitted, uses prepare_daily_data.MIN_VALID_VOLUME)."
        ),
    )
    parser.add_argument(
        "--min-avg-dollar-volume-3m",
        type=float,
        default=float(pdd.MIN_AVG_DOLLAR_VOLUME_3m),
        help=(
            "Minimum average dollar volume per 3-month bucket "
            f"(default: {float(pdd.MIN_AVG_DOLLAR_VOLUME_3m)})."
        ),
    )
    return parser.parse_args(argv)


def _concat_datetime(parts: List[pd.DatetimeIndex]) -> pd.DatetimeIndex:
    if not parts:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(np.concatenate([idx.to_numpy() for idx in parts]))


def _flush_shard(
    output_dir: Path,
    output_prefix: str,
    shard_idx: int,
    lookback_seq: int,
    lookback_decomp: int,
    horizon: int,
    stride: int,
    feature_cols_seq: List[str],
    feature_cols_decomp: List[str],
    extra_payload: Optional[Dict[str, np.ndarray]],
    shard_tickers: List[str],
    seq_arrays: List[np.ndarray],
    y_seq_list: List[np.ndarray],
    idx_seq_list: List[pd.DatetimeIndex],
    ticker_ids_seq: List[np.ndarray],
    decomp_arrays: List[np.ndarray],
    y_decomp_list: List[np.ndarray],
    idx_decomp_list: List[pd.DatetimeIndex],
    ticker_ids_decomp: List[np.ndarray],
    seq_dual_arrays: List[np.ndarray],
    decomp_dual_arrays: List[np.ndarray],
    y_dual_list: List[np.ndarray],
    idx_dual_list: List[pd.DatetimeIndex],
    ticker_ids_dual: List[np.ndarray],
    chronological_assembly: bool,
    compress: bool,
    decomp_only: bool = False,
    shard_ticker_global_ids: Optional[List[int]] = None,
) -> Dict[str, object]:
    if bool(decomp_only):
        if not decomp_arrays:
            return {}
    elif not seq_arrays:
        return {}

    if bool(decomp_only):
        if bool(chronological_assembly):
            idx_decomp, ticker_ids_decomp_arr, (X_decomp, y_decomp) = (
                pdd.assemble_samples_chronologically(
                    idx_decomp_list,
                    ticker_ids_decomp,
                    decomp_arrays,
                    y_decomp_list,
                )
            )
        else:
            X_decomp = np.concatenate(decomp_arrays, axis=0)
            y_decomp = np.concatenate(y_decomp_list, axis=0)
            idx_decomp = _concat_datetime(idx_decomp_list)
            ticker_ids_decomp_arr = np.concatenate(ticker_ids_decomp, axis=0)
    else:
        if bool(chronological_assembly):
            idx_seq, ticker_ids_seq_arr, (X_seq, y_seq) = (
                pdd.assemble_samples_chronologically(
                    idx_seq_list,
                    ticker_ids_seq,
                    seq_arrays,
                    y_seq_list,
                )
            )
            idx_decomp, ticker_ids_decomp_arr, (X_decomp, y_decomp) = (
                pdd.assemble_samples_chronologically(
                    idx_decomp_list,
                    ticker_ids_decomp,
                    decomp_arrays,
                    y_decomp_list,
                )
            )
            idx_dual, ticker_ids_dual_arr, (X_seq_dual, X_decomp_dual, y_dual) = (
                pdd.assemble_samples_chronologically(
                    idx_dual_list,
                    ticker_ids_dual,
                    seq_dual_arrays,
                    decomp_dual_arrays,
                    y_dual_list,
                )
            )
        else:
            X_seq = np.concatenate(seq_arrays, axis=0)
            y_seq = np.concatenate(y_seq_list, axis=0)
            idx_seq = _concat_datetime(idx_seq_list)
            ticker_ids_seq_arr = np.concatenate(ticker_ids_seq, axis=0)

            X_decomp = np.concatenate(decomp_arrays, axis=0)
            y_decomp = np.concatenate(y_decomp_list, axis=0)
            idx_decomp = _concat_datetime(idx_decomp_list)
            ticker_ids_decomp_arr = np.concatenate(ticker_ids_decomp, axis=0)

            X_seq_dual = np.concatenate(seq_dual_arrays, axis=0)
            X_decomp_dual = np.concatenate(decomp_dual_arrays, axis=0)
            y_dual = np.concatenate(y_dual_list, axis=0)
            idx_dual = _concat_datetime(idx_dual_list)
            ticker_ids_dual_arr = np.concatenate(ticker_ids_dual, axis=0)

    base_npz_path = output_dir / f"{output_prefix}_shard{shard_idx:04d}.npz"
    decomp_npz_path = pdd.derive_decomposition_npz_path(base_npz_path)
    dual_npz_path = pdd.derive_dual_npz_path(base_npz_path)

    if not decomp_only:
        extra_base = {}
        if shard_ticker_global_ids is not None:
            extra_base["global_ticker_ids"] = np.array(shard_ticker_global_ids, dtype=np.int32)
        save_dataset_npz(
            base_npz_path,
            X_seq,
            y_seq,
            idx_seq,
            feature_cols_seq,
            pdd.LABEL_COLS,
            lookback_seq,
            horizon,
            stride,
            ticker_ids_seq_arr,
            shard_tickers,
            compress=compress,
            extra_payload=extra_base or None,
        )

    extra_payload = extra_payload or {}
    extra_decomp = dict(extra_payload)
    if shard_ticker_global_ids is not None:
        extra_decomp["global_ticker_ids"] = np.array(shard_ticker_global_ids, dtype=np.int32)
    save_dataset_npz(
        decomp_npz_path,
        X_decomp,
        y_decomp,
        idx_decomp,
        feature_cols_decomp,
        pdd.LABEL_COLS,
        lookback_decomp,
        horizon,
        stride,
        ticker_ids_decomp_arr,
        shard_tickers,
        compress=compress,
        extra_payload=extra_decomp,
    )

    if not decomp_only:
        extra_dual = dict(extra_payload)
        if shard_ticker_global_ids is not None:
            extra_dual["global_ticker_ids"] = np.array(shard_ticker_global_ids, dtype=np.int32)
        save_dual_dataset_npz(
            dual_npz_path,
            X_seq_dual,
            X_decomp_dual,
            y_dual,
            idx_dual,
            feature_cols_seq,
            feature_cols_decomp,
            pdd.LABEL_COLS,
            lookback_seq,
            horizon,
            stride,
            ticker_ids_dual_arr,
            shard_tickers,
            compress=compress,
            extra_payload=extra_dual,
        )

    return {
        "shard_index": shard_idx,
        "base": None if decomp_only else str(base_npz_path),
        "decomp": str(decomp_npz_path),
        "dual": None if decomp_only else str(dual_npz_path),
        "tickers": list(shard_tickers),
        "samples_seq": 0 if decomp_only else int(X_seq.shape[0]),
        "samples_decomp": int(X_decomp.shape[0]),
        "samples_dual": 0 if decomp_only else int(X_seq_dual.shape[0]),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    decomp_only = bool(args.decomp_only)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lookback_override = (
        int(args.lookback) if args.lookback is not None else None
    )
    lookback_seq = (
        int(args.lookback_seq)
        if args.lookback_seq is not None
        else (
            int(lookback_override)
            if lookback_override is not None
            else int(pdd.DEFAULT_LOOKBACK_SEQ)
        )
    )
    lookback_decomp = (
        int(args.lookback_decomp)
        if args.lookback_decomp is not None
        else (
            int(lookback_override)
            if lookback_override is not None
            else int(pdd.DEFAULT_LOOKBACK_DECOMP)
        )
    )
    if int(lookback_seq) < 1:
        raise ValueError("lookback_seq must be >= 1")
    if int(lookback_decomp) < 1:
        raise ValueError("lookback_decomp must be >= 1")
    label_mode = pdd.resolve_label_mode(args.label_mode)
    effective_label_horizon = pdd.resolve_effective_label_horizon(
        horizon=int(args.horizon),
        label_mode=label_mode,
    )
    min_rows_for_one_sample = pdd.resolve_min_rows_for_one_labeled_sample(
        lookback=max(int(lookback_seq), int(lookback_decomp)),
        horizon=int(args.horizon),
        label_mode=label_mode,
    )
    print(
        "effective lookbacks: "
        f"seq={int(lookback_seq)} decomp={int(lookback_decomp)} "
        f"(min rows for >=1 labeled sample: {int(min_rows_for_one_sample)})"
    )
    print(
        "label mode: "
        f"{label_mode} (effective_label_horizon={int(effective_label_horizon)})"
    )
    liquidity_feature = (
        pdd.resolve_liquidity_feature(args.liquidity_feature)
        if args.liquidity_feature is not None
        else pdd.resolve_liquidity_feature(pdd.LIQUIDITY_FEATURE_DEFAULT)
    )
    decomp_normalization = (
        pdd.resolve_decomp_normalization(args.decomp_normalization)
        if args.decomp_normalization is not None
        else pdd.resolve_decomp_normalization(pdd.DECOMP_NORMALIZATION)
    )
    final_candle_move_filter_enabled = (
        bool(args.final_candle_move_filter_enabled)
        if args.final_candle_move_filter_enabled is not None
        else bool(pdd.FINAL_CANDLE_MOVE_FILTER_ENABLED)
    )
    final_candle_move_threshold = (
        float(args.final_candle_move_threshold)
        if args.final_candle_move_threshold is not None
        else float(pdd.FINAL_CANDLE_MOVE_THRESHOLD)
    )
    final_candle_move_scales = pdd.resolve_final_candle_move_filter_scales(
        args.final_candle_move_filter_scales
        if args.final_candle_move_filter_scales is not None
        else pdd.FINAL_CANDLE_MOVE_FILTER_SCALES
    )
    global_ma_n_window = (
        int(args.global_ma_n_window)
        if args.global_ma_n_window is not None
        else int(pdd.GLOBAL_MA_N_WINDOW)
    )
    decomp_scale_aware_ma_feature_enabled = (
        bool(args.decomp_scale_aware_ma_feature_enabled)
        if args.decomp_scale_aware_ma_feature_enabled is not None
        else bool(pdd.DECOMP_SCALE_AWARE_MA_FEATURE_ENABLED)
    )
    decomp_include_ma = (
        bool(args.decomp_include_ma)
        if args.decomp_include_ma is not None
        else bool(pdd.DECOMP_INCLUDE_MA_DEFAULT)
    )
    min_valid_volume = (
        float(args.min_valid_volume)
        if args.min_valid_volume is not None
        else float(pdd.MIN_VALID_VOLUME)
    )

    if int(global_ma_n_window) < 1:
        raise ValueError("global_ma_n_window must be >= 1")
    if float(final_candle_move_threshold) < 0.0:
        raise ValueError("final_candle_move_threshold must be >= 0")
    start_date = pdd.normalize_start_date(args.start_date)

    pdd.MIN_VALID_VOLUME = float(min_valid_volume)
    pdd.GLOBAL_MA_N_WINDOW = int(global_ma_n_window)
    pdd.DECOMP_SCALE_AWARE_MA_FEATURE_ENABLED = bool(
        decomp_scale_aware_ma_feature_enabled
    )
    pdd.FINAL_CANDLE_MOVE_FILTER_ENABLED = bool(final_candle_move_filter_enabled)
    pdd.FINAL_CANDLE_MOVE_THRESHOLD = float(final_candle_move_threshold)
    pdd.FINAL_CANDLE_MOVE_FILTER_SCALES = final_candle_move_scales

    csv_paths = pdd.iter_ticker_files(input_dir)
    if not csv_paths:
        raise ValueError(f"no csv files found in {input_dir}")

    max_samples = int(args.max_samples_per_shard)
    max_tickers = int(args.max_tickers_per_shard)

    seq_arrays: List[np.ndarray] = []
    y_seq_list: List[np.ndarray] = []
    idx_seq_list: List[pd.DatetimeIndex] = []
    ticker_ids_seq: List[np.ndarray] = []

    decomp_arrays: List[np.ndarray] = []
    y_decomp_list: List[np.ndarray] = []
    idx_decomp_list: List[pd.DatetimeIndex] = []
    ticker_ids_decomp: List[np.ndarray] = []

    seq_dual_arrays: List[np.ndarray] = []
    decomp_dual_arrays: List[np.ndarray] = []
    y_dual_list: List[np.ndarray] = []
    idx_dual_list: List[pd.DatetimeIndex] = []
    ticker_ids_dual: List[np.ndarray] = []

    shard_tickers: List[str] = []
    shard_ticker_global_ids: List[int] = []
    global_tickers: List[str] = []

    feature_cols_seq: Optional[List[str]] = None
    feature_cols_decomp: Optional[List[str]] = None
    extra_payload: Optional[Dict[str, np.ndarray]] = None

    shards: List[Dict[str, object]] = []
    skipped: List[str] = []
    processed = 0

    current_seq_samples = 0
    current_decomp_samples = 0
    current_dual_samples = 0
    shard_idx = 0

    for csv_path in csv_paths:
        ticker = csv_path.stem
        try:
            (
                X_seq,
                y_seq,
                seq_index,
                X_decomp,
                y_decomp,
                decomp_index,
                cols_seq,
                cols_decomp,
                extra,
            ) = pdd.process_ticker(
                csv_path,
                lookback=None,
                horizon=args.horizon,
                stride=args.stride,
                liquidity_feature=liquidity_feature,
                decomp_windows=args.decomp_windows,
                decomp_scales=args.decomp_scales,
                decomp_normalization=decomp_normalization,
                decomp_include_ma=decomp_include_ma,
                lookback_seq=lookback_seq,
                lookback_decomp=lookback_decomp,
                include_synthetic_features=args.include_synthetic_features,
                disable_turnover_backfill=bool(args.disable_turnover_backfill),
                disable_no_split_history_fallback=bool(
                    args.disable_no_split_history_fallback
                ),
                turnover_fallback_backfill=not bool(
                    args.disable_turnover_fallback_backfill
                ),
                min_avg_dollar_volume_6m=float(args.min_avg_dollar_volume_3m),
                decomp_only=decomp_only,
                start_date=start_date,
                label_mode=label_mode,
            )
        except Exception as exc:
            skipped.append(f"{ticker}: {exc}")
            continue

        if not decomp_only:
            try:
                X_seq_dual, X_decomp_dual, y_seq_dual, y_decomp_dual, dual_index = (
                    pdd.align_dual_samples(
                        seq_index,
                        decomp_index,
                        X_seq,
                        X_decomp,
                        y_seq,
                        y_decomp,
                    )
                )
            except Exception as exc:
                skipped.append(f"{ticker}: {exc}")
                continue

            if y_seq_dual.shape != y_decomp_dual.shape:
                skipped.append(f"{ticker}: aligned label shapes do not match")
                continue
            if not np.allclose(y_seq_dual, y_decomp_dual, equal_nan=True):
                skipped.append(f"{ticker}: aligned labels differ")
                continue

            if feature_cols_seq is None:
                feature_cols_seq = cols_seq
            elif feature_cols_seq != cols_seq:
                raise ValueError(f"feature cols mismatch for {ticker}")

        if feature_cols_decomp is None:
            feature_cols_decomp = cols_decomp
        elif feature_cols_decomp != cols_decomp:
            raise ValueError(f"decomp feature cols mismatch for {ticker}")

        if extra_payload is None:
            extra_payload = extra

        n_decomp = len(decomp_index)
        n_seq = 0
        n_dual = 0
        max_next = n_decomp
        current_max = current_decomp_samples
        if not decomp_only:
            n_seq = len(seq_index)
            n_dual = len(dual_index)
            max_next = max(n_seq, n_decomp, n_dual)
            current_max = max(current_seq_samples, current_decomp_samples, current_dual_samples)
        tickers_full = max_tickers > 0 and len(shard_tickers) >= max_tickers

        if (current_max > 0 and current_max + max_next > max_samples) or tickers_full:
            shard_info = _flush_shard(
                output_dir,
                args.output_prefix,
                shard_idx,
                lookback_seq,
                lookback_decomp,
                effective_label_horizon,
                args.stride,
                feature_cols_seq or [],
                feature_cols_decomp,
                extra_payload,
                shard_tickers,
                seq_arrays,
                y_seq_list,
                idx_seq_list,
                ticker_ids_seq,
                decomp_arrays,
                y_decomp_list,
                idx_decomp_list,
                ticker_ids_decomp,
                seq_dual_arrays,
                decomp_dual_arrays,
                y_dual_list,
                idx_dual_list,
                ticker_ids_dual,
                chronological_assembly=args.chronological_assembly,
                compress=args.compress,
                decomp_only=decomp_only,
                shard_ticker_global_ids=shard_ticker_global_ids,
            )
            if shard_info:
                shards.append(shard_info)
                shard_idx += 1

            seq_arrays = []
            y_seq_list = []
            idx_seq_list = []
            ticker_ids_seq = []
            decomp_arrays = []
            y_decomp_list = []
            idx_decomp_list = []
            ticker_ids_decomp = []
            seq_dual_arrays = []
            decomp_dual_arrays = []
            y_dual_list = []
            idx_dual_list = []
            ticker_ids_dual = []
            shard_tickers = []
            shard_ticker_global_ids = []
            current_seq_samples = 0
            current_decomp_samples = 0
            current_dual_samples = 0

        if max_next > max_samples and current_max == 0:
            print(
                f"[warn] {ticker}: {max_next} samples exceed max shard size "
                f"({max_samples}); writing as oversized shard",
                flush=True,
            )

        ticker_id_local = len(shard_tickers)
        global_id = len(global_tickers)
        global_tickers.append(ticker)
        shard_tickers.append(ticker)
        shard_ticker_global_ids.append(global_id)

        if not decomp_only:
            seq_arrays.append(X_seq)
            y_seq_list.append(y_seq)
            idx_seq_list.append(seq_index)
            ticker_ids_seq.append(np.full(n_seq, ticker_id_local, dtype=np.int32))
            current_seq_samples += n_seq

        decomp_arrays.append(X_decomp)
        y_decomp_list.append(y_decomp)
        idx_decomp_list.append(decomp_index)
        ticker_ids_decomp.append(np.full(n_decomp, ticker_id_local, dtype=np.int32))
        current_decomp_samples += n_decomp

        if not decomp_only:
            seq_dual_arrays.append(X_seq_dual)
            decomp_dual_arrays.append(X_decomp_dual)
            y_dual_list.append(y_seq_dual)
            idx_dual_list.append(dual_index)
            ticker_ids_dual.append(np.full(n_dual, ticker_id_local, dtype=np.int32))
            current_dual_samples += n_dual

        processed += 1

    shard_info = _flush_shard(
        output_dir,
        args.output_prefix,
        shard_idx,
        lookback_seq,
        lookback_decomp,
        effective_label_horizon,
        args.stride,
        feature_cols_seq or [],
        feature_cols_decomp or [],
        extra_payload,
        shard_tickers,
        seq_arrays,
        y_seq_list,
        idx_seq_list,
        ticker_ids_seq,
        decomp_arrays,
        y_decomp_list,
        idx_decomp_list,
        ticker_ids_decomp,
        seq_dual_arrays,
        decomp_dual_arrays,
        y_dual_list,
        idx_dual_list,
        ticker_ids_dual,
        chronological_assembly=args.chronological_assembly,
        compress=args.compress,
        decomp_only=decomp_only,
        shard_ticker_global_ids=shard_ticker_global_ids,
    )
    if shard_info:
        shards.append(shard_info)

    skip_reason_counts: Dict[str, int] = {}
    for item in skipped:
        reason = str(item)
        if ": " in reason:
            _, reason = reason.split(": ", 1)
        reason = reason.strip() or "unknown_error"
        skip_reason_counts[reason] = int(skip_reason_counts.get(reason, 0)) + 1

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "output_prefix": args.output_prefix,
        "args": {
            "lookback": (
                None if lookback_override is None else int(lookback_override)
            ),
            "lookback_seq": int(lookback_seq),
            "lookback_decomp": int(lookback_decomp),
            "label_mode": label_mode,
            "horizon": int(args.horizon),
            "effective_label_horizon": int(effective_label_horizon),
            "stride": int(args.stride),
            "liquidity_feature": liquidity_feature,
            "decomp_windows": int(args.decomp_windows),
            "decomp_scales": None if args.decomp_scales is None else int(args.decomp_scales),
            "decomp_normalization": decomp_normalization,
            "final_candle_move_filter_enabled": bool(
                final_candle_move_filter_enabled
            ),
            "final_candle_move_threshold": float(final_candle_move_threshold),
            "final_candle_move_filter_scales": final_candle_move_scales,
            "global_ma_n_window": int(global_ma_n_window),
            "decomp_scale_aware_ma_feature_enabled": bool(
                decomp_scale_aware_ma_feature_enabled
            ),
            "decomp_include_ma": bool(decomp_include_ma),
            "include_synthetic_features": bool(args.include_synthetic_features),
            "disable_turnover_backfill": bool(args.disable_turnover_backfill),
            "disable_no_split_history_fallback": bool(
                args.disable_no_split_history_fallback
            ),
            "disable_turnover_fallback_backfill": bool(
                args.disable_turnover_fallback_backfill
            ),
            "chronological_assembly": bool(args.chronological_assembly),
            "max_samples_per_shard": int(max_samples),
            "max_tickers_per_shard": int(max_tickers),
            "compress": bool(args.compress),
            "decomp_only": bool(args.decomp_only),
            "start_date": (
                None if start_date is None else start_date.date().isoformat()
            ),
            "min_valid_volume": float(min_valid_volume),
            "min_avg_dollar_volume_3m": float(args.min_avg_dollar_volume_3m),
            "min_rows_for_one_sample": int(min_rows_for_one_sample),
        },
        "processed_tickers": int(processed),
        "skipped": list(skipped),
        "skip_reason_counts": dict(
            sorted(skip_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "global_tickers": list(global_tickers),
        "shards": shards,
    }

    manifest_path = output_dir / f"{args.output_prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"processed: {processed}, skipped: {len(skipped)}, shards: {len(shards)}")
    if skip_reason_counts:
        top_reasons = sorted(
            skip_reason_counts.items(), key=lambda kv: (-kv[1], kv[0])
        )[:10]
        details = "; ".join([f"{reason} x{count}" for reason, count in top_reasons])
        print(f"skip reasons (top): {details}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
