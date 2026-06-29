#!/usr/bin/env python3
"""Prepare daily candle datasets (sequence, decomposition, dual) from ticker CSVs.

Lean pipeline for daily OHLCV + turnover data:
- Loads each ticker CSV
- Engineers daily features
- Builds sequence and synthetic-decomposition tensors
- Aligns and combines into a single dataset across tickers
"""

from __future__ import annotations

import argparse
import heapq
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_LOOKBACK_SEQ = 30
DEFAULT_LOOKBACK_DECOMP = 60
DEFAULT_HORIZON = 5
DEFAULT_STRIDE = 1
DEFAULT_DECOMP_WINDOWS = 5

DEFAULT_INPUT_DIR = "tickers/"

LIQUIDITY_FEATURE_DEFAULT = "volume"

DECOMP_NORMALIZATION = "lookback"
DECOMP_NORMALIZATION_CHOICES = ("synthetic", "lookback", "none")
DECOMP_INCLUDE_MA_DEFAULT = True

ENTRY_OFFSET = 1
RET_PCT_INCLUDE_PRECEDING_GAP_DEFAULT = True
LABEL_MODE_RANGE_ATR = "range_atr"
LABEL_MODE_NEXT_DAY_CLOSE_RETURN = "next_day_close_return"
LABEL_MODE_CHOICES = (
    LABEL_MODE_RANGE_ATR,
    LABEL_MODE_NEXT_DAY_CLOSE_RETURN,
)

FINAL_CANDLE_MOVE_FILTER_ENABLED = False
FINAL_CANDLE_MOVE_THRESHOLD = 0.10
FINAL_CANDLE_MOVE_FILTER_SCALES = "0"
SCALE0_VOLUME_LT1_MAX_ALLOWED = 1

GLOBAL_MA_N_WINDOW = 5
DECOMP_SCALE_AWARE_MA_FEATURE_ENABLED = True

DAYS_IN_WEEK = 5.0
EPS = 1e-9
MIN_VALID_PRICE = 1e-5
MIN_VALID_VOLUME = 1
MIN_VALID_SHARES_OUTSTANDING = 0
MIN_AVG_DOLLAR_VOLUME_3m = 1

FEATURE_COLS_BASE = [
    "open",
    "high",
    "low",
    "close",
    "ma_n",
    "volume",
    "open_ratio",
    "high_ratio",
    "low_ratio",
    "close_ratio",
    "month",
    "week",
]

SYNTHETIC_PRICE_COLS = [
    "open_syn",
    "close_syn",
    "low_syn",
    "high_syn",
    "ma_n_syn",
]
SYNTHETIC_PRICE_INPUT_COLS = ["open", "high", "low", "close", "ma_n"]

LABEL_COLS = [
    "mfe",
    "mae",
    "y_raw",
    "ret_atr",
    "avg_ret_atr",
    "log_avg_ret_atr",
    "ret_pct",
]
RET_PCT_LABEL_INDEX = LABEL_COLS.index("ret_pct")

DECOMP_FEATURE_COLS_SYN = [
    "open_syn",
    "high_syn",
    "low_syn",
    "close_syn",
    "turnover",
]
DECOMP_FEATURE_COLS_SYN_WITH_MA = [
    "open_syn",
    "high_syn",
    "low_syn",
    "close_syn",
    "ma_n_syn",
    "turnover",
]
DECOMP_FEATURE_COLS_LOOKBACK = ["open", "high", "low", "close", "turnover"]
DECOMP_FEATURE_COLS_LOOKBACK_WITH_MA = [
    "open",
    "high",
    "low",
    "close",
    "ma_n",
    "turnover",
]
MOVING_AVERAGE_MODE_GLOBAL = "trailing_sma_min_periods_window"
MOVING_AVERAGE_MODE_SCALE_AWARE = (
    "trailing_sma_min_periods_window_scale_aware_image_days"
)


def resolve_liquidity_feature(value: Optional[str]) -> str:
    if value is None:
        return "turnover"
    value = str(value).strip().lower()
    if value in ("turnover", "volume"):
        return value
    raise ValueError("liquidity_feature must be 'turnover' or 'volume'")


def resolve_decomp_normalization(value: object) -> str:
    mode = str(value).strip().lower()
    if mode in DECOMP_NORMALIZATION_CHOICES:
        return mode
    allowed = ", ".join([f"'{x}'" for x in DECOMP_NORMALIZATION_CHOICES])
    raise ValueError(f"normalization must be one of: {allowed}")


def resolve_label_mode(value: object) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    alias_map = {
        "default": LABEL_MODE_RANGE_ATR,
        "range_atr": LABEL_MODE_RANGE_ATR,
        "next_day_close_return": LABEL_MODE_NEXT_DAY_CLOSE_RETURN,
        "1d_return": LABEL_MODE_NEXT_DAY_CLOSE_RETURN,
        "1d_returns": LABEL_MODE_NEXT_DAY_CLOSE_RETURN,
        "close_to_next_close": LABEL_MODE_NEXT_DAY_CLOSE_RETURN,
        "signal_close_to_next_close": LABEL_MODE_NEXT_DAY_CLOSE_RETURN,
    }
    resolved = alias_map.get(mode)
    if resolved is not None:
        return resolved
    allowed = ", ".join([f"'{x}'" for x in LABEL_MODE_CHOICES])
    raise ValueError(f"label_mode must be one of: {allowed}")


def resolve_effective_label_horizon(horizon: int, label_mode: object) -> int:
    mode = resolve_label_mode(label_mode)
    if mode == LABEL_MODE_NEXT_DAY_CLOSE_RETURN:
        return 1
    horizon_value = int(horizon)
    if horizon_value < 1:
        raise ValueError("horizon must be >= 1")
    return horizon_value


def resolve_min_rows_for_one_labeled_sample(
    lookback: int,
    horizon: int,
    label_mode: object,
    entry_offset: int = ENTRY_OFFSET,
) -> int:
    lookback_value = int(lookback)
    if lookback_value < 1:
        raise ValueError("lookback must be >= 1")
    effective_horizon = resolve_effective_label_horizon(horizon, label_mode)
    mode = resolve_label_mode(label_mode)
    if mode == LABEL_MODE_NEXT_DAY_CLOSE_RETURN:
        return int(lookback_value + effective_horizon)
    return int(lookback_value + effective_horizon + int(entry_offset))


def swap_liquidity_feature_cols(cols: Sequence[str], liquidity_feature: str) -> List[str]:
    liquidity_feature = resolve_liquidity_feature(liquidity_feature)
    replace_from = "turnover" if liquidity_feature == "volume" else "volume"
    replace_to = liquidity_feature
    mapped = [replace_to if col == replace_from else col for col in cols]
    seen = set()
    deduped = []
    for col in mapped:
        if col in seen:
            continue
        deduped.append(col)
        seen.add(col)
    return deduped


def normalize_stride(value: Optional[int]) -> int:
    if value is None:
        return DEFAULT_STRIDE
    if isinstance(value, bool):
        raise ValueError("stride must be an integer >= 1")
    stride_float = float(value)
    if not np.isfinite(stride_float):
        raise ValueError("stride must be an integer >= 1")
    stride_int = int(stride_float)
    if stride_int < 1 or stride_int != stride_float:
        raise ValueError("stride must be an integer >= 1")
    return stride_int


def _coerce_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= 1")
    value_float = float(value)
    if not np.isfinite(value_float):
        raise ValueError(f"{name} must be an integer >= 1")
    value_int = int(value_float)
    if value_int < 1 or value_int != value_float:
        raise ValueError(f"{name} must be an integer >= 1")
    return value_int


def resolve_branch_lookbacks(
    lookback: Optional[int],
    lookback_seq: Optional[int],
    lookback_decomp: Optional[int],
) -> Tuple[int, int]:
    default_seq = _coerce_positive_int("DEFAULT_LOOKBACK_SEQ", DEFAULT_LOOKBACK_SEQ)
    default_decomp = _coerce_positive_int(
        "DEFAULT_LOOKBACK_DECOMP", DEFAULT_LOOKBACK_DECOMP
    )
    base = (
        _coerce_positive_int("lookback", lookback)
        if lookback is not None
        else default_seq
    )
    seq = (
        _coerce_positive_int("lookback_seq", lookback_seq)
        if lookback_seq is not None
        else base
    )
    decomp = (
        _coerce_positive_int("lookback_decomp", lookback_decomp)
        if lookback_decomp is not None
        else (base if lookback is not None else default_decomp)
    )
    return int(seq), int(decomp)


def normalize_start_date(value: Optional[object]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        dt = pd.to_datetime(text, errors="raise")
    except Exception as exc:  # pragma: no cover - defensive parse guard
        raise ValueError(
            f"start_date must be a valid date string (e.g. YYYY-MM-DD); got: {value!r}"
        ) from exc
    ts = pd.Timestamp(dt)
    if ts.tz is not None:
        ts = ts.tz_convert(None)
    return ts.normalize()


def coerce_datetime(values: Sequence[object] | pd.Index | pd.Series) -> pd.DatetimeIndex:
    # Prefer pandas mixed parser when available to avoid noisy inference warnings.
    try:
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    except TypeError:
        parsed = pd.to_datetime(values, errors="coerce")
    return pd.DatetimeIndex(parsed)


def load_daily_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    if df.empty:
        return df

    # Normalize index to datetime
    idx = coerce_datetime(df.index)
    if idx.isna().all():
        date_col = None
        for col in df.columns:
            if str(col).strip().lower() in ("date", "time", "datetime"):
                date_col = col
                break
        if date_col is None:
            raise ValueError(f"could not find date column in {csv_path}")
        idx = coerce_datetime(df[date_col])
        df = df.drop(columns=[date_col])
    df.index = idx
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Normalize column names
    def canon(col: str) -> str:
        key = str(col).strip().lower().replace(" ", "_")
        key = key.replace("-", "_")
        if key == "adj_close" or key == "adjclose":
            return "adj_close"
        if key in ("sharesoutstanding", "shares_outstanding"):
            return "shares_outstanding"
        return key

    df = df.rename(columns={col: canon(col) for col in df.columns})
    if "adj_close" in df.columns and "close" in df.columns:
        df = df.drop(columns=["adj_close"])
    return df


def extract_splits_from_df(df: pd.DataFrame) -> pd.Series:
    split_cols = ["stock_splits", "splits", "stock_split", "split"]
    for col in split_cols:
        if col in df.columns:
            splits = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            splits = splits[splits != 0]
            if splits.empty:
                return pd.Series(dtype=float)
            splits.index = pd.to_datetime(splits.index, errors="coerce")
            splits = splits[~splits.index.isna()]
            splits = splits.sort_index()
            return splits.astype(float)
    return pd.Series(dtype=float)


def fetch_splits_from_yfinance(ticker: str) -> pd.Series:
    try:
        import yfinance as yf  # optional dependency
    except Exception:
        return pd.Series(dtype=float)
    try:
        splits = yf.Ticker(ticker).splits
    except Exception:
        return pd.Series(dtype=float)
    if splits is None or len(splits) == 0:
        return pd.Series(dtype=float)
    splits = splits.astype(float)
    splits.index = pd.to_datetime(splits.index, errors="coerce").tz_localize(None)
    splits = splits[~splits.index.isna()]
    splits = splits.sort_index()
    return splits


def build_split_factor(index: pd.DatetimeIndex, splits: pd.Series) -> pd.Series:
    idx = pd.DatetimeIndex(index).normalize()
    factor = pd.Series(1.0, index=idx)
    if splits is None or splits.empty:
        return factor
    splits = splits.copy()
    splits.index = pd.to_datetime(splits.index, errors="coerce").normalize()
    splits = splits[~splits.index.isna()]
    for date, ratio in splits.items():
        if not np.isfinite(ratio) or ratio <= 0:
            continue
        if date in factor.index:
            factor.loc[date] *= float(ratio)
            continue
        pos = idx.get_indexer([date], method="bfill")[0]
        if pos != -1:
            factor.iloc[pos] *= float(ratio)
    return factor


def backfill_shares_outstanding(
    df: pd.DataFrame,
    ticker: str,
    disable_backfill: bool,
    disable_no_split_history_fallback: bool = False,
) -> pd.DataFrame:
    if disable_backfill:
        return df
    if "shares_outstanding" not in df.columns:
        return df
    shares = pd.to_numeric(df["shares_outstanding"], errors="coerce")
    shares = shares.where(shares > 0)
    if shares.isna().all():
        return df
    missing_mask = shares.isna()
    if not missing_mask.any():
        return df
    first_valid = shares.first_valid_index()
    if first_valid is None:
        return df

    splits = extract_splits_from_df(df)
    if splits.empty:
        splits = fetch_splits_from_yfinance(ticker)
    no_split_history = splits.empty
    if no_split_history and disable_no_split_history_fallback:
        print(f"[warn] {ticker}: no split history; skipping flat shares backfill")
        return df

    split_factor = build_split_factor(df.index, splits)
    if no_split_history:
        print(f"[warn] {ticker}: no split history; applying flat shares backfill")
    base_factor = split_factor.loc[pd.Timestamp(first_valid).normalize()]
    if not np.isfinite(base_factor) or base_factor <= 0:
        print(f"[warn] {ticker}: invalid split factor; applying flat shares backfill")
        split_factor = pd.Series(
            1.0, index=pd.DatetimeIndex(df.index).normalize(), dtype=float
        )
        base_factor = 1.0
    scale = split_factor / base_factor
    filled = shares.copy()
    filled.loc[missing_mask] = float(shares.loc[first_valid]) * scale.loc[missing_mask]
    df = df.copy()
    df["shares_outstanding"] = filled
    return df


def filter_invalid_daily_rows(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if df.empty:
        return df, {}

    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"missing required column '{col}'")

    cleaned = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    if "shares_outstanding" in cleaned.columns:
        cleaned["shares_outstanding"] = pd.to_numeric(
            cleaned["shares_outstanding"], errors="coerce"
        )

    invalid_price = pd.Series(False, index=cleaned.index)
    for col in ("open", "high", "low", "close"):
        values = cleaned[col]
        invalid_price = invalid_price | (~np.isfinite(values)) | (values < MIN_VALID_PRICE)

    volume = cleaned["volume"]
    invalid_volume = (~np.isfinite(volume)) | (volume < MIN_VALID_VOLUME)

    invalid_shares = pd.Series(False, index=cleaned.index)
    if "shares_outstanding" in cleaned.columns:
        shares = cleaned["shares_outstanding"].to_numpy(dtype=float, copy=False)
        shares_finite = np.isfinite(shares)
        # Missing shares_outstanding is allowed here; split-based backfill can fill it upstream.
        invalid_shares = pd.Series(
            shares_finite & (shares <= MIN_VALID_SHARES_OUTSTANDING),
            index=cleaned.index,
        )

    invalid_any = invalid_price | invalid_volume | invalid_shares
    if not invalid_any.any():
        return cleaned, {}

    drop_counts = {
        "invalid_price_rows": int(invalid_price.sum()),
        "invalid_volume_rows": int(invalid_volume.sum()),
    }
    if "shares_outstanding" in cleaned.columns:
        drop_counts["invalid_shares_outstanding_rows"] = int(invalid_shares.sum())
    drop_counts["invalid_rows_total"] = int(invalid_any.sum())

    cleaned = cleaned.loc[~invalid_any].copy()
    return cleaned, drop_counts


def filter_low_dollar_volume_half_year_periods(
    df: pd.DataFrame,
    min_avg_dollar_volume_6m: float,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if df.empty:
        return df, {}
    threshold = float(min_avg_dollar_volume_6m)
    if threshold <= 0:
        return df, {}
    if "volume" not in df.columns or "close" not in df.columns:
        raise ValueError("missing required columns for dollar-volume filter: volume/close")

    volume = pd.to_numeric(df["volume"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    dollar_volume = volume * close

    idx = pd.DatetimeIndex(df.index)
    quarter = (((idx.month - 1) // 3) + 1).astype(np.int8)
    period_key = (
        pd.Series(idx.year.astype(np.int32), index=df.index).astype(str)
        + "Q"
        + pd.Series(quarter, index=df.index).astype(str)
    )

    period_means = dollar_volume.groupby(period_key).mean()
    low_periods = period_means[period_means < threshold]
    if low_periods.empty:
        return df, {}

    drop_mask = period_key.isin(low_periods.index)
    dropped_rows = int(drop_mask.sum())
    if dropped_rows == 0:
        return df, {}

    filtered = df.loc[~drop_mask].copy()
    stats = {
        "liquidity_periods_dropped": int(low_periods.shape[0]),
        "liquidity_rows_dropped": dropped_rows,
    }
    return filtered, stats


def compute_wilder_atr(df: pd.DataFrame, period: int = 30) -> pd.Series:
    if period < 1:
        raise ValueError("period must be >= 1")
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    atr_values = atr.to_numpy(dtype=float, copy=True)
    tr_values = tr.to_numpy(dtype=float)
    if np.isnan(atr_values).all():
        return atr
    start = np.where(~np.isnan(atr_values))[0][0]
    for i in range(start + 1, len(tr_values)):
        if np.isnan(tr_values[i]) or np.isnan(atr_values[i - 1]):
            continue
        atr_values[i] = ((atr_values[i - 1] * (period - 1)) + tr_values[i]) / period
    return pd.Series(atr_values, index=df.index)


def compute_ma_n_window(lookback: int) -> int:
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    return max(1, int(GLOBAL_MA_N_WINDOW))


def add_ma_n_feature(features: pd.DataFrame, lookback: int) -> pd.DataFrame:
    window = compute_ma_n_window(lookback)
    features = features.copy()
    features["ma_n"] = features["close"].rolling(window=window, min_periods=window).mean()
    return features


def compute_labels_daily(
    df: pd.DataFrame,
    horizon: int,
    atr_period: int = 30,
    edge_lambda: float = 1.0,
    entry_offset: int = ENTRY_OFFSET,
    # Backward-compatible parameter name retained; now controls all return labels.
    include_preceding_gap_in_ret_pct: bool = RET_PCT_INCLUDE_PRECEDING_GAP_DEFAULT,
    label_mode: str = LABEL_MODE_RANGE_ATR,
) -> pd.DataFrame:
    label_mode_resolved = resolve_label_mode(label_mode)
    horizon = resolve_effective_label_horizon(horizon, label_mode_resolved)
    if entry_offset < 0:
        raise ValueError("entry_offset must be >= 0")

    labels = pd.DataFrame(index=df.index, columns=LABEL_COLS, dtype=float)
    atr = compute_wilder_atr(df, period=atr_period)
    atr_safe = atr.where(atr > 0.0)
    if label_mode_resolved == LABEL_MODE_NEXT_DAY_CLOSE_RETURN:
        signal_close = df["close"]
        future_high = df["high"].shift(-1)
        future_low = df["low"].shift(-1)
        future_close = df["close"].shift(-1)
        return_base_price = signal_close
        return_base_price_safe = return_base_price.where(np.abs(return_base_price) > EPS)
        mfe_points = future_high - signal_close
        mae_points = signal_close - future_low
        mfe_atr = mfe_points / atr_safe
        mae_atr = mae_points / atr_safe
        ret_atr = (future_close - signal_close) / atr_safe
        avg_ret_atr = ret_atr
        atr_return = atr_safe / return_base_price_safe
        log_avg_ret_atr = np.log(future_close / return_base_price_safe) / atr_return
        ret_pct = (future_close - signal_close) / return_base_price_safe
        y_raw = ret_pct
    else:
        label_start_offset = entry_offset + (1 if entry_offset == 0 else 0)
        if entry_offset == 0:
            entry_price = df["close"].shift(-entry_offset)
        else:
            entry_price = df["open"].shift(-entry_offset)

        future_high = (
            df["high"]
            .shift(-label_start_offset)
            .rolling(window=horizon, min_periods=horizon)
            .max()
            .shift(-(horizon - 1))
        )
        future_low = (
            df["low"]
            .shift(-label_start_offset)
            .rolling(window=horizon, min_periods=horizon)
            .min()
            .shift(-(horizon - 1))
        )
        future_close = (
            df["close"]
            .shift(-label_start_offset)
            .rolling(window=horizon, min_periods=horizon)
            .apply(lambda x: x[-1], raw=True)
            .shift(-(horizon - 1))
        )
        future_mean = (
            df["close"]
            .shift(-label_start_offset)
            .rolling(window=horizon, min_periods=horizon)
            .mean()
            .shift(-(horizon - 1))
        )

        mfe_points = future_high - entry_price
        mae_points = entry_price - future_low
        mfe_atr = mfe_points / atr_safe
        mae_atr = mae_points / atr_safe
        y_raw = mfe_atr - (edge_lambda * mae_atr)
        return_base_price = entry_price
        if bool(include_preceding_gap_in_ret_pct) and int(entry_offset) > 0:
            # Include the gap immediately before horizon-start entry
            # (e.g., close[t] -> open[t+1] for entry_offset=1).
            return_base_price = df["close"].shift(-(int(entry_offset) - 1))
        return_base_price_safe = return_base_price.where(np.abs(return_base_price) > EPS)
        ret_atr = (future_close - return_base_price) / atr_safe
        avg_ret_atr = (future_mean - return_base_price) / atr_safe
        atr_return = atr_safe / return_base_price_safe
        log_avg_ret_atr = np.log(future_mean / return_base_price_safe) / atr_return
        ret_pct = (future_close - return_base_price) / return_base_price_safe

    labels["mfe"] = mfe_atr
    labels["mae"] = mae_atr
    labels["y_raw"] = y_raw
    labels["ret_atr"] = ret_atr
    labels["avg_ret_atr"] = avg_ret_atr
    labels["log_avg_ret_atr"] = log_avg_ret_atr
    labels["ret_pct"] = ret_pct
    return labels


def build_labeled_sample_valid_mask(
    contiguous_mask: np.ndarray,
    label_values: np.ndarray,
) -> np.ndarray:
    if label_values.ndim != 2 or label_values.shape[1] != len(LABEL_COLS):
        raise ValueError(
            "unexpected label_values shape for sample filtering: "
            f"{label_values.shape}"
        )
    valid_mask = contiguous_mask & (~np.isnan(label_values).any(axis=1))
    valid_mask &= label_values[:, RET_PCT_LABEL_INDEX] != 0.0
    return valid_mask


def engineer_features_daily(
    df: pd.DataFrame,
    lookback: int,
    liquidity_feature: str,
    turnover_fallback_backfill: bool = True,
) -> pd.DataFrame:
    liquidity_feature = resolve_liquidity_feature(liquidity_feature)
    features = df.copy()

    for col in ("open", "high", "low", "close", "volume"):
        if col not in features.columns:
            raise ValueError(f"missing required column '{col}'")
        features[col] = pd.to_numeric(features[col], errors="coerce")

    if liquidity_feature == "turnover":
        if "turnover" not in features.columns:
            features["turnover"] = np.nan
        if turnover_fallback_backfill and "shares_outstanding" in features.columns:
            shares = pd.to_numeric(features["shares_outstanding"], errors="coerce")
            shares = shares.where(shares > 0)
            turnover_fallback = features["volume"] / shares
            features["turnover"] = features["turnover"].where(
                ~features["turnover"].isna(), turnover_fallback
            )
        else:
            features["turnover"] = features["turnover"]
    else:
        features["volume"] = pd.to_numeric(features["volume"], errors="coerce")

    prev_close = features["close"].shift(1)
    prev_close_safe = prev_close + EPS
    features["open_ratio"] = (features["open"] - prev_close) / prev_close_safe
    features["high_ratio"] = (features["high"] - prev_close) / prev_close_safe
    features["low_ratio"] = (features["low"] - prev_close) / prev_close_safe
    features["close_ratio"] = (features["close"] - prev_close) / prev_close_safe

    idx = pd.DatetimeIndex(features.index)
    features["month"] = idx.month / 12.0
    week_in_month = ((idx.day - 1) // 7) + 1
    features["week"] = week_in_month / 5.0
    features["dow_frac"] = idx.dayofweek / DAYS_IN_WEEK

    features = add_ma_n_feature(features, lookback)
    return features


def apply_ohlc_lookback_norm(
    X: np.ndarray,
    feature_cols: Sequence[str],
    enabled: bool = True,
) -> np.ndarray:
    if not enabled:
        return X
    cols = list(feature_cols)
    if "close" not in cols:
        return X
    target_names = ["open", "high", "low", "close", "ma_n"]
    indices = [cols.index(name) for name in target_names if name in cols]
    if not indices:
        return X
    close_idx = cols.index("close")
    base = X[:, close_idx, -1]
    base = np.where(np.abs(base) < EPS, 1.0, base)
    X = X.copy()
    for idx in indices:
        X[:, idx, :] = X[:, idx, :] / base[:, None]
    return X


def build_synthetic_price_series(
    X: np.ndarray,
    feature_cols: Sequence[str],
) -> np.ndarray:
    cols = list(feature_cols)
    missing = [name for name in SYNTHETIC_PRICE_INPUT_COLS if name not in cols]
    if missing:
        raise ValueError(
            "synthetic price series requires features "
            f"{SYNTHETIC_PRICE_INPUT_COLS} (missing {missing})"
        )
    idx = {name: cols.index(name) for name in SYNTHETIC_PRICE_INPUT_COLS}
    close = X[:, idx["close"], :]
    base = close[:, 0]
    base = np.where(np.abs(base) < EPS, 1.0, base)
    scale = (1.0 / base)[:, None]
    open_syn = X[:, idx["open"], :] * scale
    close_syn = close * scale
    low_syn = X[:, idx["low"], :] * scale
    high_syn = X[:, idx["high"], :] * scale
    ma_n_syn = X[:, idx["ma_n"], :] * scale
    return np.stack(
        [open_syn, close_syn, low_syn, high_syn, ma_n_syn], axis=1
    )


def normalize_window_view(
    windows_view: np.ndarray, lookback: int, feature_count: int
) -> np.ndarray:
    if windows_view.ndim != 3:
        raise ValueError("window view must be 3D")
    if windows_view.shape[1] == lookback and windows_view.shape[2] == feature_count:
        return windows_view
    if windows_view.shape[1] == feature_count and windows_view.shape[2] == lookback:
        return np.swapaxes(windows_view, 1, 2)
    raise ValueError(
        f"unexpected window view shape {windows_view.shape} "
        f"(lookback={lookback}, features={feature_count})"
    )


def contiguous_window_mask_from_source_rows(
    source_rows: np.ndarray,
    lookback: int,
) -> np.ndarray:
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    src = np.asarray(source_rows, dtype=np.int64).reshape(-1)
    if src.shape[0] < lookback:
        return np.empty((0,), dtype=bool)
    if lookback == 1:
        return np.ones(src.shape[0], dtype=bool)
    src_windows = np.lib.stride_tricks.sliding_window_view(src, window_shape=lookback)
    return np.all(np.diff(src_windows, axis=1) == 1, axis=1)


def resolve_source_row_series(
    features_index: pd.Index,
    source_row_ids: Optional[pd.Series],
) -> pd.Series:
    if source_row_ids is None:
        raw = pd.Series(np.arange(len(features_index), dtype=np.int64), index=features_index)
    elif isinstance(source_row_ids, pd.Series):
        raw = source_row_ids.reindex(features_index)
    else:
        arr = np.asarray(source_row_ids).reshape(-1)
        if arr.shape[0] != len(features_index):
            raise ValueError(
                "source_row_ids length mismatch: "
                f"got {arr.shape[0]} rows for {len(features_index)} feature rows"
            )
        raw = pd.Series(arr, index=features_index)
    return pd.to_numeric(raw, errors="coerce")


def resolve_input_source_row_ids(df: pd.DataFrame) -> pd.Series:
    """Resolve per-row source ids from input CSV columns when available.

    Priority:
    1) `_source_row_id_orig` (recommended for pre-filtered datasets)
    2) `_source_row_id`
    3) Fallback to in-file positional row index.
    """
    for col in ("_source_row_id_orig", "_source_row_id"):
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if bool(values.notna().all()):
            return pd.Series(
                values.to_numpy(dtype=np.int64, copy=False),
                index=df.index,
            )
    return pd.Series(np.arange(df.shape[0], dtype=np.int64), index=df.index)


def build_sequence_dataset(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    lookback: int,
    horizon: int,
    stride: int,
    liquidity_feature: str,
    include_synthetic_features: bool,
    production_no_labels: bool = False,
    source_row_ids: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str], List[str]]:
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    base_feature_cols = swap_liquidity_feature_cols(
        FEATURE_COLS_BASE, liquidity_feature
    )
    syn_input_cols: List[str] = (
        list(SYNTHETIC_PRICE_INPUT_COLS) if include_synthetic_features else []
    )
    required_cols = list(base_feature_cols)
    for col in syn_input_cols:
        if col not in required_cols:
            required_cols.append(col)
    missing = [col for col in required_cols if col not in features.columns]
    if missing:
        raise ValueError("missing expected feature columns: " + ", ".join(missing))

    features_clean = features[required_cols].dropna()
    source_series = resolve_source_row_series(features.index, source_row_ids)
    source_clean = source_series.reindex(features_clean.index)
    source_valid = ~source_clean.isna()
    if not bool(source_valid.all()):
        features_clean = features_clean.loc[source_valid[source_valid].index]
        source_clean = source_clean.loc[source_valid[source_valid].index]
    if features_clean.shape[0] < lookback:
        raise ValueError("lookback exceeds available rows")

    source_values = source_clean.to_numpy(dtype=np.int64, copy=False)
    feature_values = features_clean[base_feature_cols].to_numpy(dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(
        feature_values, window_shape=lookback, axis=0
    )
    windows = normalize_window_view(windows, lookback, feature_values.shape[1])
    X = np.swapaxes(windows, 1, 2)
    sample_index = features_clean.index[lookback - 1 :]
    contiguous_mask = contiguous_window_mask_from_source_rows(source_values, lookback)
    if contiguous_mask.shape[0] != sample_index.shape[0]:
        raise ValueError("unexpected contiguous-window mask shape mismatch")

    if production_no_labels:
        label_values = np.full(
            (sample_index.shape[0], len(LABEL_COLS)),
            np.nan,
            dtype=float,
        )
        valid_mask = contiguous_mask.copy()
    else:
        labels_clean = labels[LABEL_COLS]
        labels_aligned = labels_clean.reindex(sample_index)
        label_values = labels_aligned.to_numpy(dtype=float)
        valid_mask = build_labeled_sample_valid_mask(contiguous_mask, label_values)

    X = X[valid_mask]
    y = label_values[valid_mask]
    sample_index = pd.DatetimeIndex(sample_index[valid_mask])

    synthetic_series = None
    if include_synthetic_features:
        syn_values = features_clean[syn_input_cols].to_numpy(dtype=float)
        syn_windows = np.lib.stride_tricks.sliding_window_view(
            syn_values, window_shape=lookback, axis=0
        )
        syn_windows = normalize_window_view(
            syn_windows, lookback, syn_values.shape[1]
        )
        X_syn = np.swapaxes(syn_windows, 1, 2)
        X_syn = X_syn[valid_mask]
        synthetic_series = build_synthetic_price_series(X_syn, syn_input_cols)

    if stride > 1:
        take = np.arange(0, sample_index.shape[0], stride)
        X = X[take]
        y = y[take]
        sample_index = sample_index[take]
        if synthetic_series is not None:
            synthetic_series = synthetic_series[take]

    X = apply_ohlc_lookback_norm(X, base_feature_cols, enabled=True)
    feature_cols = list(base_feature_cols)
    if synthetic_series is not None:
        X = np.concatenate([X, synthetic_series], axis=1)
        feature_cols = feature_cols + SYNTHETIC_PRICE_COLS
    return X, y, sample_index, feature_cols, LABEL_COLS


def compute_decomposition_window_sizes(
    lookback: int,
    windows: int,
    scales: Optional[int] = None,
) -> List[int]:
    if windows < 1:
        raise ValueError("windows must be >= 1")
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if scales is None:
        if windows < 2:
            scales = 1
        else:
            scales = int(math.ceil(math.log(float(lookback), float(windows))))
            scales = max(1, scales)
    else:
        scales = int(scales)
        if scales < 1:
            raise ValueError("scales must be >= 1")
    max_window = lookback // windows
    if max_window < 1:
        raise ValueError("lookback must be >= windows")
    sizes = []
    for i in range(scales):
        size = windows**i
        if size > max_window:
            size = max_window
        sizes.append(int(size))
    return sizes


def compute_scale_aware_ma_periods(
    window_sizes: Sequence[int],
    windows: int,
) -> List[int]:
    if windows < 1:
        raise ValueError("windows must be >= 1")
    out: List[int] = []
    for window_size in window_sizes:
        ws = int(window_size)
        if ws < 1:
            raise ValueError("window_size must be >= 1")
        out.append(int(ws * int(windows)))
    return out


def resolve_final_candle_move_filter_scales(value: object) -> str:
    mode = str(value).strip().lower()
    if mode in ("0", "all"):
        return mode
    raise ValueError(
        "FINAL_CANDLE_MOVE_FILTER_SCALES must be one of: '0', 'all'; "
        f"got {value!r}"
    )


def compute_final_candle_offsets_and_lags(
    lookback: int,
    windows: int,
    scales: Optional[int] = None,
    scale_selection: object = FINAL_CANDLE_MOVE_FILTER_SCALES,
) -> Tuple[List[int], List[int], List[int]]:
    mode = resolve_final_candle_move_filter_scales(scale_selection)
    window_sizes_all = compute_decomposition_window_sizes(
        lookback=lookback,
        windows=windows,
        scales=scales,
    )
    if mode == "0":
        window_sizes = window_sizes_all[:1]
    else:
        window_sizes = window_sizes_all
    final_offsets: set[int] = set()
    for window_size in window_sizes:
        take = int(window_size) * int(windows)
        start_base = int(lookback) - int(take)
        for j in range(int(windows)):
            start = int(start_base + j * int(window_size))
            final_offsets.add(int(start + int(window_size) - 1))
    offsets_sorted = sorted(int(v) for v in final_offsets)
    lags_sorted = sorted(int(lookback - 1 - v) for v in offsets_sorted)
    return [int(v) for v in window_sizes], offsets_sorted, lags_sorted


def compute_final_candle_large_move_drop_mask(
    index: pd.DatetimeIndex,
    close: pd.Series,
    lookback: int,
    decomp_windows: int,
    decomp_scales: Optional[int],
    threshold: float,
    scale_selection: object = FINAL_CANDLE_MOVE_FILTER_SCALES,
) -> Tuple[pd.Series, Dict[str, object]]:
    idx = pd.DatetimeIndex(index).tz_localize(None)
    close_num = pd.to_numeric(close, errors="coerce")
    close_num = close_num.reindex(idx)

    daily_ret = close_num.pct_change()
    event = (daily_ret > float(threshold)) & np.isfinite(daily_ret)
    event_values = event.to_numpy(dtype=bool)

    window_sizes, final_offsets, lags = compute_final_candle_offsets_and_lags(
        lookback=int(lookback),
        windows=int(decomp_windows),
        scales=decomp_scales,
        scale_selection=scale_selection,
    )

    drop_values = np.zeros((event_values.shape[0],), dtype=bool)
    for lag in lags:
        lag_i = int(lag)
        if lag_i == 0:
            drop_values |= event_values
        else:
            drop_values[lag_i:] |= event_values[:-lag_i]

    drop_series = pd.Series(drop_values, index=idx)
    mode = resolve_final_candle_move_filter_scales(scale_selection)
    if mode == "0":
        scale_indices = [0] if len(window_sizes) > 0 else []
    else:
        scale_indices = list(range(len(window_sizes)))
    meta: Dict[str, object] = {
        "scale_selection": str(mode),
        "scale_indices": [int(x) for x in scale_indices],
        "window_sizes": [int(x) for x in window_sizes],
        "final_offsets": [int(x) for x in final_offsets],
        "lags": [int(x) for x in lags],
        "raw_rows_gt_threshold": int(event.sum()),
        "sample_rows_marked": int(drop_series.sum()),
    }
    return drop_series, meta


def build_sample_keep_mask_from_drop_dates(
    sample_index: pd.DatetimeIndex,
    drop_dates: pd.Series,
) -> np.ndarray:
    if sample_index.size <= 0:
        return np.zeros((0,), dtype=bool)
    if drop_dates.empty:
        return np.ones((sample_index.shape[0],), dtype=bool)
    idx = pd.DatetimeIndex(sample_index).tz_localize(None)
    mapped_drop = drop_dates.reindex(idx, fill_value=False).to_numpy(dtype=bool)
    return ~mapped_drop


def build_scale0_volume_lt1_keep_mask(
    X_decomp: np.ndarray,
    feature_cols_decomp: Sequence[str],
    max_lt1: int = SCALE0_VOLUME_LT1_MAX_ALLOWED,
) -> Tuple[np.ndarray, bool]:
    X = np.asarray(X_decomp)
    if X.ndim != 4:
        raise ValueError(
            f"expected decomp tensor shape (samples, scales, features, windows), got {X.shape}"
        )
    n = int(X.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=bool), False
    cols = list(feature_cols_decomp)
    if "volume" not in cols:
        return np.ones((n,), dtype=bool), False
    volume_idx = int(cols.index("volume"))
    if volume_idx < 0 or volume_idx >= int(X.shape[2]):
        raise ValueError(
            "volume feature index out of bounds for decomp tensor: "
            f"volume_idx={volume_idx} features={int(X.shape[2])}"
        )
    max_lt1_i = int(max_lt1)
    if max_lt1_i < 0:
        raise ValueError("max_lt1 must be >= 0")
    scale0_volume = np.asarray(X[:, 0, volume_idx, :], dtype=np.float64)
    low_volume_mask = (~np.isfinite(scale0_volume)) | (scale0_volume < 1.0)
    low_volume_count = np.sum(low_volume_mask, axis=1, dtype=np.int64)
    return np.asarray(low_volume_count <= max_lt1_i, dtype=bool), True


def build_synthetic_windows(
    windows_view: np.ndarray,
    feature_order: Sequence[str],
) -> np.ndarray:
    idx = {name: i for i, name in enumerate(feature_order)}
    open_vals = windows_view[:, :, idx["open"]]
    high_vals = windows_view[:, :, idx["high"]]
    low_vals = windows_view[:, :, idx["low"]]
    close_vals = windows_view[:, :, idx["close"]]
    liquidity_vals = windows_view[:, :, idx["liquidity"]]
    ma_vals = windows_view[:, :, idx["ma_n"]] if "ma_n" in idx else None

    base = close_vals[:, 0]
    base = np.where(np.abs(base) < EPS, 1.0, base)
    scale = (1.0 / base)[:, None]

    open_syn = open_vals * scale
    high_syn = high_vals * scale
    low_syn = low_vals * scale
    close_syn = close_vals * scale
    if ma_vals is None:
        return np.stack(
            [open_syn, high_syn, low_syn, close_syn, liquidity_vals], axis=2
        )
    ma_syn = ma_vals * scale
    return np.stack(
        [open_syn, high_syn, low_syn, close_syn, ma_syn, liquidity_vals], axis=2
    )


def compute_trailing_sma_strict(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError("window must be >= 1")
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    if arr.shape[0] < window:
        return out
    csum = np.cumsum(arr, dtype=np.float64)
    prev = np.concatenate(([0.0], csum[:-window]))
    out[window - 1 :] = (csum[window - 1 :] - prev) / float(window)
    return out


def compute_scale_aware_ma_values(
    close_values: np.ndarray,
    sample_end_positions: np.ndarray,
    window_size: int,
    window_count: int,
) -> Tuple[np.ndarray, int]:
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if window_count < 1:
        raise ValueError("window_count must be >= 1")
    period = int(window_size) * int(window_count)
    close_arr = np.asarray(close_values, dtype=np.float64).reshape(-1)
    sample_end = np.asarray(sample_end_positions, dtype=np.int64).reshape(-1)
    if sample_end.shape[0] == 0:
        return np.empty((0, int(window_count)), dtype=np.float64), int(period)
    offsets = (
        -int(period)
        + np.arange(1, int(window_count) + 1, dtype=np.int64) * int(window_size)
    )
    positions = sample_end[:, None] + offsets[None, :]
    if int(np.min(positions)) < 0 or int(np.max(positions)) >= int(close_arr.shape[0]):
        raise ValueError("scale-aware MA endpoint positions exceed source close array bounds")
    ma_series = compute_trailing_sma_strict(close_arr, window=int(period))
    return ma_series[positions], int(period)


def build_ohlcv_submap(
    windows_view: np.ndarray,
    lookback: int,
    window_size: int,
    window_count: int,
    include_ma: bool = False,
    ma_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if window_count < 1:
        raise ValueError("window_count must be >= 1")
    take = window_size * window_count
    if take > lookback:
        raise ValueError("window_size * window_count exceeds lookback")

    sample_count = windows_view.shape[0]
    feature_count = int(windows_view.shape[2])
    if bool(include_ma) and ma_values is None:
        expected_feature_count = 6
        if feature_count != expected_feature_count:
            raise ValueError(
                "unexpected windows_view feature count for decomposition submap: "
                f"got={feature_count} expected={expected_feature_count}"
            )
    else:
        if feature_count not in (5, 6):
            raise ValueError(
                "unexpected windows_view feature count for decomposition submap: "
                f"got={feature_count} expected one of (5, 6)"
            )
    open_idx, high_idx, low_idx, close_idx = 0, 1, 2, 3
    ma_idx = 4 if (bool(include_ma) and ma_values is None) else None
    liq_idx = 5 if feature_count == 6 else 4

    start_base = lookback - take
    dtype = windows_view.dtype
    open_vals = np.empty((sample_count, window_count), dtype=dtype)
    high_vals = np.empty((sample_count, window_count), dtype=dtype)
    low_vals = np.empty((sample_count, window_count), dtype=dtype)
    close_vals = np.empty((sample_count, window_count), dtype=dtype)
    ma_n_vals = None
    if bool(include_ma):
        if ma_values is None:
            ma_n_vals = np.empty((sample_count, window_count), dtype=dtype)
        else:
            ma_arr = np.asarray(ma_values, dtype=dtype)
            if ma_arr.shape != (sample_count, window_count):
                raise ValueError(
                    "ma_values shape mismatch for decomposition submap: "
                    f"got={ma_arr.shape} expected={(sample_count, window_count)}"
                )
            ma_n_vals = ma_arr
    liquidity_vals = np.empty((sample_count, window_count), dtype=dtype)

    for j in range(window_count):
        start = start_base + j * window_size
        end = start + window_size
        window = windows_view[:, start:end, :]
        open_vals[:, j] = window[:, 0, open_idx]
        high_vals[:, j] = window[:, :, high_idx].max(axis=1)
        low_vals[:, j] = window[:, :, low_idx].min(axis=1)
        close_vals[:, j] = window[:, -1, close_idx]
        if ma_idx is not None and ma_n_vals is not None:
            ma_n_vals[:, j] = window[:, -1, ma_idx]
        liquidity_vals[:, j] = window[:, :, liq_idx].sum(axis=1)
    if ma_n_vals is None:
        return np.stack(
            [open_vals, high_vals, low_vals, close_vals, liquidity_vals], axis=1
        )
    return np.stack(
        [open_vals, high_vals, low_vals, close_vals, ma_n_vals, liquidity_vals], axis=1
    )


def build_decomposition_dataset(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    lookback: int,
    horizon: int,
    stride: int,
    liquidity_feature: str,
    windows: int,
    scales: Optional[int],
    normalization: str,
    include_ma: bool = False,
    production_no_labels: bool = False,
    source_row_ids: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, List[str], List[str], Dict[str, np.ndarray]]:
    norm_mode = resolve_decomp_normalization(normalization)
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    liquidity_feature = resolve_liquidity_feature(liquidity_feature)
    include_ma = bool(include_ma)
    use_scale_aware_ma = bool(
        include_ma and DECOMP_SCALE_AWARE_MA_FEATURE_ENABLED
    )
    if norm_mode == "synthetic":
        decomp_cols_ref = (
            DECOMP_FEATURE_COLS_SYN_WITH_MA
            if include_ma
            else DECOMP_FEATURE_COLS_SYN
        )
        decomp_feature_cols = swap_liquidity_feature_cols(
            decomp_cols_ref, liquidity_feature
        )
    else:
        decomp_cols_ref = (
            DECOMP_FEATURE_COLS_LOOKBACK_WITH_MA
            if include_ma
            else DECOMP_FEATURE_COLS_LOOKBACK
        )
        decomp_feature_cols = swap_liquidity_feature_cols(
            decomp_cols_ref, liquidity_feature
        )

    base_cols = ["open", "high", "low", "close", liquidity_feature]
    if include_ma and not use_scale_aware_ma:
        base_cols.insert(4, "ma_n")
    missing = [col for col in base_cols if col not in features.columns]
    if missing:
        raise ValueError("missing decomposition features: " + ", ".join(missing))

    features_subset = features[base_cols].dropna()
    source_series = resolve_source_row_series(features.index, source_row_ids)
    source_clean = source_series.reindex(features_subset.index)
    source_valid = ~source_clean.isna()
    if not bool(source_valid.all()):
        features_subset = features_subset.loc[source_valid[source_valid].index]
        source_clean = source_clean.loc[source_valid[source_valid].index]
    if features_subset.shape[0] < lookback:
        raise ValueError("lookback exceeds available rows")

    source_values = source_clean.to_numpy(dtype=np.int64, copy=False)
    feature_values = features_subset.to_numpy(dtype=float)
    windows_view = np.lib.stride_tricks.sliding_window_view(
        feature_values, window_shape=lookback, axis=0
    )
    windows_view = normalize_window_view(
        windows_view, lookback, feature_values.shape[1]
    )

    if norm_mode == "synthetic":
        feature_order = ["open", "high", "low", "close"]
        if include_ma and not use_scale_aware_ma:
            feature_order.append("ma_n")
        feature_order.append("liquidity")
        syn_view = build_synthetic_windows(
            windows_view, feature_order=feature_order
        )
        window_source = syn_view
    else:
        window_source = windows_view

    sample_index = features_subset.index[lookback - 1 :]
    sample_end_positions = np.arange(
        int(lookback - 1),
        int(lookback - 1 + sample_index.shape[0]),
        dtype=np.int64,
    )
    contiguous_mask = contiguous_window_mask_from_source_rows(source_values, lookback)
    if contiguous_mask.shape[0] != sample_index.shape[0]:
        raise ValueError("unexpected contiguous-window mask shape mismatch")
    if production_no_labels:
        label_values = np.full(
            (sample_index.shape[0], len(LABEL_COLS)),
            np.nan,
            dtype=float,
        )
        valid_mask = contiguous_mask.copy()
    else:
        labels_aligned = labels[LABEL_COLS].reindex(sample_index)
        label_values = labels_aligned.to_numpy(dtype=float)
        valid_mask = build_labeled_sample_valid_mask(contiguous_mask, label_values)

    window_source = window_source[valid_mask]
    y = label_values[valid_mask]
    sample_index = pd.DatetimeIndex(sample_index[valid_mask])
    sample_end_positions = sample_end_positions[valid_mask]

    if stride > 1:
        take = np.arange(0, sample_index.shape[0], stride)
        window_source = window_source[take]
        y = y[take]
        sample_index = sample_index[take]
        sample_end_positions = sample_end_positions[take]

    window_sizes = compute_decomposition_window_sizes(
        lookback, windows=windows, scales=scales
    )
    scale_ma_periods: List[int] = []
    scale_ma_values: List[np.ndarray] = []
    if use_scale_aware_ma:
        close_values = features_subset["close"].to_numpy(dtype=np.float64, copy=False)
        sample_start_positions = sample_end_positions - int(lookback - 1)
        if sample_start_positions.shape[0] > 0 and int(np.min(sample_start_positions)) < 0:
            raise ValueError("sample_start_positions has negative values")
        base_close = close_values[sample_start_positions]
        base_close_safe = np.where(np.abs(base_close) < EPS, 1.0, base_close)
        for window_size in window_sizes:
            ma_vals, period = compute_scale_aware_ma_values(
                close_values=close_values,
                sample_end_positions=sample_end_positions,
                window_size=int(window_size),
                window_count=int(windows),
            )
            if norm_mode == "synthetic":
                ma_vals = ma_vals / base_close_safe[:, None]
            scale_ma_values.append(ma_vals.astype(window_source.dtype, copy=False))
            scale_ma_periods.append(int(period))
        if scale_ma_values:
            ma_valid_mask = np.ones((sample_index.shape[0],), dtype=bool)
            for ma_vals in scale_ma_values:
                ma_valid_mask &= np.isfinite(ma_vals).all(axis=1)
            if not bool(ma_valid_mask.all()):
                window_source = window_source[ma_valid_mask]
                y = y[ma_valid_mask]
                sample_index = pd.DatetimeIndex(sample_index[ma_valid_mask])
                sample_end_positions = sample_end_positions[ma_valid_mask]
                scale_ma_values = [vals[ma_valid_mask] for vals in scale_ma_values]
    scale_maps = []
    for scale_idx, window_size in enumerate(window_sizes):
        ma_values = (
            scale_ma_values[scale_idx]
            if use_scale_aware_ma
            else None
        )
        sub_map = build_ohlcv_submap(
            window_source,
            lookback,
            window_size,
            windows,
            include_ma=include_ma,
            ma_values=ma_values,
        )
        if norm_mode == "lookback":
            sub_map = apply_ohlc_lookback_norm(sub_map, decomp_feature_cols, enabled=True)
        scale_maps.append(sub_map)

    X = np.stack(scale_maps, axis=1)
    if use_scale_aware_ma:
        moving_average_window = (
            int(max(scale_ma_periods))
            if scale_ma_periods
            else int(compute_ma_n_window(lookback))
        )
        moving_average_mode = MOVING_AVERAGE_MODE_SCALE_AWARE
    else:
        moving_average_window = int(compute_ma_n_window(lookback))
        moving_average_mode = MOVING_AVERAGE_MODE_GLOBAL
    extra_payload = {
        "decomposition_scales": np.array([len(window_sizes)]),
        "decomposition_windows": np.array([windows]),
        "decomposition_normalization": np.array([norm_mode]),
        "decomp_include_ma": np.array([int(include_ma)], dtype=np.int8),
        "moving_average_window": np.array([int(moving_average_window)], dtype=np.int64),
        "moving_average_mode": np.array([moving_average_mode], dtype=object),
        "moving_average_global_window": np.array(
            [int(compute_ma_n_window(lookback))], dtype=np.int64
        ),
        "moving_average_scale_aware": np.array(
            [int(use_scale_aware_ma)], dtype=np.int8
        ),
        "moving_average_windows_by_scale": np.asarray(
            scale_ma_periods if use_scale_aware_ma else [],
            dtype=np.int64,
        ),
    }
    return X, y, sample_index, decomp_feature_cols, LABEL_COLS, extra_payload


def align_dual_samples(
    seq_index: pd.DatetimeIndex,
    decomp_index: pd.DatetimeIndex,
    X_seq: np.ndarray,
    X_decomp: np.ndarray,
    y_seq: np.ndarray,
    y_decomp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    seq_idx = pd.DatetimeIndex(seq_index)
    decomp_idx = pd.DatetimeIndex(decomp_index)
    common_index = seq_idx.intersection(decomp_idx, sort=False)
    if common_index.empty:
        raise ValueError("no overlapping samples between seq and decomp inputs")
    seq_pos = seq_idx.get_indexer(common_index)
    decomp_pos = decomp_idx.get_indexer(common_index)
    if (seq_pos < 0).any() or (decomp_pos < 0).any():
        raise ValueError("alignment failed for dual inputs")
    X_seq = X_seq[seq_pos]
    X_decomp = X_decomp[decomp_pos]
    y_seq = y_seq[seq_pos]
    y_decomp = y_decomp[decomp_pos]
    return X_seq, X_decomp, y_seq, y_decomp, common_index


def derive_decomposition_npz_path(npz_path: Path) -> Path:
    if npz_path.suffix != ".npz":
        raise ValueError("npz_path must end with .npz")
    stem = npz_path.with_suffix("")
    return Path(str(stem) + "_decomp.npz")


def derive_dual_npz_path(npz_path: Path) -> Path:
    if npz_path.suffix != ".npz":
        raise ValueError("npz_path must end with .npz")
    stem = npz_path.with_suffix("")
    return Path(str(stem) + "_dual.npz")


def build_chronological_merge_order(
    index_parts: Sequence[pd.DatetimeIndex],
    ticker_id_parts: Sequence[np.ndarray],
) -> np.ndarray:
    if len(index_parts) != len(ticker_id_parts):
        raise ValueError(
            "index/ticker part count mismatch: "
            f"len(index_parts)={len(index_parts)} "
            f"len(ticker_id_parts)={len(ticker_id_parts)}"
        )
    part_count = len(index_parts)
    if part_count == 0:
        return np.empty((0,), dtype=np.int64)

    offsets = np.zeros((part_count,), dtype=np.int64)
    lengths = np.zeros((part_count,), dtype=np.int64)
    ts_parts: list[np.ndarray] = []
    tid_parts: list[np.ndarray] = []

    total = 0
    for i, (idx_raw, tid_raw) in enumerate(zip(index_parts, ticker_id_parts)):
        idx = pd.DatetimeIndex(idx_raw)
        tid_arr = np.asarray(tid_raw).reshape(-1)
        n = int(idx.shape[0])
        if int(tid_arr.shape[0]) != n:
            raise ValueError(
                "array length mismatch while assembling chronologically: "
                f"len(index)={n} len(ticker_ids)={int(tid_arr.shape[0])} part={i}"
            )
        offsets[i] = int(total)
        lengths[i] = int(n)
        total += int(n)
        if n <= 0:
            ts_parts.append(np.empty((0,), dtype=np.int64))
            tid_parts.append(np.empty((0,), dtype=np.int64))
            continue
        if bool(idx.isna().any()):
            raise ValueError(
                "cannot assemble chronologically: timestamps contain invalid values"
            )
        ts = idx.asi8
        if bool(np.any(ts[1:] < ts[:-1])):
            raise ValueError(
                "cannot assemble chronologically: part timestamps are not monotonic "
                f"increasing (part={i})"
            )
        try:
            tid64 = tid_arr.astype(np.int64, copy=False)
        except Exception as exc:
            raise ValueError(
                "ticker_ids must be integer-like for chronological assembly"
            ) from exc
        ts_parts.append(ts)
        tid_parts.append(tid64)

    if total <= 0:
        return np.empty((0,), dtype=np.int64)
    if total == 1:
        return np.array([0], dtype=np.int64)

    heap: list[tuple[int, int, int, int]] = []
    for part in range(part_count):
        if int(lengths[part]) <= 0:
            continue
        heapq.heappush(
            heap,
            (
                int(ts_parts[part][0]),
                int(tid_parts[part][0]),
                int(part),
                0,
            ),
        )

    order = np.empty((int(total),), dtype=np.int64)
    w = 0
    while heap:
        _, _, part, local = heapq.heappop(heap)
        order[w] = int(offsets[part]) + int(local)
        w += 1
        nxt = int(local) + 1
        if nxt < int(lengths[part]):
            heapq.heappush(
                heap,
                (
                    int(ts_parts[part][nxt]),
                    int(tid_parts[part][nxt]),
                    int(part),
                    int(nxt),
                ),
            )

    if w != int(total):
        raise RuntimeError(
            "chronological merge order construction failed: "
            f"expected={int(total)} built={int(w)}"
        )
    return order


def assemble_samples_chronologically(
    index_parts: Sequence[pd.DatetimeIndex],
    ticker_id_parts: Sequence[np.ndarray],
    *array_parts: Sequence[np.ndarray],
) -> tuple[pd.DatetimeIndex, np.ndarray, tuple[np.ndarray, ...]]:
    if len(index_parts) != len(ticker_id_parts):
        raise ValueError(
            "index/ticker part count mismatch: "
            f"len(index_parts)={len(index_parts)} "
            f"len(ticker_id_parts)={len(ticker_id_parts)}"
        )
    for arrays in array_parts:
        if len(arrays) != len(index_parts):
            raise ValueError(
                "array part count mismatch while assembling chronologically: "
                f"len(arrays)={len(arrays)} len(index_parts)={len(index_parts)}"
            )

    n_total = int(sum(int(pd.DatetimeIndex(idx).shape[0]) for idx in index_parts))
    if n_total <= 0:
        ticker_dtype = (
            np.asarray(ticker_id_parts[0]).dtype
            if len(ticker_id_parts) > 0
            else np.int32
        )
        empty_arrays = []
        for arrays in array_parts:
            if len(arrays) <= 0:
                empty_arrays.append(np.empty((0,), dtype=np.float32))
                continue
            arr0 = np.asarray(arrays[0])
            empty_arrays.append(
                np.empty((0,) + tuple(arr0.shape[1:]), dtype=arr0.dtype)
            )
        return (
            pd.DatetimeIndex([]),
            np.empty((0,), dtype=ticker_dtype),
            tuple(empty_arrays),
        )

    idx_concat = pd.DatetimeIndex(
        np.concatenate([pd.DatetimeIndex(idx).to_numpy() for idx in index_parts])
    )
    ticker_concat = np.concatenate(
        [np.asarray(t).reshape(-1) for t in ticker_id_parts], axis=0
    )
    arrays_concat = tuple(np.concatenate(arrays, axis=0) for arrays in array_parts)
    if n_total <= 1:
        return idx_concat, ticker_concat, arrays_concat

    order = build_chronological_merge_order(index_parts, ticker_id_parts)
    if int(order.shape[0]) != n_total:
        raise RuntimeError(
            "chronological assembly order size mismatch: "
            f"order={int(order.shape[0])} expected={n_total}"
        )
    idx_out = pd.DatetimeIndex(idx_concat.to_numpy()[order])
    ticker_out = ticker_concat[order]
    arrays_out = tuple(arr[order] for arr in arrays_concat)
    return idx_out, ticker_out, arrays_out


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
    extra_payload: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = sample_index.astype(str).to_numpy()
    payload = {
        "X": X,
        "y_raw": np.asarray(y_raw),
        "timestamps": timestamps,
        "feature_cols": np.array(feature_cols, dtype=object),
        "label_cols": np.array(label_cols, dtype=object),
        "lookback": np.array([lookback]),
        "horizon": np.array([horizon]),
        "stride": np.array([normalize_stride(stride)]),
        "entry_offset": np.array([ENTRY_OFFSET]),
        "ticker_ids": ticker_ids.astype(np.int32),
        "tickers": np.array(list(tickers), dtype=object),
    }
    if extra_payload:
        payload.update(extra_payload)
    np.savez_compressed(output_path, **payload)


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
    extra_payload: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = sample_index.astype(str).to_numpy()
    payload = {
        "X_seq": X_seq,
        "X_decomp": X_decomp,
        "y_raw": np.asarray(y_raw),
        "timestamps": timestamps,
        "feature_cols_seq": np.array(feature_cols_seq, dtype=object),
        "feature_cols_decomp": np.array(feature_cols_decomp, dtype=object),
        "label_cols": np.array(label_cols, dtype=object),
        "lookback": np.array([lookback]),
        "horizon": np.array([horizon]),
        "stride": np.array([normalize_stride(stride)]),
        "entry_offset": np.array([ENTRY_OFFSET]),
        "ticker_ids": ticker_ids.astype(np.int32),
        "tickers": np.array(list(tickers), dtype=object),
    }
    if extra_payload:
        payload.update(extra_payload)
    np.savez_compressed(output_path, **payload)


def iter_ticker_files(input_dir: Path) -> List[Path]:
    csvs = sorted(
        [
            p
            for p in input_dir.glob("*.csv")
            if not p.name.startswith("_")
        ]
    )
    return csvs


def process_ticker(
    csv_path: Path,
    lookback: Optional[int],
    horizon: int,
    stride: int,
    liquidity_feature: str,
    decomp_windows: int,
    decomp_scales: Optional[int],
    decomp_normalization: str,
    include_synthetic_features: bool,
    disable_turnover_backfill: bool,
    disable_no_split_history_fallback: bool = False,
    turnover_fallback_backfill: bool = True,
    min_avg_dollar_volume_6m: float = MIN_AVG_DOLLAR_VOLUME_3m,
    production_no_labels: bool = False,
    decomp_only: bool = False,
    decomp_include_ma: bool = False,
    start_date: Optional[pd.Timestamp] = None,
    lookback_seq: Optional[int] = None,
    lookback_decomp: Optional[int] = None,
    include_preceding_gap_in_ret_pct: bool = RET_PCT_INCLUDE_PRECEDING_GAP_DEFAULT,
    label_mode: str = LABEL_MODE_RANGE_ATR,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    pd.DatetimeIndex,
    np.ndarray,
    np.ndarray,
    pd.DatetimeIndex,
    List[str],
    List[str],
    Dict[str, np.ndarray],
]:
    label_mode_resolved = resolve_label_mode(label_mode)
    effective_label_horizon = resolve_effective_label_horizon(
        horizon=horizon,
        label_mode=label_mode_resolved,
    )
    lookback_seq_resolved, lookback_decomp_resolved = resolve_branch_lookbacks(
        lookback=lookback,
        lookback_seq=lookback_seq,
        lookback_decomp=lookback_decomp,
    )
    df = load_daily_csv(csv_path)
    start_date = normalize_start_date(start_date)
    if start_date is not None:
        df = df.loc[df.index >= start_date]
        if df.empty:
            raise ValueError(
                f"no rows on/after start_date {start_date.date().isoformat()}"
            )
    if df.empty:
        raise ValueError("empty dataframe")
    df = df.copy()
    df["_source_row_id"] = resolve_input_source_row_ids(df).to_numpy(
        dtype=np.int64,
        copy=False,
    )

    df = backfill_shares_outstanding(
        df,
        ticker=csv_path.stem,
        disable_backfill=disable_turnover_backfill,
        disable_no_split_history_fallback=disable_no_split_history_fallback,
    )
    df, filter_counts = filter_invalid_daily_rows(df)
    if filter_counts:
        details = ", ".join([f"{k}={v}" for k, v in sorted(filter_counts.items())])
        print(f"[clean] {csv_path.stem}: {details}")
    if df.empty:
        raise ValueError("no rows remaining after data-quality filtering")

    df, liquidity_counts = filter_low_dollar_volume_half_year_periods(
        df,
        min_avg_dollar_volume_6m=min_avg_dollar_volume_6m,
    )
    if liquidity_counts:
        details = ", ".join([f"{k}={v}" for k, v in sorted(liquidity_counts.items())])
        print(f"[liquidity] {csv_path.stem}: {details}")
    if df.empty:
        raise ValueError("no rows remaining after liquidity-period filtering")
    source_row_ids = pd.to_numeric(df["_source_row_id"], errors="coerce")
    drop_by_date = pd.Series(dtype=bool)
    move_filter_meta: Dict[str, object] = {}
    if bool(FINAL_CANDLE_MOVE_FILTER_ENABLED):
        drop_by_date, move_filter_meta = compute_final_candle_large_move_drop_mask(
            index=pd.DatetimeIndex(df.index),
            close=df["close"],
            lookback=int(lookback_decomp_resolved),
            decomp_windows=int(decomp_windows),
            decomp_scales=decomp_scales,
            threshold=float(FINAL_CANDLE_MOVE_THRESHOLD),
            scale_selection=FINAL_CANDLE_MOVE_FILTER_SCALES,
        )
        if int(move_filter_meta.get("raw_rows_gt_threshold", 0)) > 0:
            print(
                "[sample_filter] "
                f"{csv_path.stem}: enabled=1 threshold={float(FINAL_CANDLE_MOVE_THRESHOLD):.6f} "
                f"scales={str(move_filter_meta.get('scale_selection', FINAL_CANDLE_MOVE_FILTER_SCALES))} "
                f"raw_rows_gt_threshold={int(move_filter_meta.get('raw_rows_gt_threshold', 0))} "
                f"candidate_sample_rows={int(move_filter_meta.get('sample_rows_marked', 0))}"
            )

    features = engineer_features_daily(
        df,
        lookback=max(int(lookback_seq_resolved), int(lookback_decomp_resolved)),
        liquidity_feature=liquidity_feature,
        turnover_fallback_backfill=turnover_fallback_backfill,
    )
    if production_no_labels:
        labels = pd.DataFrame(index=df.index, columns=LABEL_COLS, dtype=float)
    else:
        labels = compute_labels_daily(
            df,
            horizon=horizon,
            entry_offset=ENTRY_OFFSET,
            include_preceding_gap_in_ret_pct=bool(include_preceding_gap_in_ret_pct),
            label_mode=label_mode_resolved,
        )

    X_seq = np.empty((0, 0, 0), dtype=np.float32)
    y_seq = np.empty((0, len(LABEL_COLS)), dtype=np.float32)
    seq_index = pd.DatetimeIndex([])
    feature_cols_seq: List[str] = []
    label_cols_seq: List[str] = list(LABEL_COLS)
    if not decomp_only:
        X_seq, y_seq, seq_index, feature_cols_seq, label_cols_seq = build_sequence_dataset(
            features,
            labels,
            lookback=lookback_seq_resolved,
            horizon=effective_label_horizon,
            stride=stride,
            # Keep sequence/non-decomp features on raw volume regardless of
            # decomposition liquidity mode.
            liquidity_feature="volume",
            include_synthetic_features=include_synthetic_features,
            production_no_labels=production_no_labels,
            source_row_ids=source_row_ids,
        )

    X_decomp, y_decomp, decomp_index, feature_cols_decomp, label_cols_decomp, extra_payload = (
        build_decomposition_dataset(
            features,
            labels,
            lookback=lookback_decomp_resolved,
            horizon=effective_label_horizon,
            stride=stride,
            liquidity_feature=liquidity_feature,
            windows=decomp_windows,
            scales=decomp_scales,
            normalization=decomp_normalization,
            include_ma=bool(decomp_include_ma),
            production_no_labels=production_no_labels,
            source_row_ids=source_row_ids,
        )
    )
    if bool(FINAL_CANDLE_MOVE_FILTER_ENABLED) and not drop_by_date.empty:
        decomp_keep = build_sample_keep_mask_from_drop_dates(decomp_index, drop_by_date)

        seq_removed = 0
        if not decomp_only:
            seq_keep = build_sample_keep_mask_from_drop_dates(seq_index, drop_by_date)
            seq_removed = int(seq_keep.shape[0] - int(seq_keep.sum()))
        decomp_removed = int(decomp_keep.shape[0] - int(decomp_keep.sum()))
        if seq_removed > 0 or decomp_removed > 0:
            print(
                "[sample_filter] "
                f"{csv_path.stem}: seq_removed={seq_removed} decomp_removed={decomp_removed}"
            )

        if not decomp_only:
            X_seq = X_seq[seq_keep]
            y_seq = y_seq[seq_keep]
            seq_index = pd.DatetimeIndex(seq_index[seq_keep])

        X_decomp = X_decomp[decomp_keep]
        y_decomp = y_decomp[decomp_keep]
        decomp_index = pd.DatetimeIndex(decomp_index[decomp_keep])

        if (not decomp_only) and X_seq.shape[0] < 1:
            raise ValueError("no sequence samples remaining after final-candle move filter")
        if X_decomp.shape[0] < 1:
            raise ValueError("no decomposition samples remaining after final-candle move filter")

        extra_payload = dict(extra_payload)
        extra_payload["final_candle_move_filter_enabled"] = np.array([1], dtype=np.int8)
        extra_payload["final_candle_move_filter_threshold"] = np.array(
            [float(FINAL_CANDLE_MOVE_THRESHOLD)], dtype=np.float32
        )
        lags = move_filter_meta.get("lags", [])
        extra_payload["final_candle_move_filter_lags"] = np.asarray(lags, dtype=np.int64)
        extra_payload["final_candle_move_filter_scales"] = np.array(
            [str(move_filter_meta.get("scale_selection", FINAL_CANDLE_MOVE_FILTER_SCALES))],
            dtype=object,
        )
        extra_payload["final_candle_move_filter_scale_indices"] = np.asarray(
            move_filter_meta.get("scale_indices", []), dtype=np.int64
        )
    else:
        extra_payload = dict(extra_payload)
        extra_payload["final_candle_move_filter_enabled"] = np.array([0], dtype=np.int8)
        extra_payload["final_candle_move_filter_threshold"] = np.array(
            [float(FINAL_CANDLE_MOVE_THRESHOLD)], dtype=np.float32
        )
        extra_payload["final_candle_move_filter_lags"] = np.asarray([], dtype=np.int64)
        extra_payload["final_candle_move_filter_scales"] = np.array(
            [str(resolve_final_candle_move_filter_scales(FINAL_CANDLE_MOVE_FILTER_SCALES))],
            dtype=object,
        )
        extra_payload["final_candle_move_filter_scale_indices"] = np.asarray(
            [], dtype=np.int64
        )

    decomp_keep_low_volume, low_volume_filter_applied = build_scale0_volume_lt1_keep_mask(
        X_decomp=X_decomp,
        feature_cols_decomp=feature_cols_decomp,
        max_lt1=int(SCALE0_VOLUME_LT1_MAX_ALLOWED),
    )
    low_volume_decomp_removed = int(
        decomp_keep_low_volume.shape[0] - int(decomp_keep_low_volume.sum())
    )
    if low_volume_decomp_removed > 0:
        drop_dates_low_volume = pd.Series(
            True,
            index=pd.DatetimeIndex(decomp_index[~decomp_keep_low_volume]).tz_localize(None),
        )
        low_volume_seq_removed = 0
        if not decomp_only:
            seq_keep_low_volume = build_sample_keep_mask_from_drop_dates(
                seq_index, drop_dates_low_volume
            )
            low_volume_seq_removed = int(
                seq_keep_low_volume.shape[0] - int(seq_keep_low_volume.sum())
            )
        print(
            "[sample_filter] "
            f"{csv_path.stem}: scale0_volume_lt1_max={int(SCALE0_VOLUME_LT1_MAX_ALLOWED)} "
            f"seq_removed={low_volume_seq_removed} decomp_removed={low_volume_decomp_removed}"
        )
        if not decomp_only:
            X_seq = X_seq[seq_keep_low_volume]
            y_seq = y_seq[seq_keep_low_volume]
            seq_index = pd.DatetimeIndex(seq_index[seq_keep_low_volume])
        X_decomp = X_decomp[decomp_keep_low_volume]
        y_decomp = y_decomp[decomp_keep_low_volume]
        decomp_index = pd.DatetimeIndex(decomp_index[decomp_keep_low_volume])
        if (not decomp_only) and X_seq.shape[0] < 1:
            raise ValueError("no sequence samples remaining after scale0-volume filter")
        if X_decomp.shape[0] < 1:
            raise ValueError("no decomposition samples remaining after scale0-volume filter")
    extra_payload = dict(extra_payload)
    extra_payload["scale0_volume_lt1_filter_enabled"] = np.array(
        [1 if bool(low_volume_filter_applied) else 0], dtype=np.int8
    )
    extra_payload["scale0_volume_lt1_max_allowed"] = np.array(
        [int(SCALE0_VOLUME_LT1_MAX_ALLOWED)], dtype=np.int64
    )
    extra_payload["label_mode"] = np.array([label_mode_resolved])
    extra_payload["effective_label_horizon"] = np.array(
        [int(effective_label_horizon)], dtype=np.int64
    )
    effective_include_preceding_gap = bool(include_preceding_gap_in_ret_pct)
    if label_mode_resolved == LABEL_MODE_NEXT_DAY_CLOSE_RETURN:
        effective_include_preceding_gap = True
    returns_include_preceding_gap = np.array(
        [1 if effective_include_preceding_gap else 0], dtype=np.int8
    )
    extra_payload["return_labels_include_preceding_gap"] = returns_include_preceding_gap
    # Backward-compatible metadata key retained for existing consumers.
    extra_payload["ret_pct_includes_preceding_gap"] = returns_include_preceding_gap

    if (not decomp_only) and label_cols_seq != label_cols_decomp:
        raise ValueError("label cols mismatch between seq and decomp datasets")

    return (
        X_seq,
        y_seq,
        seq_index,
        X_decomp,
        y_decomp,
        decomp_index,
        feature_cols_seq,
        feature_cols_decomp,
        extra_payload,
    )


def build_combined_datasets(
    input_dir: Path,
    output_path: Path,
    lookback: Optional[int],
    horizon: int,
    stride: int,
    liquidity_feature: str,
    decomp_windows: int,
    decomp_scales: Optional[int],
    decomp_normalization: str,
    include_synthetic_features: bool,
    disable_turnover_backfill: bool,
    disable_no_split_history_fallback: bool = False,
    turnover_fallback_backfill: bool = True,
    min_avg_dollar_volume_6m: float = MIN_AVG_DOLLAR_VOLUME_3m,
    production_no_labels: bool = False,
    chronological_assembly: bool = True,
    decomp_only: bool = False,
    decomp_include_ma: bool = False,
    start_date: Optional[pd.Timestamp] = None,
    lookback_seq: Optional[int] = None,
    lookback_decomp: Optional[int] = None,
    include_preceding_gap_in_ret_pct: bool = RET_PCT_INCLUDE_PRECEDING_GAP_DEFAULT,
    label_mode: str = LABEL_MODE_RANGE_ATR,
) -> Tuple[Path, Path, Path]:
    start_date = normalize_start_date(start_date)
    label_mode_resolved = resolve_label_mode(label_mode)
    effective_label_horizon = resolve_effective_label_horizon(
        horizon=horizon,
        label_mode=label_mode_resolved,
    )
    lookback_seq_resolved, lookback_decomp_resolved = resolve_branch_lookbacks(
        lookback=lookback,
        lookback_seq=lookback_seq,
        lookback_decomp=lookback_decomp,
    )
    csv_paths = iter_ticker_files(input_dir)
    if not csv_paths:
        raise ValueError(f"no csv files found in {input_dir}")

    X_seq_all = []
    y_seq_all = []
    idx_seq_all = []
    ticker_ids_seq = []

    X_decomp_all = []
    y_decomp_all = []
    idx_decomp_all = []
    ticker_ids_decomp = []

    X_seq_dual_all = []
    X_decomp_dual_all = []
    y_dual_all = []
    idx_dual_all = []
    ticker_ids_dual = []

    tickers = []
    feature_cols_seq = None
    feature_cols_decomp = None
    extra_payload = None
    skip_counts: Dict[str, int] = {}

    def record_skip(reason: object) -> None:
        key = str(reason).strip() or "unknown_error"
        skip_counts[key] = int(skip_counts.get(key, 0)) + 1

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
            ) = process_ticker(
                csv_path,
                lookback=lookback,
                horizon=horizon,
                stride=stride,
                liquidity_feature=liquidity_feature,
                decomp_windows=decomp_windows,
                decomp_scales=decomp_scales,
                decomp_normalization=decomp_normalization,
                decomp_include_ma=bool(decomp_include_ma),
                include_synthetic_features=include_synthetic_features,
                disable_turnover_backfill=disable_turnover_backfill,
                disable_no_split_history_fallback=disable_no_split_history_fallback,
                turnover_fallback_backfill=turnover_fallback_backfill,
                min_avg_dollar_volume_6m=min_avg_dollar_volume_6m,
                production_no_labels=production_no_labels,
                decomp_only=bool(decomp_only),
                start_date=start_date,
                lookback_seq=lookback_seq_resolved,
                lookback_decomp=lookback_decomp_resolved,
                include_preceding_gap_in_ret_pct=bool(include_preceding_gap_in_ret_pct),
                label_mode=label_mode_resolved,
            )
        except Exception as exc:
            print(f"[skip] {ticker}: {exc}")
            record_skip(exc)
            continue

        if not decomp_only:
            try:
                X_seq_dual, X_decomp_dual, y_seq_dual, y_decomp_dual, dual_index = (
                    align_dual_samples(seq_index, decomp_index, X_seq, X_decomp, y_seq, y_decomp)
                )
            except Exception as exc:
                print(f"[skip] {ticker}: {exc}")
                record_skip(exc)
                continue

        ticker_id = len(tickers)
        if not decomp_only:
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

        tickers.append(ticker)
        if not decomp_only:
            X_seq_all.append(X_seq)
            y_seq_all.append(y_seq)
            idx_seq_all.append(seq_index)
            ticker_ids_seq.append(np.full(len(seq_index), ticker_id, dtype=np.int32))

        X_decomp_all.append(X_decomp)
        y_decomp_all.append(y_decomp)
        idx_decomp_all.append(decomp_index)
        ticker_ids_decomp.append(np.full(len(decomp_index), ticker_id, dtype=np.int32))
        if not decomp_only:
            if y_seq_dual.shape != y_decomp_dual.shape:
                raise ValueError(f"aligned label shapes do not match for {ticker}")
            if not np.allclose(y_seq_dual, y_decomp_dual, equal_nan=True):
                raise ValueError(f"aligned labels differ for {ticker}")

            X_seq_dual_all.append(X_seq_dual)
            X_decomp_dual_all.append(X_decomp_dual)
            y_dual_all.append(y_seq_dual)
            idx_dual_all.append(dual_index)
            ticker_ids_dual.append(np.full(len(dual_index), ticker_id, dtype=np.int32))

    if not X_decomp_all:
        skip_summary = sorted(skip_counts.items(), key=lambda x: (-x[1], x[0]))
        top_reasons = "; ".join([f"{reason} x{count}" for reason, count in skip_summary[:5]])
        hint = (
            "no tickers produced datasets. "
            "Most common skip reasons: "
            f"{top_reasons if top_reasons else 'none recorded'}"
        )
        raise ValueError(hint)

    if bool(decomp_only):
        if bool(chronological_assembly):
            idx_decomp, ticker_ids_decomp_arr, (X_decomp, y_decomp) = (
                assemble_samples_chronologically(
                    idx_decomp_all,
                    ticker_ids_decomp,
                    X_decomp_all,
                    y_decomp_all,
                )
            )
        else:
            X_decomp = np.concatenate(X_decomp_all, axis=0)
            y_decomp = np.concatenate(y_decomp_all, axis=0)
            idx_decomp = pd.DatetimeIndex(
                np.concatenate([idx.to_numpy() for idx in idx_decomp_all])
            )
            ticker_ids_decomp_arr = np.concatenate(ticker_ids_decomp, axis=0)
    else:
        if bool(chronological_assembly):
            idx_seq, ticker_ids_seq_arr, (X_seq, y_seq) = (
                assemble_samples_chronologically(
                    idx_seq_all,
                    ticker_ids_seq,
                    X_seq_all,
                    y_seq_all,
                )
            )
            idx_decomp, ticker_ids_decomp_arr, (X_decomp, y_decomp) = (
                assemble_samples_chronologically(
                    idx_decomp_all,
                    ticker_ids_decomp,
                    X_decomp_all,
                    y_decomp_all,
                )
            )
            idx_dual, ticker_ids_dual_arr, (X_seq_dual, X_decomp_dual, y_dual) = (
                assemble_samples_chronologically(
                    idx_dual_all,
                    ticker_ids_dual,
                    X_seq_dual_all,
                    X_decomp_dual_all,
                    y_dual_all,
                )
            )
        else:
            X_seq = np.concatenate(X_seq_all, axis=0)
            y_seq = np.concatenate(y_seq_all, axis=0)
            idx_seq = pd.DatetimeIndex(np.concatenate([idx.to_numpy() for idx in idx_seq_all]))
            ticker_ids_seq_arr = np.concatenate(ticker_ids_seq, axis=0)

            X_decomp = np.concatenate(X_decomp_all, axis=0)
            y_decomp = np.concatenate(y_decomp_all, axis=0)
            idx_decomp = pd.DatetimeIndex(
                np.concatenate([idx.to_numpy() for idx in idx_decomp_all])
            )
            ticker_ids_decomp_arr = np.concatenate(ticker_ids_decomp, axis=0)

            X_seq_dual = np.concatenate(X_seq_dual_all, axis=0)
            X_decomp_dual = np.concatenate(X_decomp_dual_all, axis=0)
            y_dual = np.concatenate(y_dual_all, axis=0)
            idx_dual = pd.DatetimeIndex(np.concatenate([idx.to_numpy() for idx in idx_dual_all]))
            ticker_ids_dual_arr = np.concatenate(ticker_ids_dual, axis=0)

    base_npz_path = output_path
    decomp_npz_path = derive_decomposition_npz_path(base_npz_path)
    dual_npz_path = derive_dual_npz_path(base_npz_path)

    extra_payload = dict(extra_payload or {})
    extra_payload["lookback_seq"] = np.array([int(lookback_seq_resolved)], dtype=np.int64)
    extra_payload["lookback_decomp"] = np.array(
        [int(lookback_decomp_resolved)], dtype=np.int64
    )

    if not decomp_only:
        save_dataset_npz(
            base_npz_path,
            X_seq,
            y_seq,
            idx_seq,
            feature_cols_seq,
            LABEL_COLS,
            int(lookback_seq_resolved),
            effective_label_horizon,
            stride,
            ticker_ids_seq_arr,
            tickers,
            extra_payload=extra_payload,
        )

    save_dataset_npz(
        decomp_npz_path,
        X_decomp,
        y_decomp,
        idx_decomp,
        feature_cols_decomp,
        LABEL_COLS,
        int(lookback_decomp_resolved),
        effective_label_horizon,
        stride,
        ticker_ids_decomp_arr,
        tickers,
        extra_payload=extra_payload,
    )

    if not decomp_only:
        save_dual_dataset_npz(
            dual_npz_path,
            X_seq_dual,
            X_decomp_dual,
            y_dual,
            idx_dual,
            feature_cols_seq,
            feature_cols_decomp,
            LABEL_COLS,
            int(lookback_seq_resolved),
            effective_label_horizon,
            stride,
            ticker_ids_dual_arr,
            tickers,
            extra_payload=extra_payload,
        )

    return base_npz_path, decomp_npz_path, dual_npz_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare daily candle datasets.")
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Directory with ticker CSVs (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-path",
        default="data/synthetic_dataset.npz",
        help="Base output .npz path (sequence).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        help=(
            "Legacy single lookback override applied to both seq/decomp when set. "
            "If omitted, --lookback-seq/--lookback-decomp defaults are used."
        ),
    )
    parser.add_argument(
        "--lookback-seq",
        type=int,
        default=DEFAULT_LOOKBACK_SEQ,
        help=f"Lookback for sequence dataset construction (default: {DEFAULT_LOOKBACK_SEQ}).",
    )
    parser.add_argument(
        "--lookback-decomp",
        type=int,
        default=DEFAULT_LOOKBACK_DECOMP,
        help=(
            "Lookback for decomposition dataset construction "
            f"(default: {DEFAULT_LOOKBACK_DECOMP})."
        ),
    )
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
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
        "--label-mode",
        type=resolve_label_mode,
        default=LABEL_MODE_RANGE_ATR,
        choices=list(LABEL_MODE_CHOICES),
        help=(
            "Label construction mode. "
            "'range_atr' preserves the existing horizon-based label logic. "
            "'next_day_close_return' uses signal-day close -> next-day close returns "
            "and forces an effective label horizon of 1."
        ),
    )
    parser.add_argument(
        "--ret-pct-include-preceding-gap",
        action=argparse.BooleanOptionalAction,
        default=RET_PCT_INCLUDE_PRECEDING_GAP_DEFAULT,
        help=(
            "Include the gap immediately before output-horizon start when computing "
            "return labels (ret_pct/ret_atr/avg_ret_atr/log_avg_ret_atr). "
            "For entry_offset=1, this switches the return base from open[t+1] to close[t]. "
            "Ignored when --label-mode=next_day_close_return."
        ),
    )
    parser.add_argument(
        "--liquidity-feature",
        default=LIQUIDITY_FEATURE_DEFAULT,
        choices=["turnover", "volume"],
    )
    parser.add_argument("--decomp-windows", type=int, default=DEFAULT_DECOMP_WINDOWS)
    parser.add_argument("--decomp-scales", type=int, default=None)
    parser.add_argument(
        "--decomp-normalization",
        type=resolve_decomp_normalization,
        default=DECOMP_NORMALIZATION,
        help=(
            "Decomposition normalization: 'synthetic', 'lookback', or 'none' "
            "(raw decomposition values without normalization)."
        ),
    )
    parser.add_argument(
        "--decomp-include-ma",
        action=argparse.BooleanOptionalAction,
        default=DECOMP_INCLUDE_MA_DEFAULT,
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
    parser.add_argument(
        "--production-no-labels",
        action="store_true",
        help=(
            "Production mode: do not require forward-looking labels for sample "
            "creation; y_raw and label columns are emitted as NaN."
        ),
    )
    parser.add_argument(
        "--decomp-only",
        action="store_true",
        help="Build and save only the decomposition dataset output.",
    )
    parser.set_defaults(chronological_assembly=True)
    parser.add_argument(
        "--chronological-assembly",
        dest="chronological_assembly",
        action="store_true",
        help=(
            "Assemble combined dataset samples in ascending timestamp order across "
            "tickers (default: enabled)."
        ),
    )
    parser.add_argument(
        "--disable-chronological-assembly",
        dest="chronological_assembly",
        action="store_false",
        help=(
            "Disable chronological assembly and keep ticker-grouped concatenation "
            "order."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    base_npz_path, decomp_npz_path, dual_npz_path = build_combined_datasets(
        input_dir=Path(args.input_dir),
        output_path=Path(args.output_path),
        lookback=(None if args.lookback is None else int(args.lookback)),
        lookback_seq=int(args.lookback_seq),
        lookback_decomp=int(args.lookback_decomp),
        horizon=int(args.horizon),
        stride=int(args.stride),
        liquidity_feature=args.liquidity_feature,
        decomp_windows=int(args.decomp_windows),
        decomp_scales=args.decomp_scales,
        decomp_normalization=args.decomp_normalization,
        decomp_include_ma=bool(args.decomp_include_ma),
        include_synthetic_features=bool(args.include_synthetic_features),
        disable_turnover_backfill=bool(args.disable_turnover_backfill),
        disable_no_split_history_fallback=bool(args.disable_no_split_history_fallback),
        turnover_fallback_backfill=not bool(args.disable_turnover_fallback_backfill),
        production_no_labels=bool(args.production_no_labels),
        chronological_assembly=bool(args.chronological_assembly),
        decomp_only=bool(args.decomp_only),
        start_date=args.start_date,
        include_preceding_gap_in_ret_pct=bool(args.ret_pct_include_preceding_gap),
        label_mode=args.label_mode,
    )

    if not bool(args.decomp_only):
        print(f"saved: {base_npz_path}")
    print(f"saved: {decomp_npz_path}")
    if not bool(args.decomp_only):
        print(f"saved: {dual_npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
