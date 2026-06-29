#!/usr/bin/env python3
"""Run daily cross-sectional and walk-forward simulation evals for a completed run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

# ----------------------------
# Evaluation constants
# ----------------------------
DAILY_CROSS_SECTIONAL_TOP_PCT = [.01]
DAILY_CROSS_SECTIONAL_BOTTOM_PCT = [.01]
DAILY_CROSS_SECTIONAL_MIN_PER_SIDE = 1
DAILY_CROSS_SECTIONAL_MIN_NAMES_PER_DAY = 2
DAILY_CROSS_SECTIONAL_ANNUALIZATION_DAYS = 252.0
# For horizon=5 labels, only 20% notional is deployed per day so five
# overlapping vintages sum to roughly full capital.
DAILY_CROSS_SECTIONAL_SPREAD_COMPOUND_DAILY_CAPITAL_FRACTION = 0.2

WALKFORWARD_ROLLING_THRESHOLD_ENABLED = 1
WALKFORWARD_ROLLING_TOP_PCT = [.01]
WALKFORWARD_ROLLING_BOTTOM_PCT = [.01]
WALKFORWARD_ROLLING_LOOKBACK_DAYS = 45
WALKFORWARD_ROLLING_MIN_HISTORY_DAYS = 20
WALKFORWARD_ROLLING_MIN_PER_SIDE = 1
WALKFORWARD_ROLLING_MIN_PER_SIDE_MODE = "either"
WALKFORWARD_ROLLING_MIN_NAMES_PER_DAY = 2
WALKFORWARD_ROLLING_ANNUALIZATION_DAYS = 252.0
WALKFORWARD_ROLLING_THRESHOLD_METHOD = "pooled"
WALKFORWARD_ROLLING_FALLBACK_TO_DAILY_RANK = 0
WALKFORWARD_ROLLING_ENFORCE_NON_OVERLAP = 1

WALKFORWARD_SPY_NON_OVERLAP_ENABLED = 1
WALKFORWARD_SPY_NON_OVERLAP_HORIZON_DAYS = 5
WALKFORWARD_SPY_DAILY_CSV_PATH = str(
    (Path(__file__).resolve().parent / "tickers" / "SPY.csv")
)
WALKFORWARD_SPY_DATE_COL = "date"
WALKFORWARD_SPY_CLOSE_COL = "close"
EVAL_TICKERS_DIR = Path(__file__).resolve().parent / "tickers"
DAILY_CROSS_SECTIONAL_BOTTOM_RUNUP_LOOKBACK_DAYS = 5
DAILY_CROSS_SECTIONAL_BOTTOM_RUNUP_MIN_PCT = 60.0
WALKFORWARD_BOTTOM_RUNUP_LOOKBACK_DAYS = 5
WALKFORWARD_BOTTOM_RUNUP_MIN_PCT = 30.0

EVAL_TAIL_RET_OUTLIER_ZSCORE_THRESHOLD = 6.0

_WALKFORWARD_SPY_RET_PCT_CACHE: dict[tuple[str, int, str, str], dict[str, float]] = {}
_TICKER_RUNUP_PCT_CACHE: dict[tuple[str, int], dict[str, float]] = {}
def resolve_daily_cross_sectional_tail_pct(value: float, name: str) -> float:
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite")
    if v <= 0.0 or v >= 50.0:
        raise ValueError(f"{name} must be in (0, 50), got {v}")
    return v


def resolve_tail_pct_values(
    value: float | Sequence[float] | str,
    name: str,
) -> list[float]:
    raw_values: list[float | str]
    if isinstance(value, str):
        text = str(value).strip()
        if not text:
            raise ValueError(f"{name} must contain at least one percentile")
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        raw_values = [tok.strip() for tok in text.split(",") if str(tok).strip()]
    elif isinstance(value, np.ndarray):
        raw_values = list(np.asarray(value, dtype=object).reshape(-1).tolist())
    elif isinstance(value, Sequence):
        raw_values = list(value)
    else:
        raw_values = [value]
    if not raw_values:
        raise ValueError(f"{name} must contain at least one percentile")

    out: list[float] = []
    for i, raw in enumerate(raw_values):
        v = resolve_daily_cross_sectional_tail_pct(float(raw), f"{name}[{i}]")
        if v not in out:
            out.append(v)
    if not out:
        raise ValueError(f"{name} must contain at least one valid percentile")
    return out


def resolve_tail_pct_pairs(
    top_value: float | Sequence[float] | str,
    bottom_value: float | Sequence[float] | str,
    context: str,
) -> list[tuple[float, float]]:
    top_vals = resolve_tail_pct_values(top_value, f"{context}.top_pct")
    bottom_vals = resolve_tail_pct_values(bottom_value, f"{context}.bottom_pct")
    if len(top_vals) == len(bottom_vals):
        pairs = list(zip(top_vals, bottom_vals))
    elif len(top_vals) == 1:
        pairs = [(top_vals[0], b) for b in bottom_vals]
    elif len(bottom_vals) == 1:
        pairs = [(t, bottom_vals[0]) for t in top_vals]
    else:
        raise ValueError(
            f"{context} percentile lists must match in length, or one side must have a single value "
            f"(got top={len(top_vals)} bottom={len(bottom_vals)})"
        )
    out: list[tuple[float, float]] = []
    for pair in pairs:
        if pair not in out:
            out.append(pair)
    return out


def tail_pct_config_value(value: float | Sequence[float] | str, name: str) -> float | list[float]:
    vals = resolve_tail_pct_values(value, name)
    if len(vals) == 1:
        return float(vals[0])
    return [float(v) for v in vals]


def format_pct_tag(value: float) -> str:
    txt = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if not txt:
        txt = "0"
    return txt.replace(".", "p")


def format_pct_value(value: float) -> str:
    txt = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return txt if txt else "0"


def resolve_walkforward_threshold_method(value: str) -> str:
    method = str(value).strip().lower()
    if method in ("pooled", "daily_median", "daily_mean"):
        return method
    raise ValueError(
        "walk-forward threshold method must be one of: pooled, daily_median, daily_mean"
    )


def resolve_walkforward_min_per_side_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in ("both", "either"):
        return mode
    raise ValueError("walk-forward min_per_side_mode must be one of: both, either")


def build_walkforward_spy_non_overlap_default_info(
    enabled: bool = bool(WALKFORWARD_SPY_NON_OVERLAP_ENABLED),
    horizon_days: int = WALKFORWARD_SPY_NON_OVERLAP_HORIZON_DAYS,
) -> dict[str, float | int | str]:
    h = max(1, int(horizon_days))
    return {
        "spy_non_overlap_enabled": int(bool(enabled)),
        "spy_non_overlap_horizon_days": int(h),
        # Number of executed trades used for the summary stats.
        "spy_non_overlap_trades": 0,
        "spy_non_overlap_trade_mean": float("nan"),
        "spy_non_overlap_trade_median": float("nan"),
        # Percent return across chained non-overlapping trades, e.g. 12.3 means +12.3%.
        "spy_non_overlap_compounded_return": float("nan"),
        "spy_non_overlap_compounded_multiple": float("nan"),
    }


def select_non_overlapping_days(
    signal_days: Sequence[str],
    ordered_dates: Sequence[str],
    horizon_days: int,
) -> list[str]:
    date_to_idx: dict[str, int] = {}
    for i, day in enumerate(ordered_dates):
        day_key = str(day)[:10]
        if len(day_key) == 10 and day_key not in date_to_idx:
            date_to_idx[day_key] = int(i)
    if not date_to_idx:
        return []

    uniq_signal_days = sorted(
        {str(day)[:10] for day in signal_days if str(day)[:10] in date_to_idx},
        key=lambda d: date_to_idx[d],
    )
    if not uniq_signal_days:
        return []

    h = max(1, int(horizon_days))
    chosen_days: list[str] = []
    next_allowed_idx = -10**9
    for day in uniq_signal_days:
        idx = int(date_to_idx[day])
        if idx >= next_allowed_idx:
            chosen_days.append(day)
            next_allowed_idx = idx + h
    return chosen_days


def build_walkforward_real_non_overlap_default_info(
    horizon_days: int = WALKFORWARD_SPY_NON_OVERLAP_HORIZON_DAYS,
) -> dict[str, float | int]:
    _ = max(1, int(horizon_days))
    return {
        "real_num_trades": 0,
        "real_per_trade_returns": float("nan"),
        "non_adj_returns_sharpe": float("nan"),
        "real_returns_sharpe": float("nan"),
        "real_nonoverlap_compounded_returns": float("nan"),
    }


def load_ticker_runup_pct_map(
    ticker: str,
    lookback_days: int,
    tickers_dir: Path = EVAL_TICKERS_DIR,
) -> dict[str, float]:
    ticker_key = str(ticker).strip().upper()
    lookback_v = max(1, int(lookback_days))
    cache_key = (ticker_key, lookback_v)
    cached = _TICKER_RUNUP_PCT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    out: dict[str, float] = {}
    if not ticker_key:
        _TICKER_RUNUP_PCT_CACHE[cache_key] = out
        return out

    path = Path(tickers_dir) / f"{ticker_key}.csv"
    if not path.is_file():
        _TICKER_RUNUP_PCT_CACHE[cache_key] = out
        return out

    try:
        raw = np.genfromtxt(
            path,
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
    except Exception:
        _TICKER_RUNUP_PCT_CACHE[cache_key] = out
        return out

    dtype_names = getattr(getattr(raw, "dtype", None), "names", None)
    if (
        getattr(raw, "size", 0) == 0
        or dtype_names is None
        or "date" not in dtype_names
        or "close" not in dtype_names
    ):
        _TICKER_RUNUP_PCT_CACHE[cache_key] = out
        return out

    arr = np.atleast_1d(raw)
    date_vals = np.asarray(arr["date"], dtype=object).reshape(-1)
    close_vals = np.asarray(arr["close"], dtype=np.float64).reshape(-1)
    if date_vals.size != close_vals.size or date_vals.size <= lookback_v:
        _TICKER_RUNUP_PCT_CACHE[cache_key] = out
        return out

    for i in range(lookback_v, int(date_vals.size)):
        prev_close = float(close_vals[i - lookback_v])
        cur_close = float(close_vals[i])
        if not np.isfinite(prev_close) or not np.isfinite(cur_close) or prev_close == 0.0:
            continue
        out[str(date_vals[i])[:10]] = float((cur_close / prev_close) - 1.0)

    _TICKER_RUNUP_PCT_CACHE[cache_key] = out
    return out


def compute_selected_runup_subset_metrics(
    *,
    dates: np.ndarray,
    tickers: np.ndarray | None,
    returns: np.ndarray,
    selected_indices: np.ndarray,
    lookback_days: int,
    min_runup_pct: float,
) -> tuple[int, float]:
    if tickers is None or selected_indices.size == 0:
        return 0, float("nan")

    subset_returns: list[float] = []
    lookback_v = max(1, int(lookback_days))
    threshold = float(min_runup_pct) / 100.0
    for idx_val in np.asarray(selected_indices, dtype=np.int64).reshape(-1):
        if idx_val < 0 or idx_val >= int(returns.shape[0]):
            continue
        date_key = str(dates[int(idx_val)])[:10]
        ticker_key = str(tickers[int(idx_val)]).strip().upper()
        if len(date_key) != 10 or not ticker_key:
            continue
        runup_map = load_ticker_runup_pct_map(ticker_key, lookback_v)
        runup_val = runup_map.get(date_key)
        if runup_val is None or not np.isfinite(runup_val) or runup_val < threshold:
            continue
        ret_val = float(returns[int(idx_val)])
        if np.isfinite(ret_val):
            subset_returns.append(ret_val)

    if not subset_returns:
        return 0, float("nan")
    arr = np.asarray(subset_returns, dtype=np.float64)
    return int(arr.size), float(np.mean(arr))


def load_walkforward_spy_ret_pct_map(
    spy_csv_path: str = WALKFORWARD_SPY_DAILY_CSV_PATH,
    horizon_days: int = WALKFORWARD_SPY_NON_OVERLAP_HORIZON_DAYS,
    date_col: str = WALKFORWARD_SPY_DATE_COL,
    close_col: str = WALKFORWARD_SPY_CLOSE_COL,
) -> dict[str, float]:
    h = max(1, int(horizon_days))
    path_txt = str(spy_csv_path).strip()
    date_col_key = str(date_col).strip().lower()
    close_col_key = str(close_col).strip().lower()
    cache_key = (path_txt, int(h), date_col_key, close_col_key)
    cached = _WALKFORWARD_SPY_RET_PCT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not path_txt:
        _WALKFORWARD_SPY_RET_PCT_CACHE[cache_key] = {}
        return {}
    path = Path(path_txt).expanduser()
    if not path.is_file():
        _WALKFORWARD_SPY_RET_PCT_CACHE[cache_key] = {}
        return {}

    close_by_date: dict[str, float] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = [str(x) for x in (reader.fieldnames or []) if x is not None]
            if not fieldnames:
                _WALKFORWARD_SPY_RET_PCT_CACHE[cache_key] = {}
                return {}
            field_lookup = {str(name).strip().lower(): str(name) for name in fieldnames}
            src_date_col = field_lookup.get(date_col_key)
            src_close_col = field_lookup.get(close_col_key)
            if src_date_col is None or src_close_col is None:
                _WALKFORWARD_SPY_RET_PCT_CACHE[cache_key] = {}
                return {}
            for row in reader:
                raw_date = str(row.get(src_date_col, "")).strip()
                raw_close = str(row.get(src_close_col, "")).strip()
                day = raw_date[:10]
                if len(day) != 10:
                    continue
                try:
                    close_val = float(raw_close)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(close_val):
                    continue
                close_by_date[day] = float(close_val)
    except OSError:
        _WALKFORWARD_SPY_RET_PCT_CACHE[cache_key] = {}
        return {}

    ordered_days = sorted(close_by_date.keys())
    if len(ordered_days) <= h:
        _WALKFORWARD_SPY_RET_PCT_CACHE[cache_key] = {}
        return {}

    ret_map: dict[str, float] = {}
    for i in range(len(ordered_days) - h):
        day = ordered_days[i]
        base = float(close_by_date[day])
        fut = float(close_by_date[ordered_days[i + h]])
        if (not np.isfinite(base)) or (not np.isfinite(fut)) or abs(base) <= 1e-12:
            continue
        ret_map[day] = float((fut - base) / base)

    _WALKFORWARD_SPY_RET_PCT_CACHE[cache_key] = ret_map
    return ret_map


def select_non_overlapping_signal_days(
    rows: Sequence[dict[str, float | int | str]],
    ordered_dates: Sequence[str],
    horizon_days: int,
) -> list[str]:
    signal_days: list[str] = []
    for row in rows:
        day = str(row.get("date", ""))[:10]
        try:
            top_count = int(row.get("top_count", 0))
        except (TypeError, ValueError):
            top_count = 0
        if top_count > 0:
            signal_days.append(day)
    return select_non_overlapping_days(
        signal_days=signal_days,
        ordered_dates=ordered_dates,
        horizon_days=horizon_days,
    )


def compute_walkforward_spy_non_overlap_info(
    rows: Sequence[dict[str, float | int | str]],
    ordered_dates: Sequence[str],
    enabled: bool = bool(WALKFORWARD_SPY_NON_OVERLAP_ENABLED),
    horizon_days: int = WALKFORWARD_SPY_NON_OVERLAP_HORIZON_DAYS,
    spy_csv_path: str = WALKFORWARD_SPY_DAILY_CSV_PATH,
    spy_date_col: str = WALKFORWARD_SPY_DATE_COL,
    spy_close_col: str = WALKFORWARD_SPY_CLOSE_COL,
) -> dict[str, float | int | str]:
    info = build_walkforward_spy_non_overlap_default_info(
        enabled=enabled,
        horizon_days=horizon_days,
    )
    if not bool(enabled):
        return info

    h = max(1, int(horizon_days))
    chosen_days = select_non_overlapping_signal_days(
        rows=rows,
        ordered_dates=ordered_dates,
        horizon_days=h,
    )
    if not chosen_days:
        return info

    spy_ret_map = load_walkforward_spy_ret_pct_map(
        spy_csv_path=spy_csv_path,
        horizon_days=h,
        date_col=spy_date_col,
        close_col=spy_close_col,
    )
    if not spy_ret_map:
        return info

    trade_returns: list[float] = []
    for day in chosen_days:
        ret_val = spy_ret_map.get(day)
        if ret_val is None or not np.isfinite(ret_val):
            continue
        trade_returns.append(float(ret_val))
    if not trade_returns:
        return info

    arr = np.asarray(trade_returns, dtype=np.float64)
    compounded_multiple = float(np.prod(1.0 + arr))
    info["spy_non_overlap_trades"] = int(arr.size)
    info["spy_non_overlap_trade_mean"] = float(np.mean(arr))
    info["spy_non_overlap_trade_median"] = float(np.median(arr))
    info["spy_non_overlap_compounded_multiple"] = float(compounded_multiple)
    info["spy_non_overlap_compounded_return"] = float((compounded_multiple - 1.0) * 100.0)
    return info


def compute_walkforward_real_non_overlap_info(
    rows: Sequence[dict[str, float | int | str]],
    ordered_dates: Sequence[str],
    horizon_days: int = WALKFORWARD_SPY_NON_OVERLAP_HORIZON_DAYS,
    annualization_days: float = WALKFORWARD_ROLLING_ANNUALIZATION_DAYS,
) -> dict[str, float | int]:
    info = build_walkforward_real_non_overlap_default_info(horizon_days=horizon_days)
    h = max(1, int(horizon_days))
    annualization = float(annualization_days)

    day_ret_map: dict[str, float] = {}
    for row in rows:
        day = str(row.get("date", ""))[:10]
        if len(day) != 10:
            continue
        try:
            top_count = int(row.get("top_count", 0))
        except (TypeError, ValueError):
            top_count = 0
        try:
            bottom_count = int(row.get("bottom_count", 0))
        except (TypeError, ValueError):
            bottom_count = 0
        try:
            top_ret = float(row.get("top_ret_pct_mean", float("nan")))
        except (TypeError, ValueError):
            top_ret = float("nan")
        try:
            bottom_ret = float(row.get("bottom_ret_pct_mean", float("nan")))
        except (TypeError, ValueError):
            bottom_ret = float("nan")

        has_top = top_count > 0 and np.isfinite(top_ret)
        has_bottom = bottom_count > 0 and np.isfinite(bottom_ret)
        if not has_top and not has_bottom:
            continue
        if has_top and has_bottom:
            # Paired day: equal-weight long top and short bottom.
            day_ret_map[day] = float(0.5 * top_ret + 0.5 * (-bottom_ret))
        elif has_top:
            day_ret_map[day] = float(top_ret)
        else:
            day_ret_map[day] = float(-bottom_ret)

    if not day_ret_map:
        return info

    chosen_days = select_non_overlapping_days(
        signal_days=list(day_ret_map.keys()),
        ordered_dates=ordered_dates,
        horizon_days=h,
    )
    if not chosen_days:
        return info

    trade_returns = [float(day_ret_map[d]) for d in chosen_days if np.isfinite(day_ret_map[d])]
    if not trade_returns:
        return info

    arr = np.asarray(trade_returns, dtype=np.float64)
    compounded_multiple = float(np.prod(1.0 + arr))
    trade_std = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
    trade_sharpe_non_adj = float("nan")
    trade_sharpe_horizon_adj = float("nan")
    if (
        np.isfinite(trade_std)
        and trade_std > 0.0
        and np.isfinite(annualization)
        and annualization > 0.0
    ):
        mean_ret = float(np.mean(arr))
        trade_sharpe_non_adj = float((mean_ret / trade_std) * np.sqrt(annualization))
        trade_sharpe_horizon_adj = float(
            (mean_ret / trade_std) * np.sqrt(annualization / float(h))
        )

    info["real_num_trades"] = int(arr.size)
    info["real_per_trade_returns"] = float(np.mean(arr))
    info["non_adj_returns_sharpe"] = float(trade_sharpe_non_adj)
    info["real_returns_sharpe"] = float(trade_sharpe_horizon_adj)
    info["real_nonoverlap_compounded_returns"] = float((compounded_multiple - 1.0) * 100.0)
    return info


def filter_eval_rows_by_tail_ret_zscore(
    rows: list[dict[str, float | int | str]],
    zscore_threshold: float = EVAL_TAIL_RET_OUTLIER_ZSCORE_THRESHOLD,
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int | str]]:
    robust_scale_const = 0.67448975
    rows_in = list(rows)
    days_before = int(len(rows_in))
    threshold = float(zscore_threshold)
    meta: dict[str, float | int | str] = {
        "outlier_filter_method": "tail_ret_pct_mean_zscore_two_sided",
        "outlier_filter_scope": "top_and_bottom",
        "outlier_filter_tail": "two_sided_abs",
        "outlier_filter_zscore_threshold": float(threshold),
        "outlier_filter_enabled": 0,
        "outlier_filter_reason": "no_rows",
        "outlier_filter_top_scale_method": "none",
        "outlier_filter_bottom_scale_method": "none",
        "outlier_filter_finite_top_count": 0,
        "outlier_filter_finite_bottom_count": 0,
        "outlier_filter_top_center_value": float("nan"),
        "outlier_filter_top_scale_value": float("nan"),
        "outlier_filter_top_lower_threshold_value": float("nan"),
        "outlier_filter_top_upper_threshold_value": float("nan"),
        "outlier_filter_bottom_center_value": float("nan"),
        "outlier_filter_bottom_scale_value": float("nan"),
        "outlier_filter_bottom_lower_threshold_value": float("nan"),
        "outlier_filter_bottom_upper_threshold_value": float("nan"),
        "outlier_filter_days_before": int(days_before),
        "outlier_filter_days_removed_by_top_only": 0,
        "outlier_filter_days_removed_by_bottom_only": 0,
        "outlier_filter_days_removed_by_both": 0,
        "outlier_filter_days_removed": 0,
        "outlier_filter_days_after": int(days_before),
    }
    if days_before == 0:
        return rows_in, meta

    if (not np.isfinite(threshold)) or threshold <= 0.0:
        meta["outlier_filter_reason"] = "disabled_invalid_threshold"
        return rows_in, meta

    def build_two_sided_z(values: np.ndarray) -> tuple[np.ndarray, dict[str, float | int | str]]:
        z = np.full((days_before,), np.nan, dtype=np.float64)
        info: dict[str, float | int | str] = {
            "finite_count": 0,
            "scale_method": "none",
            "center": float("nan"),
            "scale": float("nan"),
            "lower_threshold_value": float("nan"),
            "upper_threshold_value": float("nan"),
            "reason": "no_values",
        }
        finite_mask = np.isfinite(values)
        finite_count = int(np.sum(finite_mask))
        info["finite_count"] = int(finite_count)
        if finite_count < 2:
            info["reason"] = "insufficient_finite_values"
            return z, info

        center = float(np.median(values[finite_mask]))
        scale = float(np.median(np.abs(values[finite_mask] - center)))
        if np.isfinite(scale) and scale > 0.0:
            z[finite_mask] = (
                robust_scale_const * (values[finite_mask] - center) / scale
            )
            info["scale_method"] = "robust_mad"
            info["reason"] = "applied"
            info["center"] = float(center)
            info["scale"] = float(scale)
            info["lower_threshold_value"] = float(
                center - (threshold / robust_scale_const) * scale
            )
            info["upper_threshold_value"] = float(
                center + (threshold / robust_scale_const) * scale
            )
            return z, info

        center = float(np.mean(values[finite_mask]))
        scale = float(np.std(values[finite_mask], ddof=1))
        if (not np.isfinite(scale)) or scale <= 0.0:
            info["reason"] = "zero_or_nonfinite_scale"
            return z, info
        z[finite_mask] = (values[finite_mask] - center) / scale
        info["scale_method"] = "standard_std_fallback"
        info["reason"] = "applied"
        info["center"] = float(center)
        info["scale"] = float(scale)
        info["lower_threshold_value"] = float(center - threshold * scale)
        info["upper_threshold_value"] = float(center + threshold * scale)
        return z, info

    top = np.asarray(
        [
            float(
                r.get(
                    "top_ret_pct_mean",
                    r.get("top_ret_atr_mean", float("nan")),
                )
            )
            for r in rows_in
        ],
        dtype=np.float64,
    )
    bottom = np.asarray(
        [
            float(
                r.get(
                    "bottom_ret_pct_mean",
                    r.get("bottom_ret_atr_mean", float("nan")),
                )
            )
            for r in rows_in
        ],
        dtype=np.float64,
    )
    top_z, top_info = build_two_sided_z(top)
    bottom_z, bottom_info = build_two_sided_z(bottom)
    meta["outlier_filter_finite_top_count"] = int(top_info["finite_count"])
    meta["outlier_filter_finite_bottom_count"] = int(bottom_info["finite_count"])
    meta["outlier_filter_top_scale_method"] = str(top_info["scale_method"])
    meta["outlier_filter_bottom_scale_method"] = str(bottom_info["scale_method"])
    meta["outlier_filter_top_center_value"] = float(top_info["center"])
    meta["outlier_filter_top_scale_value"] = float(top_info["scale"])
    meta["outlier_filter_top_lower_threshold_value"] = float(
        top_info["lower_threshold_value"]
    )
    meta["outlier_filter_top_upper_threshold_value"] = float(
        top_info["upper_threshold_value"]
    )
    meta["outlier_filter_bottom_center_value"] = float(bottom_info["center"])
    meta["outlier_filter_bottom_scale_value"] = float(bottom_info["scale"])
    meta["outlier_filter_bottom_lower_threshold_value"] = float(
        bottom_info["lower_threshold_value"]
    )
    meta["outlier_filter_bottom_upper_threshold_value"] = float(
        bottom_info["upper_threshold_value"]
    )

    top_outlier = np.isfinite(top_z) & (np.abs(top_z) > threshold)
    bottom_outlier = np.isfinite(bottom_z) & (np.abs(bottom_z) > threshold)
    drop_mask = top_outlier | bottom_outlier
    keep_mask = ~drop_mask
    filtered_rows = [row for i, row in enumerate(rows_in) if bool(keep_mask[i])]

    removed = int(days_before - len(filtered_rows))
    removed_top_only = int(np.sum(top_outlier & (~bottom_outlier)))
    removed_bottom_only = int(np.sum((~top_outlier) & bottom_outlier))
    removed_both = int(np.sum(top_outlier & bottom_outlier))
    has_any_z = bool(np.any(np.isfinite(top_z) | np.isfinite(bottom_z)))
    meta["outlier_filter_enabled"] = int(has_any_z)
    if has_any_z:
        meta["outlier_filter_reason"] = "applied"
    else:
        meta["outlier_filter_reason"] = "insufficient_finite_top_and_bottom_values"
    meta["outlier_filter_days_removed_by_top_only"] = int(removed_top_only)
    meta["outlier_filter_days_removed_by_bottom_only"] = int(removed_bottom_only)
    meta["outlier_filter_days_removed_by_both"] = int(removed_both)
    meta["outlier_filter_days_removed"] = int(removed)
    meta["outlier_filter_days_after"] = int(len(filtered_rows))
    return filtered_rows, meta


def compute_daily_cross_sectional_metrics(
    pred: dict[str, np.ndarray | float],
    top_pct: float = DAILY_CROSS_SECTIONAL_TOP_PCT,
    bottom_pct: float = DAILY_CROSS_SECTIONAL_BOTTOM_PCT,
    min_per_side: int = DAILY_CROSS_SECTIONAL_MIN_PER_SIDE,
    min_names_per_day: int = DAILY_CROSS_SECTIONAL_MIN_NAMES_PER_DAY,
    annualization_days: float = DAILY_CROSS_SECTIONAL_ANNUALIZATION_DAYS,
    outlier_zscore_threshold: float = EVAL_TAIL_RET_OUTLIER_ZSCORE_THRESHOLD,
) -> tuple[dict[str, float | int], list[dict[str, float | int | str]]]:
    top_pct_v = resolve_daily_cross_sectional_tail_pct(top_pct, "top_pct")
    bottom_pct_v = resolve_daily_cross_sectional_tail_pct(bottom_pct, "bottom_pct")
    per_side_min = max(1, int(min_per_side))
    names_per_day_min = max(2, int(min_names_per_day))
    annualization = float(annualization_days)
    spread_compound_daily_capital_fraction = float(
        DAILY_CROSS_SECTIONAL_SPREAD_COMPOUND_DAILY_CAPITAL_FRACTION
    )
    _, default_outlier_filter_info = filter_eval_rows_by_tail_ret_zscore(
        [],
        zscore_threshold=outlier_zscore_threshold,
    )
    if not np.isfinite(annualization) or annualization <= 0.0:
        raise ValueError("annualization_days must be > 0")
    if (
        not np.isfinite(spread_compound_daily_capital_fraction)
        or spread_compound_daily_capital_fraction < 0.0
        or spread_compound_daily_capital_fraction > 1.0
    ):
        raise ValueError(
            "DAILY_CROSS_SECTIONAL_SPREAD_COMPOUND_DAILY_CAPITAL_FRACTION "
            "must be in [0,1]"
        )

    def build_payload(
        *,
        days_total: int,
        days_used: int,
        mean_top_count: float,
        mean_bottom_count: float,
        spread_mean: float,
        spread_median: float,
        spread_std: float,
        spread_sharpe: float,
        spread_hit_rate: float,
        mean_top_pct: float,
        spread_compounded_returns: float,
        top_sharpe: float,
        top_hit_rate: float,
        mean_bottom_pct: float,
        bottom_sharpe: float,
        bottom_hit_rate: float,
        bottom_runup5_ge60_days_used: int,
        bottom_runup5_ge60_mean_pct: float,
        bottom_runup5_ge60_sharpe: float,
        bottom_runup5_ge60_hit_rate: float,
        bottom_runup5_ge60_mean_count: float,
        bottom_runup5_ge60_trade_count: int,
        outlier_filter_info: dict[str, float | int | str],
    ) -> dict[str, float | int | str]:
        payload = {
            "top_percentile": float(top_pct_v),
            "bottom_percentile": float(bottom_pct_v),
            "top_pct": float(top_pct_v),
            "bottom_pct": float(bottom_pct_v),
            "min_per_side": int(per_side_min),
            "min_names_per_day": int(names_per_day_min),
            "annualization_days": float(annualization),
            "days_total": int(days_total),
            "days_used": int(days_used),
            "mean_top_count": float(mean_top_count),
            "mean_bottom_count": float(mean_bottom_count),
            "spread_mean": float(spread_mean),
            "spread_median": float(spread_median),
            "spread_std": float(spread_std),
            "spread_sharpe": float(spread_sharpe),
            "spread_sharpe_annualized": float(spread_sharpe),
            "spread_hit_rate": float(spread_hit_rate),
            "mean_top_pct": float(mean_top_pct),
            "spread_compounded_returns": float(spread_compounded_returns),
            "spread_compounded_daily_capital_fraction": float(
                spread_compound_daily_capital_fraction
            ),
            "top_sharpe": float(top_sharpe),
            "top_sharpe_annualized": float(top_sharpe),
            "top_hit_rate": float(top_hit_rate),
            "mean_bottom_pct": float(mean_bottom_pct),
            "bottom_sharpe": float(bottom_sharpe),
            "bottom_sharpe_annualized": float(bottom_sharpe),
            "bottom_hit_rate": float(bottom_hit_rate),
            "bottom_runup5_ge60_lookback_days": int(
                DAILY_CROSS_SECTIONAL_BOTTOM_RUNUP_LOOKBACK_DAYS
            ),
            "bottom_runup5_ge60_threshold_pct": float(
                DAILY_CROSS_SECTIONAL_BOTTOM_RUNUP_MIN_PCT
            ),
            "bottom_runup5_ge60_days_used": int(bottom_runup5_ge60_days_used),
            "bottom_runup5_ge60_mean_count": float(bottom_runup5_ge60_mean_count),
            "bottom_runup5_ge60_trade_count": int(bottom_runup5_ge60_trade_count),
            "bottom_runup5_ge60_mean_pct": float(bottom_runup5_ge60_mean_pct),
            "bottom_runup5_ge60_sharpe": float(bottom_runup5_ge60_sharpe),
            "bottom_runup5_ge60_sharpe_annualized": float(bottom_runup5_ge60_sharpe),
            "bottom_runup5_ge60_hit_rate": float(bottom_runup5_ge60_hit_rate),
            # Backward-compatible aliases.
            "top_ret_pct_mean": float(mean_top_pct),
            "bottom_ret_pct_mean": float(mean_bottom_pct),
        }
        payload.update(outlier_filter_info)
        payload["outlier_filter_zscore_used"] = float(
            outlier_filter_info.get("outlier_filter_zscore_threshold", float("nan"))
        )
        return payload

    prob = np.asarray(pred["prob"], dtype=np.float64)
    if "ret_pct_true" in pred:
        ret_target = np.asarray(pred["ret_pct_true"], dtype=np.float64)
    else:
        # Backward-compatible fallback for older prediction payloads.
        ret_target = np.asarray(pred["ret_atr_true"], dtype=np.float64)
    timestamps = np.asarray(pred.get("timestamps", np.empty((0,), dtype=object)), dtype=object)
    if prob.ndim != 2 or prob.shape[1] != 2:
        raise ValueError("prob must have shape (n,2)")
    n = int(prob.shape[0])
    if n == 0:
        payload = build_payload(
            days_total=0,
            days_used=0,
            mean_top_count=float("nan"),
            mean_bottom_count=float("nan"),
            spread_mean=float("nan"),
            spread_median=float("nan"),
            spread_std=float("nan"),
            spread_sharpe=float("nan"),
            spread_hit_rate=float("nan"),
            mean_top_pct=float("nan"),
            spread_compounded_returns=float("nan"),
            top_sharpe=float("nan"),
            top_hit_rate=float("nan"),
            mean_bottom_pct=float("nan"),
            bottom_sharpe=float("nan"),
            bottom_hit_rate=float("nan"),
            bottom_runup5_ge60_days_used=0,
            bottom_runup5_ge60_mean_pct=float("nan"),
            bottom_runup5_ge60_sharpe=float("nan"),
            bottom_runup5_ge60_hit_rate=float("nan"),
            bottom_runup5_ge60_mean_count=float("nan"),
            bottom_runup5_ge60_trade_count=0,
            outlier_filter_info=default_outlier_filter_info,
        )
        return payload, []
    if int(timestamps.shape[0]) != n:
        raise ValueError("timestamps/prob length mismatch")
    raw_tickers = pred.get("tickers")
    tickers: np.ndarray | None = None
    if raw_tickers is not None:
        tickers_arr = np.asarray(raw_tickers, dtype=object).reshape(-1)
        if int(tickers_arr.shape[0]) == n:
            tickers = tickers_arr

    p1 = prob[:, 1]
    dates = np.asarray([str(x)[:10] for x in timestamps], dtype=object)
    valid = np.isfinite(p1) & np.isfinite(ret_target) & (dates != "")
    p1 = p1[valid]
    ret = ret_target[valid]
    dates = dates[valid]
    if tickers is not None:
        tickers = tickers[valid]
    if p1.size == 0:
        payload = build_payload(
            days_total=0,
            days_used=0,
            mean_top_count=float("nan"),
            mean_bottom_count=float("nan"),
            spread_mean=float("nan"),
            spread_median=float("nan"),
            spread_std=float("nan"),
            spread_sharpe=float("nan"),
            spread_hit_rate=float("nan"),
            mean_top_pct=float("nan"),
            spread_compounded_returns=float("nan"),
            top_sharpe=float("nan"),
            top_hit_rate=float("nan"),
            mean_bottom_pct=float("nan"),
            bottom_sharpe=float("nan"),
            bottom_hit_rate=float("nan"),
            bottom_runup5_ge60_days_used=0,
            bottom_runup5_ge60_mean_pct=float("nan"),
            bottom_runup5_ge60_sharpe=float("nan"),
            bottom_runup5_ge60_hit_rate=float("nan"),
            bottom_runup5_ge60_mean_count=float("nan"),
            bottom_runup5_ge60_trade_count=0,
            outlier_filter_info=default_outlier_filter_info,
        )
        return payload, []

    unique_dates = np.unique(dates)
    rows: list[dict[str, float | int | str]] = []
    for d in sorted(unique_dates.tolist()):
        idx = np.where(dates == d)[0]
        day_n = int(idx.size)
        if day_n < names_per_day_min:
            continue
        day_p = p1[idx]
        day_r = ret[idx]
        order = np.argsort(day_p, kind="mergesort")
        top_count = max(per_side_min, int(np.ceil(day_n * (top_pct_v / 100.0))))
        bottom_count = max(per_side_min, int(np.ceil(day_n * (bottom_pct_v / 100.0))))
        max_side = day_n // 2
        if max_side < 1:
            continue
        top_count = min(top_count, max_side)
        bottom_count = min(bottom_count, max_side)
        top_idx = order[-top_count:]
        bottom_idx = order[:bottom_count]
        top_mean = float(np.mean(day_r[top_idx]))
        bottom_mean = float(np.mean(day_r[bottom_idx]))
        bottom_runup5_ge60_count, bottom_runup5_ge60_mean = compute_selected_runup_subset_metrics(
            dates=dates,
            tickers=tickers,
            returns=ret,
            selected_indices=idx[bottom_idx],
            lookback_days=DAILY_CROSS_SECTIONAL_BOTTOM_RUNUP_LOOKBACK_DAYS,
            min_runup_pct=DAILY_CROSS_SECTIONAL_BOTTOM_RUNUP_MIN_PCT,
        )
        spread = float(top_mean - bottom_mean)
        rows.append(
            {
                "date": str(d),
                "count": int(day_n),
                "top_count": int(top_count),
                "bottom_count": int(bottom_count),
                "top_prob_min": float(np.min(day_p[top_idx])),
                "top_prob_max": float(np.max(day_p[top_idx])),
                "bottom_prob_min": float(np.min(day_p[bottom_idx])),
                "bottom_prob_max": float(np.max(day_p[bottom_idx])),
                "top_ret_pct_mean": float(top_mean),
                "bottom_ret_pct_mean": float(bottom_mean),
                "bottom_runup5_ge60_count": int(bottom_runup5_ge60_count),
                "bottom_runup5_ge60_ret_pct_mean": float(bottom_runup5_ge60_mean),
                "spread": float(spread),
            }
        )

    rows, outlier_filter_info = filter_eval_rows_by_tail_ret_zscore(
        rows,
        zscore_threshold=outlier_zscore_threshold,
    )
    spread_compounded_returns = float("nan")
    if rows:
        spread_trade_returns: list[float] = []
        for row in rows:
            try:
                ret_val = float(row.get("spread", float("nan")))
            except (TypeError, ValueError):
                ret_val = float("nan")
            if np.isfinite(ret_val):
                spread_trade_returns.append(float(ret_val))
        if spread_trade_returns:
            arr = np.asarray(spread_trade_returns, dtype=np.float64)
            arr = arr * float(spread_compound_daily_capital_fraction)
            compounded_multiple = float(np.prod(1.0 + arr))
            spread_compounded_returns = float((compounded_multiple - 1.0) * 100.0)

    days_used = int(len(rows))
    if days_used == 0:
        payload = build_payload(
            days_total=int(unique_dates.size),
            days_used=0,
            mean_top_count=float("nan"),
            mean_bottom_count=float("nan"),
            spread_mean=float("nan"),
            spread_median=float("nan"),
            spread_std=float("nan"),
            spread_sharpe=float("nan"),
            spread_hit_rate=float("nan"),
            mean_top_pct=float("nan"),
            spread_compounded_returns=float(spread_compounded_returns),
            top_sharpe=float("nan"),
            top_hit_rate=float("nan"),
            mean_bottom_pct=float("nan"),
            bottom_sharpe=float("nan"),
            bottom_hit_rate=float("nan"),
            bottom_runup5_ge60_days_used=0,
            bottom_runup5_ge60_mean_pct=float("nan"),
            bottom_runup5_ge60_sharpe=float("nan"),
            bottom_runup5_ge60_hit_rate=float("nan"),
            bottom_runup5_ge60_mean_count=float("nan"),
            bottom_runup5_ge60_trade_count=0,
            outlier_filter_info=outlier_filter_info,
        )
        return payload, rows

    top_count_arr = np.asarray([r["top_count"] for r in rows], dtype=np.float64)
    bottom_count_arr = np.asarray([r["bottom_count"] for r in rows], dtype=np.float64)
    top_mean_arr = np.asarray([r["top_ret_pct_mean"] for r in rows], dtype=np.float64)
    bottom_mean_arr = np.asarray([r["bottom_ret_pct_mean"] for r in rows], dtype=np.float64)
    spread_arr = np.asarray([r["spread"] for r in rows], dtype=np.float64)

    spread_mean = float(np.mean(spread_arr)) if spread_arr.size > 0 else float("nan")
    spread_median = float(np.median(spread_arr)) if spread_arr.size > 0 else float("nan")
    spread_std = float(np.std(spread_arr, ddof=1)) if spread_arr.size > 1 else float("nan")
    spread_sharpe = float("nan")
    if np.isfinite(spread_std) and spread_std > 0.0:
        spread_sharpe = float((spread_mean / spread_std) * np.sqrt(annualization))

    top_finite_mask = np.isfinite(top_mean_arr)
    bottom_finite_mask = np.isfinite(bottom_mean_arr)
    top_positions = float(np.sum(top_count_arr[top_finite_mask]))
    bottom_positions = float(np.sum(bottom_count_arr[bottom_finite_mask]))
    mean_top_pct = (
        float(np.sum(top_mean_arr[top_finite_mask] * top_count_arr[top_finite_mask]) / top_positions)
        if top_positions > 0.0
        else float("nan")
    )
    mean_bottom_pct = (
        float(
            np.sum(bottom_mean_arr[bottom_finite_mask] * bottom_count_arr[bottom_finite_mask])
            / bottom_positions
        )
        if bottom_positions > 0.0
        else float("nan")
    )

    top_vals = top_mean_arr[top_finite_mask]
    bottom_vals = bottom_mean_arr[bottom_finite_mask]
    bottom_runup5_ge60_count_arr = np.asarray(
        [r.get("bottom_runup5_ge60_count", 0) for r in rows], dtype=np.float64
    )
    bottom_runup5_ge60_mean_arr = np.asarray(
        [r.get("bottom_runup5_ge60_ret_pct_mean", float("nan")) for r in rows],
        dtype=np.float64,
    )
    top_std = float(np.std(top_vals, ddof=1)) if top_vals.size > 1 else float("nan")
    bottom_std = (
        float(np.std(bottom_vals, ddof=1)) if bottom_vals.size > 1 else float("nan")
    )
    top_sharpe = float("nan")
    if np.isfinite(top_std) and top_std > 0.0:
        top_sharpe = float((float(np.mean(top_vals)) / top_std) * np.sqrt(annualization))
    bottom_sharpe = float("nan")
    if np.isfinite(bottom_std) and bottom_std > 0.0:
        bottom_sharpe = float(
            (float(np.mean(bottom_vals)) / bottom_std) * np.sqrt(annualization)
        )
    top_hit_rate = float(np.mean(top_vals > 0.0)) if top_vals.size > 0 else float("nan")
    bottom_hit_rate = (
        float(np.mean(bottom_vals < 0.0)) if bottom_vals.size > 0 else float("nan")
    )
    bottom_runup5_ge60_finite_mask = np.isfinite(bottom_runup5_ge60_mean_arr) & (
        bottom_runup5_ge60_count_arr > 0.0
    )
    bottom_runup5_ge60_vals = bottom_runup5_ge60_mean_arr[bottom_runup5_ge60_finite_mask]
    bottom_runup5_ge60_positions = float(
        np.sum(bottom_runup5_ge60_count_arr[bottom_runup5_ge60_finite_mask])
    )
    bottom_runup5_ge60_mean_pct = (
        float(
            np.sum(
                bottom_runup5_ge60_mean_arr[bottom_runup5_ge60_finite_mask]
                * bottom_runup5_ge60_count_arr[bottom_runup5_ge60_finite_mask]
            )
            / bottom_runup5_ge60_positions
        )
        if bottom_runup5_ge60_positions > 0.0
        else float("nan")
    )
    bottom_runup5_ge60_std = (
        float(np.std(bottom_runup5_ge60_vals, ddof=1))
        if bottom_runup5_ge60_vals.size > 1
        else float("nan")
    )
    bottom_runup5_ge60_sharpe = float("nan")
    if np.isfinite(bottom_runup5_ge60_std) and bottom_runup5_ge60_std > 0.0:
        bottom_runup5_ge60_sharpe = float(
            (float(np.mean(bottom_runup5_ge60_vals)) / bottom_runup5_ge60_std)
            * np.sqrt(annualization)
        )
    bottom_runup5_ge60_hit_rate = (
        float(np.mean(bottom_runup5_ge60_vals < 0.0))
        if bottom_runup5_ge60_vals.size > 0
        else float("nan")
    )

    payload = build_payload(
        days_total=int(unique_dates.size),
        days_used=int(days_used),
        mean_top_count=float(np.mean(top_count_arr)),
        mean_bottom_count=float(np.mean(bottom_count_arr)),
        spread_mean=float(spread_mean),
        spread_median=float(spread_median),
        spread_std=float(spread_std),
        spread_sharpe=float(spread_sharpe),
        spread_hit_rate=float(np.mean(spread_arr > 0.0)),
        mean_top_pct=float(mean_top_pct),
        spread_compounded_returns=float(spread_compounded_returns),
        top_sharpe=float(top_sharpe),
        top_hit_rate=float(top_hit_rate),
        mean_bottom_pct=float(mean_bottom_pct),
        bottom_sharpe=float(bottom_sharpe),
        bottom_hit_rate=float(bottom_hit_rate),
        bottom_runup5_ge60_days_used=int(bottom_runup5_ge60_vals.size),
        bottom_runup5_ge60_mean_pct=float(bottom_runup5_ge60_mean_pct),
        bottom_runup5_ge60_sharpe=float(bottom_runup5_ge60_sharpe),
        bottom_runup5_ge60_hit_rate=float(bottom_runup5_ge60_hit_rate),
        bottom_runup5_ge60_mean_count=float(
            np.mean(bottom_runup5_ge60_count_arr[bottom_runup5_ge60_finite_mask])
        )
        if bottom_runup5_ge60_vals.size > 0
        else float("nan"),
        bottom_runup5_ge60_trade_count=int(bottom_runup5_ge60_positions),
        outlier_filter_info=outlier_filter_info,
    )
    return payload, rows


def extract_daily_cross_section_topline_eval_stats(
    payload: dict[str, float | int | str],
) -> dict[str, float | int]:
    keys = [
        "top_percentile",
        "bottom_percentile",
        "mean_top_count",
        "mean_bottom_count",
        "spread_mean",
        "spread_median",
        "spread_std",
        "spread_sharpe",
        "spread_hit_rate",
        "mean_top_pct",
        "spread_compounded_returns",
        "spread_compounded_daily_capital_fraction",
        "top_sharpe",
        "top_hit_rate",
        "mean_bottom_pct",
        "bottom_sharpe",
        "bottom_hit_rate",
        "bottom_runup5_ge60_days_used",
        "bottom_runup5_ge60_mean_count",
        "bottom_runup5_ge60_trade_count",
        "bottom_runup5_ge60_mean_pct",
        "bottom_runup5_ge60_sharpe",
        "bottom_runup5_ge60_hit_rate",
        "outlier_filter_enabled",
        "outlier_filter_zscore_used",
        "outlier_filter_top_lower_threshold_value",
        "outlier_filter_top_upper_threshold_value",
        "outlier_filter_bottom_lower_threshold_value",
        "outlier_filter_bottom_upper_threshold_value",
        "outlier_filter_days_removed",
    ]
    out: dict[str, float | int] = {}
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float, np.integer, np.floating)):
            out[key] = int(value) if isinstance(value, (int, np.integer)) else float(value)
        else:
            out[key] = float("nan")
    return out


def write_daily_cross_sectional_metrics_best(
    out_dir: Path,
    best_epoch: int,
    payload: dict[str, float | int],
    rows: list[dict[str, float | int | str]],
    file_stem: str | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(file_stem).strip() if file_stem is not None else ""
    if stem:
        csv_path = out_dir / f"{stem}.csv"
        json_path = out_dir / f"{stem}.json"
    else:
        csv_path = out_dir / "val_daily_cross_sectional_metrics_best_epoch.csv"
        json_path = out_dir / "daily_cross_section_topline.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "best_epoch",
                "date",
                "count",
                "top_count",
                "bottom_count",
                "top_prob_min",
                "top_prob_max",
                "bottom_prob_min",
                "bottom_prob_max",
                "top_ret_pct_mean",
                "bottom_ret_pct_mean",
                "bottom_runup5_ge60_count",
                "bottom_runup5_ge60_ret_pct_mean",
                "spread",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    int(best_epoch),
                    str(row["date"]),
                    int(row["count"]),
                    int(row["top_count"]),
                    int(row["bottom_count"]),
                    float(row["top_prob_min"]),
                    float(row["top_prob_max"]),
                    float(row["bottom_prob_min"]),
                    float(row["bottom_prob_max"]),
                    float(row["top_ret_pct_mean"]),
                    float(row["bottom_ret_pct_mean"]),
                    int(row.get("bottom_runup5_ge60_count", 0)),
                    float(row.get("bottom_runup5_ge60_ret_pct_mean", float("nan"))),
                    float(row["spread"]),
                ]
            )

    topline_payload = extract_daily_cross_section_topline_eval_stats(payload)
    json_payload = {"best_epoch": int(best_epoch), **topline_payload, "rows": rows}
    json_path.write_text(json.dumps(json_payload, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def compute_walkforward_rolling_threshold_metrics(
    pred: dict[str, np.ndarray | float],
    enabled: bool = bool(WALKFORWARD_ROLLING_THRESHOLD_ENABLED),
    top_pct: float = WALKFORWARD_ROLLING_TOP_PCT,
    bottom_pct: float = WALKFORWARD_ROLLING_BOTTOM_PCT,
    lookback_days: int = WALKFORWARD_ROLLING_LOOKBACK_DAYS,
    min_history_days: int = WALKFORWARD_ROLLING_MIN_HISTORY_DAYS,
    min_per_side: int = WALKFORWARD_ROLLING_MIN_PER_SIDE,
    min_per_side_mode: str = WALKFORWARD_ROLLING_MIN_PER_SIDE_MODE,
    min_names_per_day: int = WALKFORWARD_ROLLING_MIN_NAMES_PER_DAY,
    annualization_days: float = WALKFORWARD_ROLLING_ANNUALIZATION_DAYS,
    threshold_method: str = WALKFORWARD_ROLLING_THRESHOLD_METHOD,
    fallback_to_daily_rank: bool = bool(WALKFORWARD_ROLLING_FALLBACK_TO_DAILY_RANK),
    enforce_non_overlap: bool = bool(WALKFORWARD_ROLLING_ENFORCE_NON_OVERLAP),
    outlier_zscore_threshold: float = EVAL_TAIL_RET_OUTLIER_ZSCORE_THRESHOLD,
    spy_non_overlap_enabled: bool = bool(WALKFORWARD_SPY_NON_OVERLAP_ENABLED),
    spy_non_overlap_horizon_days: int = WALKFORWARD_SPY_NON_OVERLAP_HORIZON_DAYS,
    spy_daily_csv: str = WALKFORWARD_SPY_DAILY_CSV_PATH,
    spy_date_col: str = WALKFORWARD_SPY_DATE_COL,
    spy_close_col: str = WALKFORWARD_SPY_CLOSE_COL,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    enabled_v = bool(enabled)
    top_pct_v = resolve_daily_cross_sectional_tail_pct(top_pct, "top_pct")
    bottom_pct_v = resolve_daily_cross_sectional_tail_pct(bottom_pct, "bottom_pct")
    lookback_days_v = max(1, int(lookback_days))
    min_history_days_v = max(1, int(min_history_days))
    if min_history_days_v > lookback_days_v:
        min_history_days_v = lookback_days_v
    # Allow zero so top/bottom can be empty on a day.
    per_side_min = max(0, int(min_per_side))
    per_side_mode_v = resolve_walkforward_min_per_side_mode(min_per_side_mode)
    names_per_day_min = max(2, int(min_names_per_day))
    annualization = float(annualization_days)
    threshold_method_v = resolve_walkforward_threshold_method(threshold_method)
    fallback_rank_v = bool(fallback_to_daily_rank)
    enforce_non_overlap_v = bool(enforce_non_overlap)
    spy_non_overlap_enabled_v = bool(spy_non_overlap_enabled)
    spy_non_overlap_horizon_days_v = max(1, int(spy_non_overlap_horizon_days))
    spy_daily_csv_v = str(spy_daily_csv).strip()
    spy_date_col_v = str(spy_date_col).strip()
    spy_close_col_v = str(spy_close_col).strip()
    _, default_outlier_filter_info = filter_eval_rows_by_tail_ret_zscore(
        [],
        zscore_threshold=outlier_zscore_threshold,
    )
    default_spy_non_overlap_info = build_walkforward_spy_non_overlap_default_info(
        enabled=spy_non_overlap_enabled_v,
        horizon_days=spy_non_overlap_horizon_days_v,
    )
    default_real_non_overlap_info = build_walkforward_real_non_overlap_default_info(
        horizon_days=spy_non_overlap_horizon_days_v,
    )
    if not np.isfinite(annualization) or annualization <= 0.0:
        raise ValueError("annualization_days must be > 0")

    def passes_per_side_min(top_count: int, bottom_count: int) -> bool:
        if per_side_min <= 0:
            return True
        if per_side_mode_v == "both":
            return top_count >= per_side_min and bottom_count >= per_side_min
        return top_count >= per_side_min or bottom_count >= per_side_min

    def build_payload(
        *,
        days_total: int,
        days_used: int,
        days_skipped_insufficient_history: int,
        days_skipped_min_names: int,
        days_skipped_min_per_side: int,
        days_used_rank_fallback: int,
        days_with_overlap_before_resolution: int,
        mean_history_count: float,
        mean_top_count: float,
        mean_bottom_count: float,
        mean_upper_threshold: float,
        mean_lower_threshold: float,
        mean_overlap_count_before_resolution: float,
        spread_mean: float,
        spread_median: float,
        spread_std: float,
        spread_sharpe_annualized: float,
        spread_hit_rate: float,
        outlier_filter_info: dict[str, float | int | str],
        spy_non_overlap_info: dict[str, float | int | str] | None = None,
        real_non_overlap_info: dict[str, float | int] | None = None,
        top_days_used: int = 0,
        bottom_days_used: int = 0,
        paired_days_used: int = 0,
        mean_top_pct: float = float("nan"),
        top_sharpe: float = float("nan"),
        top_hit_rate: float = float("nan"),
        mean_bottom_pct: float = float("nan"),
        bottom_sharpe: float = float("nan"),
        bottom_hit_rate: float = float("nan"),
        paired_mean_top_pct: float = float("nan"),
        paired_mean_bottom_pct: float = float("nan"),
        single_pick_mean_top_pct: float = float("nan"),
        single_pick_mean_bottom_pct: float = float("nan"),
        single_pick_top_sharpe: float = float("nan"),
        single_pick_top_days_used: int = 0,
        single_pick_bottom_days_used: int = 0,
        single_pick_bottom_sharpe: float = float("nan"),
        bottom_runup5_ge30_days_used: int = 0,
        bottom_runup5_ge30_mean_pct: float = float("nan"),
        bottom_runup5_ge30_sharpe: float = float("nan"),
        bottom_runup5_ge30_hit_rate: float = float("nan"),
        bottom_runup5_ge30_mean_count: float = float("nan"),
        bottom_runup5_ge30_trade_count: int = 0,
    ) -> dict[str, float | int | str]:
        payload = {
            "enabled": int(enabled_v),
            "top_percentile": float(top_pct_v),
            "bottom_percentile": float(bottom_pct_v),
            "top_pct": float(top_pct_v),
            "bottom_pct": float(bottom_pct_v),
            "lookback_days": int(lookback_days_v),
            "min_history_days": int(min_history_days_v),
            "min_per_side": int(per_side_min),
            "min_per_side_mode": str(per_side_mode_v),
            "min_names_per_day": int(names_per_day_min),
            "annualization_days": float(annualization),
            "threshold_method": str(threshold_method_v),
            "fallback_to_daily_rank": int(fallback_rank_v),
            "enforce_non_overlap": int(enforce_non_overlap_v),
            "days_total": int(days_total),
            "days_used": int(days_used),
            "days_skipped_insufficient_history": int(days_skipped_insufficient_history),
            "days_skipped_min_names": int(days_skipped_min_names),
            "days_skipped_min_per_side": int(days_skipped_min_per_side),
            "days_used_rank_fallback": int(days_used_rank_fallback),
            "days_with_overlap_before_resolution": int(days_with_overlap_before_resolution),
            "mean_history_count": float(mean_history_count),
            "mean_top_count": float(mean_top_count),
            "mean_bottom_count": float(mean_bottom_count),
            "mean_upper_threshold": float(mean_upper_threshold),
            "mean_lower_threshold": float(mean_lower_threshold),
            "mean_overlap_count_before_resolution": float(
                mean_overlap_count_before_resolution
            ),
            "spread_mean": float(spread_mean),
            "spread_median": float(spread_median),
            "spread_std": float(spread_std),
            "spread_sharpe": float(spread_sharpe_annualized),
            "spread_sharpe_annualized": float(spread_sharpe_annualized),
            "spread_hit_rate": float(spread_hit_rate),
            "top_days_used": int(top_days_used),
            "bottom_days_used": int(bottom_days_used),
            "paired_days_used": int(paired_days_used),
            "mean_top_pct": float(mean_top_pct),
            "top_sharpe": float(top_sharpe),
            "top_hit_rate": float(top_hit_rate),
            "mean_bottom_pct": float(mean_bottom_pct),
            "bottom_sharpe": float(bottom_sharpe),
            "bottom_hit_rate": float(bottom_hit_rate),
            "paired_mean_top_pct": float(paired_mean_top_pct),
            "paired_mean_bottom_pct": float(paired_mean_bottom_pct),
            "single_pick_mean_top_pct": float(single_pick_mean_top_pct),
            "single_pick_mean_bottom_pct": float(single_pick_mean_bottom_pct),
            "single_pick_top_sharpe": float(single_pick_top_sharpe),
            "single_pick_top_days_used": int(single_pick_top_days_used),
            "single_pick_bottom_days_used": int(single_pick_bottom_days_used),
            "single_pick_bottom_sharpe": float(single_pick_bottom_sharpe),
            "bottom_runup5_ge30_lookback_days": int(
                WALKFORWARD_BOTTOM_RUNUP_LOOKBACK_DAYS
            ),
            "bottom_runup5_ge30_threshold_pct": float(
                WALKFORWARD_BOTTOM_RUNUP_MIN_PCT
            ),
            "bottom_runup5_ge30_days_used": int(bottom_runup5_ge30_days_used),
            "bottom_runup5_ge30_mean_count": float(bottom_runup5_ge30_mean_count),
            "bottom_runup5_ge30_trade_count": int(bottom_runup5_ge30_trade_count),
            "bottom_runup5_ge30_mean_pct": float(bottom_runup5_ge30_mean_pct),
            "bottom_runup5_ge30_sharpe": float(bottom_runup5_ge30_sharpe),
            "bottom_runup5_ge30_sharpe_annualized": float(
                bottom_runup5_ge30_sharpe
            ),
            "bottom_runup5_ge30_hit_rate": float(bottom_runup5_ge30_hit_rate),
        }
        payload.update(outlier_filter_info)
        payload["outlier_filter_zscore_used"] = float(
            outlier_filter_info.get("outlier_filter_zscore_threshold", float("nan"))
        )
        payload.update(
            spy_non_overlap_info
            if spy_non_overlap_info is not None
            else default_spy_non_overlap_info
        )
        payload.update(
            real_non_overlap_info
            if real_non_overlap_info is not None
            else default_real_non_overlap_info
        )
        return payload

    prob = np.asarray(pred["prob"], dtype=np.float64)
    if "ret_pct_true" in pred:
        ret_target = np.asarray(pred["ret_pct_true"], dtype=np.float64)
    else:
        # Backward-compatible fallback for older prediction payloads.
        ret_target = np.asarray(pred["ret_atr_true"], dtype=np.float64)
    timestamps = np.asarray(pred.get("timestamps", np.empty((0,), dtype=object)), dtype=object)
    if prob.ndim != 2 or prob.shape[1] != 2:
        raise ValueError("prob must have shape (n,2)")
    n = int(prob.shape[0])
    if n == 0:
        payload = build_payload(
            days_total=0,
            days_used=0,
            days_skipped_insufficient_history=0,
            days_skipped_min_names=0,
            days_skipped_min_per_side=0,
            days_used_rank_fallback=0,
            days_with_overlap_before_resolution=0,
            mean_history_count=float("nan"),
            mean_top_count=float("nan"),
            mean_bottom_count=float("nan"),
            mean_upper_threshold=float("nan"),
            mean_lower_threshold=float("nan"),
            mean_overlap_count_before_resolution=float("nan"),
            spread_mean=float("nan"),
            spread_median=float("nan"),
            spread_std=float("nan"),
            spread_sharpe_annualized=float("nan"),
            spread_hit_rate=float("nan"),
            outlier_filter_info=default_outlier_filter_info,
        )
        return payload, []
    if int(timestamps.shape[0]) != n:
        raise ValueError("timestamps/prob length mismatch")
    raw_tickers = pred.get("tickers")
    tickers: np.ndarray | None = None
    if raw_tickers is not None:
        tickers_arr = np.asarray(raw_tickers, dtype=object).reshape(-1)
        if int(tickers_arr.shape[0]) == n:
            tickers = tickers_arr

    p1 = prob[:, 1]
    dates = np.asarray([str(x)[:10] for x in timestamps], dtype=object)
    valid = np.isfinite(p1) & np.isfinite(ret_target) & (dates != "")
    p1 = p1[valid]
    ret = ret_target[valid]
    dates = dates[valid]
    if tickers is not None:
        tickers = tickers[valid]
    if p1.size == 0:
        payload = build_payload(
            days_total=0,
            days_used=0,
            days_skipped_insufficient_history=0,
            days_skipped_min_names=0,
            days_skipped_min_per_side=0,
            days_used_rank_fallback=0,
            days_with_overlap_before_resolution=0,
            mean_history_count=float("nan"),
            mean_top_count=float("nan"),
            mean_bottom_count=float("nan"),
            mean_upper_threshold=float("nan"),
            mean_lower_threshold=float("nan"),
            mean_overlap_count_before_resolution=float("nan"),
            spread_mean=float("nan"),
            spread_median=float("nan"),
            spread_std=float("nan"),
            spread_sharpe_annualized=float("nan"),
            spread_hit_rate=float("nan"),
            outlier_filter_info=default_outlier_filter_info,
        )
        return payload, []

    unique_dates = sorted(np.unique(dates).tolist())
    if not enabled_v:
        payload = build_payload(
            days_total=int(len(unique_dates)),
            days_used=0,
            days_skipped_insufficient_history=0,
            days_skipped_min_names=0,
            days_skipped_min_per_side=0,
            days_used_rank_fallback=0,
            days_with_overlap_before_resolution=0,
            mean_history_count=float("nan"),
            mean_top_count=float("nan"),
            mean_bottom_count=float("nan"),
            mean_upper_threshold=float("nan"),
            mean_lower_threshold=float("nan"),
            mean_overlap_count_before_resolution=float("nan"),
            spread_mean=float("nan"),
            spread_median=float("nan"),
            spread_std=float("nan"),
            spread_sharpe_annualized=float("nan"),
            spread_hit_rate=float("nan"),
            outlier_filter_info=default_outlier_filter_info,
        )
        return payload, []

    day_index_map: dict[str, np.ndarray] = {}
    for d in unique_dates:
        day_index_map[str(d)] = np.where(dates == d)[0]

    rows: list[dict[str, float | int | str]] = []
    skipped_history = 0
    skipped_min_names = 0
    skipped_min_per_side = 0
    days_used_rank_fallback = 0
    days_with_overlap_before_resolution = 0

    for i, day in enumerate(unique_dates):
        day_str = str(day)
        day_idx = day_index_map[day_str]
        day_n = int(day_idx.size)
        if day_n < names_per_day_min:
            skipped_min_names += 1
            continue

        history_days = unique_dates[max(0, i - lookback_days_v) : i]
        if len(history_days) < min_history_days_v:
            skipped_history += 1
            continue
        history_idx_parts = [day_index_map[str(d)] for d in history_days]
        if not history_idx_parts:
            skipped_history += 1
            continue
        history_idx = np.concatenate(history_idx_parts, axis=0)
        history_count = int(history_idx.size)
        if history_count <= 0:
            skipped_history += 1
            continue

        if threshold_method_v == "pooled":
            hist_p = p1[history_idx]
            lower_thr = float(np.percentile(hist_p, bottom_pct_v))
            upper_thr = float(np.percentile(hist_p, 100.0 - top_pct_v))
        else:
            daily_upper_vals: list[float] = []
            daily_lower_vals: list[float] = []
            for hd in history_days:
                hidx = day_index_map[str(hd)]
                if hidx.size == 0:
                    continue
                hp = p1[hidx]
                daily_lower_vals.append(float(np.percentile(hp, bottom_pct_v)))
                daily_upper_vals.append(float(np.percentile(hp, 100.0 - top_pct_v)))
            if not daily_upper_vals or not daily_lower_vals:
                skipped_history += 1
                continue
            if threshold_method_v == "daily_mean":
                lower_thr = float(np.mean(np.asarray(daily_lower_vals, dtype=np.float64)))
                upper_thr = float(np.mean(np.asarray(daily_upper_vals, dtype=np.float64)))
            else:
                lower_thr = float(np.median(np.asarray(daily_lower_vals, dtype=np.float64)))
                upper_thr = float(np.median(np.asarray(daily_upper_vals, dtype=np.float64)))

        day_p = p1[day_idx]
        day_r = ret[day_idx]
        top_mask = day_p >= upper_thr
        bottom_mask = day_p <= lower_thr
        overlap_before = int(np.sum(top_mask & bottom_mask))
        if overlap_before > 0:
            days_with_overlap_before_resolution += 1
        if enforce_non_overlap_v and overlap_before > 0:
            top_mask = day_p > upper_thr
            bottom_mask = day_p < lower_thr

        top_count = int(np.sum(top_mask))
        bottom_count = int(np.sum(bottom_mask))
        selection_mode = "threshold"
        meets_per_side_min = passes_per_side_min(top_count, bottom_count)
        need_rank_fallback = (
            (not meets_per_side_min)
            or (enforce_non_overlap_v and bool(np.any(top_mask & bottom_mask)))
        )
        if need_rank_fallback:
            if not fallback_rank_v:
                skipped_min_per_side += 1
                continue
            max_side = day_n // 2
            if max_side < 1:
                skipped_min_per_side += 1
                continue
            rank_top_count = max(per_side_min, int(np.ceil(day_n * (top_pct_v / 100.0))))
            rank_bottom_count = max(
                per_side_min, int(np.ceil(day_n * (bottom_pct_v / 100.0)))
            )
            rank_top_count = min(rank_top_count, max_side)
            rank_bottom_count = min(rank_bottom_count, max_side)
            order = np.argsort(day_p, kind="mergesort")
            top_idx_rank = order[-rank_top_count:]
            bottom_idx_rank = order[:rank_bottom_count]
            top_mask = np.zeros(day_n, dtype=bool)
            bottom_mask = np.zeros(day_n, dtype=bool)
            top_mask[top_idx_rank] = True
            bottom_mask[bottom_idx_rank] = True
            top_count = int(np.sum(top_mask))
            bottom_count = int(np.sum(bottom_mask))
            selection_mode = "rank_fallback"
            days_used_rank_fallback += 1
            if not passes_per_side_min(top_count, bottom_count):
                skipped_min_per_side += 1
                continue

        top_ret = day_r[top_mask]
        bottom_ret = day_r[bottom_mask]
        top_mean = float(np.mean(top_ret)) if int(top_ret.size) > 0 else float("nan")
        bottom_mean = (
            float(np.mean(bottom_ret)) if int(bottom_ret.size) > 0 else float("nan")
        )
        bottom_runup5_ge30_count, bottom_runup5_ge30_mean = compute_selected_runup_subset_metrics(
            dates=dates,
            tickers=tickers,
            returns=ret,
            selected_indices=day_idx[bottom_mask],
            lookback_days=WALKFORWARD_BOTTOM_RUNUP_LOOKBACK_DAYS,
            min_runup_pct=WALKFORWARD_BOTTOM_RUNUP_MIN_PCT,
        )
        spread = (
            float(top_mean - bottom_mean)
            if np.isfinite(top_mean) and np.isfinite(bottom_mean)
            else float("nan")
        )

        rows.append(
            {
                "date": day_str,
                "count": int(day_n),
                "history_days": int(len(history_days)),
                "history_count": int(history_count),
                "upper_threshold": float(upper_thr),
                "lower_threshold": float(lower_thr),
                "top_count": int(top_count),
                "bottom_count": int(bottom_count),
                "selection_mode": str(selection_mode),
                "overlap_count_before_resolution": int(overlap_before),
                "top_prob_min": (
                    float(np.min(day_p[top_mask])) if int(np.sum(top_mask)) > 0 else float("nan")
                ),
                "top_prob_max": (
                    float(np.max(day_p[top_mask])) if int(np.sum(top_mask)) > 0 else float("nan")
                ),
                "bottom_prob_min": (
                    float(np.min(day_p[bottom_mask]))
                    if int(np.sum(bottom_mask)) > 0
                    else float("nan")
                ),
                "bottom_prob_max": (
                    float(np.max(day_p[bottom_mask]))
                    if int(np.sum(bottom_mask)) > 0
                    else float("nan")
                ),
                "top_ret_pct_mean": float(top_mean),
                "bottom_ret_pct_mean": float(bottom_mean),
                "bottom_runup5_ge30_count": int(bottom_runup5_ge30_count),
                "bottom_runup5_ge30_ret_pct_mean": float(bottom_runup5_ge30_mean),
                "spread": float(spread),
            }
        )

    rows, outlier_filter_info = filter_eval_rows_by_tail_ret_zscore(
        rows,
        zscore_threshold=outlier_zscore_threshold,
    )
    spy_non_overlap_info = compute_walkforward_spy_non_overlap_info(
        rows=rows,
        ordered_dates=[str(d) for d in unique_dates],
        enabled=spy_non_overlap_enabled_v,
        horizon_days=spy_non_overlap_horizon_days_v,
        spy_csv_path=spy_daily_csv_v,
        spy_date_col=spy_date_col_v,
        spy_close_col=spy_close_col_v,
    )
    real_non_overlap_info = compute_walkforward_real_non_overlap_info(
        rows=rows,
        ordered_dates=[str(d) for d in unique_dates],
        horizon_days=spy_non_overlap_horizon_days_v,
        annualization_days=annualization,
    )

    days_total = int(len(unique_dates))
    days_used = int(len(rows))
    if days_used == 0:
        payload = build_payload(
            days_total=int(days_total),
            days_used=0,
            days_skipped_insufficient_history=int(skipped_history),
            days_skipped_min_names=int(skipped_min_names),
            days_skipped_min_per_side=int(skipped_min_per_side),
            days_used_rank_fallback=0,
            days_with_overlap_before_resolution=0,
            mean_history_count=float("nan"),
            mean_top_count=float("nan"),
            mean_bottom_count=float("nan"),
            mean_upper_threshold=float("nan"),
            mean_lower_threshold=float("nan"),
            mean_overlap_count_before_resolution=float("nan"),
            spread_mean=float("nan"),
            spread_median=float("nan"),
            spread_std=float("nan"),
            spread_sharpe_annualized=float("nan"),
            spread_hit_rate=float("nan"),
            outlier_filter_info=outlier_filter_info,
            spy_non_overlap_info=spy_non_overlap_info,
            real_non_overlap_info=real_non_overlap_info,
        )
        return payload, rows

    history_count_arr = np.asarray([r["history_count"] for r in rows], dtype=np.float64)
    top_count_arr = np.asarray([r["top_count"] for r in rows], dtype=np.float64)
    bottom_count_arr = np.asarray([r["bottom_count"] for r in rows], dtype=np.float64)
    upper_threshold_arr = np.asarray([r["upper_threshold"] for r in rows], dtype=np.float64)
    lower_threshold_arr = np.asarray([r["lower_threshold"] for r in rows], dtype=np.float64)
    overlap_before_arr = np.asarray(
        [r["overlap_count_before_resolution"] for r in rows], dtype=np.float64
    )
    days_used_rank_fallback_filtered = int(
        np.sum(np.asarray([str(r["selection_mode"]) == "rank_fallback" for r in rows]))
    )
    days_with_overlap_before_resolution_filtered = int(np.sum(overlap_before_arr > 0.0))

    top_mean_arr = np.asarray([r["top_ret_pct_mean"] for r in rows], dtype=np.float64)
    bottom_mean_arr = np.asarray([r["bottom_ret_pct_mean"] for r in rows], dtype=np.float64)
    top_finite_mask = np.isfinite(top_mean_arr)
    bottom_finite_mask = np.isfinite(bottom_mean_arr)
    pair_finite_mask = top_finite_mask & bottom_finite_mask

    spread_arr = top_mean_arr[pair_finite_mask] - bottom_mean_arr[pair_finite_mask]
    spread_days = int(spread_arr.size)
    spread_mean = float(np.mean(spread_arr)) if spread_days > 0 else float("nan")
    spread_median = float(np.median(spread_arr)) if spread_days > 0 else float("nan")
    spread_std = float(np.std(spread_arr, ddof=1)) if spread_days > 1 else float("nan")
    spread_sharpe = float("nan")
    if np.isfinite(spread_std) and spread_std > 0.0:
        spread_sharpe = float((spread_mean / spread_std) * np.sqrt(annualization))

    top_days_used = int(np.sum(top_finite_mask))
    bottom_days_used = int(np.sum(bottom_finite_mask))
    paired_days_used = int(np.sum(pair_finite_mask))
    paired_mean_top_pct = (
        float(np.mean(top_mean_arr[pair_finite_mask]))
        if paired_days_used > 0
        else float("nan")
    )
    paired_mean_bottom_pct = (
        float(np.mean(bottom_mean_arr[pair_finite_mask]))
        if paired_days_used > 0
        else float("nan")
    )

    top_positions = float(np.sum(top_count_arr[top_finite_mask]))
    bottom_positions = float(np.sum(bottom_count_arr[bottom_finite_mask]))
    mean_top_pct = (
        float(np.sum(top_mean_arr[top_finite_mask] * top_count_arr[top_finite_mask]) / top_positions)
        if top_positions > 0.0
        else float("nan")
    )
    mean_bottom_pct = (
        float(
            np.sum(bottom_mean_arr[bottom_finite_mask] * bottom_count_arr[bottom_finite_mask])
            / bottom_positions
        )
        if bottom_positions > 0.0
        else float("nan")
    )

    top_vals = top_mean_arr[top_finite_mask]
    bottom_vals = bottom_mean_arr[bottom_finite_mask]
    top_std = float(np.std(top_vals, ddof=1)) if top_vals.size > 1 else float("nan")
    bottom_std = (
        float(np.std(bottom_vals, ddof=1)) if bottom_vals.size > 1 else float("nan")
    )
    top_sharpe = float("nan")
    if np.isfinite(top_std) and top_std > 0.0:
        top_sharpe = float((float(np.mean(top_vals)) / top_std) * np.sqrt(annualization))
    bottom_sharpe = float("nan")
    if np.isfinite(bottom_std) and bottom_std > 0.0:
        bottom_sharpe = float(
            (float(np.mean(bottom_vals)) / bottom_std) * np.sqrt(annualization)
        )
    top_hit_rate = float(np.mean(top_vals > 0.0)) if top_vals.size > 0 else float("nan")
    bottom_hit_rate = (
        float(np.mean(bottom_vals < 0.0)) if bottom_vals.size > 0 else float("nan")
    )
    bottom_runup5_ge30_count_arr = np.asarray(
        [r.get("bottom_runup5_ge30_count", 0) for r in rows], dtype=np.float64
    )
    bottom_runup5_ge30_mean_arr = np.asarray(
        [r.get("bottom_runup5_ge30_ret_pct_mean", float("nan")) for r in rows],
        dtype=np.float64,
    )
    bottom_runup5_ge30_finite_mask = np.isfinite(bottom_runup5_ge30_mean_arr) & (
        bottom_runup5_ge30_count_arr > 0.0
    )
    bottom_runup5_ge30_vals = bottom_runup5_ge30_mean_arr[
        bottom_runup5_ge30_finite_mask
    ]
    bottom_runup5_ge30_positions = float(
        np.sum(bottom_runup5_ge30_count_arr[bottom_runup5_ge30_finite_mask])
    )
    bottom_runup5_ge30_mean_pct = (
        float(
            np.sum(
                bottom_runup5_ge30_mean_arr[bottom_runup5_ge30_finite_mask]
                * bottom_runup5_ge30_count_arr[bottom_runup5_ge30_finite_mask]
            )
            / bottom_runup5_ge30_positions
        )
        if bottom_runup5_ge30_positions > 0.0
        else float("nan")
    )
    bottom_runup5_ge30_std = (
        float(np.std(bottom_runup5_ge30_vals, ddof=1))
        if bottom_runup5_ge30_vals.size > 1
        else float("nan")
    )
    bottom_runup5_ge30_sharpe = float("nan")
    if np.isfinite(bottom_runup5_ge30_std) and bottom_runup5_ge30_std > 0.0:
        bottom_runup5_ge30_sharpe = float(
            (float(np.mean(bottom_runup5_ge30_vals)) / bottom_runup5_ge30_std)
            * np.sqrt(annualization)
        )
    bottom_runup5_ge30_hit_rate = (
        float(np.mean(bottom_runup5_ge30_vals < 0.0))
        if bottom_runup5_ge30_vals.size > 0
        else float("nan")
    )

    single_top_mask = top_finite_mask & (top_count_arr == 1.0)
    single_bottom_mask = bottom_finite_mask & (bottom_count_arr == 1.0)
    single_top_vals = top_mean_arr[single_top_mask]
    single_bottom_vals = bottom_mean_arr[single_bottom_mask]
    single_top_days_used = int(single_top_vals.size)
    single_bottom_days_used = int(single_bottom_vals.size)
    single_pick_mean_top_pct = (
        float(np.mean(single_top_vals)) if single_top_days_used > 0 else float("nan")
    )
    single_pick_mean_bottom_pct = (
        float(np.mean(single_bottom_vals))
        if single_bottom_days_used > 0
        else float("nan")
    )
    single_pick_top_sharpe = float("nan")
    if single_top_days_used > 1:
        single_top_std = float(np.std(single_top_vals, ddof=1))
        if np.isfinite(single_top_std) and single_top_std > 0.0:
            single_pick_top_sharpe = float(
                (float(np.mean(single_top_vals)) / single_top_std) * np.sqrt(annualization)
            )
    single_pick_bottom_sharpe = float("nan")
    if single_bottom_days_used > 1:
        single_bottom_std = float(np.std(single_bottom_vals, ddof=1))
        if np.isfinite(single_bottom_std) and single_bottom_std > 0.0:
            single_pick_bottom_sharpe = float(
                (float(np.mean(single_bottom_vals)) / single_bottom_std)
                * np.sqrt(annualization)
            )

    payload = build_payload(
        days_total=int(days_total),
        days_used=int(days_used),
        days_skipped_insufficient_history=int(skipped_history),
        days_skipped_min_names=int(skipped_min_names),
        days_skipped_min_per_side=int(skipped_min_per_side),
        days_used_rank_fallback=int(days_used_rank_fallback_filtered),
        days_with_overlap_before_resolution=int(
            days_with_overlap_before_resolution_filtered
        ),
        mean_history_count=float(np.mean(history_count_arr)),
        mean_top_count=float(np.mean(top_count_arr)),
        mean_bottom_count=float(np.mean(bottom_count_arr)),
        mean_upper_threshold=float(np.mean(upper_threshold_arr)),
        mean_lower_threshold=float(np.mean(lower_threshold_arr)),
        mean_overlap_count_before_resolution=float(np.mean(overlap_before_arr)),
        spread_mean=float(spread_mean),
        spread_median=float(spread_median),
        spread_std=float(spread_std),
        spread_sharpe_annualized=float(spread_sharpe),
        spread_hit_rate=(
            float(np.mean(spread_arr > 0.0)) if spread_days > 0 else float("nan")
        ),
        top_days_used=int(top_days_used),
        bottom_days_used=int(bottom_days_used),
        paired_days_used=int(paired_days_used),
        mean_top_pct=float(mean_top_pct),
        top_sharpe=float(top_sharpe),
        top_hit_rate=float(top_hit_rate),
        mean_bottom_pct=float(mean_bottom_pct),
        bottom_sharpe=float(bottom_sharpe),
        bottom_hit_rate=float(bottom_hit_rate),
        paired_mean_top_pct=float(paired_mean_top_pct),
        paired_mean_bottom_pct=float(paired_mean_bottom_pct),
        single_pick_mean_top_pct=float(single_pick_mean_top_pct),
        single_pick_mean_bottom_pct=float(single_pick_mean_bottom_pct),
        single_pick_top_sharpe=float(single_pick_top_sharpe),
        single_pick_top_days_used=int(single_top_days_used),
        single_pick_bottom_days_used=int(single_bottom_days_used),
        single_pick_bottom_sharpe=float(single_pick_bottom_sharpe),
        bottom_runup5_ge30_days_used=int(bottom_runup5_ge30_vals.size),
        bottom_runup5_ge30_mean_pct=float(bottom_runup5_ge30_mean_pct),
        bottom_runup5_ge30_sharpe=float(bottom_runup5_ge30_sharpe),
        bottom_runup5_ge30_hit_rate=float(bottom_runup5_ge30_hit_rate),
        bottom_runup5_ge30_mean_count=float(
            np.mean(bottom_runup5_ge30_count_arr[bottom_runup5_ge30_finite_mask])
        )
        if bottom_runup5_ge30_vals.size > 0
        else float("nan"),
        bottom_runup5_ge30_trade_count=int(bottom_runup5_ge30_positions),
        outlier_filter_info=outlier_filter_info,
        spy_non_overlap_info=spy_non_overlap_info,
        real_non_overlap_info=real_non_overlap_info,
    )
    return payload, rows


def extract_walkforward_topline_eval_stats(
    payload: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    keys = [
        "top_percentile",
        "bottom_percentile",
        "min_per_side",
        "min_per_side_mode",
        "days_total",
        "days_used",
        "top_days_used",
        "bottom_days_used",
        "paired_days_used",
        "mean_history_count",
        "mean_top_count",
        "mean_bottom_count",
        "mean_upper_threshold",
        "mean_lower_threshold",
        "spread_mean",
        "spread_median",
        "spread_std",
        "spread_sharpe",
        "spread_hit_rate",
        "mean_top_pct",
        "top_sharpe",
        "top_hit_rate",
        "mean_bottom_pct",
        "bottom_sharpe",
        "bottom_hit_rate",
        "paired_mean_top_pct",
        "paired_mean_bottom_pct",
        "single_pick_mean_top_pct",
        "single_pick_mean_bottom_pct",
        "single_pick_top_sharpe",
        "single_pick_top_days_used",
        "single_pick_bottom_sharpe",
        "single_pick_bottom_days_used",
        "bottom_runup5_ge30_days_used",
        "bottom_runup5_ge30_mean_count",
        "bottom_runup5_ge30_trade_count",
        "bottom_runup5_ge30_mean_pct",
        "bottom_runup5_ge30_sharpe",
        "bottom_runup5_ge30_hit_rate",
        "spy_non_overlap_trades",
        "spy_non_overlap_trade_mean",
        "spy_non_overlap_trade_median",
        "spy_non_overlap_compounded_return",
        "spy_non_overlap_compounded_multiple",
        "real_num_trades",
        "real_per_trade_returns",
        "non_adj_returns_sharpe",
        "real_returns_sharpe",
        "real_nonoverlap_compounded_returns",
        "outlier_filter_enabled",
        "outlier_filter_zscore_used",
        "outlier_filter_top_lower_threshold_value",
        "outlier_filter_top_upper_threshold_value",
        "outlier_filter_bottom_lower_threshold_value",
        "outlier_filter_bottom_upper_threshold_value",
        "outlier_filter_days_removed",
    ]
    out: dict[str, float | int | str] = {}
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            out[key] = value
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            out[key] = int(value) if isinstance(value, (int, np.integer)) else float(value)
        else:
            out[key] = float("nan")
    return out


def write_walkforward_rolling_threshold_metrics_best(
    out_dir: Path,
    best_epoch: int,
    payload: dict[str, float | int | str],
    rows: list[dict[str, float | int | str]],
    file_stem: str | None = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(file_stem).strip() if file_stem is not None else ""
    if stem:
        csv_path = out_dir / f"{stem}.csv"
        json_path = out_dir / f"{stem}.json"
    else:
        csv_path = out_dir / "val_walkforward_rolling_threshold_metrics_best_epoch.csv"
        json_path = out_dir / "walk_forward_topline.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "best_epoch",
                "date",
                "count",
                "history_days",
                "history_count",
                "upper_threshold",
                "lower_threshold",
                "top_count",
                "bottom_count",
                "selection_mode",
                "overlap_count_before_resolution",
                "top_prob_min",
                "top_prob_max",
                "bottom_prob_min",
                "bottom_prob_max",
                "top_ret_pct_mean",
                "bottom_ret_pct_mean",
                "bottom_runup5_ge30_count",
                "bottom_runup5_ge30_ret_pct_mean",
                "spread",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    int(best_epoch),
                    str(row["date"]),
                    int(row["count"]),
                    int(row["history_days"]),
                    int(row["history_count"]),
                    float(row["upper_threshold"]),
                    float(row["lower_threshold"]),
                    int(row["top_count"]),
                    int(row["bottom_count"]),
                    str(row["selection_mode"]),
                    int(row["overlap_count_before_resolution"]),
                    float(row["top_prob_min"]),
                    float(row["top_prob_max"]),
                    float(row["bottom_prob_min"]),
                    float(row["bottom_prob_max"]),
                    float(row["top_ret_pct_mean"]),
                    float(row["bottom_ret_pct_mean"]),
                    int(row.get("bottom_runup5_ge30_count", 0)),
                    float(row.get("bottom_runup5_ge30_ret_pct_mean", float("nan"))),
                    float(row["spread"]),
                ]
            )

    topline_payload = extract_walkforward_topline_eval_stats(payload)
    json_payload = {"best_epoch": int(best_epoch), **topline_payload, "rows": rows}
    json_path.write_text(json.dumps(json_payload, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def _load_run_config_blob(run_dir: Path) -> dict[str, object]:
    config_path = run_dir / "config.json"
    if config_path.is_file():
        blob = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(blob, dict):
            raise ValueError(f"run config must be a JSON object: {config_path}")
        return blob

    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        blob = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(blob, dict):
            raise ValueError(f"run metrics must be a JSON object: {metrics_path}")
        cfg_val = blob.get("config", {})
        eval_val = blob.get("evaluation", {})
        return {
            "config": cfg_val if isinstance(cfg_val, dict) else {},
            "evaluation": eval_val if isinstance(eval_val, dict) else {},
            "best_epoch": blob.get("best_epoch", 1),
        }

    raise FileNotFoundError(
        f"run config not found: {config_path} (or metrics fallback {metrics_path})"
    )


def _resolve_best_epoch(
    run_dir: Path,
    config_blob: dict[str, object],
    best_epoch: int | None,
) -> int:
    if best_epoch is not None:
        return int(best_epoch)

    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(summary, dict):
                return int(summary.get("best_epoch", 1))
        except Exception:
            pass

    try:
        return int(config_blob.get("best_epoch", 1))
    except Exception:
        return 1


def _load_val_pred(
    pred_npz_path: Path,
    data_dir: str | None = None,
) -> dict[str, np.ndarray | float]:
    with np.load(pred_npz_path, allow_pickle=True) as data:
        raw: dict[str, np.ndarray | float] = {k: data[k] for k in data.files}

    out: dict[str, np.ndarray | float] = dict(raw)

    if "prob" in raw:
        prob = np.asarray(raw["prob"], dtype=np.float32)
        if prob.ndim != 2 or int(prob.shape[1]) != 2:
            raise ValueError(f"{pred_npz_path}: prob must have shape (n,2)")
        out["prob"] = prob
    elif "prob_1" in raw:
        prob_1 = np.asarray(raw["prob_1"], dtype=np.float32).reshape(-1)
        prob = np.stack((1.0 - prob_1, prob_1), axis=1).astype(np.float32, copy=False)
        out["prob"] = prob
    else:
        raise ValueError(f"{pred_npz_path}: missing probability field (need prob or prob_1)")

    sample_indices_arr: np.ndarray | None = None
    if "sample_indices" in raw:
        sample_indices_arr = np.asarray(raw["sample_indices"], dtype=np.int64).reshape(-1)
    elif "sample_index" in raw:
        sample_indices_arr = np.asarray(raw["sample_index"], dtype=np.int64).reshape(-1)
        out["sample_indices"] = sample_indices_arr

    if "pred_cls" not in raw and "pred" in raw:
        out["pred_cls"] = np.asarray(raw["pred"], dtype=np.int64).reshape(-1)

    if "timestamps" not in raw:
        data_dir_v = str(data_dir or "").strip()
        if sample_indices_arr is not None and data_dir_v:
            ts_path = Path(data_dir_v) / "timestamps.npy"
            if ts_path.is_file():
                all_timestamps = np.load(ts_path, mmap_mode="r", allow_pickle=False)
                if np.any(sample_indices_arr < 0) or np.any(sample_indices_arr >= all_timestamps.shape[0]):
                    raise ValueError(
                        f"{pred_npz_path}: sample indices out of bounds for {ts_path}"
                    )
                mapped_timestamps = np.asarray(all_timestamps[sample_indices_arr], dtype=object)
                out["timestamps"] = mapped_timestamps

    return out


def _run_evals_for_run(run_dir: Path, best_epoch: int | None = None) -> dict[str, object]:
    config_blob = _load_run_config_blob(run_dir)
    cfg_val = config_blob.get("config", {})
    cfg = dict(cfg_val) if isinstance(cfg_val, dict) else {}

    best_epoch = _resolve_best_epoch(run_dir, config_blob, best_epoch)

    pred_npz_path = run_dir / "preds" / "val_preds_best.npz"
    if not pred_npz_path.is_file():
        legacy_pred_npz_path = run_dir / "preds" / "val_preds_best_epoch.npz"
        if legacy_pred_npz_path.is_file():
            pred_npz_path = legacy_pred_npz_path
        else:
            raise FileNotFoundError(
                "validation predictions npz not found: "
                f"{pred_npz_path} (or legacy {legacy_pred_npz_path})"
            )
    pred = _load_val_pred(pred_npz_path, data_dir=str(cfg.get("data_dir", "")))

    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Eval settings are intentionally sourced from this script's constants
    # so post-run re-evals and end-of-train evals always share one control point.
    daily_top_pct = DAILY_CROSS_SECTIONAL_TOP_PCT
    daily_bottom_pct = DAILY_CROSS_SECTIONAL_BOTTOM_PCT
    daily_min_per_side = int(DAILY_CROSS_SECTIONAL_MIN_PER_SIDE)
    daily_min_names_per_day = int(DAILY_CROSS_SECTIONAL_MIN_NAMES_PER_DAY)
    daily_annualization_days = float(DAILY_CROSS_SECTIONAL_ANNUALIZATION_DAYS)
    daily_outlier_z = float(EVAL_TAIL_RET_OUTLIER_ZSCORE_THRESHOLD)

    wf_enabled = bool(int(WALKFORWARD_ROLLING_THRESHOLD_ENABLED))
    wf_top_pct = WALKFORWARD_ROLLING_TOP_PCT
    wf_bottom_pct = WALKFORWARD_ROLLING_BOTTOM_PCT
    wf_lookback_days = int(WALKFORWARD_ROLLING_LOOKBACK_DAYS)
    wf_min_history_days = int(WALKFORWARD_ROLLING_MIN_HISTORY_DAYS)
    wf_min_per_side = int(WALKFORWARD_ROLLING_MIN_PER_SIDE)
    wf_min_per_side_mode = str(WALKFORWARD_ROLLING_MIN_PER_SIDE_MODE)
    wf_min_names_per_day = int(WALKFORWARD_ROLLING_MIN_NAMES_PER_DAY)
    wf_annualization_days = float(WALKFORWARD_ROLLING_ANNUALIZATION_DAYS)
    wf_threshold_method = str(WALKFORWARD_ROLLING_THRESHOLD_METHOD)
    wf_fallback = bool(int(WALKFORWARD_ROLLING_FALLBACK_TO_DAILY_RANK))
    wf_enforce_non_overlap = bool(int(WALKFORWARD_ROLLING_ENFORCE_NON_OVERLAP))
    wf_outlier_z = float(EVAL_TAIL_RET_OUTLIER_ZSCORE_THRESHOLD)

    spy_non_overlap_enabled = bool(int(WALKFORWARD_SPY_NON_OVERLAP_ENABLED))
    spy_non_overlap_horizon_days = int(WALKFORWARD_SPY_NON_OVERLAP_HORIZON_DAYS)
    spy_daily_csv = str(WALKFORWARD_SPY_DAILY_CSV_PATH)
    spy_date_col = str(WALKFORWARD_SPY_DATE_COL)
    spy_close_col = str(WALKFORWARD_SPY_CLOSE_COL)

    daily_pct_pairs = resolve_tail_pct_pairs(
        daily_top_pct,
        daily_bottom_pct,
        context="daily_cross_sectional",
    )
    daily_eval_runs: list[dict[str, object]] = []
    for top_pct_eval, bottom_pct_eval in daily_pct_pairs:
        payload, rows = compute_daily_cross_sectional_metrics(
            pred,
            top_pct=float(top_pct_eval),
            bottom_pct=float(bottom_pct_eval),
            min_per_side=int(daily_min_per_side),
            min_names_per_day=int(daily_min_names_per_day),
            annualization_days=float(daily_annualization_days),
            outlier_zscore_threshold=float(daily_outlier_z),
        )
        daily_stem = f"daily_cs_{format_pct_tag(top_pct_eval)}"
        csv_path, json_path = write_daily_cross_sectional_metrics_best(
            out_dir=eval_dir,
            best_epoch=best_epoch,
            payload=payload,
            rows=rows,
            file_stem=daily_stem,
        )
        daily_eval_runs.append(
            {
                "top_pct": float(top_pct_eval),
                "bottom_pct": float(bottom_pct_eval),
                "payload": payload,
                "csv_path": csv_path,
                "json_path": json_path,
            }
        )

    wf_pct_pairs = resolve_tail_pct_pairs(
        wf_top_pct,
        wf_bottom_pct,
        context="walkforward_rolling_threshold",
    )
    wf_eval_runs: list[dict[str, object]] = []
    for top_pct_eval, bottom_pct_eval in wf_pct_pairs:
        payload, rows = compute_walkforward_rolling_threshold_metrics(
            pred,
            enabled=bool(wf_enabled),
            top_pct=float(top_pct_eval),
            bottom_pct=float(bottom_pct_eval),
            lookback_days=int(wf_lookback_days),
            min_history_days=int(wf_min_history_days),
            min_per_side=int(wf_min_per_side),
            min_per_side_mode=str(wf_min_per_side_mode),
            min_names_per_day=int(wf_min_names_per_day),
            annualization_days=float(wf_annualization_days),
            threshold_method=str(wf_threshold_method),
            fallback_to_daily_rank=bool(wf_fallback),
            enforce_non_overlap=bool(wf_enforce_non_overlap),
            outlier_zscore_threshold=float(wf_outlier_z),
            spy_non_overlap_enabled=bool(spy_non_overlap_enabled),
            spy_non_overlap_horizon_days=int(spy_non_overlap_horizon_days),
            spy_daily_csv=str(spy_daily_csv),
            spy_date_col=str(spy_date_col),
            spy_close_col=str(spy_close_col),
        )
        wf_stem = f"wf_{format_pct_tag(top_pct_eval)}"
        csv_path, json_path = write_walkforward_rolling_threshold_metrics_best(
            out_dir=eval_dir,
            best_epoch=best_epoch,
            payload=payload,
            rows=rows,
            file_stem=wf_stem,
        )
        wf_eval_runs.append(
            {
                "top_pct": float(top_pct_eval),
                "bottom_pct": float(bottom_pct_eval),
                "payload": payload,
                "csv_path": csv_path,
                "json_path": json_path,
            }
        )

    if not daily_eval_runs:
        raise RuntimeError("daily evaluation produced no runs")
    if not wf_eval_runs:
        raise RuntimeError("walk-forward evaluation produced no runs")

    daily_primary = daily_eval_runs[0]
    daily_cs_payload = dict(daily_primary["payload"])  # type: ignore[arg-type]
    daily_cs_csv_rel = str(Path(daily_primary["csv_path"]).relative_to(run_dir).as_posix())  # type: ignore[arg-type]
    daily_cs_json_rel = str(Path(daily_primary["json_path"]).relative_to(run_dir).as_posix())  # type: ignore[arg-type]

    wf_primary = wf_eval_runs[0]
    wf_payload = dict(wf_primary["payload"])  # type: ignore[arg-type]
    wf_csv_rel = str(Path(wf_primary["csv_path"]).relative_to(run_dir).as_posix())  # type: ignore[arg-type]
    wf_json_rel = str(Path(wf_primary["json_path"]).relative_to(run_dir).as_posix())  # type: ignore[arg-type]

    daily_outputs_summary = [
        {
            "top_pct": float(item["top_pct"]),
            "bottom_pct": float(item["bottom_pct"]),
            "csv": str(Path(item["csv_path"]).relative_to(run_dir).as_posix()),
            "json": str(Path(item["json_path"]).relative_to(run_dir).as_posix()),
            "spread_sharpe": float(item["payload"]["spread_sharpe"]),  # type: ignore[index]
            "days_used": int(item["payload"]["days_used"]),  # type: ignore[index]
            "spread_compounded_returns": float(item["payload"]["spread_compounded_returns"]),  # type: ignore[index]
        }
        for item in daily_eval_runs
    ]
    wf_outputs_summary = [
        {
            "top_pct": float(item["top_pct"]),
            "bottom_pct": float(item["bottom_pct"]),
            "csv": str(Path(item["csv_path"]).relative_to(run_dir).as_posix()),
            "json": str(Path(item["json_path"]).relative_to(run_dir).as_posix()),
            "spread_sharpe": float(item["payload"]["spread_sharpe"]),  # type: ignore[index]
            "days_used": int(item["payload"]["days_used"]),  # type: ignore[index]
            "spy_non_overlap_trades": int(item["payload"]["spy_non_overlap_trades"]),  # type: ignore[index]
            "spy_non_overlap_trade_mean": float(item["payload"]["spy_non_overlap_trade_mean"]),  # type: ignore[index]
            "spy_non_overlap_trade_median": float(item["payload"]["spy_non_overlap_trade_median"]),  # type: ignore[index]
            "spy_non_overlap_compounded_return": float(item["payload"]["spy_non_overlap_compounded_return"]),  # type: ignore[index]
            "real_num_trades": int(item["payload"]["real_num_trades"]),  # type: ignore[index]
            "real_per_trade_returns": float(item["payload"]["real_per_trade_returns"]),  # type: ignore[index]
            "non_adj_returns_sharpe": float(item["payload"]["non_adj_returns_sharpe"]),  # type: ignore[index]
            "real_returns_sharpe": float(item["payload"]["real_returns_sharpe"]),  # type: ignore[index]
            "real_nonoverlap_compounded_returns": float(item["payload"]["real_nonoverlap_compounded_returns"]),  # type: ignore[index]
        }
        for item in wf_eval_runs
    ]

    ordered_pairs: list[tuple[float, float]] = []
    for item in daily_eval_runs:
        key = (float(item["top_pct"]), float(item["bottom_pct"]))
        if key not in ordered_pairs:
            ordered_pairs.append(key)
    for item in wf_eval_runs:
        key = (float(item["top_pct"]), float(item["bottom_pct"]))
        if key not in ordered_pairs:
            ordered_pairs.append(key)

    daily_sharpe_map: dict[tuple[float, float], float] = {
        (float(item["top_pct"]), float(item["bottom_pct"])): float(item["payload"]["spread_sharpe"])  # type: ignore[index]
        for item in daily_eval_runs
    }
    wf_real_sharpe_map: dict[tuple[float, float], float] = {
        (float(item["top_pct"]), float(item["bottom_pct"])): float(item["payload"]["real_returns_sharpe"])  # type: ignore[index]
        for item in wf_eval_runs
    }
    wf_real_per_trade_map: dict[tuple[float, float], float] = {
        (float(item["top_pct"]), float(item["bottom_pct"])): float(item["payload"]["real_per_trade_returns"])  # type: ignore[index]
        for item in wf_eval_runs
    }

    console_lines: list[str] = []
    for top_pct_eval, bottom_pct_eval in ordered_pairs:
        daily_sharpe_val = float(daily_sharpe_map.get((top_pct_eval, bottom_pct_eval), float("nan")))
        wf_real_sharpe_val = float(
            wf_real_sharpe_map.get((top_pct_eval, bottom_pct_eval), float("nan"))
        )
        wf_real_per_trade_val = float(
            wf_real_per_trade_map.get((top_pct_eval, bottom_pct_eval), float("nan"))
        )
        console_lines.append(
            f"daily_cs_sharpe: {daily_sharpe_val}, "
            f"real_returns_sharpe: {wf_real_sharpe_val}, "
            f"real_per_trade_returns: {wf_real_per_trade_val}"
        )

    effective_settings = {
        "daily_cross_sectional": {
            "top_pct": tail_pct_config_value(daily_top_pct, "daily_cross_sectional.top_pct"),
            "bottom_pct": tail_pct_config_value(daily_bottom_pct, "daily_cross_sectional.bottom_pct"),
            "min_per_side": int(daily_min_per_side),
            "min_names_per_day": int(daily_min_names_per_day),
            "annualization_days": float(daily_annualization_days),
            "outlier_zscore_threshold": float(daily_outlier_z),
            "spread_compounded_daily_capital_fraction": float(
                DAILY_CROSS_SECTIONAL_SPREAD_COMPOUND_DAILY_CAPITAL_FRACTION
            ),
        },
        "walkforward_rolling_threshold": {
            "enabled": int(bool(wf_enabled)),
            "top_pct": tail_pct_config_value(
                wf_top_pct,
                "walkforward_rolling_threshold.top_pct",
            ),
            "bottom_pct": tail_pct_config_value(
                wf_bottom_pct,
                "walkforward_rolling_threshold.bottom_pct",
            ),
            "lookback_days": int(wf_lookback_days),
            "min_history_days": int(wf_min_history_days),
            "min_per_side": int(wf_min_per_side),
            "min_per_side_mode": str(wf_min_per_side_mode),
            "min_names_per_day": int(wf_min_names_per_day),
            "annualization_days": float(wf_annualization_days),
            "threshold_method": str(wf_threshold_method),
            "fallback_to_daily_rank": int(bool(wf_fallback)),
            "enforce_non_overlap": int(bool(wf_enforce_non_overlap)),
            "spy_non_overlap_enabled": int(bool(spy_non_overlap_enabled)),
            "spy_non_overlap_horizon_days": int(spy_non_overlap_horizon_days),
            "spy_daily_csv": str(spy_daily_csv),
            "spy_date_col": str(spy_date_col),
            "spy_close_col": str(spy_close_col),
            "outlier_zscore_threshold": float(wf_outlier_z),
        },
    }

    result = {
        "best_epoch": int(best_epoch),
        "settings_source": "script_constants",
        "effective_settings": effective_settings,
        "daily_primary": {
            "top_pct": float(daily_primary["top_pct"]),
            "bottom_pct": float(daily_primary["bottom_pct"]),
            "csv": str(daily_cs_csv_rel),
            "json": str(daily_cs_json_rel),
            "payload": daily_cs_payload,
        },
        "wf_primary": {
            "top_pct": float(wf_primary["top_pct"]),
            "bottom_pct": float(wf_primary["bottom_pct"]),
            "csv": str(wf_csv_rel),
            "json": str(wf_json_rel),
            "payload": wf_payload,
        },
        "daily_outputs_summary": daily_outputs_summary,
        "wf_outputs_summary": wf_outputs_summary,
        "console_lines": console_lines,
    }

    results_path = eval_dir / "sim_evals_results.json"
    results_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for line in console_lines:
        print(line)

    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run simulation evals for an existing run")
    p.add_argument("--run-id", type=str, default="", help="Run id under --runs-root (e.g. 39)")
    p.add_argument(
        "--runs-root",
        type=str,
        default="runs",
        help="Root directory that contains run subdirectories (default: runs)",
    )
    p.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Explicit run directory path (overrides --run-id/--runs-root)",
    )
    p.add_argument(
        "--best-epoch",
        type=int,
        default=-1,
        help="Override best epoch used in emitted eval CSV/JSON payloads",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir_txt = str(args.run_dir).strip()
    if run_dir_txt:
        run_dir = Path(run_dir_txt)
    else:
        run_id_txt = str(args.run_id).strip()
        if not run_id_txt:
            raise SystemExit("Provide --run-id or --run-dir")
        run_dir = Path(str(args.runs_root)).expanduser() / run_id_txt

    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    best_epoch = None if int(args.best_epoch) < 0 else int(args.best_epoch)
    _run_evals_for_run(run_dir=run_dir, best_epoch=best_epoch)


if __name__ == "__main__":
    main()
