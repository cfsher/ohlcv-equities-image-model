#!/usr/bin/env python3
"""Minimal decomp-image model (MSF-only) for OHLC image shards.

Expected input directory:
  data_dir/
    manifest.json
    shards/shard_*.npz

Each shard must include:
  X_img: (samples, scales, height, width), uint8/float
  y_raw: (samples, labels)

Label target for training:
  binary target in {0,1} derived from ret_pct:
    y_cls = 1 if ret_pct > threshold else 0
Loss:
  Cross-entropy on 2-class logits.
Output:
  Softmax probabilities.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from jagged_width_config import (
    JAGGED_2D_CONCAT_OPTION,
    JAGGED_IMAGE_WIDTHS_ENABLED,
    JAGGED_OPTION3_TARGET_WIDTH,
)

# ----------------------------
# Constants (requested)
# ----------------------------
DATA_DIR = '/ephemeral/images/55_percent'
VIX_DAILY_CSV_PATH = str((Path(__file__).resolve().parent / "daily_vix.csv"))

EPOCHS = 100
PATIENCE = 3
LEARNING_RATE = 3e-5

CONV_CHANNELS = [64, 128, 256]
MSF_SCALE_WEIGHTS = [0.5,.25,.25]
READOUT_DIM = 16
HEAD_MLP_DIMS_DEFAULT = [2048,512,128]

# READOUT_DIM = 528
# HEAD_MLP_DIMS_DEFAULT = [12288,8192,2048,512]

MODEL_BRANCH_MODE = "dual"  # valid: decomp, seq, dual
DUAL_REGRESSION_MSE_WEIGHT = 0.15
TS_WIDTH_CROP_KEEP_SIDE = "right"  # valid: left, right
SEQ_REGRESSION_LOSS_MODE = "mse"  # valid: smoothl1, mse

SEQ_VOLUME_NORM_LOG1P_ENABLED = False
SEQ_VOLUME_NORM_ROBUST_ZSCORE_ENABLED = True
SEQ_VOLUME_NORM_ROBUST_SCALE = 0.6745
SEQ_VOLUME_NORM_ROBUST_EPS = 1e-6
SEQ_VOLUME_NORM_ROBUST_CLIP = 11.0

SEQ_INPUT_ABS_CLIP = 60000.0
SEQ_RATIO_FEATURE_ABS_CLIP = 5
SEQ_FEATURE_STANDARDIZATION_ENABLED = True
SEQ_FEATURE_STANDARDIZATION_EPS = 1e-6
SEQ_FEATURE_STANDARDIZATION_CLIP = 8.0
SEQ_TARGET_CLIP_ENABLED = True
SEQ_TARGET_CLIP_LOWER_PCT = 0.2
SEQ_TARGET_CLIP_UPPER_PCT = 99.8

BATCH_NORM_ENABLED = False

VIX_IMAGE_CONV_CHANNELS = [64, 128, 128]
VIX_IMAGE_POOL_KERNELS = [3, 2, 2]
VIX_IMAGE_HEIGHT_DEFAULT = 96
VIX_IMAGE_WIDTH_DEFAULT = 180
VIX_IMAGE_BARS_DEFAULT = 60

VIX_FUSION_MODE_DEFAULT = "none" # valid modes: ['film', 'late_concat', 'none']
VIX_EMBED_DIM_DEFAULT = 8
VIX_NORM_METHOD_DEFAULT = "robust_zscore"
VIX_NORM_CLIP_DEFAULT = 5
VIX_LOG1P_DEFAULT = False
VIX_BETA_SCALING_ENABLED = False
VIX_DATE_COL_DEFAULT = "date"
VIX_VALUE_COL_DEFAULT = "close"
TICKER_CSV_DIR_PATH = str((Path(__file__).resolve().parent / "tickers"))
TICKER_DATE_COL_DEFAULT = "date"
TICKER_BETA_COL_DEFAULT = "beta"

KERNEL_SIZE = 5
KERNEL_WIDTH = 3
POOL_KERNEL = 2
POOL_STRIDE = 2
POOL_DIM = "height"
LRELU_SLOPE = 0.05
WEIGHT_INIT = "xavier_uniform"
XAVIER_GAIN = float(
    os.getenv("XAVIER_GAIN", str(nn.init.calculate_gain("leaky_relu", LRELU_SLOPE)))
)

# Training defaults
BATCH_SIZE = 256
WEIGHT_DECAY = 0.0
FC_DROPOUT = .5
# ELASTIC_NET_L1 = 3e-7
# ELASTIC_NET_L2 = 3e-6
ELASTIC_NET_L1 = 3e-9
ELASTIC_NET_L2 = 3e-7

SEED = 7
AMP_ENABLED_DEFAULT = 1
CLASS_WEIGHTED_CE_ENABLED_DEFAULT = 0
WRITE_ESTIMATED_EPOCH_TIME_ALONE_DEFAULT = 1
TRAIN_MIXED_SHARD_BATCHING_ENABLED = True
TRAIN_MIXED_SHARD_ACTIVE_SHARDS = 16
BUNDLE_ITER_CHUNK_SAMPLES_DEFAULT = 65536
KERAS_INTRA_EPOCH_LOGGING_ENABLED = True
KERAS_PROGRESS_BAR_WIDTH = 30
KERAS_PROGRESS_UPDATE_EVERY_STEPS = 50
KERAS_VAL_PROGRESS_ENABLED = True
EPOCH_STAGE_TIMING_LOGGING_ENABLED = bool(
    int(os.getenv("EPOCH_STAGE_TIMING_LOGGING_ENABLED", "1"))
)
# Keep prediction export in fp32 by default to avoid fp16/bf16 quantization flattening.
PREDS_EXPORT_AMP_ENABLED = bool(int(os.getenv("PREDS_EXPORT_AMP_ENABLED", "0")))

# Per-epoch-alone estimate defaults (single-run, no concurrent jobs).
# These are intentionally simple knobs so behavior can be toggled/retuned quickly.
ESTIMATED_EPOCH_TRAIN_SPS_ALONE_DEFAULT = 5600.0
ESTIMATED_EPOCH_VAL_SPS_ALONE_DEFAULT = 14000.0
ESTIMATED_EPOCH_FIXED_OVERHEAD_SEC_DEFAULT = 8.0

# Split defaults
VAL_FRACTION = 0.18
TEST_FRACTION = 0.001
PLOT_QUANTILES_PCT = [
    1,
    2,
    5,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    95,
    98,
    99,
]
DAILY_CROSS_SECTIONAL_TOP_PCT = [.01]
DAILY_CROSS_SECTIONAL_BOTTOM_PCT = [.01]
DAILY_CROSS_SECTIONAL_MIN_PER_SIDE = 1
DAILY_CROSS_SECTIONAL_MIN_NAMES_PER_DAY = 2
DAILY_CROSS_SECTIONAL_ANNUALIZATION_DAYS = 252.0
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
WALKFORWARD_SPY_DAILY_CSV_PATH = os.getenv(
    "WALKFORWARD_SPY_DAILY_CSV",
    str((Path(__file__).resolve().parent / "tickers" / "SPY.csv")),
)
WALKFORWARD_SPY_DATE_COL = "date"
WALKFORWARD_SPY_CLOSE_COL = "close"
EVAL_TAIL_RET_OUTLIER_ZSCORE_THRESHOLD = 6.0
# Controls end-of-training post-run simulation evaluations (sim_evals.py).
# Set to False to skip launching post-run evals.
POST_RUN_EVALS_ENABLED = True

# Overfit sanity defaults
OVERFIT_SANITY_DEFAULT = 0
OVERFIT_SAMPLE_START = 0
OVERFIT_SAMPLE_SIZE = 3000000
OVERFIT_VAL_SAMPLE_START = 5089000
OVERFIT_VAL_SAMPLE_SIZE = 400000

RET_ATR_THRESHOLD = 0.0
RUNS_ROOT = Path(os.getenv("RUNS_ROOT", "runs"))

_WALKFORWARD_SPY_RET_PCT_CACHE: dict[tuple[str, int, str, str], dict[str, float]] = {}


@dataclass(frozen=True)
class ShardInfo:
    file: str
    sample_start: int
    sample_end: int
    count: int


@dataclass(frozen=True)
class SplitRanges:
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]


@dataclass
class ShardBatchState:
    x: np.ndarray
    y: np.ndarray
    perm: np.ndarray
    sample_indices: np.ndarray
    ret_atr: np.ndarray | None
    ret_pct: np.ndarray | None
    timestamps: np.ndarray | None
    ticker_ids: np.ndarray | None
    vix: np.ndarray | None
    vix_img: np.ndarray | None
    cursor: int = 0


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str
    epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    elastic_net_l1: float
    elastic_net_l2: float
    batch_size: int
    fc_dropout: float
    readout_dim: int
    head_dims: list[int]
    seed: int
    amp_enabled: bool
    device: str
    distributed: bool
    ret_atr_threshold: float
    val_fraction: float
    test_fraction: float
    overfit_sanity: bool
    overfit_sample_start: int
    overfit_sample_size: int
    overfit_val_sample_start: int
    overfit_val_sample_size: int
    multi_gpu: bool
    torch_compile: bool
    class_weighted_ce_enabled: bool
    vix_daily_csv: str
    vix_date_col: str
    vix_value_col: str
    vix_fusion_mode: str
    vix_embed_dim: int
    vix_norm_method: str
    vix_norm_clip: float
    vix_log1p: bool
    walkforward_min_per_side_mode: str
    shuffle_train_labels: bool
    zero_image: bool


def parse_int_list(text: str) -> list[int]:
    if not text:
        return []
    out = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return int(default)


def resolve_vix_fusion_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in ("", "none", "off", "disabled", "false", "0"):
        return "none"
    if mode in ("film", "late_concat"):
        return mode
    raise ValueError(
        "vix_fusion_mode must be one of: none, film, late_concat; "
        f"got {value!r}"
    )


def resolve_vix_norm_method(value: str) -> str:
    method = str(value).strip().lower()
    if method in ("", "none", "off", "disabled", "false", "0"):
        return "none"
    if method in ("robust_zscore", "zscore"):
        return method
    raise ValueError(
        "vix_norm_method must be one of: robust_zscore, zscore, none; "
        f"got {value!r}"
    )


def resolve_seq_regression_loss_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in ("smoothl1", "smooth_l1", "huber"):
        return "smoothl1"
    if mode in ("mse", "l2"):
        return "mse"
    raise ValueError(
        "seq regression loss mode must be one of: smoothl1, mse; "
        f"got {value!r}"
    )


def resolve_model_branch_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in ("decomp", "image", "classifier"):
        return "decomp"
    if mode in ("seq", "sequence", "regression"):
        return "seq"
    if mode in ("dual", "joint", "multitask", "multi_task"):
        return "dual"
    raise ValueError(
        "model branch mode must be one of: decomp, seq, dual; "
        f"got {value!r}"
    )


def resolve_ts_crop_keep_side(value: str) -> str:
    side = str(value).strip().lower()
    if side in ("left", "l"):
        return "left"
    if side in ("right", "r"):
        return "right"
    raise ValueError(
        "TS width crop keep side must be one of: left, right; "
        f"got {value!r}"
    )


def derive_dual_npz_from_decomp_source(source_npz: str | Path) -> Path:
    path = Path(str(source_npz))
    if str(path.suffix).lower() != ".npz":
        raise ValueError(f"decomp source path must end with .npz; got {path}")
    stem = str(path.with_suffix(""))
    suffix = "_decomp"
    if not stem.endswith(suffix):
        raise ValueError(
            "decomp source path must end with '_decomp.npz' so a paired dual path can be derived; "
            f"got {path}"
        )
    return Path(stem[: -len(suffix)] + "_dual.npz")


@dataclass(frozen=True)
class VixNormStats:
    method: str
    center: float
    scale: float
    clip: float
    log1p: bool
    raw_train_count: int
    raw_train_min: float
    raw_train_max: float
    raw_train_mean: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float32).reshape(-1)
        out = np.zeros_like(x, dtype=np.float32)
        if x.size <= 0:
            return out
        finite = np.isfinite(x)
        if not np.any(finite):
            return out
        xv = x[finite].astype(np.float64, copy=False)
        if bool(self.log1p):
            xv = np.log1p(np.clip(xv, a_min=0.0, a_max=None))
        scale = float(self.scale)
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        z = (xv - float(self.center)) / scale
        clip = float(self.clip)
        if clip > 0.0:
            z = np.clip(z, -clip, clip)
        out[finite] = z.astype(np.float32, copy=False)
        return out


def fit_vix_norm_stats(
    raw_values: np.ndarray,
    method: str,
    clip: float,
    log1p: bool,
) -> VixNormStats:
    x = np.asarray(raw_values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size <= 0:
        raise ValueError("cannot fit VIX normalizer: no finite values")

    raw_min = float(np.min(x))
    raw_max = float(np.max(x))
    raw_mean = float(np.mean(x))
    tx = np.log1p(np.clip(x, a_min=0.0, a_max=None)) if bool(log1p) else x
    norm_method = resolve_vix_norm_method(method)

    if norm_method == "none":
        center = 0.0
        scale = 1.0
    elif norm_method == "zscore":
        center = float(np.mean(tx))
        scale = float(np.std(tx))
    else:
        center = float(np.median(tx))
        q25, q75 = np.percentile(tx, [25.0, 75.0])
        # Convert IQR to a Gaussian-equivalent std estimate for scale stability.
        scale = float((q75 - q25) / 1.349)
        if (not np.isfinite(scale)) or scale <= 0.0:
            scale = float(np.std(tx))

    if (not np.isfinite(scale)) or scale <= 0.0:
        scale = 1.0
    clip_v = max(0.0, float(clip))

    return VixNormStats(
        method=str(norm_method),
        center=float(center),
        scale=float(scale),
        clip=float(clip_v),
        log1p=bool(log1p),
        raw_train_count=int(x.size),
        raw_train_min=float(raw_min),
        raw_train_max=float(raw_max),
        raw_train_mean=float(raw_mean),
    )


def load_daily_vix_lookup_csv(
    csv_path: Path,
    date_col: str = VIX_DATE_COL_DEFAULT,
    value_col: str = VIX_VALUE_COL_DEFAULT,
) -> dict[str, float]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"daily VIX CSV not found: {path}")

    date_key = str(date_col).strip()
    value_key = str(value_col).strip()
    if not date_key:
        raise ValueError("vix_date_col must be non-empty")
    if not value_key:
        raise ValueError("vix_value_col must be non-empty")

    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"daily VIX CSV has no header row: {path}")
        header_map = {str(name).strip().lower(): str(name) for name in reader.fieldnames}
        date_col_key = header_map.get(date_key.lower())
        value_col_key = header_map.get(value_key.lower())
        if date_col_key is None:
            raise ValueError(
                f"daily VIX CSV missing date column {date_key!r}: {path} "
                f"(available={reader.fieldnames})"
            )
        if value_col_key is None:
            raise ValueError(
                f"daily VIX CSV missing value column {value_key!r}: {path} "
                f"(available={reader.fieldnames})"
            )

        for row in reader:
            raw_date = str(row.get(date_col_key, "")).strip()
            iso_date = coerce_to_iso_date(raw_date)
            if iso_date is None:
                continue
            raw_val = str(row.get(value_col_key, "")).strip()
            if not raw_val:
                continue
            try:
                value = float(raw_val)
            except ValueError:
                continue
            if not np.isfinite(value):
                continue
            out[iso_date] = float(value)

    if not out:
        raise ValueError(
            f"daily VIX CSV contained no usable rows for columns "
            f"{date_key!r}/{value_key!r}: {path}"
        )
    return out


def coerce_to_iso_date(raw: str) -> str | None:
    s = str(raw).strip()
    if not s:
        return None
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    for fmt in ("%m/%d/%Y", "%Y/%m/%d", "%m-%d-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:10], fmt).date().isoformat()
        except ValueError:
            continue
    return None


def normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper().replace(".", "-")


def load_ticker_lookup(data_dir: Path) -> list[str] | None:
    tickers_path = Path(data_dir) / "tickers.npy"
    if not tickers_path.is_file():
        return None
    try:
        arr = np.load(tickers_path, allow_pickle=True)
    except Exception:
        return None
    flat = np.asarray(arr).reshape(-1)
    return [str(v) for v in flat]


def load_daily_beta_lookup_csv(
    csv_path: Path,
    date_col: str = TICKER_DATE_COL_DEFAULT,
    beta_col: str = TICKER_BETA_COL_DEFAULT,
) -> dict[str, float]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"ticker CSV not found: {path}")

    date_key = str(date_col).strip()
    beta_key = str(beta_col).strip()
    if not date_key:
        raise ValueError("ticker date column name must be non-empty")
    if not beta_key:
        raise ValueError("ticker beta column name must be non-empty")

    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"ticker CSV has no header row: {path}")
        header_map = {str(name).strip().lower(): str(name) for name in reader.fieldnames}
        date_col_key = header_map.get(date_key.lower())
        beta_col_key = header_map.get(beta_key.lower())
        if date_col_key is None:
            raise ValueError(
                f"ticker CSV missing date column {date_key!r}: {path} "
                f"(available={reader.fieldnames})"
            )
        if beta_col_key is None:
            raise ValueError(
                f"ticker CSV missing beta column {beta_key!r}: {path} "
                f"(available={reader.fieldnames})"
            )

        for row in reader:
            raw_date = str(row.get(date_col_key, "")).strip()
            iso_date = coerce_to_iso_date(raw_date)
            if iso_date is None:
                continue
            raw_beta = str(row.get(beta_col_key, "")).strip()
            if not raw_beta:
                continue
            try:
                beta = float(raw_beta)
            except ValueError:
                continue
            if not np.isfinite(beta):
                continue
            out[iso_date] = float(beta)

    if not out:
        raise ValueError(
            f"ticker CSV contained no usable beta rows for columns "
            f"{date_key!r}/{beta_key!r}: {path}"
        )
    return out


def load_first_finite_beta_value_csv(
    csv_path: Path,
    beta_col: str = TICKER_BETA_COL_DEFAULT,
) -> float | None:
    path = Path(csv_path)
    if not path.is_file():
        return None
    beta_key = str(beta_col).strip()
    if not beta_key:
        return None
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return None
        header_map = {str(name).strip().lower(): str(name) for name in reader.fieldnames}
        beta_col_key = header_map.get(beta_key.lower())
        if beta_col_key is None:
            return None
        for row in reader:
            raw_beta = str(row.get(beta_col_key, "")).strip()
            if not raw_beta:
                continue
            try:
                beta = float(raw_beta)
            except ValueError:
                continue
            if np.isfinite(beta):
                return float(beta)
    return None


def resolve_image_width_per_scale(
    manifest: dict,
    scales: int,
    image_width: int,
) -> list[int]:
    if not bool(JAGGED_IMAGE_WIDTHS_ENABLED):
        day_width = int(manifest.get("day_width", 0) or 0)
        windows = int(manifest.get("decomposition_windows", 0) or 0)
        width = int(day_width * windows) if day_width > 0 and windows > 0 else 0
        if width < 1:
            width = int(image_width)
        if width < 1:
            raw = manifest.get("image_width_per_scale")
            if isinstance(raw, list) and raw:
                width = int(max(int(x) for x in raw))
            else:
                width = 1
        return [int(width) for _ in range(int(scales))]

    raw = manifest.get("image_width_per_scale")
    if isinstance(raw, list) and len(raw) == int(scales):
        widths = [int(x) for x in raw]
    else:
        day_width = int(manifest.get("day_width", 0) or 0)
        windows = int(manifest.get("decomposition_windows", 0) or 0)
        if day_width > 0 and windows > 0:
            base_width = int(day_width * windows)
            widths = [base_width for _ in range(int(scales))]
            if bool(manifest.get("weekend_feature_enabled", False)):
                gap_width = int(manifest.get("weekend_gap_width", 0) or 0)
                weekend_scales = manifest.get("weekend_gap_scale_indices", [])
                weekend_set = {int(x) for x in weekend_scales}
                if gap_width > 0:
                    for s in range(int(scales)):
                        if s in weekend_set:
                            widths[s] += int(gap_width)
        else:
            fallback = int(image_width) if int(image_width) > 0 else 1
            widths = [fallback for _ in range(int(scales))]

    if len(widths) != int(scales):
        raise ValueError(
            "image width spec does not match decomposition scales: "
            f"len(widths)={len(widths)} scales={scales}"
        )
    if any(int(w) < 1 for w in widths):
        raise ValueError(f"image widths must be >= 1, got {widths}")
    return [int(w) for w in widths]


def resolve_jagged_concat_option(value: int) -> int:
    opt = int(value)
    if opt not in (1, 2, 3, 4):
        raise ValueError(
            "JAGGED_2D_CONCAT_OPTION must be one of: 1, 2, 3, 4; "
            f"got {value!r}"
        )
    return opt


def jagged_concat_option_name(option: int) -> str:
    opt = int(option)
    if opt == 1:
        return "option1_pad_to_max_width"
    if opt == 2:
        return "option2_crop_to_min_width"
    if opt == 3:
        return "option3_resize_to_target_width"
    if opt == 4:
        return "option4_pad_to_max_with_valid_width_mask"
    return "unknown"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_msf_scale_weights(scale_count: int) -> list[float]:
    count = int(scale_count)
    if count <= 0:
        raise ValueError(f"scale_count must be >= 1, got {scale_count}")
    cfg = [float(v) for v in MSF_SCALE_WEIGHTS]
    if len(cfg) == 1:
        weights = [float(cfg[0]) for _ in range(count)]
    elif len(cfg) == count:
        weights = [float(v) for v in cfg]
    else:
        # Fallback keeps non-3-scale setups working if default list length differs.
        weights = [0.5 if i == 0 else 0.5 / (2 ** (i - 1)) for i in range(count)]
    for idx, weight in enumerate(weights):
        if not math.isfinite(float(weight)):
            raise ValueError(
                f"MSF scale weights must be finite; got weights={weights}"
            )
        if float(weight) < 0.0:
            raise ValueError(
                f"MSF scale weights must be >= 0; got weight={weight} at scale={idx}"
            )
    return [float(w) for w in weights]


def compute_msf_weight(scale_idx: int, scale_count: int) -> float:
    if scale_idx < 0 or scale_idx >= scale_count:
        raise ValueError("scale_idx out of range for msf weight")
    weights = resolve_msf_scale_weights(scale_count)
    return float(weights[int(scale_idx)])


def compute_msf_out_channels(weight: float, base_channels: int = CONV_CHANNELS[-1]) -> int:
    w = float(weight)
    if w <= 0.0:
        return 0
    return max(1, int(base_channels * w))


def resolve_pool_shape() -> tuple[tuple[int, int], tuple[int, int]]:
    token = str(POOL_DIM).strip().lower()
    if token in ("height", "h", "rows", "row", "time"):
        return (POOL_KERNEL, 1), (POOL_STRIDE, 1)
    if token in ("width", "w", "cols", "col", "window"):
        return (1, POOL_KERNEL), (1, POOL_STRIDE)
    if token in ("both", "hw", "2d"):
        return (POOL_KERNEL, POOL_KERNEL), (POOL_STRIDE, POOL_STRIDE)
    raise ValueError("POOL_DIM must be one of: height, width, both")


def elastic_net_penalty(model: nn.Module, l1_lambda: float, l2_lambda: float) -> torch.Tensor:
    l1 = float(l1_lambda)
    l2 = float(l2_lambda)
    if l1 <= 0.0 and l2 <= 0.0:
        return torch.tensor(0.0)

    penalty: torch.Tensor | None = None
    for p in model.parameters():
        if not p.requires_grad or p.ndim <= 1:
            continue
        term: torch.Tensor | None = None
        if l1 > 0.0:
            t1 = p.abs().sum() * l1
            term = t1 if term is None else (term + t1)
        if l2 > 0.0:
            t2 = p.square().sum() * l2
            term = t2 if term is None else (term + t2)
        if term is not None:
            penalty = term if penalty is None else (penalty + term)

    if penalty is None:
        device = next((p.device for p in model.parameters()), torch.device("cpu"))
        return torch.tensor(0.0, device=device)
    return penalty


def init_module_weights(module: nn.Module) -> None:
    if not isinstance(module, (nn.Conv2d, nn.Linear)):
        return
    mode = WEIGHT_INIT
    if mode in ("none", "default"):
        return
    if mode == "xavier_uniform":
        nn.init.xavier_uniform_(module.weight, gain=XAVIER_GAIN)
    elif mode == "xavier_normal":
        nn.init.xavier_normal_(module.weight, gain=XAVIER_GAIN)
    else:
        raise ValueError(
            f"WEIGHT_INIT must be one of: xavier_uniform, xavier_normal, default; got {WEIGHT_INIT!r}"
        )
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class MsfBlock(nn.Module):
    """MSF block from spec:
    5x3 conv(64) -> BN -> LReLU -> 5x3 conv(128) -> BN -> LReLU -> 2x1 maxpool ->
    5x3 conv(256*weight_i) -> BN -> LReLU -> 2x1 maxpool.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        pad = (KERNEL_SIZE // 2, KERNEL_WIDTH // 2)
        pool_kernel, pool_stride = resolve_pool_shape()
        self.conv1 = nn.Conv2d(1, CONV_CHANNELS[0], kernel_size=(KERNEL_SIZE, KERNEL_WIDTH), padding=pad)
        self.bn1 = nn.BatchNorm2d(CONV_CHANNELS[0]) if bool(BATCH_NORM_ENABLED) else nn.Identity()
        self.act1 = nn.LeakyReLU(LRELU_SLOPE)
        self.conv2 = nn.Conv2d(CONV_CHANNELS[0], CONV_CHANNELS[1], kernel_size=(KERNEL_SIZE, KERNEL_WIDTH), padding=pad)
        self.bn2 = nn.BatchNorm2d(CONV_CHANNELS[1]) if bool(BATCH_NORM_ENABLED) else nn.Identity()
        self.act2 = nn.LeakyReLU(LRELU_SLOPE)
        self.pool1 = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
        self.conv3 = nn.Conv2d(CONV_CHANNELS[1], int(out_channels), kernel_size=(KERNEL_SIZE, KERNEL_WIDTH), padding=pad)
        self.bn3 = nn.BatchNorm2d(int(out_channels)) if bool(BATCH_NORM_ENABLED) else nn.Identity()
        self.act3 = nn.LeakyReLU(LRELU_SLOPE)
        self.pool2 = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
        self.out_channels = int(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        return x


class VixImageBlock(nn.Module):
    """Dedicated CNN encoder for VIX OHLC image input."""

    def __init__(self):
        super().__init__()
        channels = [int(v) for v in VIX_IMAGE_CONV_CHANNELS]
        if len(channels) != 3 or any(ch < 1 for ch in channels):
            raise ValueError(
                f"VIX_IMAGE_CONV_CHANNELS must contain 3 positive ints, got {channels}"
            )
        pools = [int(v) for v in VIX_IMAGE_POOL_KERNELS]
        if len(pools) != 3 or any(v < 1 for v in pools):
            raise ValueError(
                f"VIX_IMAGE_POOL_KERNELS must contain 3 positive ints, got {pools}"
            )

        pad = (KERNEL_SIZE // 2, KERNEL_WIDTH // 2)
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=(KERNEL_SIZE, KERNEL_WIDTH), padding=pad)
        self.bn1 = nn.BatchNorm2d(channels[0]) if bool(BATCH_NORM_ENABLED) else nn.Identity()
        self.act1 = nn.LeakyReLU(LRELU_SLOPE)
        self.pool1 = nn.MaxPool2d(kernel_size=(pools[0], pools[0]), stride=(pools[0], pools[0]))
        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=(KERNEL_SIZE, KERNEL_WIDTH), padding=pad)
        self.bn2 = nn.BatchNorm2d(channels[1]) if bool(BATCH_NORM_ENABLED) else nn.Identity()
        self.act2 = nn.LeakyReLU(LRELU_SLOPE)
        self.pool2 = nn.MaxPool2d(kernel_size=(pools[1], pools[1]), stride=(pools[1], pools[1]))
        self.conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=(KERNEL_SIZE, KERNEL_WIDTH), padding=pad)
        self.bn3 = nn.BatchNorm2d(channels[2]) if bool(BATCH_NORM_ENABLED) else nn.Identity()
        self.act3 = nn.LeakyReLU(LRELU_SLOPE)
        self.pool3 = nn.MaxPool2d(kernel_size=(pools[2], pools[2]), stride=(pools[2], pools[2]))
        self.out_channels = int(channels[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        x = self.pool3(self.act3(self.bn3(self.conv3(x))))
        return x


class DecompImageClassifier(nn.Module):
    """Decomp image model with one MSF block per scale and cls-only head."""

    def __init__(
        self,
        scales: int,
        input_height: int,
        input_width: int,
        readout_dim: int,
        head_dims: Sequence[int],
        fc_dropout: float,
        input_width_per_scale: Sequence[int] | None = None,
        vix_fusion_mode: str = VIX_FUSION_MODE_DEFAULT,
        vix_embed_dim: int = VIX_EMBED_DIM_DEFAULT,
        include_vix_image: bool = False,
        vix_image_height: int = VIX_IMAGE_HEIGHT_DEFAULT,
        vix_image_width: int = VIX_IMAGE_WIDTH_DEFAULT,
    ) -> None:
        super().__init__()
        self.jagged_image_widths_enabled = bool(JAGGED_IMAGE_WIDTHS_ENABLED)
        self.jagged_concat_option = resolve_jagged_concat_option(JAGGED_2D_CONCAT_OPTION)
        self.jagged_option3_target_width = int(JAGGED_OPTION3_TARGET_WIDTH)
        self.jagged_option4_add_valid_width_mask = bool(
            self.jagged_image_widths_enabled and self.jagged_concat_option == 4
        )
        if self.jagged_concat_option == 3 and self.jagged_option3_target_width < 1:
            raise ValueError(
                "JAGGED_OPTION3_TARGET_WIDTH must be >= 1 when JAGGED_2D_CONCAT_OPTION=3"
            )
        if not self.jagged_image_widths_enabled:
            self.fusion_mode = "legacy_concat_2d_then_flatten"
        else:
            self.fusion_mode = jagged_concat_option_name(self.jagged_concat_option)

        self.scales = int(scales)
        self.input_height = int(input_height)
        width_max = int(input_width)
        if (not self.jagged_image_widths_enabled) or input_width_per_scale is None:
            widths = [width_max for _ in range(self.scales)]
        else:
            widths = [int(w) for w in input_width_per_scale]
            if len(widths) != self.scales:
                raise ValueError(
                    "input_width_per_scale length must match scales; "
                    f"len(widths)={len(widths)} scales={self.scales}"
                )
            if width_max <= 0:
                width_max = int(max(widths))
        if any(int(w) < 1 for w in widths):
            raise ValueError(f"input widths must be >= 1, got {widths}")
        self.input_width_per_scale = tuple(int(w) for w in widths)
        self.input_width = int(max(width_max, max(self.input_width_per_scale)))
        self.vix_fusion_mode = resolve_vix_fusion_mode(str(vix_fusion_mode))
        self.vix_embed_dim = int(vix_embed_dim)
        self.include_vix_image = bool(include_vix_image)
        self.vix_image_height = int(vix_image_height)
        self.vix_image_width = int(vix_image_width)
        if self.vix_fusion_mode != "none" and self.vix_embed_dim < 1:
            raise ValueError(
                f"vix_embed_dim must be >= 1 when vix_fusion_mode={self.vix_fusion_mode!r}; "
                f"got {vix_embed_dim!r}"
            )
        if self.include_vix_image:
            if self.vix_image_height < 1 or self.vix_image_width < 1:
                raise ValueError(
                    "vix image dimensions must be >= 1 when include_vix_image is enabled; "
                    f"got h={self.vix_image_height} w={self.vix_image_width}"
                )

        self.msf_scale_weights = tuple(resolve_msf_scale_weights(self.scales))
        active_scale_indices: list[int] = []
        active_scale_weights: list[float] = []
        disabled_scale_indices: list[int] = []
        blocks: list[nn.Module] = []
        for i, w in enumerate(self.msf_scale_weights):
            out_ch = compute_msf_out_channels(w, base_channels=CONV_CHANNELS[-1])
            if out_ch <= 0:
                disabled_scale_indices.append(int(i))
                blocks.append(nn.Identity())
                continue
            active_scale_indices.append(int(i))
            active_scale_weights.append(float(w))
            blocks.append(MsfBlock(out_channels=out_ch))
        if not active_scale_indices:
            raise ValueError(
                "MSF_SCALE_WEIGHTS disables all scales; at least one scale weight must be > 0"
            )
        self.active_scale_indices = tuple(active_scale_indices)
        self.active_scale_weights = tuple(active_scale_weights)
        self.disabled_scale_indices = tuple(disabled_scale_indices)
        self.msf_blocks = nn.ModuleList(blocks)
        self.vix_image_block: nn.Module | None = (
            VixImageBlock() if self.include_vix_image else None
        )

        with torch.no_grad():
            reps_raw: list[torch.Tensor] = []
            for scale_idx in self.active_scale_indices:
                blk = self.msf_blocks[scale_idx]
                if not self.jagged_image_widths_enabled:
                    dummy = torch.zeros(1, 1, self.input_height, self.input_width)
                else:
                    dummy = torch.zeros(
                        1,
                        1,
                        self.input_height,
                        int(self.input_width_per_scale[scale_idx]),
                    )
                reps_raw.append(blk(dummy))
            reps_aligned, aligned_width = self._align_rep_maps_2d(reps_raw)
            rep = torch.cat(reps_aligned, dim=1)
            if self.include_vix_image:
                if self.vix_image_block is None:
                    raise RuntimeError("vix image block missing while include_vix_image=True")
                vix_dummy = torch.zeros(
                    1,
                    1,
                    int(self.vix_image_height),
                    int(self.vix_image_width),
                )
                vix_rep = self.vix_image_block(vix_dummy)
                vix_rep = self._align_vix_rep_map(
                    vix_rep,
                    target_h=int(rep.shape[2]),
                    target_w=int(rep.shape[3]),
                )
                rep = torch.cat([rep, vix_rep], dim=1)
            self.rep_shape = (int(rep.shape[1]), int(rep.shape[2]), int(rep.shape[3]))
            self.rep_aligned_width = int(aligned_width)
            flat_dim = int(rep.shape[1] * rep.shape[2] * rep.shape[3])

        self.readout = nn.Linear(flat_dim, int(readout_dim))
        self.readout_act = nn.LeakyReLU(LRELU_SLOPE)
        self.readout_drop = nn.Dropout(float(fc_dropout)) if fc_dropout > 0 else nn.Identity()
        self.vix_film: nn.Module | None = None
        self.vix_embed: nn.Module | None = None
        if self.vix_fusion_mode == "film":
            rep_channels = int(self.rep_shape[0])
            self.vix_film = nn.Sequential(
                nn.Linear(1, int(self.vix_embed_dim)),
                nn.LeakyReLU(LRELU_SLOPE),
                nn.Linear(int(self.vix_embed_dim), int(2 * rep_channels)),
            )
        elif self.vix_fusion_mode == "late_concat":
            self.vix_embed = nn.Sequential(
                nn.Linear(1, int(self.vix_embed_dim)),
                nn.LeakyReLU(LRELU_SLOPE),
                nn.Linear(int(self.vix_embed_dim), int(self.vix_embed_dim)),
                nn.LeakyReLU(LRELU_SLOPE),
            )

        head_input_dim = int(readout_dim)
        if self.vix_fusion_mode == "late_concat":
            head_input_dim += int(self.vix_embed_dim)
        dims = [int(head_input_dim)] + [int(v) for v in head_dims] + [2]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LeakyReLU(LRELU_SLOPE))
            if fc_dropout > 0:
                layers.append(nn.Dropout(float(fc_dropout)))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.cls_head = nn.Sequential(*layers)

        self.apply(init_module_weights)

    def _align_rep_maps_2d(
        self,
        reps: Sequence[torch.Tensor],
    ) -> tuple[list[torch.Tensor], int]:
        if not reps:
            raise ValueError("no per-scale representations to align")
        heights = {int(r.shape[2]) for r in reps}
        if len(heights) != 1:
            raise ValueError(
                "all scale representations must have equal height for 2D concat; "
                f"got heights={sorted(heights)}"
            )
        widths = [int(r.shape[3]) for r in reps]
        if any(w < 1 for w in widths):
            raise ValueError(f"invalid representation widths: {widths}")
        opt = int(self.jagged_concat_option)
        add_valid_width_mask = bool(self.jagged_image_widths_enabled and opt == 4)
        if len(set(widths)) == 1 and not add_valid_width_mask:
            return [r.contiguous() for r in reps], int(widths[0])

        if not self.jagged_image_widths_enabled:
            target_w = int(max(widths))
            out = []
            for r in reps:
                pad_w = target_w - int(r.shape[3])
                out.append(
                    F.pad(r, (0, pad_w, 0, 0)).contiguous() if pad_w > 0 else r.contiguous()
                )
            return out, int(target_w)

        if opt == 1:
            target_w = int(max(widths))
            out = []
            for r in reps:
                pad_w = target_w - int(r.shape[3])
                out.append(
                    F.pad(r, (0, pad_w, 0, 0)).contiguous() if pad_w > 0 else r.contiguous()
                )
            return out, int(target_w)
        if opt == 2:
            target_w = int(min(widths))
            return [r[:, :, :, :target_w].contiguous() for r in reps], int(target_w)
        if opt == 4:
            target_w = int(max(widths))
            out = []
            for r in reps:
                valid_w = int(r.shape[3])
                pad_w = target_w - valid_w
                rp = F.pad(r, (0, pad_w, 0, 0)).contiguous() if pad_w > 0 else r.contiguous()
                mask = torch.zeros(
                    (int(r.shape[0]), 1, int(r.shape[2]), target_w),
                    dtype=rp.dtype,
                    device=rp.device,
                )
                mask[:, :, :, :valid_w] = 1.0
                out.append(torch.cat([rp, mask], dim=1).contiguous())
            return out, int(target_w)

        target_w = int(self.jagged_option3_target_width)
        out = []
        for r in reps:
            if int(r.shape[3]) == target_w:
                out.append(r.contiguous())
                continue
            out.append(
                F.interpolate(
                    r,
                    size=(int(r.shape[2]), target_w),
                    mode="nearest",
                ).contiguous()
            )
        return out, int(target_w)

    def _align_vix_rep_map(
        self,
        vix_rep: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        if vix_rep.ndim != 4:
            raise ValueError(f"expected VIX rep shape (batch, C, H, W); got {tuple(vix_rep.shape)}")
        out = vix_rep
        th = int(target_h)
        tw = int(target_w)
        if int(out.shape[2]) != th:
            out = F.interpolate(
                out,
                size=(th, int(out.shape[3])),
                mode="nearest",
            ).contiguous()
        cur_w = int(out.shape[3])
        if cur_w < tw:
            out = F.pad(out, (0, tw - cur_w, 0, 0)).contiguous()
        elif cur_w > tw:
            out = out[:, :, :, :tw].contiguous()
        return out

    def _prepare_vix_image(
        self,
        vix_img: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor | None:
        if not self.include_vix_image:
            return None
        if vix_img is None:
            raise ValueError("vix image tensor is required when include_vix_image=True")
        if vix_img.ndim == 3:
            vix_img = vix_img.unsqueeze(1)
        elif vix_img.ndim != 4 or int(vix_img.shape[1]) != 1:
            raise ValueError(
                "expected vix_img shape (batch,H,W) or (batch,1,H,W); "
                f"got {tuple(vix_img.shape)}"
            )
        if int(vix_img.shape[0]) != int(batch_size):
            raise ValueError(
                "vix_img batch dimension must match image batch; "
                f"got vix_img={int(vix_img.shape[0])} batch={int(batch_size)}"
            )
        return vix_img.to(dtype=torch.float32)

    def _prepare_vix(self, vix: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
        if vix is None:
            if self.vix_fusion_mode != "none":
                raise ValueError(
                    f"vix tensor is required when vix_fusion_mode={self.vix_fusion_mode!r}"
                )
            return None
        if vix.ndim == 1:
            vix = vix.unsqueeze(1)
        elif vix.ndim != 2 or int(vix.shape[1]) != 1:
            raise ValueError(f"expected vix shape (batch,) or (batch,1); got {tuple(vix.shape)}")
        if int(vix.shape[0]) != int(batch_size):
            raise ValueError(
                "vix batch dimension must match image batch; "
                f"got vix={int(vix.shape[0])} batch={int(batch_size)}"
            )
        return vix.to(dtype=torch.float32)

    def _apply_vix_film(self, rep_map: torch.Tensor, vix: torch.Tensor) -> torch.Tensor:
        if self.vix_fusion_mode != "film":
            return rep_map
        if self.vix_film is None:
            raise RuntimeError("vix_film module missing while vix_fusion_mode='film'")
        film_params = self.vix_film(vix)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        gamma = 0.25 * torch.tanh(gamma)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return rep_map * (1.0 + gamma) + beta

    def _encode_msf_map(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected input shape (batch, scales, H, W); got {tuple(x.shape)}")
        if int(x.shape[1]) != self.scales:
            raise ValueError(f"expected scales={self.scales}, got {int(x.shape[1])}")
        if int(x.shape[3]) < self.input_width:
            raise ValueError(
                f"expected width >= {self.input_width}, got {int(x.shape[3])}"
            )
        reps_raw: list[torch.Tensor] = []
        for scale_idx in self.active_scale_indices:
            block = self.msf_blocks[scale_idx]
            if not self.jagged_image_widths_enabled:
                xi = x[:, scale_idx : scale_idx + 1, :, : self.input_width]
            else:
                scale_w = int(self.input_width_per_scale[scale_idx])
                xi = x[:, scale_idx : scale_idx + 1, :, :scale_w]
            reps_raw.append(block(xi))
        reps, _ = self._align_rep_maps_2d(reps_raw)
        return torch.cat(reps, dim=1)

    def _encode_from_msf_map(
        self,
        msf_map: torch.Tensor,
        vix_ready: torch.Tensor | None = None,
        vix_img_ready: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rep = msf_map
        if self.include_vix_image:
            if self.vix_image_block is None:
                raise RuntimeError("vix image block unexpectedly missing")
            if vix_img_ready is None:
                raise RuntimeError("vix image tensor unexpectedly missing")
            vix_rep = self.vix_image_block(vix_img_ready)
            vix_rep = self._align_vix_rep_map(
                vix_rep,
                target_h=int(rep.shape[2]),
                target_w=int(rep.shape[3]),
            )
            rep = torch.cat([rep, vix_rep], dim=1)
        if self.vix_fusion_mode == "film":
            if vix_ready is None:
                raise RuntimeError("vix tensor unexpectedly missing for film fusion")
            rep = self._apply_vix_film(rep, vix_ready)
        rep = rep.flatten(start_dim=1)
        rep = self.readout_drop(self.readout_act(self.readout(rep)))
        if self.vix_fusion_mode == "late_concat":
            if vix_ready is None:
                raise RuntimeError("vix tensor unexpectedly missing for late-concat fusion")
            if self.vix_embed is None:
                raise RuntimeError("vix_embed module missing while vix_fusion_mode='late_concat'")
            rep = torch.cat([rep, self.vix_embed(vix_ready)], dim=1)
        return rep

    def _encode(
        self,
        x: torch.Tensor,
        vix: torch.Tensor | None = None,
        vix_img: torch.Tensor | None = None,
    ) -> torch.Tensor:
        msf_map = self._encode_msf_map(x)
        vix_ready = self._prepare_vix(vix, batch_size=int(x.shape[0]))
        vix_img_ready = self._prepare_vix_image(vix_img, batch_size=int(x.shape[0]))
        return self._encode_from_msf_map(
            msf_map=msf_map,
            vix_ready=vix_ready,
            vix_img_ready=vix_img_ready,
        )

    def forward_logits(
        self,
        x: torch.Tensor,
        vix: torch.Tensor | None = None,
        vix_img: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rep = self._encode(x, vix=vix, vix_img=vix_img)
        return self.cls_head(rep)

    def forward(
        self,
        x: torch.Tensor,
        vix: torch.Tensor | None = None,
        vix_img: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_logits(x, vix=vix, vix_img=vix_img)

    def forward_proba(
        self,
        x: torch.Tensor,
        vix: torch.Tensor | None = None,
        vix_img: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.forward_logits(x, vix=vix, vix_img=vix_img)
        return torch.softmax(logits, dim=1)


class TsBlock(nn.Module):
    """Two-stage temporal-spatial conv stack for sequence feature maps."""

    def __init__(self) -> None:
        super().__init__()
        pad = (KERNEL_SIZE // 2, KERNEL_WIDTH // 2)
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(KERNEL_SIZE, KERNEL_WIDTH), padding=pad)
        self.act1 = nn.LeakyReLU(LRELU_SLOPE)
        self.pool1 = nn.MaxPool2d(kernel_size=(POOL_KERNEL, 1), stride=(POOL_STRIDE, 1))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(KERNEL_SIZE, KERNEL_WIDTH), padding=pad)
        self.act2 = nn.LeakyReLU(LRELU_SLOPE)
        self.pool2 = nn.MaxPool2d(kernel_size=(POOL_KERNEL, 1), stride=(POOL_STRIDE, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        return x


class SequenceRegressor(nn.Module):
    """Regression model for X_seq with a TS block + MLP head."""

    def __init__(
        self,
        input_features: int,
        input_lookback: int,
        readout_dim: int,
        head_dims: Sequence[int],
        fc_dropout: float,
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.input_lookback = int(input_lookback)
        if self.input_features < 1 or self.input_lookback < 1:
            raise ValueError(
                "sequence input dims must be >= 1; "
                f"got features={self.input_features} lookback={self.input_lookback}"
            )
        self.ts_block = TsBlock()
        with torch.no_grad():
            dummy = torch.zeros(
                1, 1, int(self.input_features), int(self.input_lookback), dtype=torch.float32
            )
            rep = self.ts_block(dummy)
            flat_dim = int(rep.shape[1] * rep.shape[2] * rep.shape[3])
            self.rep_shape = (int(rep.shape[1]), int(rep.shape[2]), int(rep.shape[3]))
        self.readout = nn.Linear(flat_dim, int(readout_dim))
        self.readout_act = nn.LeakyReLU(LRELU_SLOPE)
        self.readout_drop = nn.Dropout(float(fc_dropout)) if fc_dropout > 0 else nn.Identity()

        dims = [int(readout_dim)] + [int(v) for v in head_dims] + [1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LeakyReLU(LRELU_SLOPE))
            if fc_dropout > 0:
                layers.append(nn.Dropout(float(fc_dropout)))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.reg_head = nn.Sequential(*layers)
        self.apply(init_module_weights)

    def forward_value(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.ndim != 3:
            raise ValueError(f"expected X_seq shape (batch,F,L); got {tuple(x_seq.shape)}")
        if int(x_seq.shape[1]) != self.input_features or int(x_seq.shape[2]) != self.input_lookback:
            raise ValueError(
                "unexpected X_seq shape: "
                f"expected (*,{self.input_features},{self.input_lookback}) "
                f"got {tuple(x_seq.shape)}"
            )
        x = x_seq.unsqueeze(1)
        rep = self.ts_block(x)
        rep = rep.flatten(start_dim=1)
        rep = self.readout_drop(self.readout_act(self.readout(rep)))
        return self.reg_head(rep).squeeze(1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return self.forward_value(x_seq)


class DualBranchImageClassifier(DecompImageClassifier):
    """Dual-branch image+sequence model with dominant cls head and assisted reg head."""

    def __init__(
        self,
        scales: int,
        input_height: int,
        input_width: int,
        readout_dim: int,
        head_dims: Sequence[int],
        fc_dropout: float,
        seq_input_features: int,
        seq_input_lookback: int,
        input_width_per_scale: Sequence[int] | None = None,
        vix_fusion_mode: str = VIX_FUSION_MODE_DEFAULT,
        vix_embed_dim: int = VIX_EMBED_DIM_DEFAULT,
        include_vix_image: bool = False,
        vix_image_height: int = VIX_IMAGE_HEIGHT_DEFAULT,
        vix_image_width: int = VIX_IMAGE_WIDTH_DEFAULT,
    ) -> None:
        super().__init__(
            scales=scales,
            input_height=input_height,
            input_width=input_width,
            input_width_per_scale=input_width_per_scale,
            readout_dim=readout_dim,
            head_dims=head_dims,
            fc_dropout=fc_dropout,
            vix_fusion_mode=vix_fusion_mode,
            vix_embed_dim=vix_embed_dim,
            include_vix_image=include_vix_image,
            vix_image_height=vix_image_height,
            vix_image_width=vix_image_width,
        )
        self.seq_input_features = int(seq_input_features)
        self.seq_input_lookback = int(seq_input_lookback)
        if self.seq_input_features < 1 or self.seq_input_lookback < 1:
            raise ValueError(
                "sequence input dims must be >= 1; "
                f"got features={self.seq_input_features} lookback={self.seq_input_lookback}"
            )
        self.ts_crop_keep_side = resolve_ts_crop_keep_side(TS_WIDTH_CROP_KEEP_SIDE)
        self.ts_block = TsBlock()
        with torch.no_grad():
            dummy_img = torch.zeros(
                1,
                self.scales,
                int(self.input_height),
                int(self.input_width),
                dtype=torch.float32,
            )
            msf_map = self._encode_msf_map(dummy_img)
            msf_channels = int(msf_map.shape[1])
            msf_h = int(msf_map.shape[2])
            msf_w = int(msf_map.shape[3])
            dummy_seq = torch.zeros(
                1,
                int(self.seq_input_features),
                int(self.seq_input_lookback),
                dtype=torch.float32,
            )
            ts_map = self.ts_block(dummy_seq.unsqueeze(1))
            ts_channels = int(ts_map.shape[1])

        self.ts_projection = nn.Sequential(
            nn.Conv2d(int(ts_channels), int(msf_channels), kernel_size=1),
            nn.LeakyReLU(LRELU_SLOPE),
        )
        self.regression_map_fusion = nn.Sequential(
            nn.Conv2d(int(msf_channels * 2), int(msf_channels), kernel_size=1),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv2d(int(msf_channels), int(msf_channels), kernel_size=3, padding=1),
            nn.LeakyReLU(LRELU_SLOPE),
        )
        with torch.no_grad():
            ts_map = self.ts_block(dummy_seq.unsqueeze(1))
            ts_map = self._align_ts_rep_map(
                ts_map,
                target_h=int(msf_h),
                target_w=int(msf_w),
            )
            ts_proj = self.ts_projection(ts_map)
            fused_all_map = msf_map + self.regression_map_fusion(
                torch.cat([msf_map, ts_proj], dim=1)
            )
            self.msf_map_shape = (int(msf_channels), int(msf_h), int(msf_w))
            self.reg_fused_map_shape = (
                int(fused_all_map.shape[1]),
                int(fused_all_map.shape[2]),
                int(fused_all_map.shape[3]),
            )
            reg_flat_dim = int(
                fused_all_map.shape[1] * fused_all_map.shape[2] * fused_all_map.shape[3]
            )

        self.reg_readout = nn.Linear(reg_flat_dim, int(readout_dim))
        self.reg_readout_act = nn.LeakyReLU(LRELU_SLOPE)
        self.reg_readout_drop = nn.Dropout(float(fc_dropout)) if fc_dropout > 0 else nn.Identity()
        reg_dims = [int(readout_dim)] + [int(v) for v in head_dims] + [1]
        reg_layers: list[nn.Module] = []
        for i in range(len(reg_dims) - 2):
            reg_layers.append(nn.Linear(reg_dims[i], reg_dims[i + 1]))
            reg_layers.append(nn.LeakyReLU(LRELU_SLOPE))
            if fc_dropout > 0:
                reg_layers.append(nn.Dropout(float(fc_dropout)))
        reg_layers.append(nn.Linear(reg_dims[-2], reg_dims[-1]))
        self.reg_head = nn.Sequential(*reg_layers)
        self.reg_fusion_mode = "msf_ts_residual_conv"

        self.ts_block.apply(init_module_weights)
        self.ts_projection.apply(init_module_weights)
        self.regression_map_fusion.apply(init_module_weights)
        init_module_weights(self.reg_readout)
        self.reg_head.apply(init_module_weights)

    def _prepare_seq(self, x_seq: torch.Tensor, batch_size: int) -> torch.Tensor:
        if x_seq.ndim != 3:
            raise ValueError(f"expected X_seq shape (batch,F,L); got {tuple(x_seq.shape)}")
        if int(x_seq.shape[0]) != int(batch_size):
            raise ValueError(
                "sequence batch dimension must match image batch; "
                f"got x_seq={int(x_seq.shape[0])} batch={int(batch_size)}"
            )
        if int(x_seq.shape[1]) != self.seq_input_features or int(x_seq.shape[2]) != self.seq_input_lookback:
            raise ValueError(
                "unexpected X_seq shape: "
                f"expected (*,{self.seq_input_features},{self.seq_input_lookback}) "
                f"got {tuple(x_seq.shape)}"
            )
        return x_seq.to(dtype=torch.float32)

    def _align_ts_rep_map(
        self,
        ts_map: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        if ts_map.ndim != 4:
            raise ValueError(f"expected TS rep shape (batch, C, H, W); got {tuple(ts_map.shape)}")
        out = ts_map
        th = int(target_h)
        tw = int(target_w)
        if int(out.shape[2]) != th:
            out = F.interpolate(
                out,
                size=(th, int(out.shape[3])),
                mode="nearest",
            ).contiguous()
        cur_w = int(out.shape[3])
        if cur_w < tw:
            out = F.pad(out, (0, tw - cur_w, 0, 0)).contiguous()
        elif cur_w > tw:
            if self.ts_crop_keep_side == "right":
                out = out[:, :, :, -tw:].contiguous()
            else:
                out = out[:, :, :, :tw].contiguous()
        return out

    def _fuse_regression_map(self, msf_map: torch.Tensor, x_seq: torch.Tensor) -> torch.Tensor:
        seq_ready = self._prepare_seq(x_seq, batch_size=int(msf_map.shape[0]))
        ts_map = self.ts_block(seq_ready.unsqueeze(1))
        ts_map = self._align_ts_rep_map(
            ts_map,
            target_h=int(msf_map.shape[2]),
            target_w=int(msf_map.shape[3]),
        )
        ts_proj = self.ts_projection(ts_map)
        return msf_map + self.regression_map_fusion(torch.cat([msf_map, ts_proj], dim=1))

    def forward_dual(
        self,
        x: torch.Tensor,
        x_seq: torch.Tensor,
        vix: torch.Tensor | None = None,
        vix_img: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        msf_map = self._encode_msf_map(x)
        vix_ready = self._prepare_vix(vix, batch_size=int(x.shape[0]))
        vix_img_ready = self._prepare_vix_image(vix_img, batch_size=int(x.shape[0]))
        cls_rep = self._encode_from_msf_map(
            msf_map=msf_map,
            vix_ready=vix_ready,
            vix_img_ready=vix_img_ready,
        )
        logits = self.cls_head(cls_rep)

        reg_map = self._fuse_regression_map(msf_map=msf_map, x_seq=x_seq)
        reg_rep = reg_map.flatten(start_dim=1)
        reg_rep = self.reg_readout_drop(self.reg_readout_act(self.reg_readout(reg_rep)))
        reg_value = self.reg_head(reg_rep).squeeze(1)
        return logits, reg_value

    def forward(
        self,
        x: torch.Tensor,
        vix: torch.Tensor | None = None,
        vix_img: torch.Tensor | None = None,
        x_seq: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x_seq is None:
            return self.forward_logits(x, vix=vix, vix_img=vix_img)
        return self.forward_dual(x=x, x_seq=x_seq, vix=vix, vix_img=vix_img)


class SequenceDualAlignedStore:
    """Sequence loader aligned to image shards via sample_indices and dual NPZ."""

    def __init__(
        self,
        data_dir: Path,
        ret_atr_threshold: float,
    ) -> None:
        self.image_store = ShardedImageStore(
            data_dir=Path(data_dir),
            ret_atr_threshold=float(ret_atr_threshold),
            vix_daily_csv="",
        )
        self.subset_start = int(self.image_store.subset_start)
        self.subset_end = int(self.image_store.subset_end)
        self.subset_count = int(self.image_store.subset_count)

        source_npz_raw = self.image_store.manifest.get("source_npz")
        source_npz = str(source_npz_raw).strip() if source_npz_raw is not None else ""
        if not source_npz:
            raise ValueError(
                "image manifest must include source_npz for mandatory dual alignment in seq mode"
            )
        self.decomp_source_npz = Path(source_npz)
        self.dual_npz_path = derive_dual_npz_from_decomp_source(self.decomp_source_npz)
        if not self.dual_npz_path.is_file():
            raise FileNotFoundError(
                "seq mode requires paired dual dataset but file is missing: "
                f"{self.dual_npz_path}"
            )

        dual_npz = np.load(self.dual_npz_path, allow_pickle=True)
        required_keys = ("X_seq", "y_raw")
        missing = [k for k in required_keys if k not in dual_npz.files]
        if missing:
            raise KeyError(
                "dual dataset missing required arrays for seq mode: " + ", ".join(missing)
            )
        self._dual_npz = dual_npz
        self.X_seq = np.asarray(dual_npz["X_seq"])
        self.y_raw_dual = np.asarray(dual_npz["y_raw"])
        self.feature_cols_seq = (
            [str(x) for x in dual_npz["feature_cols_seq"].tolist()]
            if "feature_cols_seq" in dual_npz.files
            else []
        )
        self.timestamps_dual = (
            np.asarray(dual_npz["timestamps"]).astype(str)
            if "timestamps" in dual_npz.files
            else None
        )
        self.ticker_ids_dual = (
            np.asarray(dual_npz["ticker_ids"], dtype=np.int64)
            if "ticker_ids" in dual_npz.files
            else None
        )

        if self.X_seq.ndim != 3:
            raise ValueError(
                "dual X_seq must have shape (N,F,L); "
                f"got shape={tuple(self.X_seq.shape)}"
            )
        if self.y_raw_dual.ndim != 2:
            raise ValueError(
                "dual y_raw must have shape (N,labels); "
                f"got shape={tuple(self.y_raw_dual.shape)}"
            )
        if int(self.X_seq.shape[0]) != int(self.y_raw_dual.shape[0]):
            raise ValueError(
                "dual X_seq/y_raw sample count mismatch: "
                f"X_seq={int(self.X_seq.shape[0])} y_raw={int(self.y_raw_dual.shape[0])}"
            )
        if self.timestamps_dual is not None and int(self.timestamps_dual.shape[0]) != int(self.X_seq.shape[0]):
            raise ValueError(
                "dual timestamps sample count mismatch: "
                f"timestamps={int(self.timestamps_dual.shape[0])} X_seq={int(self.X_seq.shape[0])}"
            )
        if self.ticker_ids_dual is not None and int(self.ticker_ids_dual.shape[0]) != int(self.X_seq.shape[0]):
            raise ValueError(
                "dual ticker_ids sample count mismatch: "
                f"ticker_ids={int(self.ticker_ids_dual.shape[0])} X_seq={int(self.X_seq.shape[0])}"
            )

        label_cols_dual = (
            [str(x) for x in dual_npz["label_cols"].tolist()]
            if "label_cols" in dual_npz.files
            else []
        )
        if "ret_pct" in label_cols_dual:
            self.ret_pct_idx_dual = int(label_cols_dual.index("ret_pct"))
        else:
            self.ret_pct_idx_dual = int(self.image_store.ret_pct_idx)
            if self.ret_pct_idx_dual >= int(self.y_raw_dual.shape[1]):
                raise ValueError(
                    "ret_pct index is out of bounds in dual y_raw: "
                    f"ret_pct_idx={self.ret_pct_idx_dual} y_raw_cols={int(self.y_raw_dual.shape[1])}"
                )

        self.seq_feature_count = int(self.X_seq.shape[1])
        self.seq_lookback = int(self.X_seq.shape[2])
        self.dual_sample_count = int(self.X_seq.shape[0])
        effective_end = min(int(self.subset_end), int(self.dual_sample_count))
        if effective_end <= int(self.subset_start):
            raise ValueError(
                "no overlapping aligned samples between image subset and dual dataset: "
                f"subset_start={int(self.subset_start)} subset_end={int(self.subset_end)} "
                f"dual_count={int(self.dual_sample_count)}"
            )
        self.dropped_image_samples_for_seq = int(self.subset_end) - int(effective_end)
        self.subset_end = int(effective_end)
        self.subset_count = int(self.subset_end - int(self.subset_start))
        self.seq_volume_feature_idx = (
            int(self.feature_cols_seq.index("volume"))
            if "volume" in self.feature_cols_seq
            else -1
        )
        self.seq_ratio_feature_indices = [
            int(self.feature_cols_seq.index(name))
            for name in ("open_ratio", "high_ratio", "low_ratio", "close_ratio")
            if name in self.feature_cols_seq
        ]
        self.seq_feature_standardization_enabled = bool(
            SEQ_FEATURE_STANDARDIZATION_ENABLED
        )
        self.seq_feature_standardization_eps = float(
            max(1e-12, float(SEQ_FEATURE_STANDARDIZATION_EPS))
        )
        self.seq_feature_standardization_clip = float(
            max(0.0, float(SEQ_FEATURE_STANDARDIZATION_CLIP))
        )
        self.seq_feature_standardizer_mean: np.ndarray | None = None
        self.seq_feature_standardizer_scale: np.ndarray | None = None
        self.seq_feature_standardizer_fitted = False
        self.seq_target_clip_enabled = bool(SEQ_TARGET_CLIP_ENABLED)
        self.seq_target_clip_lower_pct = float(SEQ_TARGET_CLIP_LOWER_PCT)
        self.seq_target_clip_upper_pct = float(SEQ_TARGET_CLIP_UPPER_PCT)
        self.seq_target_clip_low = 0.0
        self.seq_target_clip_high = 0.0
        self.seq_target_clip_fitted = False

    def _resolve_absolute_split_bounds(
        self,
        split_range: tuple[int, int],
    ) -> tuple[int, int]:
        rel_start = max(0, int(split_range[0]))
        rel_end = min(int(self.subset_count), int(split_range[1]))
        if rel_end <= rel_start:
            raise ValueError(
                "invalid split_range for seq feature standardizer fit: "
                f"split_range={split_range} subset_count={int(self.subset_count)}"
            )
        abs_start = int(self.subset_start + rel_start)
        abs_end = int(min(int(self.subset_start + rel_end), int(self.subset_end)))
        if abs_end <= abs_start:
            raise ValueError(
                "no aligned samples available for seq feature standardizer fit: "
                f"abs_start={abs_start} abs_end={abs_end}"
            )
        return abs_start, abs_end

    def fit_seq_feature_standardizer(
        self,
        split_range: tuple[int, int],
        chunk_size: int = 2048,
    ) -> dict[str, object]:
        if not bool(self.seq_feature_standardization_enabled):
            self.seq_feature_standardizer_mean = None
            self.seq_feature_standardizer_scale = None
            self.seq_feature_standardizer_fitted = False
            return {
                "enabled": 0,
                "fitted": 0,
                "feature_count": int(self.seq_feature_count),
                "seq_feature_standardization_eps": float(
                    self.seq_feature_standardization_eps
                ),
                "seq_feature_standardization_clip": float(
                    self.seq_feature_standardization_clip
                ),
                "reason": "disabled",
            }

        abs_start, abs_end = self._resolve_absolute_split_bounds(split_range)
        feature_count = int(self.seq_feature_count)
        chunk_n = max(1, int(chunk_size))
        sums = np.zeros((feature_count,), dtype=np.float64)
        sums_sq = np.zeros((feature_count,), dtype=np.float64)
        counts = np.zeros((feature_count,), dtype=np.int64)
        non_finite_value_count = 0

        for row_start in range(abs_start, abs_end, chunk_n):
            row_end = min(abs_end, row_start + chunk_n)
            x_chunk = np.asarray(self.X_seq[row_start:row_end], dtype=np.float32).copy()
            x_chunk = apply_seq_input_transforms_inplace(
                x_seq=x_chunk,
                seq_volume_feature_idx=self.seq_volume_feature_idx,
                seq_ratio_feature_indices=self.seq_ratio_feature_indices,
            )
            feature_view = np.transpose(x_chunk, (1, 0, 2)).reshape(feature_count, -1)
            finite = np.isfinite(feature_view)
            non_finite_value_count += int(np.size(finite) - int(np.sum(finite)))
            safe_view = np.where(finite, feature_view, 0.0).astype(np.float64, copy=False)
            sums += np.sum(safe_view, axis=1, dtype=np.float64)
            sums_sq += np.sum(safe_view * safe_view, axis=1, dtype=np.float64)
            counts += np.sum(finite, axis=1, dtype=np.int64)

        if int(np.min(counts)) <= 0:
            zero_count_indices = np.where(counts <= 0)[0].tolist()
            raise ValueError(
                "cannot fit seq feature standardizer: feature has no finite training values "
                f"(indices={zero_count_indices})"
            )

        means = sums / np.maximum(counts.astype(np.float64, copy=False), 1.0)
        variances = (sums_sq / np.maximum(counts.astype(np.float64, copy=False), 1.0)) - (
            means * means
        )
        tiny_negative = (variances < 0.0) & (np.abs(variances) <= 1e-12)
        variances[tiny_negative] = 0.0
        variances = np.maximum(variances, 0.0)
        stds = np.sqrt(variances).astype(np.float64, copy=False)
        means = np.where(np.isfinite(means), means, 0.0)
        valid_scales = np.isfinite(stds) & (stds > float(self.seq_feature_standardization_eps))
        fallback_scale_feature_count = int(np.size(valid_scales) - int(np.sum(valid_scales)))
        stds = np.where(valid_scales, stds, 1.0)

        self.seq_feature_standardizer_mean = np.ascontiguousarray(
            means.astype(np.float32, copy=False)
        )
        self.seq_feature_standardizer_scale = np.ascontiguousarray(
            stds.astype(np.float32, copy=False)
        )
        self.seq_feature_standardizer_fitted = True

        rel_start = max(0, int(split_range[0]))
        rel_end = min(int(self.subset_count), int(split_range[1]))
        return {
            "enabled": 1,
            "fitted": 1,
            "feature_count": int(feature_count),
            "train_split_rel": [int(rel_start), int(rel_end)],
            "train_split_abs": [int(abs_start), int(abs_end)],
            "train_value_count": int(np.sum(counts)),
            "non_finite_value_count": int(non_finite_value_count),
            "fallback_scale_feature_count": int(fallback_scale_feature_count),
            "seq_feature_standardization_eps": float(
                self.seq_feature_standardization_eps
            ),
            "seq_feature_standardization_clip": float(
                self.seq_feature_standardization_clip
            ),
        }

    def get_seq_feature_standardizer_state(self) -> dict[str, object] | None:
        if not bool(self.seq_feature_standardization_enabled):
            return {"enabled": 0}
        if not bool(self.seq_feature_standardizer_fitted):
            return None
        if (
            self.seq_feature_standardizer_mean is None
            or self.seq_feature_standardizer_scale is None
        ):
            return None
        return {
            "enabled": 1,
            "eps": float(self.seq_feature_standardization_eps),
            "clip": float(self.seq_feature_standardization_clip),
            "mean": np.asarray(self.seq_feature_standardizer_mean, dtype=np.float32),
            "scale": np.asarray(self.seq_feature_standardizer_scale, dtype=np.float32),
        }

    def load_seq_feature_standardizer_state(
        self,
        state: dict[str, object] | None,
    ) -> None:
        if state is None:
            self.seq_feature_standardizer_mean = None
            self.seq_feature_standardizer_scale = None
            self.seq_feature_standardizer_fitted = False
            return

        enabled = bool(int(state.get("enabled", 1)))
        self.seq_feature_standardization_enabled = bool(enabled)
        self.seq_feature_standardization_eps = float(
            max(1e-12, float(state.get("eps", self.seq_feature_standardization_eps)))
        )
        self.seq_feature_standardization_clip = float(
            max(0.0, float(state.get("clip", self.seq_feature_standardization_clip)))
        )
        if not enabled:
            self.seq_feature_standardizer_mean = None
            self.seq_feature_standardizer_scale = None
            self.seq_feature_standardizer_fitted = False
            return

        if "mean" not in state or "scale" not in state:
            raise ValueError(
                "invalid seq feature standardizer state: missing 'mean' or 'scale'"
            )
        mean_arr = np.asarray(state["mean"], dtype=np.float32).reshape(-1)
        scale_arr = np.asarray(state["scale"], dtype=np.float32).reshape(-1)
        if int(mean_arr.size) != int(self.seq_feature_count):
            raise ValueError(
                "seq feature standardizer mean shape mismatch: "
                f"expected={int(self.seq_feature_count)} got={int(mean_arr.size)}"
            )
        if int(scale_arr.size) != int(self.seq_feature_count):
            raise ValueError(
                "seq feature standardizer scale shape mismatch: "
                f"expected={int(self.seq_feature_count)} got={int(scale_arr.size)}"
            )
        scale_arr = np.where(
            np.isfinite(scale_arr)
            & (scale_arr > float(self.seq_feature_standardization_eps)),
            scale_arr,
            1.0,
        ).astype(np.float32, copy=False)
        mean_arr = np.where(np.isfinite(mean_arr), mean_arr, 0.0).astype(
            np.float32, copy=False
        )
        self.seq_feature_standardizer_mean = np.ascontiguousarray(mean_arr)
        self.seq_feature_standardizer_scale = np.ascontiguousarray(scale_arr)
        self.seq_feature_standardizer_fitted = True

    def fit_seq_target_clipper(
        self,
        split_range: tuple[int, int],
    ) -> dict[str, object]:
        if not bool(self.seq_target_clip_enabled):
            self.seq_target_clip_fitted = False
            self.seq_target_clip_low = 0.0
            self.seq_target_clip_high = 0.0
            return {
                "enabled": 0,
                "fitted": 0,
                "lower_pct": float(self.seq_target_clip_lower_pct),
                "upper_pct": float(self.seq_target_clip_upper_pct),
                "reason": "disabled",
            }

        lower_pct = float(self.seq_target_clip_lower_pct)
        upper_pct = float(self.seq_target_clip_upper_pct)
        if not (0.0 <= lower_pct < upper_pct <= 100.0):
            raise ValueError(
                "invalid seq target clip percentiles: "
                f"lower_pct={lower_pct} upper_pct={upper_pct}"
            )

        abs_start, abs_end = self._resolve_absolute_split_bounds(split_range)
        y = np.asarray(
            self.y_raw_dual[abs_start:abs_end, self.ret_pct_idx_dual],
            dtype=np.float64,
        ).reshape(-1)
        finite = np.isfinite(y)
        y_finite = y[finite]
        if y_finite.size <= 0:
            raise ValueError(
                "cannot fit seq target clipper: no finite ret_pct values in train split"
            )

        clip_low = float(np.percentile(y_finite, lower_pct))
        clip_high = float(np.percentile(y_finite, upper_pct))
        if (not np.isfinite(clip_low)) or (not np.isfinite(clip_high)):
            raise ValueError(
                "cannot fit seq target clipper: non-finite clip bounds "
                f"(low={clip_low}, high={clip_high})"
            )
        if clip_high <= clip_low:
            raise ValueError(
                "cannot fit seq target clipper: upper bound must exceed lower bound "
                f"(low={clip_low}, high={clip_high})"
            )

        clipped_count = int(np.sum((y_finite < clip_low) | (y_finite > clip_high)))
        self.seq_target_clip_low = float(clip_low)
        self.seq_target_clip_high = float(clip_high)
        self.seq_target_clip_fitted = True

        rel_start = max(0, int(split_range[0]))
        rel_end = min(int(self.subset_count), int(split_range[1]))
        return {
            "enabled": 1,
            "fitted": 1,
            "lower_pct": float(lower_pct),
            "upper_pct": float(upper_pct),
            "clip_low": float(clip_low),
            "clip_high": float(clip_high),
            "train_split_rel": [int(rel_start), int(rel_end)],
            "train_split_abs": [int(abs_start), int(abs_end)],
            "train_finite_count": int(y_finite.size),
            "train_non_finite_count": int(np.size(y) - int(y_finite.size)),
            "train_clipped_count": int(clipped_count),
            "train_clipped_frac": (
                float(clipped_count) / float(y_finite.size)
                if int(y_finite.size) > 0
                else 0.0
            ),
        }

    def get_seq_target_clipper_state(self) -> dict[str, object] | None:
        if not bool(self.seq_target_clip_enabled):
            return {"enabled": 0}
        if not bool(self.seq_target_clip_fitted):
            return None
        return {
            "enabled": 1,
            "lower_pct": float(self.seq_target_clip_lower_pct),
            "upper_pct": float(self.seq_target_clip_upper_pct),
            "clip_low": float(self.seq_target_clip_low),
            "clip_high": float(self.seq_target_clip_high),
        }

    def load_seq_target_clipper_state(
        self,
        state: dict[str, object] | None,
    ) -> None:
        if state is None:
            self.seq_target_clip_fitted = False
            self.seq_target_clip_low = 0.0
            self.seq_target_clip_high = 0.0
            return

        enabled = bool(int(state.get("enabled", 1)))
        self.seq_target_clip_enabled = bool(enabled)
        self.seq_target_clip_lower_pct = float(
            state.get("lower_pct", self.seq_target_clip_lower_pct)
        )
        self.seq_target_clip_upper_pct = float(
            state.get("upper_pct", self.seq_target_clip_upper_pct)
        )
        if not enabled:
            self.seq_target_clip_fitted = False
            self.seq_target_clip_low = 0.0
            self.seq_target_clip_high = 0.0
            return

        clip_low = float(state.get("clip_low", float("nan")))
        clip_high = float(state.get("clip_high", float("nan")))
        if (not np.isfinite(clip_low)) or (not np.isfinite(clip_high)):
            raise ValueError(
                "invalid seq target clipper state: non-finite bounds "
                f"(low={clip_low}, high={clip_high})"
            )
        if clip_high <= clip_low:
            raise ValueError(
                "invalid seq target clipper state: upper bound must exceed lower bound "
                f"(low={clip_low}, high={clip_high})"
            )
        self.seq_target_clip_low = float(clip_low)
        self.seq_target_clip_high = float(clip_high)
        self.seq_target_clip_fitted = True

    def _apply_seq_feature_standardization_inplace(
        self,
        x_seq: np.ndarray,
    ) -> np.ndarray:
        x = np.asarray(x_seq, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(
                f"expected X_seq shape (batch,F,L) for standardization; got {tuple(x.shape)}"
            )
        if not bool(self.seq_feature_standardization_enabled):
            return x
        if (
            not bool(self.seq_feature_standardizer_fitted)
            or self.seq_feature_standardizer_mean is None
            or self.seq_feature_standardizer_scale is None
        ):
            raise RuntimeError(
                "seq feature standardization is enabled but standardizer is not fitted"
            )
        if int(x.shape[1]) != int(self.seq_feature_count):
            raise ValueError(
                "seq feature dimension mismatch during standardization: "
                f"x_features={int(x.shape[1])} expected={int(self.seq_feature_count)}"
            )
        x -= self.seq_feature_standardizer_mean.reshape(1, int(self.seq_feature_count), 1)
        x /= self.seq_feature_standardizer_scale.reshape(1, int(self.seq_feature_count), 1)
        clip = float(self.seq_feature_standardization_clip)
        if clip > 0.0:
            np.clip(x, -clip, clip, out=x)
        if not np.all(np.isfinite(x)):
            n_bad = int(np.size(x) - int(np.isfinite(x).sum()))
            raise ValueError(
                f"seq feature standardization produced non-finite values (count={n_bad})"
            )
        return x

    def _apply_seq_target_clipping_inplace(
        self,
        y_reg: np.ndarray,
    ) -> np.ndarray:
        y = np.asarray(y_reg, dtype=np.float32)
        if y.ndim != 1:
            raise ValueError(
                f"expected y_reg shape (batch,) for clipping; got {tuple(y.shape)}"
            )
        if not bool(self.seq_target_clip_enabled):
            return y
        if not bool(self.seq_target_clip_fitted):
            raise RuntimeError("seq target clipping is enabled but clipper is not fitted")
        np.clip(
            y,
            float(self.seq_target_clip_low),
            float(self.seq_target_clip_high),
            out=y,
        )
        if not np.all(np.isfinite(y)):
            n_bad = int(np.size(y) - int(np.isfinite(y).sum()))
            raise ValueError(
                f"seq target clipping produced non-finite values (count={n_bad})"
            )
        return y

    def fetch_seq_reg_by_sample_indices(
        self,
        sample_indices: np.ndarray,
        strict: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        sample_indices_v = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
        if sample_indices_v.size <= 0:
            return (
                np.empty(
                    (0, int(self.seq_feature_count), int(self.seq_lookback)),
                    dtype=np.float32,
                ),
                np.empty((0,), dtype=np.float32),
            )
        in_bounds = (sample_indices_v >= 0) & (sample_indices_v < self.dual_sample_count)
        if not bool(np.all(in_bounds)):
            if bool(strict):
                lo = int(np.min(sample_indices_v))
                hi = int(np.max(sample_indices_v))
                bad_count = int(np.size(sample_indices_v) - int(np.sum(in_bounds)))
                raise IndexError(
                    "sample_indices out of dual dataset bounds in strict mode: "
                    f"min={lo} max={hi} dual_count={self.dual_sample_count} "
                    f"bad_count={bad_count}"
                )
            sample_indices_v = sample_indices_v[in_bounds]
            if sample_indices_v.size <= 0:
                return (
                    np.empty(
                        (0, int(self.seq_feature_count), int(self.seq_lookback)),
                        dtype=np.float32,
                    ),
                    np.empty((0,), dtype=np.float32),
                )

        x_seq = np.asarray(self.X_seq[sample_indices_v], dtype=np.float32)
        x_seq = apply_seq_input_transforms_inplace(
            x_seq=x_seq,
            seq_volume_feature_idx=self.seq_volume_feature_idx,
            seq_ratio_feature_indices=self.seq_ratio_feature_indices,
        )
        x_seq = self._apply_seq_feature_standardization_inplace(x_seq=x_seq)
        y_reg = np.asarray(
            self.y_raw_dual[sample_indices_v, self.ret_pct_idx_dual],
            dtype=np.float32,
        ).reshape(-1)
        y_reg = self._apply_seq_target_clipping_inplace(y_reg=y_reg)
        if x_seq.shape[0] != y_reg.shape[0]:
            raise RuntimeError(
                "aligned seq batch mismatch: "
                f"X_seq={x_seq.shape[0]} y_reg={y_reg.shape[0]}"
            )
        if not np.all(np.isfinite(y_reg)):
            n_bad = int(np.size(y_reg) - int(np.isfinite(y_reg).sum()))
            raise ValueError(
                f"dual aligned ret_pct contains non-finite values in seq mode (count={n_bad})"
            )
        return np.ascontiguousarray(x_seq), np.ascontiguousarray(y_reg)

    def iter_batches(
        self,
        split_range: tuple[int, int],
        batch_size: int,
        shuffle: bool,
        seed: int,
        return_sample_indices: bool = False,
        return_timestamps: bool = False,
        return_ticker_ids: bool = False,
    ):
        image_iter = self.image_store.iter_batches(
            split_range=split_range,
            batch_size=batch_size,
            shuffle=bool(shuffle),
            seed=int(seed),
            return_sample_indices=True,
            return_ret_atr=False,
            return_ret_pct=False,
            return_timestamps=bool(return_timestamps),
            return_ticker_ids=bool(return_ticker_ids),
            return_vix=False,
            return_vix_img=False,
        )
        for batch in image_iter:
            cursor = 0
            _x_img = batch[cursor]
            cursor += 1
            _y_cls = batch[cursor]
            cursor += 1
            sample_indices = np.asarray(batch[cursor], dtype=np.int64).reshape(-1)
            cursor += 1
            timestamps = None
            ticker_ids = None
            if return_timestamps:
                timestamps = np.asarray(batch[cursor]).astype(str)
                cursor += 1
            if return_ticker_ids:
                ticker_ids = np.asarray(batch[cursor], dtype=np.int64).reshape(-1)
                cursor += 1

            if sample_indices.size <= 0:
                continue
            in_bounds = (sample_indices >= 0) & (sample_indices < self.dual_sample_count)
            if not bool(np.all(in_bounds)):
                sample_indices = sample_indices[in_bounds]
                if sample_indices.size <= 0:
                    continue
                if timestamps is not None:
                    timestamps = np.asarray(timestamps[in_bounds]).astype(str)
                if ticker_ids is not None:
                    ticker_ids = np.asarray(ticker_ids[in_bounds], dtype=np.int64).reshape(-1)
            lo = int(np.min(sample_indices))
            hi = int(np.max(sample_indices))
            if lo < 0 or hi >= self.dual_sample_count:
                raise IndexError(
                    "sample_indices out of dual dataset bounds: "
                    f"min={lo} max={hi} dual_count={self.dual_sample_count}"
                )

            x_seq, y_reg = self.fetch_seq_reg_by_sample_indices(
                sample_indices=sample_indices,
                strict=True,
            )

            out: list[np.ndarray] = [np.ascontiguousarray(x_seq), np.ascontiguousarray(y_reg)]
            if return_sample_indices:
                out.append(np.ascontiguousarray(sample_indices))
            if return_timestamps:
                if timestamps is None:
                    raise RuntimeError("timestamps requested but unavailable")
                out.append(np.asarray(timestamps, dtype=object))
            if return_ticker_ids:
                if ticker_ids is None:
                    raise RuntimeError("ticker_ids requested but unavailable")
                out.append(np.ascontiguousarray(ticker_ids))
            yield tuple(out)


def seq_regression_loss(pred: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    if str(mode) == "smoothl1":
        return F.smooth_l1_loss(pred, target, reduction="mean")
    if str(mode) == "mse":
        return F.mse_loss(pred, target, reduction="mean")
    raise ValueError(f"unsupported seq regression loss mode: {mode!r}")


class ShardedImageStore:
    """Shard-aware batch streamer for X_img/y_raw without full dataset load."""

    def __init__(
        self,
        data_dir: Path,
        ret_atr_threshold: float,
        vix_daily_csv: str = "",
        vix_date_col: str = VIX_DATE_COL_DEFAULT,
        vix_value_col: str = VIX_VALUE_COL_DEFAULT,
    ):
        self.data_dir = Path(data_dir)
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.subset_start = int(self.manifest["subset_start"])
        self.subset_end = int(self.manifest["subset_end"])
        self.subset_count = int(self.manifest["subset_count"])
        self.scales = int(self.manifest["decomposition_scales"])
        self.height = int(self.manifest["image_height"])
        self.width = int(self.manifest["image_width"])
        self.width_per_scale = tuple(
            resolve_image_width_per_scale(
                manifest=self.manifest,
                scales=self.scales,
                image_width=self.width,
            )
        )
        if bool(JAGGED_IMAGE_WIDTHS_ENABLED):
            self.width = int(max(int(self.width), max(self.width_per_scale)))
        else:
            self.width = int(max(self.width_per_scale))
        self.include_vix_image = bool(self.manifest.get("include_vix_image", False))
        self.vix_image_height = int(
            self.manifest.get("vix_image_height", VIX_IMAGE_HEIGHT_DEFAULT)
        )
        self.vix_image_width = int(
            self.manifest.get("vix_image_width", VIX_IMAGE_WIDTH_DEFAULT)
        )
        self.vix_image_bars = int(self.manifest.get("vix_image_bars", VIX_IMAGE_BARS_DEFAULT))
        if self.include_vix_image:
            if self.vix_image_height < 1 or self.vix_image_width < 1:
                raise ValueError(
                    "manifest include_vix_image is enabled but vix image dims are invalid: "
                    f"h={self.vix_image_height} w={self.vix_image_width}"
                )
        label_cols = [str(x) for x in self.manifest.get("label_cols", [])]
        if not label_cols:
            label_cols_path = self.data_dir / "label_cols.npy"
            if label_cols_path.is_file():
                try:
                    label_cols_arr = np.load(label_cols_path, allow_pickle=True)
                    label_cols = [
                        str(x)
                        for x in np.asarray(label_cols_arr, dtype=object).reshape(-1).tolist()
                    ]
                except Exception:
                    label_cols = []
        self.ret_atr_idx = label_cols.index("ret_atr") if "ret_atr" in label_cols else 3
        self.ret_pct_idx = label_cols.index("ret_pct") if "ret_pct" in label_cols else 6
        # Backward-compatible config field name; threshold is now applied to ret_pct.
        self.ret_atr_threshold = float(ret_atr_threshold)
        self.npy_bundle_arrays: dict[str, np.ndarray] | None = None
        shard_entries = self.manifest.get("shards")
        if isinstance(shard_entries, list):
            self.shards = [
                ShardInfo(
                    file=str(entry["file"]),
                    sample_start=int(entry["sample_start"]),
                    sample_end=int(entry["sample_end"]),
                    count=int(entry["count"]),
                )
                for entry in shard_entries
            ]
        else:
            arrays_meta = self.manifest.get("arrays")
            if not isinstance(arrays_meta, dict):
                raise KeyError("manifest must contain 'shards' or 'arrays'")

            def _array_path(meta_key: str) -> Path:
                item = arrays_meta.get(meta_key)
                if not isinstance(item, dict):
                    raise ValueError(
                        f"manifest arrays metadata missing dict for key={meta_key!r}"
                    )
                rel = str(item.get("file", "")).strip()
                if not rel:
                    raise ValueError(
                        f"manifest arrays metadata missing file path for key={meta_key!r}"
                    )
                p = self.data_dir / rel
                if not p.exists():
                    raise FileNotFoundError(f"array file not found: {p}")
                return p

            x_arr = np.load(_array_path("X_img"), mmap_mode="r")
            y_arr = np.load(_array_path("y_raw"), mmap_mode="r")
            ts_arr = np.load(_array_path("timestamps"), mmap_mode="r")
            if int(x_arr.shape[0]) != int(y_arr.shape[0]) or int(x_arr.shape[0]) != int(
                ts_arr.shape[0]
            ):
                raise ValueError(
                    "npy bundle arrays must share axis-0 length "
                    f"(X_img={x_arr.shape[0]}, y_raw={y_arr.shape[0]}, "
                    f"timestamps={ts_arr.shape[0]})"
                )
            vix_img_arr = None
            if self.include_vix_image:
                vix_img_arr = np.load(_array_path("X_vix_img"), mmap_mode="r")
                if int(vix_img_arr.shape[0]) != int(x_arr.shape[0]):
                    raise ValueError(
                        "npy bundle X_vix_img must match X_img axis-0 length "
                        f"(X_img={x_arr.shape[0]}, X_vix_img={vix_img_arr.shape[0]})"
                    )

            si_arr = None
            si_item = arrays_meta.get("sample_indices")
            if isinstance(si_item, dict):
                try:
                    si_arr = np.load(_array_path("sample_indices"), mmap_mode="r")
                except Exception:
                    si_arr = None
            tid_arr = None
            tid_item = arrays_meta.get("ticker_ids")
            if isinstance(tid_item, dict):
                try:
                    tid_arr = np.load(_array_path("ticker_ids"), mmap_mode="r")
                except Exception:
                    tid_arr = None
            if tid_arr is not None and int(tid_arr.shape[0]) != int(x_arr.shape[0]):
                raise ValueError(
                    "npy bundle ticker_ids must match X_img axis-0 length "
                    f"(X_img={x_arr.shape[0]}, ticker_ids={tid_arr.shape[0]})"
                )

            self.npy_bundle_arrays = {
                "X_img": x_arr,
                "y_raw": y_arr,
                "timestamps": ts_arr,
            }
            if vix_img_arr is not None:
                self.npy_bundle_arrays["X_vix_img"] = vix_img_arr
            if si_arr is not None:
                self.npy_bundle_arrays["sample_indices"] = si_arr
            if tid_arr is not None:
                self.npy_bundle_arrays["ticker_ids"] = tid_arr

            n_bundle = int(x_arr.shape[0])
            ready_candidates: list[int] = []
            ready_raw = self.manifest.get("arrays_valid_count")
            if ready_raw is not None:
                try:
                    ready_candidates.append(int(ready_raw))
                except Exception:
                    pass
            for key in (
                "X_img",
                "X_vix_img",
                "y_raw",
                "timestamps",
                "sample_indices",
                "ticker_ids",
            ):
                item = arrays_meta.get(key)
                if not isinstance(item, dict):
                    continue
                if "ready_count" not in item:
                    continue
                try:
                    ready_candidates.append(int(item.get("ready_count")))
                except Exception:
                    continue
            n_ready = int(min(ready_candidates)) if ready_candidates else int(n_bundle)
            n_ready = int(max(0, min(int(n_bundle), int(n_ready))))
            manifest_count = int(max(0, int(self.subset_count)))
            n_visible = int(min(int(n_bundle), int(manifest_count), int(n_ready)))
            self.subset_count = int(n_visible)
            self.subset_end = int(self.subset_start + int(n_visible))
            self.shards = [
                ShardInfo(
                    file="__single_npy_bundle__",
                    sample_start=int(self.subset_start),
                    sample_end=int(self.subset_end),
                    count=int(self.subset_count),
                )
            ]

        self.shards.sort(key=lambda s: s.sample_start)
        self.shard_ends = np.asarray([int(s.sample_end) for s in self.shards], dtype=np.int64)
        self.bundle_iter_chunk_samples = int(BUNDLE_ITER_CHUNK_SAMPLES_DEFAULT)
        raw_chunk = os.getenv("BUNDLE_ITER_CHUNK_SAMPLES", "").strip()
        if raw_chunk:
            try:
                self.bundle_iter_chunk_samples = max(1, int(raw_chunk))
            except ValueError as exc:
                raise ValueError(
                    "BUNDLE_ITER_CHUNK_SAMPLES must be a positive integer; "
                    f"got {raw_chunk!r}"
                ) from exc

        self.vix_daily_csv = str(vix_daily_csv).strip()
        self.vix_date_col = str(vix_date_col).strip() or VIX_DATE_COL_DEFAULT
        self.vix_value_col = str(vix_value_col).strip() or VIX_VALUE_COL_DEFAULT
        self.vix_by_date: dict[str, float] | None = None
        self.vix_norm_stats: VixNormStats | None = None
        if self.vix_daily_csv:
            self.vix_by_date = load_daily_vix_lookup_csv(
                csv_path=Path(self.vix_daily_csv),
                date_col=self.vix_date_col,
                value_col=self.vix_value_col,
            )
        self.ticker_lookup = load_ticker_lookup(self.data_dir)
        self.ticker_csv_dir = Path(TICKER_CSV_DIR_PATH)
        self.ticker_date_col = str(TICKER_DATE_COL_DEFAULT)
        self.ticker_beta_col = str(TICKER_BETA_COL_DEFAULT)
        self._beta_by_symbol_by_date_cache: dict[str, dict[str, float] | None] = {}
        self.beta_clip_low = float("-inf")
        self.beta_clip_high = float("inf")
        self.beta_clip_symbol_count = 0
        self.vix_beta_scaling_requested = bool(VIX_BETA_SCALING_ENABLED)
        self.vix_beta_scaling_available = bool(
            bool(self.ticker_lookup) and self.ticker_csv_dir.is_dir()
        )
        self.vix_beta_scaling_enabled = bool(
            self.vix_beta_scaling_requested and self.vix_beta_scaling_available
        )
        if self.vix_beta_scaling_enabled:
            (
                self.beta_clip_low,
                self.beta_clip_high,
                self.beta_clip_symbol_count,
            ) = self._fit_beta_clip_bounds()

    @staticmethod
    def _validate_split_range(split_range: tuple[int, int]) -> tuple[int, int]:
        rel_start, rel_end = int(split_range[0]), int(split_range[1])
        if rel_start < 0 or rel_end < 0 or rel_end < rel_start:
            raise ValueError(f"invalid split range: {split_range}")
        return rel_start, rel_end

    def _iter_overlapping_shards(self, start_abs: int, end_abs: int) -> Iterable[ShardInfo]:
        for shard in self.shards:
            if shard.sample_end <= start_abs:
                continue
            if shard.sample_start >= end_abs:
                break
            yield shard

    def _chunk_single_bundle_jobs(
        self,
        shard_jobs: list[tuple[ShardInfo, tuple[int, int] | np.ndarray]],
        batch_size: int,
    ) -> list[tuple[ShardInfo, tuple[int, int] | np.ndarray]]:
        if self.npy_bundle_arrays is None:
            return shard_jobs

        chunk_n = max(int(batch_size), int(self.bundle_iter_chunk_samples))
        if chunk_n <= 0:
            return shard_jobs

        out: list[tuple[ShardInfo, tuple[int, int] | np.ndarray]] = []
        for shard, selector in shard_jobs:
            if isinstance(selector, tuple):
                local_start = int(selector[0])
                local_end = int(selector[1])
                if local_end <= local_start:
                    continue
                if (local_end - local_start) <= chunk_n:
                    out.append((shard, (local_start, local_end)))
                    continue
                cur = local_start
                while cur < local_end:
                    nxt = min(local_end, cur + chunk_n)
                    out.append((shard, (cur, nxt)))
                    cur = nxt
                continue

            local_idx = np.asarray(selector, dtype=np.int64).reshape(-1)
            if local_idx.size <= 0:
                continue
            if int(local_idx.size) <= chunk_n:
                out.append((shard, local_idx))
                continue
            for i in range(0, int(local_idx.size), chunk_n):
                out.append((shard, local_idx[i : i + chunk_n]))
        return out

    def _lookup_vix_for_timestamps(self, timestamps: np.ndarray) -> np.ndarray:
        if self.vix_by_date is None:
            raise RuntimeError("daily VIX lookup unavailable; set --vix-daily-csv first")
        ts = np.asarray(timestamps).astype(str, copy=False).reshape(-1)
        return np.fromiter(
            (
                float(self.vix_by_date.get(str(ts_i)[:10], float("nan")))
                for ts_i in ts
            ),
            dtype=np.float32,
            count=int(ts.shape[0]),
        )

    def _fit_beta_clip_bounds(self) -> tuple[float, float, int]:
        symbols_src: list[str]
        if self.ticker_lookup:
            symbols_src = [str(s) for s in self.ticker_lookup]
        else:
            symbols_src = [str(p.stem) for p in self.ticker_csv_dir.glob("*.csv")]

        seen: set[str] = set()
        vals: list[float] = []
        for raw in symbols_src:
            symbol = normalize_symbol(raw)
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            csv_path = self.ticker_csv_dir / f"{symbol}.csv"
            beta = load_first_finite_beta_value_csv(
                csv_path=csv_path,
                beta_col=self.ticker_beta_col,
            )
            if beta is None:
                continue
            if np.isfinite(float(beta)):
                vals.append(float(beta))

        if len(vals) <= 0:
            return float("-inf"), float("inf"), 0

        arr = np.asarray(vals, dtype=np.float64)
        q25 = float(np.percentile(arr, 25.0))
        q75 = float(np.percentile(arr, 75.0))
        iqr = float(q75 - q25)
        low = float("nan")
        high = float("nan")
        if np.isfinite(iqr) and iqr > 0.0:
            low = float(q25 - (1.5 * iqr))
            high = float(q75 + (1.5 * iqr))
        if (not np.isfinite(low)) or (not np.isfinite(high)) or high <= low:
            if arr.size >= 5:
                low = float(np.percentile(arr, 1.0))
                high = float(np.percentile(arr, 99.0))
            else:
                low = float(np.min(arr))
                high = float(np.max(arr))
        if (not np.isfinite(low)) or (not np.isfinite(high)) or high <= low:
            return float("-inf"), float("inf"), int(arr.size)
        # Keep neutral beta=1.0 inside the clipping range.
        low = float(min(low, 1.0))
        high = float(max(high, 1.0))
        return float(low), float(high), int(arr.size)

    def _ticker_symbol_for_id(self, ticker_id: int) -> str | None:
        lookup = self.ticker_lookup
        if lookup is None:
            return None
        idx = int(ticker_id)
        if idx < 0 or idx >= len(lookup):
            return None
        symbol = normalize_symbol(str(lookup[idx]))
        return symbol if symbol else None

    def _get_beta_by_symbol_by_date(self, symbol: str) -> dict[str, float] | None:
        symbol_key = normalize_symbol(symbol)
        cached = self._beta_by_symbol_by_date_cache.get(symbol_key, None)
        if symbol_key in self._beta_by_symbol_by_date_cache:
            return cached
        if not symbol_key:
            self._beta_by_symbol_by_date_cache[symbol_key] = None
            return None

        csv_path = self.ticker_csv_dir / f"{symbol_key}.csv"
        if not csv_path.is_file():
            self._beta_by_symbol_by_date_cache[symbol_key] = None
            return None
        try:
            beta_map = load_daily_beta_lookup_csv(
                csv_path=csv_path,
                date_col=self.ticker_date_col,
                beta_col=self.ticker_beta_col,
            )
        except Exception:
            beta_map = None
        self._beta_by_symbol_by_date_cache[symbol_key] = beta_map
        return beta_map

    def _lookup_beta_for_samples(
        self,
        ticker_ids: np.ndarray,
        timestamps: np.ndarray,
    ) -> np.ndarray:
        n = int(np.asarray(ticker_ids).shape[0])
        beta = np.ones((n,), dtype=np.float32)
        if not self.vix_beta_scaling_enabled or n <= 0:
            return beta

        tids = np.asarray(ticker_ids, dtype=np.int64).reshape(-1)
        ts = np.asarray(timestamps).astype(str, copy=False).reshape(-1)
        if int(ts.shape[0]) != int(tids.shape[0]):
            return beta

        for tid in np.unique(tids):
            symbol = self._ticker_symbol_for_id(int(tid))
            if symbol is None:
                continue
            beta_by_date = self._get_beta_by_symbol_by_date(symbol)
            if not beta_by_date:
                continue
            idx = np.flatnonzero(tids == int(tid))
            if idx.size <= 0:
                continue
            for i in idx.tolist():
                day = str(ts[int(i)])[:10]
                value = beta_by_date.get(day)
                if value is None:
                    continue
                if np.isfinite(float(value)):
                    beta[int(i)] = float(value)
        low = float(self.beta_clip_low)
        high = float(self.beta_clip_high)
        if np.isfinite(low) and np.isfinite(high) and high > low:
            beta = np.clip(beta, low, high)
        return beta

    def fit_vix_normalizer(
        self,
        split_range: tuple[int, int],
        method: str,
        clip: float,
        log1p: bool,
    ) -> dict[str, float | int | str]:
        if self.vix_by_date is None:
            raise ValueError("cannot fit VIX normalizer without --vix-daily-csv")

        rel_start, rel_end = self._validate_split_range(split_range)
        if rel_start >= rel_end:
            raise ValueError("cannot fit VIX normalizer on an empty split range")

        start_abs = self.subset_start + rel_start
        end_abs = self.subset_start + rel_end
        vix_parts: list[np.ndarray] = []
        label_valid_n = 0
        vix_lookup_n = 0
        vix_missing_n = 0

        for shard in self._iter_overlapping_shards(start_abs, end_abs):
            local_start = max(start_abs, shard.sample_start) - shard.sample_start
            local_end = min(end_abs, shard.sample_end) - shard.sample_start
            if local_end <= local_start:
                continue

            if self.npy_bundle_arrays is None:
                shard_path = self.data_dir / shard.file
                with np.load(shard_path, allow_pickle=False) as data:
                    y_raw = data["y_raw"][local_start:local_end]
                    timestamps = data["timestamps"][local_start:local_end].astype(str)
            else:
                y_raw = self.npy_bundle_arrays["y_raw"][local_start:local_end]
                timestamps = np.asarray(
                    self.npy_bundle_arrays["timestamps"][local_start:local_end]
                ).astype(str)

            ret_pct = y_raw[:, self.ret_pct_idx].astype(np.float32, copy=False)
            valid = np.isfinite(ret_pct)
            if not np.any(valid):
                continue

            ts_valid = timestamps[valid]
            label_valid_n += int(ts_valid.shape[0])
            vix_raw = self._lookup_vix_for_timestamps(ts_valid)
            vix_is_finite = np.isfinite(vix_raw)
            vix_lookup_n += int(np.sum(vix_is_finite))
            vix_missing_n += int(vix_raw.shape[0] - np.sum(vix_is_finite))
            if np.any(vix_is_finite):
                vix_parts.append(np.asarray(vix_raw[vix_is_finite], dtype=np.float32))

        if not vix_parts:
            raise ValueError(
                "daily VIX lookup produced no usable values for the train split. "
                "Check --vix-daily-csv coverage and date format."
            )

        vix_all = np.concatenate(vix_parts, axis=0)
        stats = fit_vix_norm_stats(
            raw_values=vix_all,
            method=str(method),
            clip=float(clip),
            log1p=bool(log1p),
        )
        self.vix_norm_stats = stats
        coverage = float(vix_lookup_n) / float(max(1, label_valid_n))
        return {
            "source_csv": str(self.vix_daily_csv),
            "date_col": str(self.vix_date_col),
            "value_col": str(self.vix_value_col),
            "label_valid_count": int(label_valid_n),
            "vix_found_count": int(vix_lookup_n),
            "vix_missing_count": int(vix_missing_n),
            "vix_coverage_ratio": float(coverage),
            "norm_method": str(stats.method),
            "norm_log1p": int(bool(stats.log1p)),
            "norm_clip": float(stats.clip),
            "norm_center": float(stats.center),
            "norm_scale": float(stats.scale),
            "raw_train_count": int(stats.raw_train_count),
            "raw_train_min": float(stats.raw_train_min),
            "raw_train_max": float(stats.raw_train_max),
            "raw_train_mean": float(stats.raw_train_mean),
        }

    def iter_batches(
        self,
        split_range: tuple[int, int],
        batch_size: int,
        shuffle: bool,
        seed: int,
        return_sample_indices: bool = False,
        return_ret_atr: bool = False,
        return_ret_pct: bool = False,
        return_timestamps: bool = False,
        return_ticker_ids: bool = False,
        return_vix: bool = False,
        return_vix_img: bool = False,
    ):
        rel_start, rel_end = self._validate_split_range(split_range)
        if rel_start >= rel_end:
            return
        bs = int(batch_size)
        if bs <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size!r}")
        if return_vix and self.vix_by_date is None:
            raise ValueError(
                "return_vix requested but no daily VIX lookup is configured. "
                "Set --vix-daily-csv."
            )
        if return_vix_img and not self.include_vix_image:
            raise ValueError(
                "return_vix_img requested but dataset manifest does not include VIX images."
            )

        rng = np.random.default_rng(seed)
        need_timestamps = bool(return_timestamps or return_vix)
        need_ticker_ids = bool(return_ticker_ids or return_vix)

        start_abs = self.subset_start + rel_start
        end_abs = self.subset_start + rel_end
        shard_jobs: list[tuple[ShardInfo, tuple[int, int] | np.ndarray]] = []
        for shard in self._iter_overlapping_shards(start_abs, end_abs):
            local_start = int(max(start_abs, shard.sample_start) - shard.sample_start)
            local_end = int(min(end_abs, shard.sample_end) - shard.sample_start)
            if local_end <= local_start:
                continue
            shard_jobs.append((shard, (local_start, local_end)))

        shard_jobs = self._chunk_single_bundle_jobs(shard_jobs, batch_size=bs)
        if shuffle:
            rng.shuffle(shard_jobs)

        def load_shard_state(
            shard_job: tuple[ShardInfo, tuple[int, int] | np.ndarray]
        ) -> ShardBatchState | None:
            shard, selector = shard_job
            vix_img = None
            ticker_ids: np.ndarray | None = None
            if self.npy_bundle_arrays is None:
                shard_path = self.data_dir / shard.file
                with np.load(shard_path, allow_pickle=False) as data:
                    if isinstance(selector, tuple):
                        local_start, local_end = int(selector[0]), int(selector[1])
                        if local_end <= local_start:
                            return None
                        x = data["X_img"][local_start:local_end].astype(np.float32, copy=False)
                        y_raw = data["y_raw"][local_start:local_end]
                        if "sample_indices" in data.files:
                            sample_indices = data["sample_indices"][local_start:local_end].astype(
                                np.int64, copy=False
                            )
                        else:
                            sample_indices = np.arange(
                                shard.sample_start + local_start,
                                shard.sample_start + local_end,
                                dtype=np.int64,
                            )
                        timestamps = (
                            data["timestamps"][local_start:local_end].astype(str)
                            if need_timestamps
                            else None
                        )
                        if need_ticker_ids:
                            if "ticker_ids" in data.files:
                                ticker_ids = data["ticker_ids"][local_start:local_end].astype(
                                    np.int64, copy=False
                                )
                            else:
                                ticker_ids = np.full(
                                    (int(x.shape[0]),),
                                    -1,
                                    dtype=np.int64,
                                )
                        if return_vix_img:
                            if "X_vix_img" not in data.files:
                                raise ValueError(
                                    f"return_vix_img requested but X_vix_img is missing in shard: {shard_path}"
                                )
                            vix_img = data["X_vix_img"][local_start:local_end].astype(
                                np.float32, copy=False
                            )
                    else:
                        local_idx = np.asarray(selector, dtype=np.int64).reshape(-1)
                        if local_idx.size <= 0:
                            return None
                        x = data["X_img"][local_idx].astype(np.float32, copy=False)
                        y_raw = data["y_raw"][local_idx]
                        if "sample_indices" in data.files:
                            sample_indices = data["sample_indices"][local_idx].astype(
                                np.int64, copy=False
                            )
                        else:
                            sample_indices = (local_idx + int(shard.sample_start)).astype(
                                np.int64, copy=False
                            )
                        timestamps = (
                            data["timestamps"][local_idx].astype(str)
                            if need_timestamps
                            else None
                        )
                        if need_ticker_ids:
                            if "ticker_ids" in data.files:
                                ticker_ids = data["ticker_ids"][local_idx].astype(
                                    np.int64, copy=False
                                )
                            else:
                                ticker_ids = np.full(
                                    (int(x.shape[0]),),
                                    -1,
                                    dtype=np.int64,
                                )
                        if return_vix_img:
                            if "X_vix_img" not in data.files:
                                raise ValueError(
                                    f"return_vix_img requested but X_vix_img is missing in shard: {shard_path}"
                                )
                            vix_img = data["X_vix_img"][local_idx].astype(
                                np.float32, copy=False
                            )
            else:
                bundle = self.npy_bundle_arrays
                x_all = bundle["X_img"]
                y_all = bundle["y_raw"]
                ts_all = bundle["timestamps"]
                si_all = bundle.get("sample_indices")
                tid_all = bundle.get("ticker_ids")
                vix_all = bundle.get("X_vix_img")
                if isinstance(selector, tuple):
                    local_start, local_end = int(selector[0]), int(selector[1])
                    if local_end <= local_start:
                        return None
                    x = np.asarray(x_all[local_start:local_end], dtype=np.float32)
                    y_raw = y_all[local_start:local_end]
                    if si_all is not None:
                        sample_indices = np.asarray(si_all[local_start:local_end], dtype=np.int64)
                    else:
                        sample_indices = np.arange(
                            shard.sample_start + local_start,
                            shard.sample_start + local_end,
                            dtype=np.int64,
                        )
                    timestamps = (
                        np.asarray(ts_all[local_start:local_end]).astype(str)
                        if need_timestamps
                        else None
                    )
                    if need_ticker_ids:
                        if tid_all is not None:
                            ticker_ids = np.asarray(tid_all[local_start:local_end], dtype=np.int64)
                        else:
                            ticker_ids = np.full(
                                (int(x.shape[0]),),
                                -1,
                                dtype=np.int64,
                            )
                    if return_vix_img:
                        if vix_all is None:
                            raise ValueError(
                                "return_vix_img requested but X_vix_img array is unavailable in npy bundle"
                            )
                        vix_img = np.asarray(vix_all[local_start:local_end], dtype=np.float32)
                else:
                    local_idx = np.asarray(selector, dtype=np.int64).reshape(-1)
                    if local_idx.size <= 0:
                        return None
                    x = np.asarray(x_all[local_idx], dtype=np.float32)
                    y_raw = y_all[local_idx]
                    if si_all is not None:
                        sample_indices = np.asarray(si_all[local_idx], dtype=np.int64)
                    else:
                        sample_indices = (local_idx + int(shard.sample_start)).astype(
                            np.int64, copy=False
                        )
                    timestamps = (
                        np.asarray(ts_all[local_idx]).astype(str)
                        if need_timestamps
                        else None
                    )
                    if need_ticker_ids:
                        if tid_all is not None:
                            ticker_ids = np.asarray(tid_all[local_idx], dtype=np.int64)
                        else:
                            ticker_ids = np.full(
                                (int(x.shape[0]),),
                                -1,
                                dtype=np.int64,
                            )
                    if return_vix_img:
                        if vix_all is None:
                            raise ValueError(
                                "return_vix_img requested but X_vix_img array is unavailable in npy bundle"
                            )
                        vix_img = np.asarray(vix_all[local_idx], dtype=np.float32)

            ret_pct = y_raw[:, self.ret_pct_idx].astype(np.float32, copy=False)
            valid = np.isfinite(ret_pct)
            ret_atr_valid: np.ndarray | None = None
            if return_ret_atr:
                ret_atr_all = y_raw[:, self.ret_atr_idx].astype(np.float32, copy=False)
                valid = valid & np.isfinite(ret_atr_all)
            if not np.any(valid):
                return None
            ret_pct_valid = ret_pct[valid]
            if return_ret_atr:
                ret_atr_valid = ret_atr_all[valid]
            x = x[valid]
            if return_vix_img:
                if vix_img is None:
                    raise RuntimeError("vix image requested but unavailable")
                vix_img = np.asarray(vix_img[valid], dtype=np.float32)
            y = (ret_pct_valid > self.ret_atr_threshold).astype(np.int64)
            sample_indices = sample_indices[valid]
            if need_timestamps and timestamps is not None:
                timestamps = timestamps[valid]
            if need_ticker_ids:
                if ticker_ids is None:
                    ticker_ids = np.full((int(valid.shape[0]),), -1, dtype=np.int64)
                ticker_ids = np.asarray(ticker_ids[valid], dtype=np.int64)
            vix_valid: np.ndarray | None = None
            if return_vix:
                if timestamps is None:
                    raise RuntimeError("VIX requested but timestamps are unavailable")
                vix_raw = self._lookup_vix_for_timestamps(timestamps)
                if self.vix_norm_stats is not None:
                    vix_valid = self.vix_norm_stats.transform(vix_raw)
                else:
                    vix_valid = np.asarray(vix_raw, dtype=np.float32)
                vix_valid = np.where(np.isfinite(vix_valid), vix_valid, 0.0).astype(
                    np.float32, copy=False
                )
                if ticker_ids is not None:
                    beta_valid = self._lookup_beta_for_samples(
                        ticker_ids=ticker_ids,
                        timestamps=timestamps,
                    )
                    vix_valid = np.asarray(vix_valid * beta_valid, dtype=np.float32)
            perm = np.arange(x.shape[0], dtype=np.int64)
            if shuffle:
                rng.shuffle(perm)
            return ShardBatchState(
                x=x,
                y=y,
                perm=perm,
                sample_indices=sample_indices,
                ret_atr=ret_atr_valid if return_ret_atr else None,
                ret_pct=ret_pct_valid if return_ret_pct else None,
                timestamps=(np.asarray(timestamps, dtype=object) if return_timestamps else None),
                ticker_ids=ticker_ids if need_ticker_ids else None,
                vix=vix_valid if return_vix else None,
                vix_img=vix_img if return_vix_img else None,
                cursor=0,
            )

        def emit_batch(parts: list[tuple[ShardBatchState, np.ndarray]]) -> tuple[np.ndarray, ...] | None:
            if not parts:
                return None
            xb_parts: list[np.ndarray] = []
            yb_parts: list[np.ndarray] = []
            sib_parts: list[np.ndarray] = []
            rb_parts: list[np.ndarray] = []
            rp_parts: list[np.ndarray] = []
            ts_parts: list[np.ndarray] = []
            tid_parts: list[np.ndarray] = []
            vix_parts: list[np.ndarray] = []
            vix_img_parts: list[np.ndarray] = []
            for state, b in parts:
                if b.size <= 0:
                    continue
                xb_parts.append(state.x[b])
                yb_parts.append(state.y[b])
                if return_sample_indices:
                    sib_parts.append(state.sample_indices[b])
                if return_ret_atr:
                    if state.ret_atr is None:
                        raise RuntimeError("ret_atr requested but unavailable")
                    rb_parts.append(state.ret_atr[b])
                if return_ret_pct:
                    if state.ret_pct is None:
                        raise RuntimeError("ret_pct requested but unavailable")
                    rp_parts.append(state.ret_pct[b])
                if return_timestamps:
                    if state.timestamps is None:
                        raise RuntimeError("timestamps requested but unavailable")
                    ts_parts.append(state.timestamps[b])
                if return_ticker_ids:
                    if state.ticker_ids is None:
                        raise RuntimeError("ticker_ids requested but unavailable")
                    tid_parts.append(state.ticker_ids[b])
                if return_vix:
                    if state.vix is None:
                        raise RuntimeError("vix requested but unavailable")
                    vix_parts.append(state.vix[b])
                if return_vix_img:
                    if state.vix_img is None:
                        raise RuntimeError("vix_img requested but unavailable")
                    vix_img_parts.append(state.vix_img[b])
            if not xb_parts:
                return None

            xb = np.ascontiguousarray(np.concatenate(xb_parts, axis=0))
            yb = np.ascontiguousarray(np.concatenate(yb_parts, axis=0))
            sib = (
                np.ascontiguousarray(np.concatenate(sib_parts, axis=0))
                if return_sample_indices
                else None
            )
            rb = (
                np.ascontiguousarray(np.concatenate(rb_parts, axis=0))
                if return_ret_atr
                else None
            )
            rp = (
                np.ascontiguousarray(np.concatenate(rp_parts, axis=0))
                if return_ret_pct
                else None
            )
            tsb = (
                np.asarray(np.concatenate(ts_parts, axis=0), dtype=object)
                if return_timestamps
                else None
            )
            tidb = (
                np.ascontiguousarray(np.concatenate(tid_parts, axis=0))
                if return_ticker_ids
                else None
            )
            vixb = (
                np.ascontiguousarray(np.concatenate(vix_parts, axis=0))
                if return_vix
                else None
            )
            vix_imgb = (
                np.ascontiguousarray(np.concatenate(vix_img_parts, axis=0))
                if return_vix_img
                else None
            )

            if shuffle and xb.shape[0] > 1 and len(parts) > 1:
                order = np.arange(xb.shape[0], dtype=np.int64)
                rng.shuffle(order)
                xb = np.ascontiguousarray(xb[order])
                yb = np.ascontiguousarray(yb[order])
                if sib is not None:
                    sib = np.ascontiguousarray(sib[order])
                if rb is not None:
                    rb = np.ascontiguousarray(rb[order])
                if rp is not None:
                    rp = np.ascontiguousarray(rp[order])
                if tsb is not None:
                    tsb = np.asarray(tsb[order], dtype=object)
                if tidb is not None:
                    tidb = np.ascontiguousarray(tidb[order])
                if vixb is not None:
                    vixb = np.ascontiguousarray(vixb[order])
                if vix_imgb is not None:
                    vix_imgb = np.ascontiguousarray(vix_imgb[order])

            out: list[np.ndarray] = [xb, yb]
            if return_sample_indices:
                if sib is None:
                    raise RuntimeError("sample_indices requested but unavailable")
                out.append(sib)
            if return_ret_atr:
                if rb is None:
                    raise RuntimeError("ret_atr requested but unavailable")
                out.append(rb)
            if return_ret_pct:
                if rp is None:
                    raise RuntimeError("ret_pct requested but unavailable")
                out.append(rp)
            if return_timestamps:
                if tsb is None:
                    raise RuntimeError("timestamps requested but unavailable")
                out.append(tsb)
            if return_ticker_ids:
                if tidb is None:
                    raise RuntimeError("ticker_ids requested but unavailable")
                out.append(tidb)
            if return_vix:
                if vixb is None:
                    raise RuntimeError("vix requested but unavailable")
                out.append(vixb)
            if return_vix_img:
                if vix_imgb is None:
                    raise RuntimeError("vix_img requested but unavailable")
                out.append(vix_imgb)
            return tuple(out)

        mixed_shard_batches_enabled = bool(
            shuffle
            and bool(TRAIN_MIXED_SHARD_BATCHING_ENABLED)
            and int(TRAIN_MIXED_SHARD_ACTIVE_SHARDS) > 1
        )
        if not mixed_shard_batches_enabled:
            for shard_job in shard_jobs:
                state = load_shard_state(shard_job)
                if state is None:
                    continue
                for i in range(0, state.perm.shape[0], bs):
                    b = state.perm[i : i + bs]
                    if b.size == 0:
                        continue
                    batch = emit_batch([(state, b)])
                    if batch is not None:
                        yield batch
            return

        active_shards_target = max(2, int(TRAIN_MIXED_SHARD_ACTIVE_SHARDS))
        active_states: list[ShardBatchState] = []
        shard_cursor = 0

        def refill_active_states() -> None:
            nonlocal shard_cursor
            active_states[:] = [
                s for s in active_states if int(s.cursor) < int(s.perm.shape[0])
            ]
            while len(active_states) < active_shards_target and shard_cursor < len(shard_jobs):
                state = load_shard_state(shard_jobs[shard_cursor])
                shard_cursor += 1
                if state is None:
                    continue
                if int(state.perm.shape[0]) <= 0:
                    continue
                active_states.append(state)

        refill_active_states()
        while True:
            refill_active_states()
            if not active_states:
                break
            remaining = bs
            parts: list[tuple[ShardBatchState, np.ndarray]] = []
            order = np.arange(len(active_states), dtype=np.int64)
            rng.shuffle(order)
            target_each = max(1, int(np.ceil(float(bs) / float(max(1, len(active_states))))))
            for active_idx in order:
                if remaining <= 0:
                    break
                state = active_states[int(active_idx)]
                available = int(state.perm.shape[0] - state.cursor)
                if available <= 0:
                    continue
                take = min(target_each, available, remaining)
                if take <= 0:
                    continue
                b = state.perm[state.cursor : state.cursor + take]
                state.cursor += int(take)
                parts.append((state, b))
                remaining -= int(take)

            while remaining > 0:
                refill_active_states()
                if not active_states:
                    break
                candidate_idxs = [
                    i
                    for i, state in enumerate(active_states)
                    if int(state.cursor) < int(state.perm.shape[0])
                ]
                if not candidate_idxs:
                    break
                rng.shuffle(candidate_idxs)
                made_progress = False
                for active_idx in candidate_idxs:
                    if remaining <= 0:
                        break
                    state = active_states[int(active_idx)]
                    available = int(state.perm.shape[0] - state.cursor)
                    if available <= 0:
                        continue
                    take = min(available, remaining)
                    b = state.perm[state.cursor : state.cursor + take]
                    state.cursor += int(take)
                    parts.append((state, b))
                    remaining -= int(take)
                    made_progress = True
                if not made_progress:
                    break

            if not parts:
                break
            batch = emit_batch(parts)
            if batch is not None:
                yield batch

    def compute_class_counts(self, split_range: tuple[int, int]) -> tuple[int, int]:
        """Return (class0_count, class1_count) for the provided range."""
        rel_start, rel_end = self._validate_split_range(split_range)
        if rel_start >= rel_end:
            return 0, 0

        start_abs = self.subset_start + rel_start
        end_abs = self.subset_start + rel_end
        class0_count = 0
        class1_count = 0
        for shard in self._iter_overlapping_shards(start_abs, end_abs):
            local_start = max(start_abs, shard.sample_start) - shard.sample_start
            local_end = min(end_abs, shard.sample_end) - shard.sample_start
            if local_end <= local_start:
                continue
            if self.npy_bundle_arrays is None:
                shard_path = self.data_dir / shard.file
                with np.load(shard_path, allow_pickle=False) as data:
                    y_raw = data["y_raw"][local_start:local_end]
            else:
                y_raw = self.npy_bundle_arrays["y_raw"][local_start:local_end]
            ret_pct = y_raw[:, self.ret_pct_idx].astype(np.float32, copy=False)
            valid = np.isfinite(ret_pct)
            if not np.any(valid):
                continue
            y = ret_pct[valid] > self.ret_atr_threshold
            class1 = int(np.sum(y))
            class1_count += class1
            class0_count += int(y.size - class1)
        return class0_count, class1_count


def build_split_ranges(store: ShardedImageStore, cfg: TrainConfig) -> SplitRanges:
    total = int(store.subset_count)
    if total < 3:
        raise ValueError("need at least 3 samples for train/val/test")

    if cfg.overfit_sanity:
        start = max(0, int(cfg.overfit_sample_start))
        size = max(1, int(cfg.overfit_sample_size))
        end = min(total, start + size)
        if end - start < 2:
            raise ValueError("overfit train slice must contain at least 2 samples")
        train = (start, end)
        if cfg.overfit_val_sample_size > 0:
            vsize = int(cfg.overfit_val_sample_size)
            vstart = (
                int(cfg.overfit_val_sample_start)
                if cfg.overfit_val_sample_start >= 0
                else end
            )
            if vstart >= total:
                vstart = max(0, start - vsize)
            vend = min(total, vstart + vsize)
            if vend <= vstart:
                raise ValueError("invalid overfit val slice")
            val = (vstart, vend)
        else:
            val = train
        test = val
        return SplitRanges(train=train, val=val, test=test)

    val_frac = float(cfg.val_fraction)
    test_frac = float(cfg.test_fraction)
    if val_frac <= 0 or test_frac <= 0 or val_frac + test_frac >= 1.0:
        raise ValueError("val_fraction/test_fraction must be >0 and sum <1")
    train_end = int(total * (1.0 - val_frac - test_frac))
    val_end = int(total * (1.0 - test_frac))
    train_end = max(1, min(train_end, total - 2))
    val_end = max(train_end + 1, min(val_end, total - 1))
    train = (0, train_end)
    val = (train_end, val_end)
    test = (val_end, total)
    return SplitRanges(train=train, val=val, test=test)


def build_balanced_class_weights_binary(
    class0_count: int,
    class1_count: int,
) -> torch.Tensor | None:
    n0 = int(class0_count)
    n1 = int(class1_count)
    if n0 <= 0 or n1 <= 0:
        return None
    total = float(n0 + n1)
    w0 = total / (2.0 * float(n0))
    w1 = total / (2.0 * float(n1))
    return torch.tensor([w0, w1], dtype=torch.float32)


def estimate_epoch_time_alone(
    train_samples: int,
    val_samples: int,
    enabled: bool = bool(WRITE_ESTIMATED_EPOCH_TIME_ALONE_DEFAULT),
) -> dict[str, float | int | bool] | None:
    if not bool(enabled):
        return None
    n_train = max(0, int(train_samples))
    n_val = max(0, int(val_samples))
    train_sps = max(1e-9, float(ESTIMATED_EPOCH_TRAIN_SPS_ALONE_DEFAULT))
    val_sps = max(1e-9, float(ESTIMATED_EPOCH_VAL_SPS_ALONE_DEFAULT))
    overhead_sec = max(0.0, float(ESTIMATED_EPOCH_FIXED_OVERHEAD_SEC_DEFAULT))

    train_sec = float(n_train) / train_sps
    val_sec = float(n_val) / val_sps
    total_sec = train_sec + val_sec + overhead_sec
    return {
        "enabled": True,
        "train_samples": int(n_train),
        "val_samples": int(n_val),
        "train_sps_assumed": float(train_sps),
        "val_sps_assumed": float(val_sps),
        "fixed_overhead_sec_assumed": float(overhead_sec),
        "train_sec_est": float(train_sec),
        "val_sec_est": float(val_sec),
        "total_sec_est": float(total_sec),
        "total_min_est": float(total_sec / 60.0),
    }


def apply_seq_input_transforms_inplace(
    x_seq: np.ndarray,
    seq_volume_feature_idx: int,
    seq_ratio_feature_indices: Sequence[int] | None = None,
) -> np.ndarray:
    x = np.asarray(x_seq, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"expected X_seq shape (batch,F,L); got {tuple(x.shape)}")
    if (
        int(seq_volume_feature_idx) >= 0
        and bool(SEQ_VOLUME_NORM_ROBUST_ZSCORE_ENABLED)
        and x.shape[0] > 0
    ):
        vol = x[:, int(seq_volume_feature_idx), :]
        if bool(SEQ_VOLUME_NORM_LOG1P_ENABLED):
            vol = np.log1p(np.clip(vol, a_min=0.0, a_max=None))
        vol_med = np.median(vol, axis=1, keepdims=True)
        vol_mad = np.median(np.abs(vol - vol_med), axis=1, keepdims=True)
        vol = float(SEQ_VOLUME_NORM_ROBUST_SCALE) * (vol - vol_med) / np.maximum(
            vol_mad, float(SEQ_VOLUME_NORM_ROBUST_EPS)
        )
        clip = float(SEQ_VOLUME_NORM_ROBUST_CLIP)
        if clip > 0.0:
            vol = np.clip(vol, -clip, clip)
        x[:, int(seq_volume_feature_idx), :] = vol.astype(np.float32, copy=False)
    ratio_clip = float(SEQ_RATIO_FEATURE_ABS_CLIP)
    if ratio_clip > 0.0 and seq_ratio_feature_indices is not None:
        for ratio_idx in seq_ratio_feature_indices:
            idx = int(ratio_idx)
            if idx < 0 or idx >= int(x.shape[1]):
                continue
            np.clip(x[:, idx, :], -ratio_clip, ratio_clip, out=x[:, idx, :])
    seq_clip = float(SEQ_INPUT_ABS_CLIP)
    if seq_clip > 0.0 and x.size > 0:
        np.clip(x, -seq_clip, seq_clip, out=x)
    if not np.all(np.isfinite(x)):
        n_bad = int(np.size(x) - int(np.isfinite(x).sum()))
        raise ValueError(
            f"dual aligned X_seq contains non-finite values in seq mode (count={n_bad})"
        )
    return x


def compute_seq_feature_mean_std(
    x_seq: np.ndarray,
    feature_names: Sequence[str] | None,
    sample_start: int,
    sample_end: int,
) -> list[dict[str, float | int | str]]:
    x = np.asarray(x_seq)
    if x.ndim != 3:
        raise ValueError(f"expected seq array shape (N,F,L); got {tuple(x.shape)}")
    lo = max(0, int(sample_start))
    hi = min(int(sample_end), int(x.shape[0]))
    if hi <= lo:
        return []
    feature_count = int(x.shape[1])
    names: list[str] = []
    for idx in range(feature_count):
        name = ""
        if feature_names is not None and idx < len(feature_names):
            name = str(feature_names[idx]).strip()
        names.append(name if name else f"feature_{idx:02d}")

    out: list[dict[str, float | int | str]] = []
    for idx in range(feature_count):
        vals = np.asarray(x[lo:hi, idx, :], dtype=np.float64).reshape(-1)
        finite = np.isfinite(vals)
        count = int(np.sum(finite))
        if count <= 0:
            mean = float("nan")
            std = float("nan")
        else:
            vals_finite = vals[finite]
            mean = float(np.mean(vals_finite))
            std = float(np.std(vals_finite))
        out.append(
            {
                "feature_idx": int(idx),
                "feature_name": str(names[idx]),
                "count": int(count),
                "mean": float(mean),
                "std": float(std),
            }
        )
    return out


def compute_seq_feature_mean_std_after_input_transforms(
    x_seq: np.ndarray,
    feature_names: Sequence[str] | None,
    sample_start: int,
    sample_end: int,
    seq_volume_feature_idx: int,
    seq_ratio_feature_indices: Sequence[int] | None,
    seq_feature_standardizer_mean: np.ndarray | None = None,
    seq_feature_standardizer_scale: np.ndarray | None = None,
    seq_feature_standardization_clip: float = 0.0,
    chunk_size: int = 2048,
) -> list[dict[str, float | int | str]]:
    x = np.asarray(x_seq)
    if x.ndim != 3:
        raise ValueError(f"expected seq array shape (N,F,L); got {tuple(x.shape)}")
    lo = max(0, int(sample_start))
    hi = min(int(sample_end), int(x.shape[0]))
    if hi <= lo:
        return []
    feature_count = int(x.shape[1])
    names: list[str] = []
    for idx in range(feature_count):
        name = ""
        if feature_names is not None and idx < len(feature_names):
            name = str(feature_names[idx]).strip()
        names.append(name if name else f"feature_{idx:02d}")

    chunk_n = max(1, int(chunk_size))
    sums = np.zeros((feature_count,), dtype=np.float64)
    sums_sq = np.zeros((feature_count,), dtype=np.float64)
    counts = np.zeros((feature_count,), dtype=np.int64)

    for row_start in range(lo, hi, chunk_n):
        row_end = min(hi, row_start + chunk_n)
        x_chunk = np.asarray(x[row_start:row_end], dtype=np.float32).copy()
        x_chunk = apply_seq_input_transforms_inplace(
            x_seq=x_chunk,
            seq_volume_feature_idx=int(seq_volume_feature_idx),
            seq_ratio_feature_indices=seq_ratio_feature_indices,
        )
        if (
            seq_feature_standardizer_mean is not None
            and seq_feature_standardizer_scale is not None
        ):
            mean_v = np.asarray(seq_feature_standardizer_mean, dtype=np.float32).reshape(-1)
            scale_v = np.asarray(seq_feature_standardizer_scale, dtype=np.float32).reshape(-1)
            if int(mean_v.size) != feature_count or int(scale_v.size) != feature_count:
                raise ValueError(
                    "seq feature standardizer stats shape mismatch in stats helper: "
                    f"feature_count={feature_count} mean={int(mean_v.size)} scale={int(scale_v.size)}"
                )
            x_chunk -= mean_v.reshape(1, feature_count, 1)
            x_chunk /= scale_v.reshape(1, feature_count, 1)
            clip = float(seq_feature_standardization_clip)
            if clip > 0.0:
                np.clip(x_chunk, -clip, clip, out=x_chunk)
        feature_view = (
            np.transpose(x_chunk, (1, 0, 2)).reshape(feature_count, -1).astype(np.float64, copy=False)
        )
        sums += np.sum(feature_view, axis=1, dtype=np.float64)
        sums_sq += np.sum(feature_view * feature_view, axis=1, dtype=np.float64)
        counts += np.int64(feature_view.shape[1])

    out: list[dict[str, float | int | str]] = []
    for idx in range(feature_count):
        count = int(counts[idx])
        if count <= 0:
            mean = float("nan")
            std = float("nan")
        else:
            mean = float(sums[idx] / float(count))
            var = float((sums_sq[idx] / float(count)) - (mean * mean))
            if var < 0.0 and abs(var) <= 1e-12:
                var = 0.0
            std = float(var**0.5) if np.isfinite(var) and var >= 0.0 else float("nan")
        out.append(
            {
                "feature_idx": int(idx),
                "feature_name": str(names[idx]),
                "count": int(count),
                "mean": float(mean),
                "std": float(std),
            }
        )
    return out


def _format_keras_eta(seconds: float) -> str:
    s = max(0, int(round(float(seconds))))
    if s >= 3600:
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h}:{m:02d}:{sec:02d}"
    if s >= 60:
        m = s // 60
        sec = s % 60
        return f"{m}:{sec:02d}"
    return f"{s}s"


def _format_keras_time_per_step(elapsed_sec: float, steps_done: int) -> str:
    done = max(1, int(steps_done))
    sec_per_step = float(elapsed_sec) / float(done)
    if sec_per_step >= 1.0:
        if sec_per_step >= 10.0:
            return f"{int(round(sec_per_step))}s/step"
        return f"{sec_per_step:.2f}s/step"
    ms = sec_per_step * 1000.0
    if ms >= 10.0:
        return f"{int(round(ms))}ms/step"
    return f"{ms:.1f}ms/step"


def format_keras_progbar_line(
    steps_done: int,
    total_steps: int,
    elapsed_sec: float,
    metric_pairs: Sequence[tuple[str, float]],
    final: bool,
) -> str:
    done = max(0, int(steps_done))
    total = max(1, int(total_steps))
    done_disp = min(done, total)

    bar_w = max(1, int(KERAS_PROGRESS_BAR_WIDTH))
    if final or done_disp >= total:
        bar = "=" * bar_w
    else:
        progress = float(done_disp) / float(total)
        filled = int(bar_w * progress)
        if filled <= 0:
            bar = ">" + "." * max(0, bar_w - 1)
        elif filled >= bar_w:
            bar = "=" * bar_w
        else:
            bar = "=" * max(0, filled - 1) + ">" + "." * max(0, bar_w - filled)

    if final:
        prefix = (
            f"{done_disp}/{total} [{bar}] - "
            f"{max(0, int(round(float(elapsed_sec))))}s "
            f"{_format_keras_time_per_step(elapsed_sec, done_disp)}"
        )
    else:
        remaining = max(0, total - done_disp)
        eta_sec = (float(elapsed_sec) / float(max(1, done_disp))) * float(remaining)
        prefix = f"{done_disp}/{total} [{bar}] - ETA: {_format_keras_eta(eta_sec)}"

    metrics_txt = []
    for k, v in metric_pairs:
        if np.isfinite(v):
            metrics_txt.append(f"{k}: {float(v):.4f}")
        else:
            metrics_txt.append(f"{k}: nan")
    if metrics_txt:
        return prefix + " - " + " - ".join(metrics_txt)
    return prefix


def to_device_seq_batch(
    xb: np.ndarray,
    yb: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(np.asarray(xb, dtype=np.float32))
    y = torch.from_numpy(np.asarray(yb, dtype=np.float32))
    x = x.to(device=device, non_blocking=True)
    y = y.to(device=device, non_blocking=True)
    return x, y


def run_seq_epoch(
    model: nn.Module,
    store: SequenceDualAlignedStore,
    split_range: tuple[int, int],
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    train: bool,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    seed: int,
    elastic_net_l1: float,
    elastic_net_l2: float,
    loss_mode: str,
    distributed_reduce: bool = False,
    progress_label: str | None = None,
) -> dict[str, float]:
    if train and optimizer is None:
        raise ValueError("optimizer required for train epoch")
    model.train(mode=train)

    total_n = 0
    stats_device = device if device.type == "cuda" else torch.device("cpu")
    stats_dtype = torch.float64
    loss_sum_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    data_loss_sum_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    reg_loss_sum_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    mae_sum_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    mse_sum_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    pred_sum_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    pred_sq_sum_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    pred_count_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    distributed_ready = bool(
        distributed_reduce and dist.is_available() and dist.is_initialized()
    )

    batch_iter = store.iter_batches(
        split_range=split_range,
        batch_size=batch_size,
        shuffle=train,
        seed=seed,
        return_sample_indices=False,
        return_timestamps=False,
        return_ticker_ids=False,
    )

    autocast_enabled = bool(amp_enabled and device.type == "cuda")
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
    use_ddp_join = train and isinstance(model, DDP)
    join_ctx = model.join if use_ddp_join else nullcontext
    t_epoch_start = time.perf_counter()
    step_count = 0
    split_n_hint = max(0, int(split_range[1]) - int(split_range[0]))
    steps_total_hint = (
        int(math.ceil(float(split_n_hint) / float(max(1, int(batch_size)))))
        if split_n_hint > 0
        else 0
    )
    progress_enabled = bool(progress_label) and bool(KERAS_INTRA_EPOCH_LOGGING_ENABLED)
    progress_update_every = max(1, int(KERAS_PROGRESS_UPDATE_EVERY_STEPS))
    progress_line_len = 0
    running_n_local = 0
    running_loss_local = 0.0
    running_mae_local = 0.0

    def write_progress_line(line: str) -> None:
        nonlocal progress_line_len
        pad_len = max(0, int(progress_line_len - len(line)))
        sys.stdout.write("\r" + line + (" " * pad_len))
        sys.stdout.flush()
        progress_line_len = len(line)

    if progress_enabled:
        print(str(progress_label))

    with join_ctx():
        for batch in batch_iter:
            step_count += 1
            xb_np, yb_np = batch
            xb, yb = to_device_seq_batch(xb_np, yb_np, device=device)
            if train:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=autocast_device_type, enabled=autocast_enabled
                ):
                    pred = model(xb)
                    data_loss = seq_regression_loss(pred, yb, mode=loss_mode)
                reg_loss = elastic_net_penalty(
                    model=model,
                    l1_lambda=elastic_net_l1,
                    l2_lambda=elastic_net_l2,
                ).to(device=data_loss.device, dtype=data_loss.dtype)
                loss = data_loss + reg_loss
                if autocast_enabled and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                with torch.no_grad():
                    with torch.amp.autocast(
                        device_type=autocast_device_type, enabled=autocast_enabled
                    ):
                        pred = model(xb)
                        data_loss = seq_regression_loss(pred, yb, mode=loss_mode)
                    reg_loss = torch.zeros_like(data_loss)
                    loss = data_loss

            err = (pred.detach() - yb.detach()).to(dtype=torch.float32)
            abs_err = torch.abs(err)
            sq_err = torch.square(err)
            bs = int(yb.shape[0])
            total_n += bs
            running_n_local += bs
            running_loss_local += float(loss.detach().item()) * bs
            running_mae_local += float(abs_err.sum().item())

            loss_sum_t += loss.detach().to(device=stats_device, dtype=stats_dtype) * bs
            data_loss_sum_t += data_loss.detach().to(device=stats_device, dtype=stats_dtype) * bs
            reg_loss_sum_t += reg_loss.detach().to(device=stats_device, dtype=stats_dtype) * bs
            mae_sum_t += abs_err.to(device=stats_device, dtype=stats_dtype).sum()
            mse_sum_t += sq_err.to(device=stats_device, dtype=stats_dtype).sum()
            pred_stats = pred.detach().to(device=stats_device, dtype=stats_dtype).reshape(-1)
            pred_sum_t += pred_stats.sum()
            pred_sq_sum_t += torch.square(pred_stats).sum()
            pred_count_t += torch.tensor(
                float(pred_stats.numel()), device=stats_device, dtype=stats_dtype
            )

            if progress_enabled and (
                step_count % progress_update_every == 0 or step_count == 1
            ):
                elapsed_now = float(max(1e-12, time.perf_counter() - t_epoch_start))
                total_display = max(1, int(steps_total_hint), int(step_count))
                if int(step_count) < int(total_display):
                    metric_pairs = [
                        ("loss", float(running_loss_local / max(1, running_n_local))),
                        ("mae", float(running_mae_local / max(1, running_n_local))),
                    ]
                    line = format_keras_progbar_line(
                        steps_done=int(step_count),
                        total_steps=int(total_display),
                        elapsed_sec=float(elapsed_now),
                        metric_pairs=metric_pairs,
                        final=False,
                    )
                    write_progress_line(line)

    total_n_t = torch.tensor(float(total_n), device=stats_device, dtype=stats_dtype)
    if distributed_ready:
        stats_vec = torch.stack(
            (
                loss_sum_t,
                data_loss_sum_t,
                reg_loss_sum_t,
                mae_sum_t,
                mse_sum_t,
                pred_sum_t,
                pred_sq_sum_t,
                pred_count_t,
                total_n_t,
            ),
            dim=0,
        )
        dist.all_reduce(stats_vec, op=dist.ReduceOp.SUM)
        (
            loss_sum_t,
            data_loss_sum_t,
            reg_loss_sum_t,
            mae_sum_t,
            mse_sum_t,
            pred_sum_t,
            pred_sq_sum_t,
            pred_count_t,
            total_n_t,
        ) = tuple(stats_vec.unbind(dim=0))

    total_n = int(total_n_t.item())
    elapsed_total = float(max(1e-12, time.perf_counter() - t_epoch_start))
    if total_n == 0:
        if progress_enabled:
            total_display = max(1, int(steps_total_hint), int(step_count))
            line = format_keras_progbar_line(
                steps_done=int(step_count),
                total_steps=int(total_display),
                elapsed_sec=float(elapsed_total),
                metric_pairs=[("loss", float("nan")), ("mae", float("nan"))],
                final=True,
            )
            write_progress_line(line)
            sys.stdout.write("\n")
            sys.stdout.flush()
        return {
            "loss": float("nan"),
            "data_loss": float("nan"),
            "reg_loss": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "pred_std": float("nan"),
            "n": 0.0,
        }

    loss_mean = float(loss_sum_t.item()) / float(total_n)
    data_loss_mean = float(data_loss_sum_t.item()) / float(total_n)
    reg_loss_mean = float(reg_loss_sum_t.item()) / float(total_n)
    mae_mean = float(mae_sum_t.item()) / float(total_n)
    mse_mean = float(mse_sum_t.item()) / float(total_n)
    rmse = float(mse_mean**0.5) if np.isfinite(mse_mean) and mse_mean >= 0.0 else float("nan")
    pred_count = float(pred_count_t.item())
    mean_pred = float(pred_sum_t.item()) / pred_count if pred_count > 0.0 else float("nan")
    var_pred = (
        max(0.0, (float(pred_sq_sum_t.item()) / pred_count) - (mean_pred * mean_pred))
        if pred_count > 0.0 and np.isfinite(mean_pred)
        else float("nan")
    )
    pred_std = float(var_pred**0.5) if np.isfinite(var_pred) else float("nan")
    out = {
        "loss": loss_mean,
        "data_loss": data_loss_mean,
        "reg_loss": reg_loss_mean,
        "mae": mae_mean,
        "rmse": rmse,
        "pred_std": pred_std,
        "n": float(total_n),
    }
    if progress_enabled:
        total_display = max(1, int(steps_total_hint), int(step_count))
        line = format_keras_progbar_line(
            steps_done=int(step_count),
            total_steps=int(total_display),
            elapsed_sec=float(elapsed_total),
            metric_pairs=[
                ("loss", float(out["loss"])),
                ("mae", float(out["mae"])),
            ],
            final=True,
        )
        write_progress_line(line)
        sys.stdout.write("\n")
        sys.stdout.flush()
    return out


def collect_seq_split_predictions(
    model: nn.Module,
    store: SequenceDualAlignedStore,
    split_range: tuple[int, int],
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    loss_mode: str,
) -> dict[str, np.ndarray | float]:
    model.eval()
    autocast_enabled = bool(
        amp_enabled and device.type == "cuda" and bool(PREDS_EXPORT_AMP_ENABLED)
    )
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"

    all_sample_indices: list[np.ndarray] = []
    all_y_true: list[np.ndarray] = []
    all_y_pred: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    all_ticker_ids: list[np.ndarray] = []
    total_n = 0
    loss_sum = 0.0
    mae_sum = 0.0
    mse_sum = 0.0

    with torch.no_grad():
        for batch in store.iter_batches(
            split_range=split_range,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
            return_sample_indices=True,
            return_timestamps=True,
            return_ticker_ids=True,
        ):
            xb_np, yb_np, si_np, ts_np, tid_np = batch
            xb, yb = to_device_seq_batch(xb_np, yb_np, device=device)
            with torch.amp.autocast(
                device_type=autocast_device_type, enabled=autocast_enabled
            ):
                pred = model(xb)
                loss = seq_regression_loss(pred, yb, mode=loss_mode)
            err = (pred - yb).detach()
            abs_err = torch.abs(err)
            sq_err = torch.square(err)

            bs = int(yb.shape[0])
            total_n += bs
            loss_sum += float(loss.detach().item()) * bs
            mae_sum += float(abs_err.sum().item())
            mse_sum += float(sq_err.sum().item())

            all_sample_indices.append(np.asarray(si_np, dtype=np.int64))
            all_y_true.append(yb.detach().cpu().numpy().astype(np.float32, copy=False))
            all_y_pred.append(pred.detach().cpu().numpy().astype(np.float32, copy=False))
            all_timestamps.append(np.asarray(ts_np, dtype=object))
            all_ticker_ids.append(np.asarray(tid_np, dtype=np.int64))

    if total_n == 0:
        return {
            "loss": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "n": 0.0,
            "sample_indices": np.empty((0,), dtype=np.int64),
            "y_true_ret_pct": np.empty((0,), dtype=np.float32),
            "y_pred_ret_pct": np.empty((0,), dtype=np.float32),
            "timestamps": np.empty((0,), dtype=object),
            "ticker_ids": np.empty((0,), dtype=np.int64),
        }

    sample_indices = np.concatenate(all_sample_indices, axis=0).astype(np.int64, copy=False)
    y_true = np.concatenate(all_y_true, axis=0).astype(np.float32, copy=False)
    y_pred = np.concatenate(all_y_pred, axis=0).astype(np.float32, copy=False)
    timestamps = np.asarray(np.concatenate(all_timestamps, axis=0), dtype=object)
    ticker_ids = np.concatenate(all_ticker_ids, axis=0).astype(np.int64, copy=False)
    order = np.argsort(sample_indices, kind="mergesort")
    sample_indices = sample_indices[order]
    y_true = y_true[order]
    y_pred = y_pred[order]
    timestamps = np.asarray(timestamps[order], dtype=object)
    ticker_ids = ticker_ids[order]

    return {
        "loss": float(loss_sum / total_n),
        "mae": float(mae_sum / total_n),
        "rmse": float((mse_sum / total_n) ** 0.5),
        "n": float(total_n),
        "sample_indices": sample_indices,
        "y_true_ret_pct": y_true,
        "y_pred_ret_pct": y_pred,
        "timestamps": timestamps,
        "ticker_ids": ticker_ids,
    }


def write_val_regression_preds_best(
    out_dir: Path,
    best_epoch: int,
    pred: dict[str, np.ndarray | float],
) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "val_preds_best.npz"
    csv_path = out_dir / "val_preds_best.csv"
    sample_indices = np.asarray(pred["sample_indices"], dtype=np.int64)
    y_true = np.asarray(pred["y_true_ret_pct"], dtype=np.float32)
    y_pred = np.asarray(pred["y_pred_ret_pct"], dtype=np.float32)
    timestamps = np.asarray(pred["timestamps"], dtype=object)
    ticker_ids = np.asarray(pred["ticker_ids"], dtype=np.int64)
    y_pred_finite = np.asarray(y_pred[np.isfinite(y_pred)], dtype=np.float64)
    if y_pred_finite.size > 1:
        y_pred_std = float(np.std(y_pred_finite))
        if y_pred_std == 0.0:
            print(
                "[warn] val_preds_best export has zero prediction std "
                "(all finite y_pred_ret_pct values are identical)."
            )
    # Always rank exported seq-regression predictions by predicted ret_pct descending.
    sort_key = np.where(np.isfinite(y_pred), y_pred, -np.inf).astype(np.float64, copy=False)
    order = np.argsort(-sort_key, kind="mergesort")
    sample_indices = sample_indices[order]
    y_true = y_true[order]
    y_pred = y_pred[order]
    timestamps = np.asarray(timestamps[order], dtype=object)
    ticker_ids = ticker_ids[order]
    np.savez_compressed(
        npz_path,
        best_epoch=np.array([int(best_epoch)], dtype=np.int64),
        sample_indices=sample_indices,
        y_true_ret_pct=y_true,
        y_pred_ret_pct=y_pred,
        timestamps=timestamps,
        ticker_ids=ticker_ids,
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sample_index",
                "ticker_id",
                "y_true_ret_pct",
                "y_pred_ret_pct",
                "error",
                "timestamp",
            ]
        )
        for i in range(sample_indices.shape[0]):
            err = float(y_pred[i] - y_true[i])
            w.writerow(
                [
                    int(sample_indices[i]),
                    int(ticker_ids[i]),
                    float(y_true[i]),
                    float(y_pred[i]),
                    err,
                    str(timestamps[i]),
                ]
            )
    return npz_path, csv_path


def to_device_batch(
    xb: np.ndarray,
    yb: np.ndarray,
    device: torch.device,
    vixb: np.ndarray | None = None,
    vix_imgb: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    x = torch.from_numpy(xb)
    y = torch.from_numpy(yb)
    vix: torch.Tensor | None = None
    vix_img: torch.Tensor | None = None
    if vixb is not None:
        vix = torch.from_numpy(np.asarray(vixb, dtype=np.float32))
    if vix_imgb is not None:
        vix_img = torch.from_numpy(np.asarray(vix_imgb, dtype=np.float32))
    x = x.to(device=device, non_blocking=True)
    y = y.to(device=device, non_blocking=True)
    if vix is not None:
        vix = vix.to(device=device, non_blocking=True)
    if vix_img is not None:
        vix_img = vix_img.to(device=device, non_blocking=True)
    if device.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)
        if vix_img is not None and vix_img.ndim == 4:
            vix_img = vix_img.contiguous(memory_format=torch.channels_last)
    return x, y, vix, vix_img


def run_epoch(
    model: nn.Module,
    store: ShardedImageStore,
    split_range: tuple[int, int],
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    train: bool,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    seed: int,
    elastic_net_l1: float,
    elastic_net_l2: float,
    class_weights: torch.Tensor | None = None,
    distributed_reduce: bool = False,
    progress_label: str | None = None,
    vix_enabled: bool = False,
    vix_image_enabled: bool = False,
    shuffle_train_labels: bool = False,
    zero_image: bool = False,
) -> dict[str, float]:
    if train and optimizer is None:
        raise ValueError("optimizer required for train epoch")
    model.train(mode=train)
    total_n = 0
    stats_device = device if device.type == "cuda" else torch.device("cpu")
    stats_dtype = torch.float64
    loss_sum_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    ce_loss_sum_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    reg_loss_sum_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    acc_sum_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    prob_sum_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    prob_sq_sum_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    tp_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    tn_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    fp_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    fn_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    distributed_ready = bool(
        distributed_reduce and dist.is_available() and dist.is_initialized()
    )

    batch_iter = store.iter_batches(
        split_range=split_range,
        batch_size=batch_size,
        shuffle=train,
        seed=seed,
        return_ret_atr=False,
        return_vix=bool(vix_enabled),
        return_vix_img=bool(vix_image_enabled),
    )

    autocast_enabled = bool(amp_enabled and device.type == "cuda")
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
    use_ddp_join = train and isinstance(model, DDP)
    join_ctx = model.join if use_ddp_join else nullcontext
    t_epoch_start = time.perf_counter()
    step_count = 0
    split_n_hint = max(0, int(split_range[1]) - int(split_range[0]))
    steps_total_hint = (
        int(math.ceil(float(split_n_hint) / float(max(1, int(batch_size)))))
        if split_n_hint > 0
        else 0
    )
    progress_enabled = bool(progress_label) and bool(KERAS_INTRA_EPOCH_LOGGING_ENABLED)
    progress_update_every = max(1, int(KERAS_PROGRESS_UPDATE_EVERY_STEPS))
    progress_line_len = 0
    running_n_local = 0
    running_loss_local = 0.0
    running_acc_local = 0.0

    def write_progress_line(line: str) -> None:
        nonlocal progress_line_len
        pad_len = max(0, int(progress_line_len - len(line)))
        sys.stdout.write("\r" + line + (" " * pad_len))
        sys.stdout.flush()
        progress_line_len = len(line)

    if progress_enabled:
        print(str(progress_label))

    with join_ctx():
        for batch in batch_iter:
            step_count += 1
            if vix_enabled and vix_image_enabled:
                xb_np, yb_np, vixb_np, vix_imgb_np = batch
            elif vix_enabled:
                xb_np, yb_np, vixb_np = batch
                vix_imgb_np = None
            elif vix_image_enabled:
                xb_np, yb_np, vix_imgb_np = batch
                vixb_np = None
            else:
                xb_np, yb_np = batch
                vixb_np = None
                vix_imgb_np = None
            xb, yb, vixb, vix_imgb = to_device_batch(
                xb_np,
                yb_np,
                device=device,
                vixb=vixb_np,
                vix_imgb=vix_imgb_np,
            )
            if zero_image:
                xb = xb.zero_()
            if train and shuffle_train_labels and int(yb.shape[0]) > 1:
                yb = yb[torch.randperm(int(yb.shape[0]), device=yb.device)]
            if train:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=autocast_device_type, enabled=autocast_enabled
                ):
                    logits = model(xb, vix=vixb, vix_img=vix_imgb)
                    ce_loss = F.cross_entropy(
                        logits, yb, weight=class_weights, reduction="mean"
                    )
                reg_loss = elastic_net_penalty(
                    model=model,
                    l1_lambda=elastic_net_l1,
                    l2_lambda=elastic_net_l2,
                ).to(device=ce_loss.device, dtype=ce_loss.dtype)
                loss = ce_loss + reg_loss
                if autocast_enabled and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                with torch.no_grad():
                    with torch.amp.autocast(
                        device_type=autocast_device_type, enabled=autocast_enabled
                    ):
                        logits = model(xb, vix=vixb, vix_img=vix_imgb)
                        ce_loss = F.cross_entropy(
                            logits, yb, weight=class_weights, reduction="mean"
                        )
                    reg_loss = torch.zeros_like(ce_loss)
                    loss = ce_loss

            bs = int(yb.shape[0])
            total_n += bs
            running_n_local += bs
            running_loss_local += float(loss.detach().item()) * bs
            loss_sum_t += loss.detach().to(device=stats_device, dtype=stats_dtype) * bs
            ce_loss_sum_t += ce_loss.detach().to(device=stats_device, dtype=stats_dtype) * bs
            reg_loss_sum_t += reg_loss.detach().to(device=stats_device, dtype=stats_dtype) * bs
            prob = torch.softmax(logits.detach(), dim=1)
            pred_cls = torch.argmax(prob, dim=1)
            running_acc_local += float((pred_cls == yb).sum().item())
            acc_sum_t += (pred_cls == yb).to(dtype=stats_dtype).sum()
            tp_t += ((pred_cls == 1) & (yb == 1)).to(dtype=stats_dtype).sum()
            tn_t += ((pred_cls == 0) & (yb == 0)).to(dtype=stats_dtype).sum()
            fp_t += ((pred_cls == 1) & (yb == 0)).to(dtype=stats_dtype).sum()
            fn_t += ((pred_cls == 0) & (yb == 1)).to(dtype=stats_dtype).sum()
            prob1 = prob[:, 1].to(dtype=torch.float32)
            prob1_stats = prob1.to(device=stats_device, dtype=stats_dtype)
            prob_sum_t += prob1_stats.sum()
            prob_sq_sum_t += torch.square(prob1_stats).sum()

            if progress_enabled and (
                step_count % progress_update_every == 0 or step_count == 1
            ):
                elapsed_now = float(max(1e-12, time.perf_counter() - t_epoch_start))
                total_display = max(1, int(steps_total_hint), int(step_count))
                if int(step_count) < int(total_display):
                    metric_pairs = [
                        ("loss", float(running_loss_local / max(1, running_n_local))),
                        ("acc", float(running_acc_local / max(1, running_n_local))),
                    ]
                    line = format_keras_progbar_line(
                        steps_done=int(step_count),
                        total_steps=int(total_display),
                        elapsed_sec=float(elapsed_now),
                        metric_pairs=metric_pairs,
                        final=False,
                    )
                    write_progress_line(line)

    total_n_t = torch.tensor(float(total_n), device=stats_device, dtype=stats_dtype)
    if distributed_ready:
        stats_vec = torch.stack(
            (
                loss_sum_t,
                ce_loss_sum_t,
                reg_loss_sum_t,
                acc_sum_t,
                prob_sum_t,
                prob_sq_sum_t,
                tp_t,
                tn_t,
                fp_t,
                fn_t,
                total_n_t,
            ),
            dim=0,
        )
        dist.all_reduce(stats_vec, op=dist.ReduceOp.SUM)
        (
            loss_sum_t,
            ce_loss_sum_t,
            reg_loss_sum_t,
            acc_sum_t,
            prob_sum_t,
            prob_sq_sum_t,
            tp_t,
            tn_t,
            fp_t,
            fn_t,
            total_n_t,
        ) = tuple(stats_vec.unbind(dim=0))
    total_n = int(total_n_t.item())
    elapsed_total = float(max(1e-12, time.perf_counter() - t_epoch_start))

    if total_n == 0:
        if progress_enabled:
            total_display = max(1, int(steps_total_hint), int(step_count))
            line = format_keras_progbar_line(
                steps_done=int(step_count),
                total_steps=int(total_display),
                elapsed_sec=float(elapsed_total),
                metric_pairs=[("loss", float("nan")), ("acc", float("nan"))],
                final=True,
            )
            write_progress_line(line)
            sys.stdout.write("\n")
            sys.stdout.flush()
        return {
            "loss": float("nan"),
            "ce_loss": float("nan"),
            "reg_loss": float("nan"),
            "acc": float("nan"),
            "prob_std": float("nan"),
            "ppv": float("nan"),
            "npv": float("nan"),
            "n": 0.0,
        }
    loss_sum = float(loss_sum_t.item())
    ce_loss_sum = float(ce_loss_sum_t.item())
    reg_loss_sum = float(reg_loss_sum_t.item())
    acc_sum = float(acc_sum_t.item())
    prob_sum = float(prob_sum_t.item())
    prob_sq_sum = float(prob_sq_sum_t.item())
    tp = float(tp_t.item())
    tn = float(tn_t.item())
    fp = float(fp_t.item())
    fn = float(fn_t.item())
    mean_prob = prob_sum / total_n
    var_prob = max(0.0, (prob_sq_sum / total_n) - (mean_prob * mean_prob))
    ppv_den = tp + fp
    npv_den = tn + fn
    ppv = (tp / ppv_den) if ppv_den > 0 else float("nan")
    npv = (tn / npv_den) if npv_den > 0 else float("nan")

    out = {
        "loss": loss_sum / total_n,
        "ce_loss": ce_loss_sum / total_n,
        "reg_loss": reg_loss_sum / total_n,
        "acc": acc_sum / total_n,
        "prob_std": float(var_prob ** 0.5),
        "ppv": float(ppv),
        "npv": float(npv),
        "n": float(total_n),
    }
    if progress_enabled:
        total_display = max(1, int(steps_total_hint), int(step_count))
        line = format_keras_progbar_line(
            steps_done=int(step_count),
            total_steps=int(total_display),
            elapsed_sec=float(elapsed_total),
            metric_pairs=[
                ("loss", float(out["loss"])),
                ("acc", float(out["acc"])),
            ],
            final=True,
        )
        write_progress_line(line)
        sys.stdout.write("\n")
        sys.stdout.flush()
    return out


def run_dual_epoch(
    model: nn.Module,
    store: ShardedImageStore,
    seq_store: SequenceDualAlignedStore,
    split_range: tuple[int, int],
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    train: bool,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    seed: int,
    elastic_net_l1: float,
    elastic_net_l2: float,
    dual_reg_mse_weight: float,
    class_weights: torch.Tensor | None = None,
    distributed_reduce: bool = False,
    progress_label: str | None = None,
    vix_enabled: bool = False,
    vix_image_enabled: bool = False,
    shuffle_train_labels: bool = False,
    zero_image: bool = False,
) -> dict[str, float]:
    if train and optimizer is None:
        raise ValueError("optimizer required for train epoch")
    model.train(mode=train)

    total_n = 0
    stats_device = device if device.type == "cuda" else torch.device("cpu")
    stats_dtype = torch.float64
    loss_sum_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    ce_loss_sum_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    mse_loss_sum_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    reg_loss_sum_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    acc_sum_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    prob_sum_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    prob_sq_sum_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    tp_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    tn_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    fp_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    fn_t = torch.zeros((), device=stats_device, dtype=stats_dtype)
    distributed_ready = bool(
        distributed_reduce and dist.is_available() and dist.is_initialized()
    )

    batch_iter = store.iter_batches(
        split_range=split_range,
        batch_size=batch_size,
        shuffle=train,
        seed=seed,
        return_sample_indices=True,
        return_ret_atr=False,
        return_ret_pct=False,
        return_timestamps=False,
        return_ticker_ids=False,
        return_vix=bool(vix_enabled),
        return_vix_img=bool(vix_image_enabled),
    )

    autocast_enabled = bool(amp_enabled and device.type == "cuda")
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
    use_ddp_join = train and isinstance(model, DDP)
    join_ctx = model.join if use_ddp_join else nullcontext
    t_epoch_start = time.perf_counter()
    step_count = 0
    split_n_hint = max(0, int(split_range[1]) - int(split_range[0]))
    steps_total_hint = (
        int(math.ceil(float(split_n_hint) / float(max(1, int(batch_size)))))
        if split_n_hint > 0
        else 0
    )
    progress_enabled = bool(progress_label) and bool(KERAS_INTRA_EPOCH_LOGGING_ENABLED)
    progress_update_every = max(1, int(KERAS_PROGRESS_UPDATE_EVERY_STEPS))
    progress_line_len = 0
    running_n_local = 0
    running_loss_local = 0.0
    running_acc_local = 0.0

    def write_progress_line(line: str) -> None:
        nonlocal progress_line_len
        pad_len = max(0, int(progress_line_len - len(line)))
        sys.stdout.write("\r" + line + (" " * pad_len))
        sys.stdout.flush()
        progress_line_len = len(line)

    if progress_enabled:
        print(str(progress_label))

    with join_ctx():
        for batch in batch_iter:
            step_count += 1
            if vix_enabled and vix_image_enabled:
                xb_np, yb_np, si_np, vixb_np, vix_imgb_np = batch
            elif vix_enabled:
                xb_np, yb_np, si_np, vixb_np = batch
                vix_imgb_np = None
            elif vix_image_enabled:
                xb_np, yb_np, si_np, vix_imgb_np = batch
                vixb_np = None
            else:
                xb_np, yb_np, si_np = batch
                vixb_np = None
                vix_imgb_np = None

            x_seq_np, y_reg_np = seq_store.fetch_seq_reg_by_sample_indices(
                sample_indices=np.asarray(si_np, dtype=np.int64),
                strict=True,
            )
            xb, yb, vixb, vix_imgb = to_device_batch(
                xb_np,
                yb_np,
                device=device,
                vixb=vixb_np,
                vix_imgb=vix_imgb_np,
            )
            x_seqb, y_regb = to_device_seq_batch(x_seq_np, y_reg_np, device=device)
            if zero_image:
                xb = xb.zero_()
            if train and shuffle_train_labels and int(yb.shape[0]) > 1:
                yb = yb[torch.randperm(int(yb.shape[0]), device=yb.device)]

            if train:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=autocast_device_type, enabled=autocast_enabled
                ):
                    logits, reg_pred = model(  # type: ignore[misc]
                        xb,
                        x_seq=x_seqb,
                        vix=vixb,
                        vix_img=vix_imgb,
                    )
                    ce_loss = F.cross_entropy(
                        logits, yb, weight=class_weights, reduction="mean"
                    )
                    mse_loss = F.mse_loss(reg_pred, y_regb, reduction="mean")
                reg_loss = elastic_net_penalty(
                    model=model,
                    l1_lambda=elastic_net_l1,
                    l2_lambda=elastic_net_l2,
                ).to(device=ce_loss.device, dtype=ce_loss.dtype)
                loss = ce_loss + (float(dual_reg_mse_weight) * mse_loss) + reg_loss
                if autocast_enabled and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                with torch.no_grad():
                    with torch.amp.autocast(
                        device_type=autocast_device_type, enabled=autocast_enabled
                    ):
                        logits, reg_pred = model(  # type: ignore[misc]
                            xb,
                            x_seq=x_seqb,
                            vix=vixb,
                            vix_img=vix_imgb,
                        )
                        ce_loss = F.cross_entropy(
                            logits, yb, weight=class_weights, reduction="mean"
                        )
                        mse_loss = F.mse_loss(reg_pred, y_regb, reduction="mean")
                    reg_loss = torch.zeros_like(ce_loss)
                    loss = ce_loss + (float(dual_reg_mse_weight) * mse_loss)

            bs = int(yb.shape[0])
            total_n += bs
            running_n_local += bs
            running_loss_local += float(loss.detach().item()) * bs
            loss_sum_t += loss.detach().to(device=stats_device, dtype=stats_dtype) * bs
            ce_loss_sum_t += ce_loss.detach().to(device=stats_device, dtype=stats_dtype) * bs
            mse_loss_sum_t += mse_loss.detach().to(device=stats_device, dtype=stats_dtype) * bs
            reg_loss_sum_t += reg_loss.detach().to(device=stats_device, dtype=stats_dtype) * bs
            prob = torch.softmax(logits.detach(), dim=1)
            pred_cls = torch.argmax(prob, dim=1)
            running_acc_local += float((pred_cls == yb).sum().item())
            acc_sum_t += (pred_cls == yb).to(dtype=stats_dtype).sum()
            tp_t += ((pred_cls == 1) & (yb == 1)).to(dtype=stats_dtype).sum()
            tn_t += ((pred_cls == 0) & (yb == 0)).to(dtype=stats_dtype).sum()
            fp_t += ((pred_cls == 1) & (yb == 0)).to(dtype=stats_dtype).sum()
            fn_t += ((pred_cls == 0) & (yb == 1)).to(dtype=stats_dtype).sum()
            prob1 = prob[:, 1].to(dtype=torch.float32)
            prob1_stats = prob1.to(device=stats_device, dtype=stats_dtype)
            prob_sum_t += prob1_stats.sum()
            prob_sq_sum_t += torch.square(prob1_stats).sum()

            if progress_enabled and (
                step_count % progress_update_every == 0 or step_count == 1
            ):
                elapsed_now = float(max(1e-12, time.perf_counter() - t_epoch_start))
                total_display = max(1, int(steps_total_hint), int(step_count))
                if int(step_count) < int(total_display):
                    metric_pairs = [
                        ("loss", float(running_loss_local / max(1, running_n_local))),
                        ("acc", float(running_acc_local / max(1, running_n_local))),
                    ]
                    line = format_keras_progbar_line(
                        steps_done=int(step_count),
                        total_steps=int(total_display),
                        elapsed_sec=float(elapsed_now),
                        metric_pairs=metric_pairs,
                        final=False,
                    )
                    write_progress_line(line)

    total_n_t = torch.tensor(float(total_n), device=stats_device, dtype=stats_dtype)
    if distributed_ready:
        stats_vec = torch.stack(
            (
                loss_sum_t,
                ce_loss_sum_t,
                mse_loss_sum_t,
                reg_loss_sum_t,
                acc_sum_t,
                prob_sum_t,
                prob_sq_sum_t,
                tp_t,
                tn_t,
                fp_t,
                fn_t,
                total_n_t,
            ),
            dim=0,
        )
        dist.all_reduce(stats_vec, op=dist.ReduceOp.SUM)
        (
            loss_sum_t,
            ce_loss_sum_t,
            mse_loss_sum_t,
            reg_loss_sum_t,
            acc_sum_t,
            prob_sum_t,
            prob_sq_sum_t,
            tp_t,
            tn_t,
            fp_t,
            fn_t,
            total_n_t,
        ) = tuple(stats_vec.unbind(dim=0))
    total_n = int(total_n_t.item())
    elapsed_total = float(max(1e-12, time.perf_counter() - t_epoch_start))

    if total_n == 0:
        if progress_enabled:
            total_display = max(1, int(steps_total_hint), int(step_count))
            line = format_keras_progbar_line(
                steps_done=int(step_count),
                total_steps=int(total_display),
                elapsed_sec=float(elapsed_total),
                metric_pairs=[("loss", float("nan")), ("acc", float("nan"))],
                final=True,
            )
            write_progress_line(line)
            sys.stdout.write("\n")
            sys.stdout.flush()
        return {
            "loss": float("nan"),
            "ce_loss": float("nan"),
            "mse_loss": float("nan"),
            "reg_loss": float("nan"),
            "acc": float("nan"),
            "prob_std": float("nan"),
            "ppv": float("nan"),
            "npv": float("nan"),
            "n": 0.0,
        }

    loss_sum = float(loss_sum_t.item())
    ce_loss_sum = float(ce_loss_sum_t.item())
    mse_loss_sum = float(mse_loss_sum_t.item())
    reg_loss_sum = float(reg_loss_sum_t.item())
    acc_sum = float(acc_sum_t.item())
    prob_sum = float(prob_sum_t.item())
    prob_sq_sum = float(prob_sq_sum_t.item())
    tp = float(tp_t.item())
    tn = float(tn_t.item())
    fp = float(fp_t.item())
    fn = float(fn_t.item())
    mean_prob = prob_sum / total_n
    var_prob = max(0.0, (prob_sq_sum / total_n) - (mean_prob * mean_prob))
    ppv_den = tp + fp
    npv_den = tn + fn
    ppv = (tp / ppv_den) if ppv_den > 0 else float("nan")
    npv = (tn / npv_den) if npv_den > 0 else float("nan")

    out = {
        "loss": loss_sum / total_n,
        "ce_loss": ce_loss_sum / total_n,
        "mse_loss": mse_loss_sum / total_n,
        "reg_loss": reg_loss_sum / total_n,
        "acc": acc_sum / total_n,
        "prob_std": float(var_prob ** 0.5),
        "ppv": float(ppv),
        "npv": float(npv),
        "n": float(total_n),
    }
    if progress_enabled:
        total_display = max(1, int(steps_total_hint), int(step_count))
        line = format_keras_progbar_line(
            steps_done=int(step_count),
            total_steps=int(total_display),
            elapsed_sec=float(elapsed_total),
            metric_pairs=[
                ("loss", float(out["loss"])),
                ("acc", float(out["acc"])),
            ],
            final=True,
        )
        write_progress_line(line)
        sys.stdout.write("\n")
        sys.stdout.flush()
    return out


def collect_split_predictions(
    model: nn.Module,
    store: ShardedImageStore,
    split_range: tuple[int, int],
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    class_weights: torch.Tensor | None = None,
    vix_enabled: bool = False,
    vix_image_enabled: bool = False,
    zero_image: bool = False,
) -> dict[str, np.ndarray | float]:
    model.eval()
    autocast_enabled = bool(
        amp_enabled and device.type == "cuda" and bool(PREDS_EXPORT_AMP_ENABLED)
    )
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"

    all_sample_indices: list[np.ndarray] = []
    all_y_true: list[np.ndarray] = []
    all_pred_cls: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    all_ret_atr_true: list[np.ndarray] = []
    all_ret_pct_true: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    all_ticker_ids: list[np.ndarray] = []

    total_n = 0
    loss_sum = 0.0
    acc_sum = 0.0

    with torch.no_grad():
        for batch in store.iter_batches(
            split_range=split_range,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
            return_sample_indices=True,
            return_ret_atr=True,
            return_ret_pct=True,
            return_timestamps=True,
            return_ticker_ids=True,
            return_vix=bool(vix_enabled),
            return_vix_img=bool(vix_image_enabled),
        ):
            if vix_enabled and vix_image_enabled:
                (
                    xb_np,
                    yb_np,
                    si_np,
                    ret_atr_np,
                    ret_pct_np,
                    ts_np,
                    tid_np,
                    vix_np,
                    vix_img_np,
                ) = batch
            elif vix_enabled:
                xb_np, yb_np, si_np, ret_atr_np, ret_pct_np, ts_np, tid_np, vix_np = batch
                vix_img_np = None
            elif vix_image_enabled:
                xb_np, yb_np, si_np, ret_atr_np, ret_pct_np, ts_np, tid_np, vix_img_np = batch
                vix_np = None
            else:
                xb_np, yb_np, si_np, ret_atr_np, ret_pct_np, ts_np, tid_np = batch
                vix_np = None
                vix_img_np = None
            xb, yb, vixb, vix_imgb = to_device_batch(
                xb_np,
                yb_np,
                device=device,
                vixb=vix_np,
                vix_imgb=vix_img_np,
            )
            if zero_image:
                xb = xb.zero_()
            with torch.amp.autocast(
                device_type=autocast_device_type, enabled=autocast_enabled
            ):
                logits = model(xb, vix=vixb, vix_img=vix_imgb)
                loss = F.cross_entropy(logits, yb, weight=class_weights, reduction="mean")
            prob = torch.softmax(logits, dim=1)
            pred_cls = torch.argmax(prob, dim=1)

            bs = int(yb.shape[0])
            total_n += bs
            loss_sum += float(loss.detach().item()) * bs
            acc_sum += float((pred_cls == yb).sum().item())

            all_sample_indices.append(np.asarray(si_np, dtype=np.int64))
            all_y_true.append(yb.detach().cpu().numpy().astype(np.int64, copy=False))
            all_pred_cls.append(
                pred_cls.detach().cpu().numpy().astype(np.int64, copy=False)
            )
            all_prob.append(prob.detach().cpu().numpy().astype(np.float32, copy=False))
            all_logits.append(
                logits.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            all_ret_atr_true.append(np.asarray(ret_atr_np, dtype=np.float32))
            all_ret_pct_true.append(np.asarray(ret_pct_np, dtype=np.float32))
            all_timestamps.append(np.asarray(ts_np, dtype=object))
            all_ticker_ids.append(np.asarray(tid_np, dtype=np.int64))

    if total_n == 0:
        return {
            "loss": float("nan"),
            "acc": float("nan"),
            "n": 0.0,
            "sample_indices": np.empty((0,), dtype=np.int64),
            "y_true": np.empty((0,), dtype=np.int64),
            "pred_cls": np.empty((0,), dtype=np.int64),
            "prob": np.empty((0, 2), dtype=np.float32),
            "logits": np.empty((0, 2), dtype=np.float32),
            "ret_atr_true": np.empty((0,), dtype=np.float32),
            "ret_pct_true": np.empty((0,), dtype=np.float32),
            "timestamps": np.empty((0,), dtype=object),
            "ticker_ids": np.empty((0,), dtype=np.int64),
            "tickers": np.empty((0,), dtype=object),
        }

    ticker_ids = np.concatenate(all_ticker_ids, axis=0)
    tickers = np.asarray(
        [store._ticker_symbol_for_id(int(tid)) or "" for tid in ticker_ids],
        dtype=object,
    )

    return {
        "loss": float(loss_sum / total_n),
        "acc": float(acc_sum / total_n),
        "n": float(total_n),
        "sample_indices": np.concatenate(all_sample_indices, axis=0),
        "y_true": np.concatenate(all_y_true, axis=0),
        "pred_cls": np.concatenate(all_pred_cls, axis=0),
        "prob": np.concatenate(all_prob, axis=0),
        "logits": np.concatenate(all_logits, axis=0),
        "ret_atr_true": np.concatenate(all_ret_atr_true, axis=0),
        "ret_pct_true": np.concatenate(all_ret_pct_true, axis=0),
        "timestamps": np.concatenate(all_timestamps, axis=0),
        "ticker_ids": ticker_ids,
        "tickers": tickers,
    }


def write_val_preds_best(
    out_dir: Path,
    best_epoch: int,
    pred: dict[str, np.ndarray | float],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "val_preds_best.npz"
    csv_path = out_dir / "val_preds_best.csv"

    sample_indices = np.asarray(pred["sample_indices"], dtype=np.int64)
    y_true = np.asarray(pred["y_true"], dtype=np.int64)
    pred_cls = np.asarray(pred["pred_cls"], dtype=np.int64)
    prob = np.asarray(pred["prob"], dtype=np.float32)
    if prob.ndim != 2 or prob.shape[1] != 2:
        raise ValueError("prob must have shape (n,2)")
    logits = np.asarray(pred["logits"], dtype=np.float32)
    ret_atr_true = np.asarray(pred["ret_atr_true"], dtype=np.float32)
    ret_pct_true_arr = pred.get("ret_pct_true")
    if ret_pct_true_arr is None:
        ret_pct_true = np.full((sample_indices.shape[0],), np.nan, dtype=np.float32)
    else:
        ret_pct_true = np.asarray(ret_pct_true_arr, dtype=np.float32)
        if int(ret_pct_true.shape[0]) != int(sample_indices.shape[0]):
            raise ValueError(
                "ret_pct_true length mismatch: "
                f"len={ret_pct_true.shape[0]} n={sample_indices.shape[0]}"
            )
    timestamps = np.asarray(pred.get("timestamps", np.empty((0,), dtype=object)), dtype=object)
    has_timestamps = int(timestamps.shape[0]) == int(sample_indices.shape[0])
    ticker_ids_arr = pred.get("ticker_ids")
    if ticker_ids_arr is None:
        ticker_ids = np.full((sample_indices.shape[0],), -1, dtype=np.int64)
    else:
        ticker_ids = np.asarray(ticker_ids_arr, dtype=np.int64).reshape(-1)
        if int(ticker_ids.shape[0]) != int(sample_indices.shape[0]):
            raise ValueError(
                "ticker_ids length mismatch: "
                f"len={ticker_ids.shape[0]} n={sample_indices.shape[0]}"
            )
    tickers_arr = pred.get("tickers")
    if tickers_arr is None:
        tickers = np.full((sample_indices.shape[0],), "", dtype=object)
    else:
        tickers = np.asarray(tickers_arr, dtype=object).reshape(-1)
        if int(tickers.shape[0]) != int(sample_indices.shape[0]):
            raise ValueError(
                "tickers length mismatch: "
                f"len={tickers.shape[0]} n={sample_indices.shape[0]}"
            )
        tickers = np.asarray(
            [str(v) if v is not None else "" for v in tickers],
            dtype=object,
        )

    # Always rank exported predictions by class-1 probability descending.
    p1 = np.asarray(prob[:, 1], dtype=np.float64)
    p1_finite = p1[np.isfinite(p1)]
    if p1_finite.size > 1:
        p1_std = float(np.std(p1_finite))
        if p1_std == 0.0:
            print(
                "[warn] val_preds_best export has zero prediction std "
                "(all finite prob_1 values are identical)."
            )
    sort_key = np.where(np.isfinite(p1), p1, -np.inf)
    order = np.argsort(-sort_key, kind="mergesort")
    sample_indices = sample_indices[order]
    y_true = y_true[order]
    pred_cls = pred_cls[order]
    prob = prob[order]
    logits = logits[order]
    ret_atr_true = ret_atr_true[order]
    ret_pct_true = ret_pct_true[order]
    ticker_ids = ticker_ids[order]
    tickers = tickers[order]
    if has_timestamps:
        timestamps = timestamps[order]

    npz_payload = {
        "best_epoch": np.array([int(best_epoch)], dtype=np.int64),
        "sample_indices": sample_indices,
        "y_true": y_true,
        "pred_cls": pred_cls,
        "prob": prob,
        "logits": logits,
        "ret_atr_true": ret_atr_true,
        "ret_pct_true": ret_pct_true,
        "ticker_ids": ticker_ids,
        "tickers": tickers.astype(str),
    }
    if has_timestamps:
        npz_payload["timestamps"] = timestamps.astype(str)
    np.savez_compressed(npz_path, **npz_payload)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = [
            "sample_index",
            "ticker_id",
            "ticker",
            "y_true_cls",
            "pred_cls",
            "ret_atr_true",
            "ret_pct_true",
            "prob_0",
            "prob_1",
            "logit_0",
            "logit_1",
        ]
        if has_timestamps:
            header.append("timestamp")
        w.writerow(header)
        for i in range(sample_indices.shape[0]):
            row = [
                int(sample_indices[i]),
                int(ticker_ids[i]),
                str(tickers[i]),
                int(y_true[i]),
                int(pred_cls[i]),
                float(ret_atr_true[i]),
                float(ret_pct_true[i]),
                float(prob[i, 0]),
                float(prob[i, 1]),
                float(logits[i, 0]),
                float(logits[i, 1]),
            ]
            if has_timestamps:
                row.append(str(timestamps[i]))
            w.writerow(row)

    return npz_path, csv_path


def compute_val_acc_by_threshold(
    pred: dict[str, np.ndarray | float],
    num_thresholds: int = 201,
) -> dict[str, float | list[dict[str, float]]]:
    prob = np.asarray(pred["prob"], dtype=np.float64)
    y_true = np.asarray(pred["y_true"], dtype=np.int64)
    if prob.ndim != 2 or prob.shape[1] != 2:
        raise ValueError("prob must have shape (n,2)")
    p1 = prob[:, 1]
    valid = np.isfinite(p1)
    if y_true.shape[0] != p1.shape[0]:
        raise ValueError("y_true/prob length mismatch")
    p1 = p1[valid]
    y = y_true[valid]
    n = int(y.size)
    if n == 0:
        return {
            "best_acc": float("nan"),
            "best_thresh": float("nan"),
            "rows": [],
        }

    m = max(2, int(num_thresholds))
    thresholds = np.linspace(0.0, 1.0, m, dtype=np.float64)
    rows: list[dict[str, float]] = []
    accs: list[float] = []
    for t in thresholds:
        pred_cls = (p1 >= float(t)).astype(np.int64)
        acc = float(np.mean(pred_cls == y))
        rows.append({"threshold": float(t), "val_acc": acc})
        accs.append(acc)

    acc_arr = np.asarray(accs, dtype=np.float64)
    best_acc = float(np.max(acc_arr))
    best_idx = np.where(acc_arr == best_acc)[0]
    if best_idx.size == 1:
        pick = int(best_idx[0])
    else:
        # Tie-breaker: prefer threshold closest to 0.5; then lower threshold.
        cand = thresholds[best_idx]
        dist = np.abs(cand - 0.5)
        order = np.lexsort((cand, dist))
        pick = int(best_idx[order[0]])
    best_thresh = float(thresholds[pick])
    return {
        "best_acc": best_acc,
        "best_thresh": best_thresh,
        "rows": rows,
    }


def write_val_acc_by_threshold(
    out_dir: Path,
    best_epoch: int,
    payload: dict[str, float | list[dict[str, float]]],
) -> tuple[Path, Path]:
    csv_path = out_dir / "val_best_acc_by_threshold.csv"
    json_path = out_dir / "val_best_acc_by_threshold.json"
    rows = list(payload.get("rows", []))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["best_epoch", "threshold", "val_acc"])
        for row in rows:
            w.writerow(
                [
                    int(best_epoch),
                    float(row["threshold"]),
                    float(row["val_acc"]),
                ]
            )

    json_payload = {
        "best_epoch": int(best_epoch),
        "best_acc": float(payload.get("best_acc", float("nan"))),
        "best_thresh": float(payload.get("best_thresh", float("nan"))),
        "rows": rows,
    }
    json_path.write_text(json.dumps(json_payload, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def compute_decile_metrics(pred: dict[str, np.ndarray | float]) -> list[dict[str, float | int]]:
    prob = np.asarray(pred["prob"], dtype=np.float64)
    y_true = np.asarray(pred["y_true"], dtype=np.int64)
    ret_atr_true = np.asarray(pred["ret_atr_true"], dtype=np.float64)
    if prob.ndim != 2 or prob.shape[1] != 2:
        raise ValueError("prob must have shape (n,2)")
    n = int(prob.shape[0])
    if n == 0:
        return []
    p1 = prob[:, 1]
    order = np.argsort(p1, kind="mergesort")
    pred_cls = (p1 >= 0.5).astype(np.int64)
    out: list[dict[str, float | int]] = []
    rank = np.arange(n, dtype=np.int64)
    dec = (rank * 10) // n
    for d in range(10):
        mask_sorted = dec == d
        if not np.any(mask_sorted):
            continue
        idx = order[mask_sorted]
        p = p1[idx]
        y = y_true[idx]
        yhat = pred_cls[idx]
        r = ret_atr_true[idx]
        acc = float(np.mean(yhat == y))
        out.append(
            {
                "decile": int(d),
                "count": int(idx.size),
                "prob1_min": float(np.min(p)),
                "prob1_max": float(np.max(p)),
                "prob1_mean": float(np.mean(p)),
                "ret_atr_true_mean": float(np.mean(r)),
                "acc_threshold_0p5": acc,
            }
        )
    return out


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
    _, default_outlier_filter_info = filter_eval_rows_by_tail_ret_zscore(
        [],
        zscore_threshold=outlier_zscore_threshold,
    )
    if not np.isfinite(annualization) or annualization <= 0.0:
        raise ValueError("annualization_days must be > 0")

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
        top_compounded_returns: float,
        top_sharpe: float,
        top_hit_rate: float,
        mean_bottom_pct: float,
        bottom_sharpe: float,
        bottom_hit_rate: float,
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
            "top_compounded_returns": float(top_compounded_returns),
            "top_sharpe": float(top_sharpe),
            "top_sharpe_annualized": float(top_sharpe),
            "top_hit_rate": float(top_hit_rate),
            "mean_bottom_pct": float(mean_bottom_pct),
            "bottom_sharpe": float(bottom_sharpe),
            "bottom_sharpe_annualized": float(bottom_sharpe),
            "bottom_hit_rate": float(bottom_hit_rate),
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
            top_compounded_returns=float("nan"),
            top_sharpe=float("nan"),
            top_hit_rate=float("nan"),
            mean_bottom_pct=float("nan"),
            bottom_sharpe=float("nan"),
            bottom_hit_rate=float("nan"),
            outlier_filter_info=default_outlier_filter_info,
        )
        return payload, []
    if int(timestamps.shape[0]) != n:
        raise ValueError("timestamps/prob length mismatch")

    p1 = prob[:, 1]
    dates = np.asarray([str(x)[:10] for x in timestamps], dtype=object)
    valid = np.isfinite(p1) & np.isfinite(ret_target) & (dates != "")
    p1 = p1[valid]
    ret = ret_target[valid]
    dates = dates[valid]
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
            top_compounded_returns=float("nan"),
            top_sharpe=float("nan"),
            top_hit_rate=float("nan"),
            mean_bottom_pct=float("nan"),
            bottom_sharpe=float("nan"),
            bottom_hit_rate=float("nan"),
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
                "spread": float(spread),
            }
        )

    rows, outlier_filter_info = filter_eval_rows_by_tail_ret_zscore(
        rows,
        zscore_threshold=outlier_zscore_threshold,
    )
    top_compounded_returns = float("nan")
    chosen_top_days = select_non_overlapping_signal_days(
        rows=rows,
        ordered_dates=sorted(unique_dates.tolist()),
        horizon_days=WALKFORWARD_SPY_NON_OVERLAP_HORIZON_DAYS,
    )
    if chosen_top_days:
        row_by_day: dict[str, dict[str, float | int | str]] = {}
        for row in rows:
            day = str(row.get("date", ""))[:10]
            if len(day) == 10 and day not in row_by_day:
                row_by_day[day] = row
        top_trade_returns: list[float] = []
        for day in chosen_top_days:
            row = row_by_day.get(day)
            if row is None:
                continue
            try:
                ret_val = float(row.get("top_ret_pct_mean", float("nan")))
            except (TypeError, ValueError):
                ret_val = float("nan")
            if np.isfinite(ret_val):
                top_trade_returns.append(float(ret_val))
        if top_trade_returns:
            arr = np.asarray(top_trade_returns, dtype=np.float64)
            compounded_multiple = float(np.prod(1.0 + arr))
            top_compounded_returns = float((compounded_multiple - 1.0) * 100.0)

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
            top_compounded_returns=float(top_compounded_returns),
            top_sharpe=float("nan"),
            top_hit_rate=float("nan"),
            mean_bottom_pct=float("nan"),
            bottom_sharpe=float("nan"),
            bottom_hit_rate=float("nan"),
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
        top_compounded_returns=float(top_compounded_returns),
        top_sharpe=float(top_sharpe),
        top_hit_rate=float(top_hit_rate),
        mean_bottom_pct=float(mean_bottom_pct),
        bottom_sharpe=float(bottom_sharpe),
        bottom_hit_rate=float(bottom_hit_rate),
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
        "top_compounded_returns",
        "top_sharpe",
        "top_hit_rate",
        "mean_bottom_pct",
        "bottom_sharpe",
        "bottom_hit_rate",
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

    p1 = prob[:, 1]
    dates = np.asarray([str(x)[:10] for x in timestamps], dtype=object)
    valid = np.isfinite(p1) & np.isfinite(ret_target) & (dates != "")
    p1 = p1[valid]
    ret = ret_target[valid]
    dates = dates[valid]
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
                    float(row["spread"]),
                ]
            )

    topline_payload = extract_walkforward_topline_eval_stats(payload)
    json_payload = {"best_epoch": int(best_epoch), **topline_payload, "rows": rows}
    json_path.write_text(json.dumps(json_payload, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def write_decile_metrics_best(
    out_dir: Path, best_epoch: int, rows: list[dict[str, float | int]]
) -> tuple[Path, Path]:
    csv_path = out_dir / "val_decile_metrics_best_epoch.csv"
    json_path = out_dir / "val_decile_metrics_best_epoch.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "best_epoch",
                "decile",
                "count",
                "prob1_min",
                "prob1_max",
                "prob1_mean",
                "ret_atr_true_mean",
                "acc_threshold_0p5",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    int(best_epoch),
                    int(row["decile"]),
                    int(row["count"]),
                    float(row["prob1_min"]),
                    float(row["prob1_max"]),
                    float(row["prob1_mean"]),
                    float(row["ret_atr_true_mean"]),
                    float(row["acc_threshold_0p5"]),
                ]
            )
    payload = {"best_epoch": int(best_epoch), "rows": rows}
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def write_decile_plot_best(
    out_dir: Path,
    best_epoch: int,
    rows: list[dict[str, float | int]],
) -> Path | None:
    if not rows:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] unable to create decile plot: {exc}")
        return None

    rows_sorted = sorted(rows, key=lambda r: int(r["decile"]))
    x = np.arange(len(rows_sorted), dtype=np.int64)
    labels = [f"D{int(r['decile']) + 1}" for r in rows_sorted]
    values = [float(r["ret_atr_true_mean"]) for r in rows_sorted]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=140)
    ax.plot(x, values, marker="o", linewidth=2.0, label="ret_atr_true")
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", label="zero")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Decile (by prob_1 ascending)")
    ax.set_ylabel("mean ret_atr_true")
    ax.set_title(f"Validation Deciles vs ret_atr_true (best epoch {int(best_epoch)})")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    path = out_dir / "val_decile_ret_atr_best_epoch.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_decile_class1_accuracy_plot_best(
    out_dir: Path,
    best_epoch: int,
    pred: dict[str, np.ndarray | float],
    n_deciles: int = 10,
) -> Path | None:
    p = np.asarray(pred["prob"], dtype=np.float64)
    y_true = np.asarray(pred["y_true"], dtype=np.int64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("prob must have shape (n,2)")
    if int(y_true.shape[0]) != int(p.shape[0]):
        raise ValueError("y_true/prob length mismatch")

    p1 = p[:, 1]
    valid = np.isfinite(p1)
    p1 = p1[valid]
    y = y_true[valid]
    n = int(y.size)
    if n == 0:
        return None

    deciles = max(1, int(n_deciles))
    order = np.argsort(p1, kind="mergesort")
    rank = np.arange(n, dtype=np.int64)
    dec = (rank * deciles) // n

    labels: list[str] = []
    class1_acc_values: list[float] = []
    class1_counts: list[int] = []
    for d in range(deciles):
        mask_sorted = dec == d
        if not np.any(mask_sorted):
            continue
        idx = order[mask_sorted]
        p_bin = p1[idx]
        y_bin = y[idx]
        pred_cls_bin = (p_bin >= 0.5).astype(np.int64)
        class1_mask = y_bin == 1
        class1_count = int(np.sum(class1_mask))
        class1_acc = (
            float(np.mean(pred_cls_bin[class1_mask] == 1))
            if class1_count > 0
            else float("nan")
        )
        labels.append(f"D{d + 1}")
        class1_acc_values.append(class1_acc)
        class1_counts.append(class1_count)

    if not labels:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] unable to create class1-accuracy decile plot: {exc}")
        return None

    x = np.arange(len(labels), dtype=np.int64)
    class1_acc_arr = np.asarray(class1_acc_values, dtype=np.float64)
    class1_count_arr = np.asarray(class1_counts, dtype=np.int64)
    non_empty = np.isfinite(class1_acc_arr)

    fig, (ax_acc, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(8.8, 6.4),
        dpi=140,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
        sharex=True,
    )
    ax_acc.axhline(0.5, linestyle="--", linewidth=1.0, color="gray", label="0.5")
    if np.any(non_empty):
        ax_acc.plot(
            x[non_empty],
            class1_acc_arr[non_empty],
            marker="o",
            linewidth=2.0,
            color="#1f77b4",
            label="class1 accuracy",
        )
    else:
        ax_acc.text(
            0.5,
            0.5,
            "No class1 samples in any decile",
            transform=ax_acc.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.set_ylabel("Class1 accuracy")
    ax_acc.set_title(f"Validation Decile Class1 Accuracy (best epoch {int(best_epoch)})")
    ax_acc.grid(True, alpha=0.25)
    ax_acc.legend(loc="best", fontsize=8)

    ax_hist.bar(
        x,
        class1_count_arr.astype(np.float64),
        width=0.8,
        color="#5b9bd5",
        edgecolor="white",
    )
    ax_hist.set_xticks(x)
    ax_hist.set_xticklabels(labels)
    ax_hist.set_xlabel("Decile (by prob_1 ascending)")
    ax_hist.set_ylabel("Class1 count")
    ax_hist.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    path = out_dir / "val_decile_class1_accuracy_best_epoch.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_accuracy_calibration_plot_best(
    out_dir: Path,
    best_epoch: int,
    prob: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 20,
) -> Path | None:
    p = np.asarray(prob, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("prob must have shape (n,2)")
    if int(y.shape[0]) != int(p.shape[0]):
        raise ValueError("y_true/prob length mismatch")

    p1 = p[:, 1]
    valid = np.isfinite(p1) & np.isfinite(y)
    p1 = p1[valid]
    y = y[valid]
    n = int(y.size)
    if n == 0:
        return None

    bins = max(5, int(n_bins))
    edges = np.linspace(0.0, 1.0, bins + 1, dtype=np.float64)
    bin_ids = np.digitize(p1, edges[1:-1], right=False)
    counts = np.zeros((bins,), dtype=np.int64)
    pred_mean = np.full((bins,), np.nan, dtype=np.float64)
    true_rate = np.full((bins,), np.nan, dtype=np.float64)

    for b in range(bins):
        m = bin_ids == b
        c = int(np.sum(m))
        counts[b] = c
        if c > 0:
            pred_mean[b] = float(np.mean(p1[m]))
            true_rate[b] = float(np.mean(y[m] > 0.5))

    non_empty = counts > 0
    ece = float("nan")
    if np.any(non_empty):
        w = counts[non_empty].astype(np.float64) / float(n)
        ece = float(np.sum(w * np.abs(true_rate[non_empty] - pred_mean[non_empty])))
    brier = float(np.mean((p1 - (y > 0.5).astype(np.float64)) ** 2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] unable to create accuracy calibration plot: {exc}")
        return None

    fig, (ax_cal, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(8.8, 6.4),
        dpi=140,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
        sharex=True,
    )
    ax_cal.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.2,
        color="gray",
        label="perfect calibration",
    )
    if np.any(non_empty):
        ax_cal.plot(
            pred_mean[non_empty],
            true_rate[non_empty],
            marker="o",
            linewidth=2.0,
            color="#1f77b4",
            label="binned calibration",
        )
    ax_cal.set_xlim(0.0, 1.0)
    ax_cal.set_ylim(0.0, 1.0)
    ax_cal.set_ylabel("Empirical positive rate")
    ax_cal.set_title(f"Validation Accuracy Calibration (best epoch {int(best_epoch)})")
    ax_cal.grid(True, alpha=0.25)
    ax_cal.legend(loc="lower right", fontsize=8)
    ax_cal.text(
        0.02,
        0.98,
        f"N={n}  bins={bins}  ECE={ece:.4f}  Brier={brier:.4f}",
        transform=ax_cal.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    widths = np.diff(edges)
    ax_hist.bar(
        edges[:-1],
        counts.astype(np.float64),
        width=widths,
        align="edge",
        color="#5b9bd5",
        edgecolor="white",
    )
    ax_hist.set_xlim(0.0, 1.0)
    ax_hist.set_xlabel("Predicted probability (class 1)")
    ax_hist.set_ylabel("Count")
    ax_hist.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    path = out_dir / "val_accuracy_calibration_best_epoch.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def resolve_plot_quantiles_pct(quantiles_pct: Sequence[float]) -> list[float]:
    out: list[float] = []
    for q in quantiles_pct:
        qf = float(q)
        if not np.isfinite(qf):
            raise ValueError(f"quantile is not finite: {q!r}")
        if qf <= 0.0 or qf >= 100.0:
            raise ValueError(f"quantile must be in (0, 100), got {qf}")
        out.append(qf)
    if not out:
        raise ValueError("at least one quantile is required")
    return sorted(set(out))


def write_quantile_plot_best(
    out_dir: Path,
    best_epoch: int,
    prob: np.ndarray,
    ret_atr_true: np.ndarray,
) -> Path | None:
    p = np.asarray(prob, dtype=np.float64)
    r = np.asarray(ret_atr_true, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("prob must have shape (n,2)")
    p1 = p[:, 1]
    valid = np.isfinite(p1) & np.isfinite(r)
    p1 = p1[valid]
    r = r[valid]
    n = int(r.size)
    if n == 0:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] unable to create quantile plot: {exc}")
        return None

    try:
        quantiles_pct = resolve_plot_quantiles_pct(PLOT_QUANTILES_PCT)
    except Exception as exc:
        print(f"[warn] unable to create quantile plot: invalid PLOT_QUANTILES_PCT: {exc}")
        return None
    order = np.argsort(p1, kind="mergesort")

    def mean_rank_bucket(lo_pct: float, hi_pct: float) -> float:
        start = int(np.floor((lo_pct / 100.0) * n))
        end = int(np.ceil((hi_pct / 100.0) * n))
        start = max(0, min(start, n - 1))
        end = max(start + 1, min(end, n))
        idx = order[start:end]
        return float(np.mean(r[idx]))

    q_values = [
        mean_rank_bucket(0.0, q) if q < 50.0 else mean_rank_bucket(q, 100.0)
        for q in quantiles_pct
    ]

    x = np.arange(len(quantiles_pct), dtype=np.int64)
    labels = [f"Q{q:g}" for q in quantiles_pct]

    fig, ax = plt.subplots(figsize=(10.0, 4.8), dpi=140)
    ax.plot(
        x,
        q_values,
        marker="o",
        linewidth=2.0,
        color="#cc5500",
        label="mean ret_atr_true per prob_1 quantile bucket",
    )
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", label="zero")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Tail Quantiles by prob_1 rank")
    ax.set_ylabel("mean ret_atr_true")
    ax.set_title(
        f"Validation prob_1-ranked Quantile Buckets vs ret_atr_true (best epoch {int(best_epoch)})"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    path = out_dir / "val_quantile_ret_atr_best_epoch.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def write_quantile_expected_class_plot_best(
    out_dir: Path,
    best_epoch: int,
    prob: np.ndarray,
    ret_atr_true: np.ndarray,
) -> Path | None:
    p = np.asarray(prob, dtype=np.float64)
    r = np.asarray(ret_atr_true, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("prob must have shape (n,2)")
    p1 = p[:, 1]
    valid = np.isfinite(p1) & np.isfinite(r)
    p1 = p1[valid]
    r = r[valid]
    n = int(r.size)
    if n == 0:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] unable to create expected-class quantile plot: {exc}")
        return None

    try:
        quantiles_pct = resolve_plot_quantiles_pct(PLOT_QUANTILES_PCT)
    except Exception as exc:
        print(
            f"[warn] unable to create expected-class quantile plot: invalid PLOT_QUANTILES_PCT: {exc}"
        )
        return None
    order = np.argsort(p1, kind="mergesort")
    pred_up = p1 >= 0.5

    def mean_rank_bucket_expected(lo_pct: float, hi_pct: float, expected_up: bool | None) -> float:
        start = int(np.floor((lo_pct / 100.0) * n))
        end = int(np.ceil((hi_pct / 100.0) * n))
        start = max(0, min(start, n - 1))
        end = max(start + 1, min(end, n))
        idx = order[start:end]
        if expected_up is None:
            return float(np.mean(r[idx]))
        m = pred_up[idx] if expected_up else (~pred_up[idx])
        if not np.any(m):
            return float("nan")
        return float(np.mean(r[idx][m]))

    q_values: list[float] = []
    for q in quantiles_pct:
        if q < 50.0:
            q_values.append(mean_rank_bucket_expected(0.0, q, expected_up=False))
        elif q > 50.0:
            q_values.append(mean_rank_bucket_expected(q, 100.0, expected_up=True))
        else:
            # Midpoint has no strict expected direction by the <P50/>P50 rule.
            q_values.append(float("nan"))

    x = np.arange(len(quantiles_pct), dtype=np.int64)
    labels = [f"Q{q:g}" for q in quantiles_pct]

    fig, ax = plt.subplots(figsize=(10.0, 4.8), dpi=140)
    ax.plot(
        x,
        q_values,
        marker="o",
        linewidth=2.0,
        color="#2a8f5f",
        label="mean ret_atr_true for expected predicted class",
    )
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", label="zero")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Tail Quantiles by prob_1 rank")
    ax.set_ylabel("mean ret_atr_true")
    ax.set_title(
        f"Validation Quantiles (expected-class filtered) vs ret_atr_true (best epoch {int(best_epoch)})"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    path = out_dir / "val_quantile_expected_class_ret_atr_best_epoch.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def resolve_device(token: str) -> torch.device:
    t = str(token).strip().lower()
    if t == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(t)


def init_distributed_if_needed(
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[bool, int, int, int, torch.device]:
    world_size_env = _env_int("WORLD_SIZE", 1)
    local_rank_env = _env_int("LOCAL_RANK", 0)
    use_distributed = bool(cfg.distributed or cfg.multi_gpu or world_size_env > 1)
    if not use_distributed:
        return False, 0, 1, 0, device

    if world_size_env <= 1:
        raise ValueError(
            "Multi-GPU training requires torchrun multi-process launch. "
            "Example: torchrun --standalone --nproc_per_node=2 iimage_model.py --multi-gpu"
        )
    if device.type != "cuda":
        raise ValueError("Distributed training in this script requires CUDA devices.")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available but distributed training was requested.")

    cuda_count = int(torch.cuda.device_count())
    if cuda_count <= 0:
        raise ValueError("No visible CUDA devices for distributed training.")
    if local_rank_env < 0 or local_rank_env >= cuda_count:
        raise ValueError(
            f"LOCAL_RANK={local_rank_env} is out of range for {cuda_count} visible CUDA device(s)."
        )

    torch.cuda.set_device(local_rank_env)
    local_device = torch.device(f"cuda:{local_rank_env}")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable in this PyTorch build.")
    if not dist.is_initialized():
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                device_id=local_rank_env,
            )
        except TypeError:
            dist.init_process_group(backend="nccl", init_method="env://")
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    return True, rank, world_size, local_rank_env, local_device


def distributed_cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return int(rank) == 0


def broadcast_object(value, src: int = 0):
    if not (dist.is_available() and dist.is_initialized()):
        return value
    payload = [value]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def split_range_for_rank(
    split_range: tuple[int, int],
    rank: int,
    world_size: int,
) -> tuple[int, int]:
    start, end = int(split_range[0]), int(split_range[1])
    n = max(0, end - start)
    w = max(1, int(world_size))
    r = max(0, int(rank))
    base = n // w
    rem = n % w
    if r < rem:
        local_start = start + r * (base + 1)
        local_end = local_start + (base + 1)
    else:
        local_start = start + rem * (base + 1) + (r - rem) * base
        local_end = local_start + base
    return int(local_start), int(local_end)


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, "_orig_mod"):
        return getattr(model, "_orig_mod")
    return model


def count_model_parameters(model: nn.Module) -> dict[str, int]:
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return {"total": total, "trainable": trainable, "frozen": int(total - trainable)}


def resolve_next_run_dir(root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ids: list[int] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name.strip()
        if name.isdigit():
            ids.append(int(name))
    next_id = 0 if not ids else (max(ids) + 1)
    return root / str(next_id)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.getenv("DATA_DIR", DATA_DIR),
        help=f"Input dataset directory (default: {DATA_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Defaults to runs/<next_numeric_id> when not provided.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--elastic-net-l1", type=float, default=ELASTIC_NET_L1)
    parser.add_argument("--elastic-net-l2", type=float, default=ELASTIC_NET_L2)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--fc-dropout",
        "--dropout",
        type=float,
        default=FC_DROPOUT,
        dest="fc_dropout",
        help="Dropout applied only to fully connected layers (readout + head).",
    )
    parser.add_argument("--readout-dim", type=int, default=READOUT_DIM)
    parser.add_argument(
        "--head-dims",
        type=str,
        default=",".join(str(v) for v in HEAD_MLP_DIMS_DEFAULT),
        help="Comma-separated hidden dims for classification head MLP.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable torch.distributed DDP (launch with torchrun).",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        default=False,
        help="Enable multi-GPU DDP (launch with torchrun).",
    )
    parser.add_argument("--amp-enabled", action="store_true", default=AMP_ENABLED_DEFAULT)
    parser.add_argument("--no-amp", action="store_false", dest="amp_enabled")
    parser.add_argument(
        "--ret-pct-threshold",
        "--ret-atr-threshold",
        dest="ret_atr_threshold",
        type=float,
        default=RET_ATR_THRESHOLD,
        help="Classification threshold applied to ret_pct (ret_atr flag kept as alias).",
    )
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--overfit-sanity", action="store_true", default=OVERFIT_SANITY_DEFAULT)
    parser.add_argument("--overfit-sample-start", type=int, default=OVERFIT_SAMPLE_START)
    parser.add_argument("--overfit-sample-size", type=int, default=OVERFIT_SAMPLE_SIZE)
    parser.add_argument("--overfit-val-sample-start", type=int, default=OVERFIT_VAL_SAMPLE_START)
    parser.add_argument("--overfit-val-sample-size", type=int, default=OVERFIT_VAL_SAMPLE_SIZE)
    parser.add_argument("--compile", action="store_true", dest="torch_compile")
    parser.add_argument(
        "--vix-daily-csv",
        type=str,
        default=os.getenv("VIX_DAILY_CSV", VIX_DAILY_CSV_PATH),
        help=(
            "Optional CSV with daily VIX by date. "
            "Expected columns from --vix-date-col and --vix-value-col. "
            f"(default: {VIX_DAILY_CSV_PATH})"
        ),
    )
    parser.add_argument(
        "--vix-date-col",
        type=str,
        default=VIX_DATE_COL_DEFAULT,
        help=f"Date column name in --vix-daily-csv (default: {VIX_DATE_COL_DEFAULT}).",
    )
    parser.add_argument(
        "--vix-value-col",
        type=str,
        default=VIX_VALUE_COL_DEFAULT,
        help=f"VIX value column name in --vix-daily-csv (default: {VIX_VALUE_COL_DEFAULT}).",
    )
    parser.add_argument(
        "--vix-fusion-mode",
        type=str,
        default=VIX_FUSION_MODE_DEFAULT,
        help="One of: none, film, late_concat.",
    )
    parser.add_argument("--vix-embed-dim", type=int, default=VIX_EMBED_DIM_DEFAULT)
    parser.add_argument(
        "--vix-norm-method",
        type=str,
        default=VIX_NORM_METHOD_DEFAULT,
        help="One of: robust_zscore, zscore, none.",
    )
    parser.add_argument(
        "--vix-norm-clip",
        type=float,
        default=VIX_NORM_CLIP_DEFAULT,
        help="Symmetric clip range after VIX normalization. Set <=0 to disable clipping.",
    )
    parser.add_argument(
        "--vix-log1p",
        action="store_true",
        default=bool(VIX_LOG1P_DEFAULT),
        dest="vix_log1p",
        help="Apply log1p to raw VIX before normalization (default: on).",
    )
    parser.add_argument(
        "--no-vix-log1p",
        action="store_false",
        dest="vix_log1p",
        help="Disable log1p preprocessing for raw VIX values.",
    )
    parser.add_argument(
        "--walkforward-min-per-side-mode",
        type=str,
        default=WALKFORWARD_ROLLING_MIN_PER_SIDE_MODE,
        help="One of: both, either.",
    )
    parser.add_argument(
        "--shuffle-train-labels",
        action="store_true",
        default=False,
        help=(
            "Placebo control: randomly permute labels within each training batch "
            "to break image/VIX-to-label alignment during training only."
        ),
    )
    parser.add_argument(
        "--zero-image",
        action="store_true",
        default=False,
        help=(
            "Attribution control: replace X_img with zeros for train/val/test "
            "(retains VIX inputs when enabled)."
        ),
    )
    args = parser.parse_args()

    data_dir = str(args.data_dir).strip()
    if not data_dir:
        parser.error("--data-dir must be non-empty")

    out_dir = str(args.output_dir).strip()
    if not out_dir:
        out_dir = str(resolve_next_run_dir(RUNS_ROOT))

    vix_daily_csv = str(args.vix_daily_csv).strip()
    vix_fusion_mode = resolve_vix_fusion_mode(str(args.vix_fusion_mode))
    vix_norm_method = resolve_vix_norm_method(str(args.vix_norm_method))
    walkforward_min_per_side_mode = resolve_walkforward_min_per_side_mode(
        str(args.walkforward_min_per_side_mode)
    )
    if vix_fusion_mode != "none" and not vix_daily_csv:
        parser.error("--vix-daily-csv is required when --vix-fusion-mode is not 'none'")
    if int(args.vix_embed_dim) < 1:
        parser.error("--vix-embed-dim must be >= 1")

    return TrainConfig(
        data_dir=data_dir,
        output_dir=out_dir,
        epochs=int(args.epochs),
        patience=int(args.patience),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        elastic_net_l1=float(args.elastic_net_l1),
        elastic_net_l2=float(args.elastic_net_l2),
        batch_size=int(args.batch_size),
        fc_dropout=float(args.fc_dropout),
        readout_dim=int(args.readout_dim),
        head_dims=parse_int_list(args.head_dims),
        seed=int(args.seed),
        amp_enabled=bool(args.amp_enabled),
        device=str(args.device),
        distributed=bool(args.distributed),
        ret_atr_threshold=float(args.ret_atr_threshold),
        val_fraction=float(args.val_fraction),
        test_fraction=float(args.test_fraction),
        overfit_sanity=bool(args.overfit_sanity),
        overfit_sample_start=int(args.overfit_sample_start),
        overfit_sample_size=int(args.overfit_sample_size),
        overfit_val_sample_start=int(args.overfit_val_sample_start),
        overfit_val_sample_size=int(args.overfit_val_sample_size),
        multi_gpu=bool(args.multi_gpu),
        torch_compile=bool(args.torch_compile),
        class_weighted_ce_enabled=bool(CLASS_WEIGHTED_CE_ENABLED_DEFAULT),
        vix_daily_csv=vix_daily_csv,
        vix_date_col=str(args.vix_date_col),
        vix_value_col=str(args.vix_value_col),
        vix_fusion_mode=str(vix_fusion_mode),
        vix_embed_dim=int(args.vix_embed_dim),
        vix_norm_method=str(vix_norm_method),
        vix_norm_clip=float(args.vix_norm_clip),
        vix_log1p=bool(args.vix_log1p),
        walkforward_min_per_side_mode=str(walkforward_min_per_side_mode),
        shuffle_train_labels=bool(args.shuffle_train_labels),
        zero_image=bool(args.zero_image),
    )


def run_sim_evals_script(run_dir: Path, best_epoch: int) -> dict[str, object]:
    script_path = Path(__file__).resolve().parent / "sim_evals.py"
    cmd = [
        str(sys.executable),
        str(script_path),
        "--run-dir",
        str(run_dir),
        "--best-epoch",
        str(int(best_epoch)),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"simulation eval script failed with exit code {exc.returncode}") from exc

    results_path = run_dir / "eval" / "sim_evals_results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"simulation eval results file missing: {results_path}")
    return json.loads(results_path.read_text(encoding="utf-8"))


def run_seq_branch(cfg: TrainConfig) -> None:
    out_dir = Path(cfg.output_dir)
    weights_dir = out_dir / "weights"
    preds_dir = out_dir / "preds"
    eval_dir = out_dir / "eval"
    distributed = False
    rank = 0
    world_size = 1
    local_rank = 0
    device = resolve_device(cfg.device)
    main_process = True
    loss_mode = resolve_seq_regression_loss_mode(SEQ_REGRESSION_LOSS_MODE)

    try:
        distributed, rank, world_size, local_rank, device = init_distributed_if_needed(
            cfg, device
        )
        main_process = is_main_process(rank)
        cuda_available = bool(torch.cuda.is_available())
        cuda_count = int(torch.cuda.device_count()) if cuda_available else 0

        if cfg.torch_compile and distributed:
            if main_process:
                print("[warn] --compile is disabled when --distributed is enabled.")
            cfg.torch_compile = False

        if distributed and dist.is_initialized():
            out_dir = Path(str(broadcast_object(str(out_dir), src=0)))
            weights_dir = out_dir / "weights"
            preds_dir = out_dir / "preds"
            eval_dir = out_dir / "eval"

        if main_process:
            out_dir.mkdir(parents=True, exist_ok=True)
            weights_dir.mkdir(parents=True, exist_ok=True)
            preds_dir.mkdir(parents=True, exist_ok=True)
            eval_dir.mkdir(parents=True, exist_ok=True)
        if distributed and dist.is_initialized():
            dist.barrier()

        set_seed(cfg.seed + int(rank))
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        amp_enabled = bool(cfg.amp_enabled and device.type == "cuda")

        store = SequenceDualAlignedStore(
            data_dir=Path(cfg.data_dir),
            ret_atr_threshold=cfg.ret_atr_threshold,
        )
        splits = build_split_ranges(store, cfg)
        train_split = (
            split_range_for_rank(splits.train, rank, world_size)
            if distributed
            else splits.train
        )
        seq_feature_norm_meta: dict[str, object] | None = None
        seq_feature_norm_state: dict[str, object] | None = None
        seq_target_clip_meta: dict[str, object] | None = None
        seq_target_clip_state: dict[str, object] | None = None
        if not distributed or main_process:
            seq_feature_norm_meta = store.fit_seq_feature_standardizer(
                split_range=splits.train
            )
            seq_feature_norm_state = store.get_seq_feature_standardizer_state()
            seq_target_clip_meta = store.fit_seq_target_clipper(split_range=splits.train)
            seq_target_clip_state = store.get_seq_target_clipper_state()
        if distributed:
            seq_feature_norm_meta = broadcast_object(seq_feature_norm_meta, src=0)
            seq_feature_norm_state = broadcast_object(seq_feature_norm_state, src=0)
            seq_target_clip_meta = broadcast_object(seq_target_clip_meta, src=0)
            seq_target_clip_state = broadcast_object(seq_target_clip_state, src=0)
            store.load_seq_feature_standardizer_state(seq_feature_norm_state)
            store.load_seq_target_clipper_state(seq_target_clip_state)

        train_samples_count = int(max(0, int(splits.train[1]) - int(splits.train[0])))
        val_samples_count = int(max(0, int(splits.val[1]) - int(splits.val[0])))
        epoch_time_alone_est = estimate_epoch_time_alone(
            train_samples=train_samples_count,
            val_samples=val_samples_count,
            enabled=bool(WRITE_ESTIMATED_EPOCH_TIME_ALONE_DEFAULT),
        )

        model = SequenceRegressor(
            input_features=int(store.seq_feature_count),
            input_lookback=int(store.seq_lookback),
            readout_dim=cfg.readout_dim,
            head_dims=cfg.head_dims,
            fc_dropout=cfg.fc_dropout,
        ).to(device)
        if cfg.torch_compile and hasattr(torch, "compile"):
            model = torch.compile(model)
        if distributed:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
            )
        base_model = unwrap_model(model)
        param_counts = count_model_parameters(base_model)

        if cfg.elastic_net_l1 < 0.0 or cfg.elastic_net_l2 < 0.0:
            raise ValueError("elastic-net coefficients must be >= 0")
        if cfg.elastic_net_l2 > 0.0 and cfg.weight_decay != 0.0:
            raise ValueError(
                "Set --weight-decay 0 when --elastic-net-l2 > 0 to avoid double-counting L2."
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        if main_process:
            run_meta = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": asdict(cfg),
                "runtime": {
                    "distributed": int(distributed),
                    "rank": int(rank),
                    "world_size": int(world_size),
                    "local_rank": int(local_rank),
                    "resolved_device": str(device),
                    "cuda_device_count": int(cuda_count),
                    "seq_branch_enabled": int(
                        bool(resolve_model_branch_mode(MODEL_BRANCH_MODE) == "seq")
                    ),
                    "seq_regression_loss_mode": str(loss_mode),
                    "preds_export_amp_enabled": int(bool(PREDS_EXPORT_AMP_ENABLED)),
                },
                "data": {
                    "data_dir": str(cfg.data_dir),
                    "subset_count": int(store.subset_count),
                    "dropped_image_samples_for_seq": int(store.dropped_image_samples_for_seq),
                    "decomp_source_npz": str(store.decomp_source_npz),
                    "dual_npz_path": str(store.dual_npz_path),
                    "dual_sample_count": int(store.dual_sample_count),
                    "seq_feature_count": int(store.seq_feature_count),
                    "seq_lookback": int(store.seq_lookback),
                    "ret_pct_index_dual": int(store.ret_pct_idx_dual),
                    "seq_feature_standardization_train_split": seq_feature_norm_meta,
                    "seq_target_clip_train_split": seq_target_clip_meta,
                },
                "splits": {
                    "train": list(splits.train),
                    "val": list(splits.val),
                    "test": list(splits.test),
                },
                "estimates": {
                    "per_epoch_time_alone": epoch_time_alone_est,
                },
                "model": {
                    "ts_kernel_size": [int(KERNEL_SIZE), int(KERNEL_WIDTH)],
                    "ts_pool_kernel": [int(POOL_KERNEL), 1],
                    "ts_pool_stride": [int(POOL_STRIDE), 1],
                    "fc_dropout": float(cfg.fc_dropout),
                    "readout_dim": int(cfg.readout_dim),
                    "head_dims": [int(v) for v in cfg.head_dims],
                    "rep_shape": [int(v) for v in base_model.rep_shape],
                    "param_count_total": int(param_counts["total"]),
                    "param_count_trainable": int(param_counts["trainable"]),
                    "param_count_frozen": int(param_counts["frozen"]),
                    "multi_gpu_enabled": int(bool(cfg.multi_gpu)),
                    "multi_gpu_device_ids": list(range(int(cuda_count))),
                },
            }
            (out_dir / "config.json").write_text(
                json.dumps(run_meta, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"seq_branch=on device={device} distributed={int(distributed)} "
                f"world_size={world_size} amp_enabled={int(amp_enabled)} "
                f"samples={store.subset_count} train={splits.train} val={splits.val} test={splits.test}"
            )
            if int(store.dropped_image_samples_for_seq) > 0:
                print(
                    "seq alignment note: image samples without dual counterparts were dropped: "
                    f"{int(store.dropped_image_samples_for_seq)}"
                )
            if seq_feature_norm_meta is not None:
                print(
                    "seq_feature_standardization="
                    f"{'on' if int(seq_feature_norm_meta.get('enabled', 0)) else 'off'} "
                    f"fitted={int(seq_feature_norm_meta.get('fitted', 0))} "
                    f"clip={float(store.seq_feature_standardization_clip):g} "
                    f"eps={float(store.seq_feature_standardization_eps):g}"
                )
            if seq_target_clip_meta is not None:
                print(
                    "seq_target_clipping="
                    f"{'on' if int(seq_target_clip_meta.get('enabled', 0)) else 'off'} "
                    f"fitted={int(seq_target_clip_meta.get('fitted', 0))} "
                    f"pct=[{float(store.seq_target_clip_lower_pct):g},"
                    f"{float(store.seq_target_clip_upper_pct):g}] "
                    f"bounds=[{float(store.seq_target_clip_low):.6f},"
                    f"{float(store.seq_target_clip_high):.6f}]"
                )
            seq_feature_stats = compute_seq_feature_mean_std(
                x_seq=store.X_seq,
                feature_names=store.feature_cols_seq,
                sample_start=store.subset_start,
                sample_end=store.subset_end,
            )
            print(
                "seq feature mean/std at run start (raw aligned seq) "
                f"(features={len(seq_feature_stats)} lookback={int(store.seq_lookback)}):"
            )
            if not seq_feature_stats:
                print("  [warn] no aligned seq samples available for stats")
            for row in seq_feature_stats:
                print(
                    f"  seq[{int(row['feature_idx']):02d}] {str(row['feature_name'])}: "
                    f"mean={float(row['mean']):.6f} std={float(row['std']):.6f}"
                )
            seq_feature_stats_transformed = compute_seq_feature_mean_std_after_input_transforms(
                x_seq=store.X_seq,
                feature_names=store.feature_cols_seq,
                sample_start=store.subset_start,
                sample_end=store.subset_end,
                seq_volume_feature_idx=store.seq_volume_feature_idx,
                seq_ratio_feature_indices=store.seq_ratio_feature_indices,
            )
            print(
                "seq feature mean/std at run start (post seq input transforms) "
                f"(volume_log1p={int(bool(SEQ_VOLUME_NORM_LOG1P_ENABLED))} "
                f"volume_robust_z={int(bool(SEQ_VOLUME_NORM_ROBUST_ZSCORE_ENABLED))} "
                f"volume_clip={float(SEQ_VOLUME_NORM_ROBUST_CLIP):g} "
                f"ratio_clip={float(SEQ_RATIO_FEATURE_ABS_CLIP):g} "
                f"abs_clip={float(SEQ_INPUT_ABS_CLIP):g} "
                f"features={len(seq_feature_stats_transformed)}):"
            )
            if not seq_feature_stats_transformed:
                print("  [warn] no aligned seq samples available for transformed stats")
            for row in seq_feature_stats_transformed:
                print(
                    f"  seq[{int(row['feature_idx']):02d}] {str(row['feature_name'])}: "
                    f"mean={float(row['mean']):.6f} std={float(row['std']):.6f}"
                )
            if store.seq_feature_standardizer_fitted:
                seq_feature_stats_standardized = (
                    compute_seq_feature_mean_std_after_input_transforms(
                        x_seq=store.X_seq,
                        feature_names=store.feature_cols_seq,
                        sample_start=store.subset_start,
                        sample_end=store.subset_end,
                        seq_volume_feature_idx=store.seq_volume_feature_idx,
                        seq_ratio_feature_indices=store.seq_ratio_feature_indices,
                        seq_feature_standardizer_mean=store.seq_feature_standardizer_mean,
                        seq_feature_standardizer_scale=store.seq_feature_standardizer_scale,
                        seq_feature_standardization_clip=store.seq_feature_standardization_clip,
                    )
                )
                print(
                    "seq feature mean/std at run start (post seq input transforms + train split standardization) "
                    f"(std_clip={float(store.seq_feature_standardization_clip):g} "
                    f"features={len(seq_feature_stats_standardized)}):"
                )
                if not seq_feature_stats_standardized:
                    print("  [warn] no aligned seq samples available for standardized stats")
                for row in seq_feature_stats_standardized:
                    print(
                        f"  seq[{int(row['feature_idx']):02d}] {str(row['feature_name'])}: "
                        f"mean={float(row['mean']):.6f} std={float(row['std']):.6f}"
                    )

        best_state = None
        best_val_loss = float("inf")
        best_val_reg_loss = 0.0
        best_epoch = -1
        bad_epochs = 0
        history: list[dict[str, float | int]] = []

        for epoch in range(1, cfg.epochs + 1):
            train_metrics = run_seq_epoch(
                model=model,
                store=store,
                split_range=train_split,
                batch_size=cfg.batch_size,
                device=device,
                amp_enabled=amp_enabled,
                train=True,
                optimizer=optimizer,
                scaler=scaler,
                seed=cfg.seed + epoch * 101 + rank * 100003,
                elastic_net_l1=cfg.elastic_net_l1,
                elastic_net_l2=cfg.elastic_net_l2,
                loss_mode=loss_mode,
                distributed_reduce=distributed,
                progress_label=(
                    f"Epoch {int(epoch)}/{int(cfg.epochs)}" if main_process else None
                ),
            )
            stop_now = False
            if main_process:
                val_metrics = run_seq_epoch(
                    model=base_model,
                    store=store,
                    split_range=splits.val,
                    batch_size=cfg.batch_size,
                    device=device,
                    amp_enabled=amp_enabled,
                    train=False,
                    optimizer=None,
                    scaler=None,
                    seed=cfg.seed + epoch * 307,
                    elastic_net_l1=0.0,
                    elastic_net_l2=0.0,
                    loss_mode=loss_mode,
                    distributed_reduce=False,
                    progress_label=(
                        f"Epoch {int(epoch)}/{int(cfg.epochs)} [val]"
                        if KERAS_VAL_PROGRESS_ENABLED
                        else None
                    ),
                )
                row: dict[str, float | int] = {
                    "epoch": int(epoch),
                    "train_loss": float(train_metrics["loss"]),
                    "train_data_loss": float(train_metrics["data_loss"]),
                    "train_reg_loss": float(train_metrics["reg_loss"]),
                    "train_mae": float(train_metrics["mae"]),
                    "train_rmse": float(train_metrics["rmse"]),
                    "train_n": int(train_metrics["n"]),
                    "val_loss": float(val_metrics["loss"]),
                    "val_mae": float(val_metrics["mae"]),
                    "val_rmse": float(val_metrics["rmse"]),
                    "val_preds_std": float(val_metrics["pred_std"]),
                    "val_n": int(val_metrics["n"]),
                }
                history.append(row)
                print(
                    f"epoch={epoch:03d} "
                    f"train_loss={row['train_loss']:.6f} "
                    f"train_mae={row['train_mae']:.6f} train_rmse={row['train_rmse']:.6f} "
                    f"val_loss={row['val_loss']:.6f} "
                    f"val_mae={row['val_mae']:.6f} val_rmse={row['val_rmse']:.6f} "
                    f"val_preds_std={row['val_preds_std']:.6f}"
                )
                print()

                if row["val_loss"] < best_val_loss:
                    best_val_loss = float(row["val_loss"])
                    best_epoch = int(epoch)
                    bad_epochs = 0
                    best_state = copy.deepcopy(base_model.state_dict())
                    torch.save(
                        {
                            "epoch": int(epoch),
                            "model_state_dict": best_state,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_val_loss": float(best_val_loss),
                            "config": asdict(cfg),
                        },
                        weights_dir / "best.pt",
                    )
                else:
                    bad_epochs += 1
                    if bad_epochs >= cfg.patience:
                        print(f"early_stop: no val improvement for {cfg.patience} epoch(s)")
                        stop_now = True

            if distributed:
                stop_now = bool(broadcast_object(bool(stop_now), src=0))
            if stop_now:
                break

        if main_process and best_state is not None:
            base_model.load_state_dict(best_state)

        if distributed and dist.is_initialized():
            dist.barrier()
        if not main_process:
            return

        val_pred = collect_seq_split_predictions(
            model=base_model,
            store=store,
            split_range=splits.val,
            batch_size=cfg.batch_size,
            device=device,
            amp_enabled=amp_enabled,
            loss_mode=loss_mode,
        )
        val_npz_path, val_csv_path = write_val_regression_preds_best(
            out_dir=preds_dir,
            best_epoch=best_epoch,
            pred=val_pred,
        )
        test_metrics = run_seq_epoch(
            model=base_model,
            store=store,
            split_range=splits.test,
            batch_size=cfg.batch_size,
            device=device,
            amp_enabled=amp_enabled,
            train=False,
            optimizer=None,
            scaler=None,
            seed=cfg.seed + 99991,
            elastic_net_l1=0.0,
            elastic_net_l2=0.0,
            loss_mode=loss_mode,
            distributed_reduce=False,
            progress_label=None,
        )

        summary = {
            "mode": "seq_regression",
            "seq_branch_enabled": int(
                bool(resolve_model_branch_mode(MODEL_BRANCH_MODE) == "seq")
            ),
            "seq_regression_loss_mode": str(loss_mode),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "best_val_mae": float(val_pred["mae"]),
            "best_val_rmse": float(val_pred["rmse"]),
            "best_val_n": int(val_pred["n"]),
            "best_val_preds_npz": str(val_npz_path.relative_to(out_dir).as_posix()),
            "best_val_preds_csv": str(val_csv_path.relative_to(out_dir).as_posix()),
            "test_loss": float(test_metrics["loss"]),
            "test_mae": float(test_metrics["mae"]),
            "test_rmse": float(test_metrics["rmse"]),
            "test_n": int(test_metrics["n"]),
        }
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        torch.save(
            {"model_state_dict": base_model.state_dict(), "config": asdict(cfg)},
            weights_dir / "last.pt",
        )
    finally:
        distributed_cleanup()


def main() -> None:
    cfg = parse_args()
    branch_mode = resolve_model_branch_mode(MODEL_BRANCH_MODE)
    if branch_mode == "seq":
        run_seq_branch(cfg)
        return
    dual_mode_enabled = bool(branch_mode == "dual")
    out_dir = Path(cfg.output_dir)
    weights_dir = out_dir / "weights"
    preds_dir = out_dir / "preds"
    eval_dir = out_dir / "eval"
    distributed = False
    rank = 0
    world_size = 1
    local_rank = 0
    device = resolve_device(cfg.device)
    main_process = True

    try:
        distributed, rank, world_size, local_rank, device = init_distributed_if_needed(
            cfg, device
        )
        main_process = is_main_process(rank)
        cuda_available = bool(torch.cuda.is_available())
        cuda_count = int(torch.cuda.device_count()) if cuda_available else 0

        if cfg.torch_compile and distributed:
            if main_process:
                print("[warn] --compile is disabled when --distributed is enabled.")
            cfg.torch_compile = False

        if distributed and dist.is_initialized():
            out_dir = Path(str(broadcast_object(str(out_dir), src=0)))
            weights_dir = out_dir / "weights"
            preds_dir = out_dir / "preds"
            eval_dir = out_dir / "eval"

        if main_process:
            out_dir.mkdir(parents=True, exist_ok=True)
            weights_dir.mkdir(parents=True, exist_ok=True)
            preds_dir.mkdir(parents=True, exist_ok=True)
            eval_dir.mkdir(parents=True, exist_ok=True)
        if distributed and dist.is_initialized():
            dist.barrier()

        set_seed(cfg.seed + int(rank))
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
        amp_enabled = bool(cfg.amp_enabled and device.type == "cuda")

        vix_enabled = bool(resolve_vix_fusion_mode(cfg.vix_fusion_mode) != "none")
        effective_vix_daily_csv = cfg.vix_daily_csv if vix_enabled else ""
        store = ShardedImageStore(
            Path(cfg.data_dir),
            ret_atr_threshold=cfg.ret_atr_threshold,
            vix_daily_csv=effective_vix_daily_csv,
            vix_date_col=cfg.vix_date_col,
            vix_value_col=cfg.vix_value_col,
        )
        vix_image_enabled = bool(store.include_vix_image)
        splits = build_split_ranges(store, cfg)
        train_split = (
            split_range_for_rank(splits.train, rank, world_size)
            if distributed
            else splits.train
        )
        seq_store: SequenceDualAlignedStore | None = None
        seq_feature_norm_meta: dict[str, object] | None = None
        seq_target_clip_meta: dict[str, object] | None = None
        if dual_mode_enabled:
            seq_store = SequenceDualAlignedStore(
                data_dir=Path(cfg.data_dir),
                ret_atr_threshold=cfg.ret_atr_threshold,
            )
            if int(seq_store.dropped_image_samples_for_seq) > 0:
                raise ValueError(
                    "dual mode requires full sample alignment between image and dual sequence datasets; "
                    f"dropped_image_samples_for_seq={int(seq_store.dropped_image_samples_for_seq)}"
                )
            if int(seq_store.subset_count) != int(store.subset_count):
                raise ValueError(
                    "dual mode requires identical sample counts between image and seq stores; "
                    f"image_subset_count={int(store.subset_count)} "
                    f"seq_subset_count={int(seq_store.subset_count)}"
                )
            seq_feature_norm_state: dict[str, object] | None = None
            seq_target_clip_state: dict[str, object] | None = None
            if not distributed or main_process:
                seq_feature_norm_meta = seq_store.fit_seq_feature_standardizer(
                    split_range=splits.train
                )
                seq_feature_norm_state = seq_store.get_seq_feature_standardizer_state()
                seq_target_clip_meta = seq_store.fit_seq_target_clipper(
                    split_range=splits.train
                )
                seq_target_clip_state = seq_store.get_seq_target_clipper_state()
            if distributed:
                seq_feature_norm_meta = broadcast_object(seq_feature_norm_meta, src=0)
                seq_feature_norm_state = broadcast_object(seq_feature_norm_state, src=0)
                seq_target_clip_meta = broadcast_object(seq_target_clip_meta, src=0)
                seq_target_clip_state = broadcast_object(seq_target_clip_state, src=0)
                seq_store.load_seq_feature_standardizer_state(seq_feature_norm_state)
                seq_store.load_seq_target_clipper_state(seq_target_clip_state)
        vix_norm_meta: dict[str, float | int | str] | None = None
        if vix_enabled:
            local_vix_stats: VixNormStats | None = None
            if not distributed or main_process:
                vix_norm_meta = store.fit_vix_normalizer(
                    split_range=splits.train,
                    method=cfg.vix_norm_method,
                    clip=cfg.vix_norm_clip,
                    log1p=cfg.vix_log1p,
                )
                local_vix_stats = store.vix_norm_stats
            if distributed:
                vix_norm_meta = broadcast_object(vix_norm_meta, src=0)
                local_vix_stats = broadcast_object(local_vix_stats, src=0)
            store.vix_norm_stats = local_vix_stats
            if main_process and vix_norm_meta is not None:
                print(
                    "vix_fusion=on "
                    f"mode={cfg.vix_fusion_mode} "
                    f"coverage={float(vix_norm_meta['vix_coverage_ratio']):.4f} "
                    f"norm={vix_norm_meta['norm_method']} "
                    f"center={float(vix_norm_meta['norm_center']):.6f} "
                    f"scale={float(vix_norm_meta['norm_scale']):.6f}"
                )
            if main_process:
                if store.vix_beta_scaling_enabled:
                    print(
                        "vix_beta_scaling=on "
                        f"ticker_csv_dir={store.ticker_csv_dir} "
                        f"date_col={store.ticker_date_col} "
                        f"beta_col={store.ticker_beta_col} "
                        f"clip=[{store.beta_clip_low:.6f},{store.beta_clip_high:.6f}] "
                        f"clip_symbol_count={int(store.beta_clip_symbol_count)}"
                    )
                else:
                    reason = (
                        f"disabled by VIX_BETA_SCALING_ENABLED={int(bool(VIX_BETA_SCALING_ENABLED))}"
                        if not store.vix_beta_scaling_requested
                        else "missing tickers.npy mapping or ticker CSV directory"
                    )
                    print(
                        "[warn] vix_beta_scaling=off "
                        f"({reason})."
                    )
        elif cfg.vix_daily_csv and main_process:
            print(
                "[warn] --vix-daily-csv provided but --vix-fusion-mode=none; "
                "daily VIX values will not be used."
            )
        if vix_image_enabled and main_process:
            print(
                "vix_image=on "
                f"shape=({int(store.vix_image_height)},{int(store.vix_image_width)}) "
                f"bars={int(store.vix_image_bars)}"
            )
        if main_process and cfg.shuffle_train_labels:
            print("[warn] placebo mode: --shuffle-train-labels enabled (train split only)")
        if main_process and cfg.zero_image:
            print("[warn] attribution mode: --zero-image enabled (X_img zeroed on all splits)")

        train_class0_count, train_class1_count = store.compute_class_counts(splits.train)
        class_weights = None
        if cfg.class_weighted_ce_enabled:
            class_weights = build_balanced_class_weights_binary(
                class0_count=train_class0_count,
                class1_count=train_class1_count,
            )
            if class_weights is None:
                if main_process:
                    print(
                        "[warn] class-weighted CE requested but train split has an empty class; using unweighted CE."
                    )
            else:
                class_weights = class_weights.to(device=device)
        if main_process:
            if class_weights is not None:
                print(
                    f"class_weighted_ce=on train_class0={train_class0_count} "
                    f"train_class1={train_class1_count} "
                    f"weights=[{float(class_weights[0].item()):.6f}, {float(class_weights[1].item()):.6f}]"
                )
            else:
                print(
                    f"class_weighted_ce=off train_class0={train_class0_count} "
                    f"train_class1={train_class1_count}"
                )
        class_weights_log = (
            [float(v) for v in class_weights.detach().cpu().tolist()]
            if class_weights is not None
            else None
        )

        train_samples_count = int(max(0, int(splits.train[1]) - int(splits.train[0])))
        val_samples_count = int(max(0, int(splits.val[1]) - int(splits.val[0])))
        epoch_time_alone_est = estimate_epoch_time_alone(
            train_samples=train_samples_count,
            val_samples=val_samples_count,
            enabled=bool(WRITE_ESTIMATED_EPOCH_TIME_ALONE_DEFAULT),
        )

        if dual_mode_enabled:
            if seq_store is None:
                raise RuntimeError("dual mode requires sequence store but it is unavailable")
            model = DualBranchImageClassifier(
                scales=store.scales,
                input_height=store.height,
                input_width=store.width,
                input_width_per_scale=store.width_per_scale,
                readout_dim=cfg.readout_dim,
                head_dims=cfg.head_dims,
                fc_dropout=cfg.fc_dropout,
                seq_input_features=int(seq_store.seq_feature_count),
                seq_input_lookback=int(seq_store.seq_lookback),
                vix_fusion_mode=cfg.vix_fusion_mode,
                vix_embed_dim=cfg.vix_embed_dim,
                include_vix_image=bool(vix_image_enabled),
                vix_image_height=int(store.vix_image_height),
                vix_image_width=int(store.vix_image_width),
            ).to(device)
        else:
            model = DecompImageClassifier(
                scales=store.scales,
                input_height=store.height,
                input_width=store.width,
                input_width_per_scale=store.width_per_scale,
                readout_dim=cfg.readout_dim,
                head_dims=cfg.head_dims,
                fc_dropout=cfg.fc_dropout,
                vix_fusion_mode=cfg.vix_fusion_mode,
                vix_embed_dim=cfg.vix_embed_dim,
                include_vix_image=bool(vix_image_enabled),
                vix_image_height=int(store.vix_image_height),
                vix_image_width=int(store.vix_image_width),
            ).to(device)
        if device.type == "cuda" and not distributed:
            model = model.to(memory_format=torch.channels_last)
        if cfg.torch_compile and hasattr(torch, "compile"):
            model = torch.compile(model)
        if distributed:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
            )
        base_model = unwrap_model(model)
        param_counts = count_model_parameters(base_model)

        if cfg.elastic_net_l1 < 0.0 or cfg.elastic_net_l2 < 0.0:
            raise ValueError("elastic-net coefficients must be >= 0")
        if cfg.elastic_net_l2 > 0.0 and cfg.weight_decay != 0.0:
            raise ValueError(
                "Set --weight-decay 0 when --elastic-net-l2 > 0 to avoid double-counting L2."
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        if main_process:
            run_meta = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": asdict(cfg),
                "runtime": {
                    "distributed": int(distributed),
                    "rank": int(rank),
                    "world_size": int(world_size),
                    "local_rank": int(local_rank),
                    "resolved_device": str(device),
                    "cuda_device_count": int(cuda_count),
                    "keras_train_progress_enabled": int(
                        bool(KERAS_INTRA_EPOCH_LOGGING_ENABLED)
                    ),
                    "keras_val_progress_enabled": int(bool(KERAS_VAL_PROGRESS_ENABLED)),
                    "epoch_stage_timing_logging_enabled": int(
                        bool(EPOCH_STAGE_TIMING_LOGGING_ENABLED)
                    ),
                    "preds_export_amp_enabled": int(bool(PREDS_EXPORT_AMP_ENABLED)),
                },
                "data": {
                    "data_dir": str(cfg.data_dir),
                    "subset_count": int(store.subset_count),
                    "scales": int(store.scales),
                    "height": int(store.height),
                    "width": int(store.width),
                    "width_per_scale": [int(w) for w in store.width_per_scale],
                    "jagged_image_widths_enabled": int(bool(JAGGED_IMAGE_WIDTHS_ENABLED)),
                    "ret_atr_index": int(store.ret_atr_idx),
                    "ret_pct_index": int(store.ret_pct_idx),
                    "label_column_for_classification": "ret_pct",
                    "label_threshold_for_classification": float(cfg.ret_atr_threshold),
                    "vix_enabled": int(bool(vix_enabled)),
                    "vix_image_enabled": int(bool(vix_image_enabled)),
                    "vix_image_height": int(store.vix_image_height),
                    "vix_image_width": int(store.vix_image_width),
                    "vix_image_bars": int(store.vix_image_bars),
                    "vix_daily_csv": str(cfg.vix_daily_csv),
                    "vix_date_col": str(cfg.vix_date_col),
                    "vix_value_col": str(cfg.vix_value_col),
                    "vix_beta_scaling_enabled": int(bool(store.vix_beta_scaling_enabled)),
                    "vix_beta_ticker_csv_dir": str(store.ticker_csv_dir),
                    "vix_beta_ticker_date_col": str(store.ticker_date_col),
                    "vix_beta_ticker_beta_col": str(store.ticker_beta_col),
                    "vix_beta_clip_low": float(store.beta_clip_low),
                    "vix_beta_clip_high": float(store.beta_clip_high),
                    "vix_beta_clip_symbol_count": int(store.beta_clip_symbol_count),
                    "ticker_lookup_count": int(
                        len(store.ticker_lookup) if store.ticker_lookup is not None else 0
                    ),
                    "vix_norm_train_split": vix_norm_meta,
                    "train_class0_count": int(train_class0_count),
                    "train_class1_count": int(train_class1_count),
                    "train_class_weights_for_50_50_ce": class_weights_log,
                    "train_mixed_shard_batching_enabled_constant": int(
                        bool(TRAIN_MIXED_SHARD_BATCHING_ENABLED)
                    ),
                    "train_mixed_shard_active_shards_constant": int(
                        TRAIN_MIXED_SHARD_ACTIVE_SHARDS
                    ),
                    "dual_seq_feature_count": int(seq_store.seq_feature_count)
                    if seq_store is not None
                    else 0,
                    "dual_seq_lookback": int(seq_store.seq_lookback)
                    if seq_store is not None
                    else 0,
                    "dual_sample_count": int(seq_store.dual_sample_count)
                    if seq_store is not None
                    else 0,
                    "dropped_image_samples_for_seq": int(
                        seq_store.dropped_image_samples_for_seq
                    )
                    if seq_store is not None
                    else 0,
                    "seq_feature_standardization_train_split": seq_feature_norm_meta,
                    "seq_target_clip_train_split": seq_target_clip_meta,
                },
                "splits": {
                    "train": list(splits.train),
                    "val": list(splits.val),
                    "test": list(splits.test),
                },
                "estimates": {
                    "per_epoch_time_alone": epoch_time_alone_est,
                },
                "evaluation": {
                    "post_run_evals_enabled_constant": int(
                        bool(POST_RUN_EVALS_ENABLED)
                    ),
                    "daily_cross_sectional": {
                        "top_pct": tail_pct_config_value(
                            DAILY_CROSS_SECTIONAL_TOP_PCT,
                            "daily_cross_sectional.top_pct",
                        ),
                        "bottom_pct": tail_pct_config_value(
                            DAILY_CROSS_SECTIONAL_BOTTOM_PCT,
                            "daily_cross_sectional.bottom_pct",
                        ),
                        "min_per_side": int(DAILY_CROSS_SECTIONAL_MIN_PER_SIDE),
                        "min_names_per_day": int(DAILY_CROSS_SECTIONAL_MIN_NAMES_PER_DAY),
                        "annualization_days": float(DAILY_CROSS_SECTIONAL_ANNUALIZATION_DAYS),
                        "outlier_zscore_threshold": float(EVAL_TAIL_RET_OUTLIER_ZSCORE_THRESHOLD),
                    },
                    "walkforward_rolling_threshold": {
                        "enabled": int(bool(WALKFORWARD_ROLLING_THRESHOLD_ENABLED)),
                        "top_pct": tail_pct_config_value(
                            WALKFORWARD_ROLLING_TOP_PCT,
                            "walkforward_rolling_threshold.top_pct",
                        ),
                        "bottom_pct": tail_pct_config_value(
                            WALKFORWARD_ROLLING_BOTTOM_PCT,
                            "walkforward_rolling_threshold.bottom_pct",
                        ),
                        "lookback_days": int(WALKFORWARD_ROLLING_LOOKBACK_DAYS),
                        "min_history_days": int(WALKFORWARD_ROLLING_MIN_HISTORY_DAYS),
                        "min_per_side": int(WALKFORWARD_ROLLING_MIN_PER_SIDE),
                        "min_per_side_mode": str(cfg.walkforward_min_per_side_mode),
                        "min_names_per_day": int(WALKFORWARD_ROLLING_MIN_NAMES_PER_DAY),
                        "annualization_days": float(WALKFORWARD_ROLLING_ANNUALIZATION_DAYS),
                        "threshold_method": str(WALKFORWARD_ROLLING_THRESHOLD_METHOD),
                        "fallback_to_daily_rank": int(
                            bool(WALKFORWARD_ROLLING_FALLBACK_TO_DAILY_RANK)
                        ),
                        "enforce_non_overlap": int(
                            bool(WALKFORWARD_ROLLING_ENFORCE_NON_OVERLAP)
                        ),
                        "spy_non_overlap_enabled": int(
                            bool(WALKFORWARD_SPY_NON_OVERLAP_ENABLED)
                        ),
                        "spy_non_overlap_horizon_days": int(
                            WALKFORWARD_SPY_NON_OVERLAP_HORIZON_DAYS
                        ),
                        "spy_daily_csv": str(WALKFORWARD_SPY_DAILY_CSV_PATH),
                        "spy_date_col": str(WALKFORWARD_SPY_DATE_COL),
                        "spy_close_col": str(WALKFORWARD_SPY_CLOSE_COL),
                        "outlier_zscore_threshold": float(
                            EVAL_TAIL_RET_OUTLIER_ZSCORE_THRESHOLD
                        ),
                    },
                },
                "model": {
                    "conv_channels": CONV_CHANNELS,
                    "pool_dim": POOL_DIM,
                    "pool_kernel": POOL_KERNEL,
                    "pool_stride": POOL_STRIDE,
                    "weight_init": WEIGHT_INIT,
                    "xavier_gain": float(XAVIER_GAIN),
                    "elastic_net_l1": float(cfg.elastic_net_l1),
                    "elastic_net_l2": float(cfg.elastic_net_l2),
                    "fc_dropout": float(cfg.fc_dropout),
                    "readout_dim": cfg.readout_dim,
                    "head_dims": cfg.head_dims,
                    "vix_fusion_mode": str(base_model.vix_fusion_mode),
                    "vix_embed_dim": int(base_model.vix_embed_dim),
                    "include_vix_image": int(bool(base_model.include_vix_image)),
                    "vix_image_height": int(base_model.vix_image_height),
                    "vix_image_width": int(base_model.vix_image_width),
                    "vix_image_conv_channels": [int(v) for v in VIX_IMAGE_CONV_CHANNELS],
                    "vix_image_pool_kernels": [int(v) for v in VIX_IMAGE_POOL_KERNELS],
                    "input_width_per_scale": [int(w) for w in base_model.input_width_per_scale],
                    "fusion_mode": str(base_model.fusion_mode),
                    "jagged_2d_concat_option": int(JAGGED_2D_CONCAT_OPTION),
                    "jagged_option3_target_width": int(JAGGED_OPTION3_TARGET_WIDTH),
                    "jagged_option4_add_valid_width_mask": int(
                        bool(base_model.jagged_option4_add_valid_width_mask)
                    ),
                    "rep_shape": list(base_model.rep_shape),
                    "rep_aligned_width": int(base_model.rep_aligned_width),
                    "branch_mode": str(branch_mode),
                    "dual_regression_mse_weight": float(
                        DUAL_REGRESSION_MSE_WEIGHT if dual_mode_enabled else 0.0
                    ),
                    "dual_reg_fusion_mode": str(
                        getattr(base_model, "reg_fusion_mode", "")
                    ),
                    "dual_msf_map_shape": [
                        int(v) for v in getattr(base_model, "msf_map_shape", ())
                    ],
                    "dual_reg_fused_map_shape": [
                        int(v) for v in getattr(base_model, "reg_fused_map_shape", ())
                    ],
                    "dual_seq_input_features": int(
                        getattr(base_model, "seq_input_features", 0)
                    ),
                    "dual_seq_input_lookback": int(
                        getattr(base_model, "seq_input_lookback", 0)
                    ),
                    "dual_ts_crop_keep_side": str(
                        getattr(base_model, "ts_crop_keep_side", "")
                    ),
                    "param_count_total": int(param_counts["total"]),
                    "param_count_trainable": int(param_counts["trainable"]),
                    "param_count_frozen": int(param_counts["frozen"]),
                    "msf_weights": [
                        compute_msf_weight(i, store.scales) for i in range(store.scales)
                    ],
                    "msf_out_channels": [
                        compute_msf_out_channels(
                            compute_msf_weight(i, store.scales),
                            base_channels=CONV_CHANNELS[-1],
                        )
                        for i in range(store.scales)
                    ],
                    "msf_active_scale_indices": [
                        int(v) for v in base_model.active_scale_indices
                    ],
                    "msf_active_scale_weights": [
                        float(v) for v in base_model.active_scale_weights
                    ],
                    "msf_disabled_scale_indices": [
                        int(v) for v in base_model.disabled_scale_indices
                    ],
                    "multi_gpu_enabled": int(bool(cfg.multi_gpu)),
                    "multi_gpu_device_ids": list(range(int(cuda_count))),
                },
            }
            (out_dir / "config.json").write_text(
                json.dumps(run_meta, indent=2) + "\n", encoding="utf-8"
            )

        best_state = None
        best_val_loss = float("inf")
        best_epoch = -1
        bad_epochs = 0
        history: list[dict] = []

        if main_process:
            print(
                f"device={device} distributed={int(distributed)} world_size={world_size} "
                f"amp_enabled={amp_enabled} samples={store.subset_count} "
                f"train={splits.train} val={splits.val} test={splits.test} "
                f"branch_mode={branch_mode}"
            )
            if dual_mode_enabled and seq_store is not None:
                print(
                    f"dual_mode=on seq_features={int(seq_store.seq_feature_count)} "
                    f"seq_lookback={int(seq_store.seq_lookback)} "
                    f"mse_weight={float(DUAL_REGRESSION_MSE_WEIGHT):.4f}"
                )
                if seq_feature_norm_meta is not None:
                    print(
                        "seq_feature_standardization="
                        f"{'on' if int(seq_feature_norm_meta.get('enabled', 0)) else 'off'} "
                        f"fitted={int(seq_feature_norm_meta.get('fitted', 0))} "
                        f"clip={float(seq_store.seq_feature_standardization_clip):g} "
                        f"eps={float(seq_store.seq_feature_standardization_eps):g}"
                    )
                if seq_target_clip_meta is not None:
                    print(
                        "seq_target_clipping="
                        f"{'on' if int(seq_target_clip_meta.get('enabled', 0)) else 'off'} "
                        f"fitted={int(seq_target_clip_meta.get('fitted', 0))} "
                        f"pct=[{float(seq_store.seq_target_clip_lower_pct):g},"
                        f"{float(seq_store.seq_target_clip_upper_pct):g}] "
                        f"bounds=[{float(seq_store.seq_target_clip_low):.6f},"
                        f"{float(seq_store.seq_target_clip_high):.6f}]"
                    )

        for epoch in range(1, cfg.epochs + 1):
            t_train_epoch_start = time.perf_counter()
            if dual_mode_enabled:
                if seq_store is None:
                    raise RuntimeError("dual mode requires sequence store but it is unavailable")
                train_metrics = run_dual_epoch(
                    model=model,
                    store=store,
                    seq_store=seq_store,
                    split_range=train_split,
                    batch_size=cfg.batch_size,
                    device=device,
                    amp_enabled=amp_enabled,
                    train=True,
                    optimizer=optimizer,
                    scaler=scaler,
                    seed=cfg.seed + epoch * 101 + rank * 100003,
                    elastic_net_l1=cfg.elastic_net_l1,
                    elastic_net_l2=cfg.elastic_net_l2,
                    dual_reg_mse_weight=float(DUAL_REGRESSION_MSE_WEIGHT),
                    class_weights=class_weights,
                    distributed_reduce=distributed,
                    progress_label=(
                        f"Epoch {int(epoch)}/{int(cfg.epochs)}" if main_process else None
                    ),
                    vix_enabled=bool(vix_enabled),
                    vix_image_enabled=bool(vix_image_enabled),
                    shuffle_train_labels=bool(cfg.shuffle_train_labels),
                    zero_image=bool(cfg.zero_image),
                )
            else:
                train_metrics = run_epoch(
                    model=model,
                    store=store,
                    split_range=train_split,
                    batch_size=cfg.batch_size,
                    device=device,
                    amp_enabled=amp_enabled,
                    train=True,
                    optimizer=optimizer,
                    scaler=scaler,
                    seed=cfg.seed + epoch * 101 + rank * 100003,
                    elastic_net_l1=cfg.elastic_net_l1,
                    elastic_net_l2=cfg.elastic_net_l2,
                    class_weights=class_weights,
                    distributed_reduce=distributed,
                    progress_label=(
                        f"Epoch {int(epoch)}/{int(cfg.epochs)}" if main_process else None
                    ),
                    vix_enabled=bool(vix_enabled),
                    vix_image_enabled=bool(vix_image_enabled),
                    shuffle_train_labels=bool(cfg.shuffle_train_labels),
                    zero_image=bool(cfg.zero_image),
                )
            train_epoch_sec = float(max(0.0, time.perf_counter() - t_train_epoch_start))
            stop_now = False
            if main_process:
                t_val_epoch_start = time.perf_counter()
                if dual_mode_enabled:
                    if seq_store is None:
                        raise RuntimeError("dual mode requires sequence store but it is unavailable")
                    val_metrics = run_dual_epoch(
                        model=base_model,
                        store=store,
                        seq_store=seq_store,
                        split_range=splits.val,
                        batch_size=cfg.batch_size,
                        device=device,
                        amp_enabled=amp_enabled,
                        train=False,
                        optimizer=None,
                        scaler=None,
                        seed=cfg.seed + epoch * 307,
                        elastic_net_l1=0.0,
                        elastic_net_l2=0.0,
                        dual_reg_mse_weight=float(DUAL_REGRESSION_MSE_WEIGHT),
                        class_weights=class_weights,
                        distributed_reduce=False,
                        progress_label=(
                            f"Epoch {int(epoch)}/{int(cfg.epochs)} [val]"
                            if KERAS_VAL_PROGRESS_ENABLED
                            else None
                        ),
                        vix_enabled=bool(vix_enabled),
                        vix_image_enabled=bool(vix_image_enabled),
                        shuffle_train_labels=False,
                        zero_image=bool(cfg.zero_image),
                    )
                else:
                    val_metrics = run_epoch(
                        model=base_model,
                        store=store,
                        split_range=splits.val,
                        batch_size=cfg.batch_size,
                        device=device,
                        amp_enabled=amp_enabled,
                        train=False,
                        optimizer=None,
                        scaler=None,
                        seed=cfg.seed + epoch * 307,
                        elastic_net_l1=0.0,
                        elastic_net_l2=0.0,
                        class_weights=class_weights,
                        distributed_reduce=False,
                        progress_label=(
                            f"Epoch {int(epoch)}/{int(cfg.epochs)} [val]"
                            if KERAS_VAL_PROGRESS_ENABLED
                            else None
                        ),
                        vix_enabled=bool(vix_enabled),
                        vix_image_enabled=bool(vix_image_enabled),
                        shuffle_train_labels=False,
                        zero_image=bool(cfg.zero_image),
                    )
                val_epoch_sec = float(max(0.0, time.perf_counter() - t_val_epoch_start))
                val_cls_loss = float(val_metrics["ce_loss"])
                val_reg_loss = (
                    float(val_metrics["mse_loss"])
                    if dual_mode_enabled and ("mse_loss" in val_metrics)
                    else 0.0
                )
                row = {
                    "epoch": int(epoch),
                    "train_loss": float(train_metrics["loss"]),
                    "train_ce_loss": float(train_metrics["ce_loss"]),
                    "train_reg_loss": float(train_metrics["reg_loss"]),
                    "train_mse_loss": (
                        float(train_metrics["mse_loss"])
                        if dual_mode_enabled and ("mse_loss" in train_metrics)
                        else 0.0
                    ),
                    "train_acc": float(train_metrics["acc"]),
                    "train_prob_std": float(train_metrics["prob_std"]),
                    "train_ppv": float(train_metrics["ppv"]),
                    "train_npv": float(train_metrics["npv"]),
                    "train_n": int(train_metrics["n"]),
                    "val_loss": float(val_cls_loss),
                    "val_cls_loss": float(val_cls_loss),
                    "val_reg_loss": float(val_reg_loss),
                    "val_joint_loss": float(val_metrics["loss"]),
                    "val_acc": float(val_metrics["acc"]),
                    "val_prob_std": float(val_metrics["prob_std"]),
                    "val_ppv": float(val_metrics["ppv"]),
                    "val_npv": float(val_metrics["npv"]),
                    "val_n": int(val_metrics["n"]),
                }
                history.append(row)
                if EPOCH_STAGE_TIMING_LOGGING_ENABLED:
                    print(
                        f"epoch_timing={epoch:03d} "
                        f"train_sec={train_epoch_sec:.2f} val_sec={val_epoch_sec:.2f}"
                    )
                if dual_mode_enabled:
                    print(
                        f"epoch={epoch:03d} "
                        f"train_cls_loss={row['train_ce_loss']:.6f} "
                        f"train_reg={row['train_mse_loss']:.6f} train_acc={row['train_acc']:.4f} "
                        f"train_prob_std={row['train_prob_std']:.6f} "
                        f"val_cls_loss={row['val_cls_loss']:.6f} "
                        f"val_reg_loss={row['val_reg_loss']:.6f} "
                        f"split_diff={row['train_ce_loss'] - val_cls_loss:.6f} "
                        f"val_acc={row['val_acc']:.4f} val_prob_std={row['val_prob_std']:.6f} "
                        f"val_ppv={row['val_ppv']:.4f} val_npv={row['val_npv']:.4f}"
                    )
                else:
                    print(
                        f"epoch={epoch:03d} "
                        f"train_ce_loss={row['train_ce_loss']:.6f} "
                        f"train_reg={row['train_reg_loss']:.6f} train_acc={row['train_acc']:.4f} "
                        f"train_prob_std={row['train_prob_std']:.6f} "
                        f"val_loss={row['val_loss']:.6f} split_diff={row['train_ce_loss'] - val_cls_loss:.6f} "
                        f"val_acc={row['val_acc']:.4f} val_prob_std={row['val_prob_std']:.6f} "
                        f"val_ppv={row['val_ppv']:.4f} val_npv={row['val_npv']:.4f}"
                    )
                print()

                if row["val_cls_loss"] < best_val_loss:
                    best_val_loss = float(row["val_cls_loss"])
                    best_val_reg_loss = float(row["val_reg_loss"])
                    best_epoch = int(epoch)
                    bad_epochs = 0
                    best_state = copy.deepcopy(base_model.state_dict())
                    torch.save(
                        {
                            "epoch": int(epoch),
                            "model_state_dict": best_state,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_val_loss": float(best_val_loss),
                            "best_val_cls_loss": float(row["val_cls_loss"]),
                            "best_val_reg_loss": float(row["val_reg_loss"]),
                            "config": asdict(cfg),
                        },
                        weights_dir / "best.pt",
                    )
                else:
                    bad_epochs += 1
                    if bad_epochs >= cfg.patience:
                        print(f"early_stop: no val improvement for {cfg.patience} epoch(s)")
                        stop_now = True

            if distributed:
                stop_now = bool(broadcast_object(bool(stop_now), src=0))
            if stop_now:
                break

        if main_process and best_state is not None:
            base_model.load_state_dict(best_state)

        if distributed and dist.is_initialized():
            dist.barrier()
        if not main_process:
            return

        val_pred = collect_split_predictions(
            model=base_model,
            store=store,
            split_range=splits.val,
            batch_size=cfg.batch_size,
            device=device,
            amp_enabled=amp_enabled,
            class_weights=class_weights,
            vix_enabled=bool(vix_enabled),
            vix_image_enabled=bool(vix_image_enabled),
            zero_image=bool(cfg.zero_image),
        )
        val_npz_path, val_csv_path = write_val_preds_best(
            out_dir=preds_dir,
            best_epoch=best_epoch,
            pred=val_pred,
        )
        val_acc_thr = compute_val_acc_by_threshold(val_pred)
        decile_rows = compute_decile_metrics(val_pred)
        decile_csv_path, decile_json_path = write_decile_metrics_best(
            out_dir=out_dir,
            best_epoch=best_epoch,
            rows=decile_rows,
        )
        decile_plot_path = write_decile_plot_best(
            out_dir=out_dir,
            best_epoch=best_epoch,
            rows=decile_rows,
        )
        decile_class1_accuracy_plot_path = write_decile_class1_accuracy_plot_best(
            out_dir=out_dir,
            best_epoch=best_epoch,
            pred=val_pred,
        )
        quantile_plot_path = write_quantile_plot_best(
            out_dir=out_dir,
            best_epoch=best_epoch,
            prob=np.asarray(val_pred["prob"], dtype=np.float32),
            ret_atr_true=np.asarray(val_pred["ret_atr_true"], dtype=np.float32),
        )
        accuracy_calibration_plot_path = write_accuracy_calibration_plot_best(
            out_dir=out_dir,
            best_epoch=best_epoch,
            prob=np.asarray(val_pred["prob"], dtype=np.float32),
            y_true=np.asarray(val_pred["y_true"], dtype=np.int64),
        )
        eval_summary: dict[str, object]
        if bool(POST_RUN_EVALS_ENABLED):
            sim_eval_results = run_sim_evals_script(run_dir=out_dir, best_epoch=best_epoch)
            daily_primary = dict(sim_eval_results["daily_primary"])  # type: ignore[index]
            wf_primary = dict(sim_eval_results["wf_primary"])  # type: ignore[index]
            daily_cs_payload = dict(daily_primary["payload"])  # type: ignore[index]
            wf_payload = dict(wf_primary["payload"])  # type: ignore[index]
            # sim_evals schema changed from top_compounded_returns -> spread_compounded_returns.
            daily_cs_compounded_raw = daily_cs_payload.get("spread_compounded_returns")
            if daily_cs_compounded_raw is None:
                daily_cs_compounded_raw = daily_cs_payload.get("top_compounded_returns")
            try:
                daily_cs_compounded_returns = float(daily_cs_compounded_raw)
            except (TypeError, ValueError):
                daily_cs_compounded_returns = float("nan")
            daily_cs_csv_path = out_dir / str(daily_primary["csv"])
            daily_cs_json_path = out_dir / str(daily_primary["json"])
            wf_csv_path = out_dir / str(wf_primary["csv"])
            wf_json_path = out_dir / str(wf_primary["json"])
            daily_outputs_summary = list(sim_eval_results["daily_outputs_summary"])  # type: ignore[index]
            wf_outputs_summary = list(sim_eval_results["wf_outputs_summary"])  # type: ignore[index]
            eval_summary = {
                "best_val_daily_cross_sectional_metrics_csv": str(
                    daily_cs_csv_path.relative_to(out_dir).as_posix()
                ),
                "best_val_daily_cross_sectional_metrics_json": str(
                    daily_cs_json_path.relative_to(out_dir).as_posix()
                ),
                "best_val_daily_cross_sectional_spread_mean": float(
                    daily_cs_payload["spread_mean"]
                ),
                "best_val_daily_cross_sectional_spread_median": float(
                    daily_cs_payload["spread_median"]
                ),
                "best_val_daily_cross_sectional_spread_sharpe": float(
                    daily_cs_payload["spread_sharpe_annualized"]
                ),
                "best_val_daily_cross_sectional_days_used": int(daily_cs_payload["days_used"]),
                "best_val_daily_cross_sectional_spread_compounded_returns": float(
                    daily_cs_compounded_returns
                ),
                "best_val_daily_cross_sectional_top_compounded_returns": float(
                    daily_cs_compounded_returns
                ),
                "best_val_daily_cross_sectional_outputs": daily_outputs_summary,
                "best_val_walkforward_rolling_threshold_metrics_csv": str(
                    wf_csv_path.relative_to(out_dir).as_posix()
                ),
                "best_val_walkforward_rolling_threshold_metrics_json": str(
                    wf_json_path.relative_to(out_dir).as_posix()
                ),
                "best_val_walkforward_rolling_threshold_spread_mean": float(
                    wf_payload["spread_mean"]
                ),
                "best_val_walkforward_rolling_threshold_spread_median": float(
                    wf_payload["spread_median"]
                ),
                "best_val_walkforward_rolling_threshold_spread_sharpe": float(
                    wf_payload["spread_sharpe_annualized"]
                ),
                "best_val_walkforward_rolling_threshold_days_used": int(wf_payload["days_used"]),
                "best_val_walkforward_rolling_threshold_method": str(
                    wf_payload["threshold_method"]
                ),
                "best_val_walkforward_rolling_threshold_days_used_rank_fallback": int(
                    wf_payload["days_used_rank_fallback"]
                ),
                "best_val_walkforward_rolling_threshold_spy_non_overlap_trades": int(
                    wf_payload["spy_non_overlap_trades"]
                ),
                "best_val_walkforward_rolling_threshold_spy_non_overlap_trade_mean": float(
                    wf_payload["spy_non_overlap_trade_mean"]
                ),
                "best_val_walkforward_rolling_threshold_spy_non_overlap_trade_median": float(
                    wf_payload["spy_non_overlap_trade_median"]
                ),
                "best_val_walkforward_rolling_threshold_spy_non_overlap_compounded_return": float(
                    wf_payload["spy_non_overlap_compounded_return"]
                ),
                "best_val_walkforward_rolling_threshold_real_num_trades": int(
                    wf_payload["real_num_trades"]
                ),
                "best_val_walkforward_rolling_threshold_real_per_trade_returns": float(
                    wf_payload["real_per_trade_returns"]
                ),
                "best_val_walkforward_rolling_threshold_non_adj_returns_sharpe": float(
                    wf_payload["non_adj_returns_sharpe"]
                ),
                "best_val_walkforward_rolling_threshold_real_returns_sharpe": float(
                    wf_payload["real_returns_sharpe"]
                ),
                "best_val_walkforward_rolling_threshold_real_nonoverlap_compounded_returns": float(
                    wf_payload["real_nonoverlap_compounded_returns"]
                ),
                "best_val_walkforward_rolling_threshold_outputs": wf_outputs_summary,
            }
        else:
            print("[post_run_eval] skipped (POST_RUN_EVALS_ENABLED=0)")
            eval_summary = {
                "best_val_daily_cross_sectional_metrics_csv": "",
                "best_val_daily_cross_sectional_metrics_json": "",
                "best_val_daily_cross_sectional_spread_mean": None,
                "best_val_daily_cross_sectional_spread_median": None,
                "best_val_daily_cross_sectional_spread_sharpe": None,
                "best_val_daily_cross_sectional_days_used": None,
                "best_val_daily_cross_sectional_spread_compounded_returns": None,
                "best_val_daily_cross_sectional_top_compounded_returns": None,
                "best_val_daily_cross_sectional_outputs": [],
                "best_val_walkforward_rolling_threshold_metrics_csv": "",
                "best_val_walkforward_rolling_threshold_metrics_json": "",
                "best_val_walkforward_rolling_threshold_spread_mean": None,
                "best_val_walkforward_rolling_threshold_spread_median": None,
                "best_val_walkforward_rolling_threshold_spread_sharpe": None,
                "best_val_walkforward_rolling_threshold_days_used": None,
                "best_val_walkforward_rolling_threshold_method": "",
                "best_val_walkforward_rolling_threshold_days_used_rank_fallback": None,
                "best_val_walkforward_rolling_threshold_spy_non_overlap_trades": None,
                "best_val_walkforward_rolling_threshold_spy_non_overlap_trade_mean": None,
                "best_val_walkforward_rolling_threshold_spy_non_overlap_trade_median": None,
                "best_val_walkforward_rolling_threshold_spy_non_overlap_compounded_return": None,
                "best_val_walkforward_rolling_threshold_real_num_trades": None,
                "best_val_walkforward_rolling_threshold_real_per_trade_returns": None,
                "best_val_walkforward_rolling_threshold_non_adj_returns_sharpe": None,
                "best_val_walkforward_rolling_threshold_real_returns_sharpe": None,
                "best_val_walkforward_rolling_threshold_real_nonoverlap_compounded_returns": None,
                "best_val_walkforward_rolling_threshold_outputs": [],
            }

        if dual_mode_enabled:
            if seq_store is None:
                raise RuntimeError("dual mode requires sequence store but it is unavailable")
            test_metrics = run_dual_epoch(
                model=base_model,
                store=store,
                seq_store=seq_store,
                split_range=splits.test,
                batch_size=cfg.batch_size,
                device=device,
                amp_enabled=amp_enabled,
                train=False,
                optimizer=None,
                scaler=None,
                seed=cfg.seed + 99991,
                elastic_net_l1=0.0,
                elastic_net_l2=0.0,
                dual_reg_mse_weight=float(DUAL_REGRESSION_MSE_WEIGHT),
                class_weights=class_weights,
                distributed_reduce=False,
                vix_enabled=bool(vix_enabled),
                vix_image_enabled=bool(vix_image_enabled),
                shuffle_train_labels=False,
                zero_image=bool(cfg.zero_image),
            )
        else:
            test_metrics = run_epoch(
                model=base_model,
                store=store,
                split_range=splits.test,
                batch_size=cfg.batch_size,
                device=device,
                amp_enabled=amp_enabled,
                train=False,
                optimizer=None,
                scaler=None,
                seed=cfg.seed + 99991,
                elastic_net_l1=0.0,
                elastic_net_l2=0.0,
                class_weights=class_weights,
                distributed_reduce=False,
                vix_enabled=bool(vix_enabled),
                vix_image_enabled=bool(vix_image_enabled),
                shuffle_train_labels=False,
                zero_image=bool(cfg.zero_image),
            )
        summary = {
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "best_val_cls_loss": float(best_val_loss),
            "best_val_reg_loss": float(best_val_reg_loss),
            "best_val_acc": float(val_pred["acc"]),
            "best_val_n": int(val_pred["n"]),
            "shuffle_train_labels": int(bool(cfg.shuffle_train_labels)),
            "zero_image": int(bool(cfg.zero_image)),
            "branch_mode": str(branch_mode),
            "dual_mode_enabled": int(bool(dual_mode_enabled)),
            "dual_regression_mse_weight": float(
                DUAL_REGRESSION_MSE_WEIGHT if dual_mode_enabled else 0.0
            ),
            "val_best_acc": float(val_acc_thr["best_acc"]),
            "val_best_thresh": float(val_acc_thr["best_thresh"]),
            "best_val_preds_npz": str(val_npz_path.relative_to(out_dir).as_posix()),
            "best_val_preds_csv": str(val_csv_path.relative_to(out_dir).as_posix()),
            "best_val_deciles_csv": str(decile_csv_path.name),
            "best_val_deciles_json": str(decile_json_path.name),
            "best_val_deciles_plot": (
                str(decile_plot_path.name) if decile_plot_path is not None else ""
            ),
            "best_val_decile_class1_accuracy_plot": (
                str(decile_class1_accuracy_plot_path.name)
                if decile_class1_accuracy_plot_path is not None
                else ""
            ),
            "best_val_quantiles_plot": (
                str(quantile_plot_path.name) if quantile_plot_path is not None else ""
            ),
            "best_val_accuracy_calibration_plot": (
                str(accuracy_calibration_plot_path.name)
                if accuracy_calibration_plot_path is not None
                else ""
            ),
            "post_run_evals_enabled_constant": int(bool(POST_RUN_EVALS_ENABLED)),
            "post_run_evals_ran": int(bool(POST_RUN_EVALS_ENABLED)),
            "test_loss": float(test_metrics["ce_loss"]),
            "test_cls_loss": float(test_metrics["ce_loss"]),
            "test_reg_loss": float(
                test_metrics["mse_loss"] if dual_mode_enabled and ("mse_loss" in test_metrics) else 0.0
            ),
            "test_joint_loss": float(test_metrics["loss"]),
            "test_acc": float(test_metrics["acc"]),
            "test_n": int(test_metrics["n"]),
        }
        summary.update(eval_summary)
        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        torch.save(
            {"model_state_dict": base_model.state_dict(), "config": asdict(cfg)},
            weights_dir / "last.pt",
        )
    finally:
        distributed_cleanup()


if __name__ == "__main__":
    main()
