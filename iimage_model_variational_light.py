#!/usr/bin/env python3
"""Lightweight per-scale variational image classifier.

This is a compact alternative to iimage_model.py:
- one variational encoder per decomposition scale
- same-style per-scale weighting behavior (weight=0 disables a scale)
- no decoder / reconstruction branch
- classification objective: CE + beta * weighted_KL

Dataset format is the existing image bundle format used by /ephemeral/images/mixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from iimage_model import ShardedImageStore, VixNormStats

# ----------------------------
# Constants
# ----------------------------
DATA_DIR = "/ephemeral/images/mixed_idk"

MULTI_GPU_ENABLED_DEFAULT = 1
AMP_ENABLED_DEFAULT = 1

# Same role as iimage_model: scale-indexed multipliers, scale0 first.
# Any 0.0 entry disables the corresponding scale encoder.
MSF_SCALE_WEIGHTS = [0.5, 0.25, 0.25]

EPOCHS = 4
BATCH_SIZE = 256
LEARNING_RATE = 3e-5
STOCHASTIC_MODE = 1
# BETA_KL = 0
BETA_KL = 3e-3

CONV_CHANNELS = [64, 128]
BASE_OUT_CHANNELS = 256
LATENT_DIM_PER_SCALE = 32
LATENT_PRE_MLP_DIM = 64
READOUT_DIM = 16
HEAD_DIMS = [512, 128]
FC_DROPOUT = 0.25

SCALE_POOL_DIM = "none"
VIX_POOL_DIM = "both"

VIX_FEATURES_ENABLED = 0

VIX_IMAGE_ENABLED = 1
VIX_IMAGE_EMBED_DIM = 8
VIX_CONV_CHANNELS = [32,64,64]
# Per-conv pooling stride for VIX image encoder.
# One entry per conv block in VIX_CONV_CHANNELS.
# Use stride=1 to disable pooling for a block.
VIX_POOLING = [3,2,2]

KERNEL_SIZE = 5
KERNEL_WIDTH = 3
POOL_KERNEL = 2
POOL_STRIDE = 2
LRELU_SLOPE = 0.1

VIX_DAILY_CSV_DEFAULT = str((Path(__file__).resolve().parent / "daily_vix.csv"))
VIX_DATE_COL_DEFAULT = "date"
VIX_VALUE_COL_DEFAULT = "close"
VIX_NORM_METHOD_DEFAULT = "robust_zscore"
VIX_NORM_CLIP_DEFAULT = 5.0
VIX_LOG1P_DEFAULT = 0
VIX_EMBED_DIM_DEFAULT = 12
VIX_FUSION_MODE = "none"

VAL_FRACTION = 0.18
TEST_FRACTION = 0.001
SEED = 7
KERAS_INTRA_EPOCH_LOGGING_ENABLED = True
KERAS_PROGRESS_BAR_WIDTH = 30
KERAS_PROGRESS_UPDATE_EVERY_STEPS = 20
WEIGHT_DECAY=0
RET_ATR_THRESHOLD=0


@dataclass(frozen=True)
class SplitRanges:
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]


def resolve_device(token: str) -> torch.device:
    t = str(token).strip().lower()
    if t in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if t == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but CUDA is unavailable")
    return torch.device(t)


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def count_model_parameters(model: nn.Module) -> dict[str, int]:
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return {"total": total, "trainable": trainable, "frozen": int(total - trainable)}


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, (nn.DataParallel, DDP)):
        return model.module
    if hasattr(model, "_orig_mod"):
        return getattr(model, "_orig_mod")
    return model


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name!r} must be an integer; got {raw!r}") from exc


def init_distributed_if_needed(
    device: torch.device,
    multi_gpu_enabled: bool,
) -> tuple[bool, int, int, int, torch.device]:
    world_size_env = _env_int("WORLD_SIZE", 1)
    local_rank_env = _env_int("LOCAL_RANK", 0)
    use_distributed = bool(int(world_size_env) > 1)
    if bool(use_distributed) and not bool(multi_gpu_enabled):
        raise ValueError(
            "torchrun distributed launch detected (WORLD_SIZE>1) but --multi-gpu-enabled=0. "
            "Set --multi-gpu-enabled=1 or launch a single process."
        )
    if not use_distributed:
        return False, 0, 1, 0, device
    if device.type != "cuda":
        raise ValueError("DDP in this script requires CUDA devices. Set --device=cuda or --device=auto.")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable but distributed launch was requested.")

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


def resolve_next_variational_run_dir(root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    prefix = "variational_light_"
    ids: list[int] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name.strip()
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if suffix.isdigit():
            ids.append(int(suffix))
    next_id = 1 if not ids else (max(ids) + 1)
    return root / f"{prefix}{next_id}"


def resolve_vix_fusion_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode in ("none", "film", "late_concat"):
        return mode
    raise ValueError(
        "vix_fusion_mode must be one of: none, film, late_concat; "
        f"got {value!r}"
    )


def resolve_pool_dim(value: str, name: str) -> str:
    mode = str(value).strip().lower()
    if mode in ("none", "height", "width", "both"):
        return mode
    raise ValueError(
        f"{name} must be one of: none, height, width, both; "
        f"got {value!r}"
    )


def build_pool_layer(dim_mode: str, name: str) -> nn.Module:
    mode = resolve_pool_dim(dim_mode, name)
    if mode == "none":
        return nn.Identity()
    if mode == "height":
        return nn.MaxPool2d(kernel_size=(POOL_KERNEL, 1), stride=(POOL_STRIDE, 1))
    if mode == "width":
        return nn.MaxPool2d(kernel_size=(1, POOL_KERNEL), stride=(1, POOL_STRIDE))
    return nn.MaxPool2d(
        kernel_size=(POOL_KERNEL, POOL_KERNEL),
        stride=(POOL_STRIDE, POOL_STRIDE),
    )


def build_pool_layer_with_stride(dim_mode: str, name: str, stride: int) -> nn.Module:
    mode = resolve_pool_dim(dim_mode, name)
    s = int(stride)
    if s < 1:
        raise ValueError(f"{name} stride must be >= 1; got {stride!r}")
    if mode == "none" or s == 1:
        return nn.Identity()
    if mode == "height":
        return nn.MaxPool2d(kernel_size=(s, 1), stride=(s, 1))
    if mode == "width":
        return nn.MaxPool2d(kernel_size=(1, s), stride=(1, s))
    return nn.MaxPool2d(kernel_size=(s, s), stride=(s, s))


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
        # Keep behavior compatible with iimage_model fallback.
        weights = [0.5 if i == 0 else 0.5 / (2 ** (i - 1)) for i in range(count)]
    for idx, w in enumerate(weights):
        if not math.isfinite(float(w)):
            raise ValueError(f"MSF scale weight must be finite at scale={idx}: {w!r}")
        if float(w) < 0.0:
            raise ValueError(f"MSF scale weight must be >= 0 at scale={idx}: {w!r}")
    return [float(v) for v in weights]


def compute_msf_out_channels(weight: float, base_channels: int = BASE_OUT_CHANNELS) -> int:
    w = float(weight)
    if w <= 0.0:
        return 0
    return max(1, int(base_channels * w))


def build_split_ranges(total: int, val_fraction: float, test_fraction: float) -> SplitRanges:
    n = int(total)
    if n < 3:
        raise ValueError("need at least 3 samples for train/val/test")
    vf = float(val_fraction)
    tf = float(test_fraction)
    if vf <= 0.0 or tf <= 0.0 or (vf + tf) >= 1.0:
        raise ValueError(
            "val_fraction and test_fraction must be > 0 and sum to < 1.0; "
            f"got val_fraction={vf} test_fraction={tf}"
        )
    train_end = int(n * (1.0 - vf - tf))
    val_end = int(n * (1.0 - tf))
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))
    return SplitRanges(train=(0, train_end), val=(train_end, val_end), test=(val_end, n))


def build_balanced_class_weights_binary(class0_count: int, class1_count: int) -> torch.Tensor | None:
    n0 = int(class0_count)
    n1 = int(class1_count)
    if n0 <= 0 or n1 <= 0:
        return None
    total = float(n0 + n1)
    w0 = total / (2.0 * float(n0))
    w1 = total / (2.0 * float(n1))
    return torch.tensor([w0, w1], dtype=torch.float32)


def to_device_batch(
    xb: np.ndarray,
    yb: np.ndarray,
    vixb: np.ndarray | None,
    vix_imgb: np.ndarray | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    x = torch.from_numpy(np.asarray(xb, dtype=np.float32))
    y = torch.from_numpy(np.asarray(yb, dtype=np.int64))
    vix = (
        torch.from_numpy(np.asarray(vixb, dtype=np.float32))
        if vixb is not None
        else None
    )
    vix_img = (
        torch.from_numpy(np.asarray(vix_imgb, dtype=np.float32))
        if vix_imgb is not None
        else None
    )
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


class ScaleVariationalEncoder(nn.Module):
    """Single-scale conv encoder that produces (mu, logvar, z)."""

    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        pre_mlp_dim: int = LATENT_PRE_MLP_DIM,
    ) -> None:
        super().__init__()
        out_ch = int(out_channels)
        z_dim = int(latent_dim)
        if out_ch < 1:
            raise ValueError(f"out_channels must be >= 1, got {out_channels!r}")
        if z_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim!r}")

        pad = (KERNEL_SIZE // 2, KERNEL_WIDTH // 2)
        self.conv1 = nn.Conv2d(1, CONV_CHANNELS[0], kernel_size=(KERNEL_SIZE, KERNEL_WIDTH), padding=pad)
        self.bn1 = nn.BatchNorm2d(CONV_CHANNELS[0])
        self.act1 = nn.LeakyReLU(LRELU_SLOPE)
        self.conv2 = nn.Conv2d(
            CONV_CHANNELS[0],
            CONV_CHANNELS[1],
            kernel_size=(KERNEL_SIZE, KERNEL_WIDTH),
            padding=pad,
        )
        self.bn2 = nn.BatchNorm2d(CONV_CHANNELS[1])
        self.act2 = nn.LeakyReLU(LRELU_SLOPE)
        self.pool1 = build_pool_layer(SCALE_POOL_DIM, "SCALE_POOL_DIM")
        self.conv3 = nn.Conv2d(
            CONV_CHANNELS[1],
            out_ch,
            kernel_size=(KERNEL_SIZE, KERNEL_WIDTH),
            padding=pad,
        )
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.act3 = nn.LeakyReLU(LRELU_SLOPE)
        self.pool2 = build_pool_layer(SCALE_POOL_DIM, "SCALE_POOL_DIM")
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        hidden_in = out_ch
        self.pre_latent: nn.Module
        if int(pre_mlp_dim) > 0:
            self.pre_latent = nn.Sequential(
                nn.Linear(hidden_in, int(pre_mlp_dim)),
                nn.LeakyReLU(LRELU_SLOPE),
            )
            hidden_in = int(pre_mlp_dim)
        else:
            self.pre_latent = nn.Identity()

        self.fc_mu = nn.Linear(hidden_in, z_dim)
        self.fc_logvar = nn.Linear(hidden_in, z_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not bool(sample):
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
        sample_latent: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.act1(self.bn1(self.conv1(x)))
        h = self.act2(self.bn2(self.conv2(h)))
        h = self.pool1(h)
        h = self.act3(self.bn3(self.conv3(h)))
        h = self.pool2(h)
        h = self.gap(h).flatten(start_dim=1)
        h = self.pre_latent(h)
        mu = self.fc_mu(h)
        # Clamp helps numeric stability for exp(logvar).
        logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=10.0)
        z = self.reparameterize(mu=mu, logvar=logvar, sample=sample_latent)
        return z, mu, logvar


class VixImageEncoder(nn.Module):
    """Compact conv encoder for optional VIX image input."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        emb = int(embed_dim)
        if emb < 1:
            raise ValueError(f"vix image embed_dim must be >= 1, got {embed_dim!r}")
        conv_ch = [int(v) for v in VIX_CONV_CHANNELS]
        if len(conv_ch) != 3 or any(int(v) < 1 for v in conv_ch):
            raise ValueError(
                "VIX_CONV_CHANNELS must contain exactly 3 positive ints; "
                f"got {VIX_CONV_CHANNELS!r}"
            )
        pool_cfg = [int(v) for v in VIX_POOLING]
        if len(pool_cfg) > len(conv_ch):
            raise ValueError(
                "VIX_POOLING has more entries than VIX_CONV_CHANNELS; "
                f"got len(VIX_POOLING)={len(pool_cfg)} len(VIX_CONV_CHANNELS)={len(conv_ch)}"
            )
        if any(int(v) < 1 for v in pool_cfg):
            raise ValueError(f"VIX_POOLING entries must be >=1 strides; got {VIX_POOLING!r}")
        # Missing entries mean no pooling for remaining conv blocks.
        if len(pool_cfg) < len(conv_ch):
            pool_cfg = pool_cfg + [1] * (len(conv_ch) - len(pool_cfg))
        self.vix_pooling = tuple(int(v) for v in pool_cfg)

        pad = (KERNEL_SIZE // 2, KERNEL_WIDTH // 2)
        self.conv1 = nn.Conv2d(
            1, conv_ch[0], kernel_size=(KERNEL_SIZE, KERNEL_WIDTH), padding=pad
        )
        self.bn1 = nn.BatchNorm2d(conv_ch[0])
        self.act1 = nn.LeakyReLU(LRELU_SLOPE)
        self.conv2 = nn.Conv2d(
            conv_ch[0], conv_ch[1], kernel_size=(KERNEL_SIZE, KERNEL_WIDTH), padding=pad
        )
        self.bn2 = nn.BatchNorm2d(conv_ch[1])
        self.act2 = nn.LeakyReLU(LRELU_SLOPE)
        self.pool1 = build_pool_layer_with_stride(
            VIX_POOL_DIM,
            "VIX_POOL_DIM",
            stride=int(self.vix_pooling[0]),
        )
        self.conv3 = nn.Conv2d(
            conv_ch[1], conv_ch[2], kernel_size=(KERNEL_SIZE, KERNEL_WIDTH), padding=pad
        )
        self.bn3 = nn.BatchNorm2d(conv_ch[2])
        self.act3 = nn.LeakyReLU(LRELU_SLOPE)
        self.pool2 = build_pool_layer_with_stride(
            VIX_POOL_DIM,
            "VIX_POOL_DIM",
            stride=int(self.vix_pooling[1]),
        )
        self.pool3 = build_pool_layer_with_stride(
            VIX_POOL_DIM,
            "VIX_POOL_DIM",
            stride=int(self.vix_pooling[2]),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(conv_ch[2], emb),
            nn.LeakyReLU(LRELU_SLOPE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.bn1(self.conv1(x)))
        h = self.pool1(h)
        h = self.act2(self.bn2(self.conv2(h)))
        h = self.pool2(h)
        h = self.act3(self.bn3(self.conv3(h)))
        h = self.pool3(h)
        h = self.gap(h).flatten(start_dim=1)
        return self.proj(h)


class VariationalDecompClassifier(nn.Module):
    """Per-scale variational encoders + classification head."""

    def __init__(
        self,
        scales: int,
        input_height: int,
        input_width_per_scale: Sequence[int],
        latent_dim_per_scale: int = LATENT_DIM_PER_SCALE,
        readout_dim: int = READOUT_DIM,
        head_dims: Sequence[int] = HEAD_DIMS,
        fc_dropout: float = FC_DROPOUT,
        vix_fusion_mode: str = VIX_FUSION_MODE,
        vix_embed_dim: int = VIX_EMBED_DIM_DEFAULT,
        include_vix_image: bool = False,
        vix_image_height: int = 0,
        vix_image_width: int = 0,
        vix_image_embed_dim: int = VIX_IMAGE_EMBED_DIM,
    ) -> None:
        super().__init__()
        self.scales = int(scales)
        self.input_height = int(input_height)
        self.input_width_per_scale = tuple(int(w) for w in input_width_per_scale)
        if len(self.input_width_per_scale) != self.scales:
            raise ValueError(
                "input_width_per_scale length must match scales; "
                f"len={len(self.input_width_per_scale)} scales={self.scales}"
            )
        if any(int(w) < 1 for w in self.input_width_per_scale):
            raise ValueError(f"all input widths must be >=1, got {self.input_width_per_scale}")

        self.latent_dim_per_scale = int(latent_dim_per_scale)
        if self.latent_dim_per_scale < 1:
            raise ValueError(
                f"latent_dim_per_scale must be >= 1, got {self.latent_dim_per_scale}"
            )
        self.vix_fusion_mode = resolve_vix_fusion_mode(str(vix_fusion_mode))
        self.vix_embed_dim = int(vix_embed_dim)
        if self.vix_fusion_mode != "none" and self.vix_embed_dim < 1:
            raise ValueError(
                f"vix_embed_dim must be >= 1 when vix_fusion_mode={self.vix_fusion_mode!r}; "
                f"got {vix_embed_dim!r}"
            )
        self.include_vix_image = bool(include_vix_image)
        self.vix_image_height = int(vix_image_height)
        self.vix_image_width = int(vix_image_width)
        self.vix_image_embed_dim = int(vix_image_embed_dim)
        if self.include_vix_image:
            if self.vix_image_height < 1 or self.vix_image_width < 1:
                raise ValueError(
                    "vix image dimensions must be >= 1 when include_vix_image=True; "
                    f"got h={self.vix_image_height} w={self.vix_image_width}"
                )
            if self.vix_image_embed_dim < 1:
                raise ValueError(
                    "vix_image_embed_dim must be >= 1 when include_vix_image=True; "
                    f"got {vix_image_embed_dim!r}"
                )

        self.msf_scale_weights = tuple(resolve_msf_scale_weights(self.scales))
        blocks: list[nn.Module] = []
        active_scale_indices: list[int] = []
        active_scale_weights: list[float] = []
        disabled_scale_indices: list[int] = []
        for i, w in enumerate(self.msf_scale_weights):
            out_ch = compute_msf_out_channels(w, base_channels=BASE_OUT_CHANNELS)
            if out_ch <= 0:
                blocks.append(nn.Identity())
                disabled_scale_indices.append(int(i))
                continue
            blocks.append(
                ScaleVariationalEncoder(
                    out_channels=out_ch,
                    latent_dim=self.latent_dim_per_scale,
                )
            )
            active_scale_indices.append(int(i))
            active_scale_weights.append(float(w))
        if not active_scale_indices:
            raise ValueError("all scales disabled by MSF_SCALE_WEIGHTS; need at least one active")

        self.encoders = nn.ModuleList(blocks)
        self.active_scale_indices = tuple(active_scale_indices)
        self.active_scale_weights = tuple(active_scale_weights)
        self.disabled_scale_indices = tuple(disabled_scale_indices)

        fused_dim = int(len(self.active_scale_indices) * self.latent_dim_per_scale)
        self.readout = nn.Linear(fused_dim, int(readout_dim))
        self.readout_act = nn.LeakyReLU(LRELU_SLOPE)
        self.readout_drop = nn.Dropout(float(fc_dropout)) if fc_dropout > 0 else nn.Identity()

        self.vix_film: nn.Module | None = None
        self.vix_embed: nn.Module | None = None
        self.vix_img_encoder: nn.Module | None = None
        if self.vix_fusion_mode == "film":
            self.vix_film = nn.Sequential(
                nn.Linear(1, int(self.vix_embed_dim)),
                nn.LeakyReLU(LRELU_SLOPE),
                nn.Linear(int(self.vix_embed_dim), int(2 * int(readout_dim))),
            )
        elif self.vix_fusion_mode == "late_concat":
            self.vix_embed = nn.Sequential(
                nn.Linear(1, int(self.vix_embed_dim)),
                nn.LeakyReLU(LRELU_SLOPE),
                nn.Linear(int(self.vix_embed_dim), int(self.vix_embed_dim)),
                nn.LeakyReLU(LRELU_SLOPE),
            )
        if self.include_vix_image:
            self.vix_img_encoder = VixImageEncoder(embed_dim=int(self.vix_image_embed_dim))

        head_input_dim = int(readout_dim)
        if self.include_vix_image:
            head_input_dim += int(self.vix_image_embed_dim)
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

    @staticmethod
    def _kl_per_scale(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL[q(z|x) || p(z)] with p(z)=N(0,1), per-sample.
        return -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(dim=1)

    def _weighted_kl(self, mus: Sequence[torch.Tensor], logvars: Sequence[torch.Tensor]) -> torch.Tensor:
        if not mus:
            raise RuntimeError("no active latent scales for KL computation")
        denom = float(sum(self.active_scale_weights))
        if denom <= 0.0:
            raise RuntimeError("active scale weight sum must be > 0")
        total = None
        for w, mu, logvar in zip(self.active_scale_weights, mus, logvars):
            kl_i = self._kl_per_scale(mu=mu, logvar=logvar) * float(w)
            total = kl_i if total is None else (total + kl_i)
        if total is None:
            raise RuntimeError("failed to compute KL term")
        # Return per-sample KL so parallel wrappers gather by batch dimension cleanly.
        return total / float(denom)

    def _prepare_vix(self, vix: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
        if self.vix_fusion_mode == "none":
            return None
        if vix is None:
            raise ValueError(
                f"vix tensor is required when vix_fusion_mode={self.vix_fusion_mode!r}"
            )
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

    def _apply_vix_film(self, rep: torch.Tensor, vix: torch.Tensor) -> torch.Tensor:
        if self.vix_fusion_mode != "film":
            return rep
        if self.vix_film is None:
            raise RuntimeError("vix_film module missing while vix_fusion_mode='film'")
        film_params = self.vix_film(vix)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        gamma = 0.25 * torch.tanh(gamma)
        return rep * (1.0 + gamma) + beta

    def forward(
        self,
        x: torch.Tensor,
        vix: torch.Tensor | None = None,
        vix_img: torch.Tensor | None = None,
        sample_latent: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"expected x shape (batch, scales, H, W), got {tuple(x.shape)}")
        if int(x.shape[1]) != self.scales:
            raise ValueError(f"expected scales={self.scales}, got {int(x.shape[1])}")
        if int(x.shape[2]) != self.input_height:
            raise ValueError(f"expected height={self.input_height}, got {int(x.shape[2])}")
        if int(x.shape[3]) < int(max(self.input_width_per_scale)):
            raise ValueError(
                "input width is smaller than required max scale width: "
                f"input={int(x.shape[3])} max_required={int(max(self.input_width_per_scale))}"
            )
        sample = bool(self.training if sample_latent is None else sample_latent)
        vix_ready = self._prepare_vix(vix=vix, batch_size=int(x.shape[0]))
        vix_img_ready = self._prepare_vix_image(vix_img=vix_img, batch_size=int(x.shape[0]))

        zs: list[torch.Tensor] = []
        mus: list[torch.Tensor] = []
        logvars: list[torch.Tensor] = []
        for scale_idx in self.active_scale_indices:
            enc = self.encoders[scale_idx]
            if isinstance(enc, nn.Identity):
                raise RuntimeError(f"scale {scale_idx} listed active but encoder is Identity")
            scale_w = int(self.input_width_per_scale[scale_idx])
            xi = x[:, scale_idx : scale_idx + 1, :, :scale_w]
            z, mu, logvar = enc(xi, sample_latent=sample)
            zs.append(z)
            mus.append(mu)
            logvars.append(logvar)

        z_fused = torch.cat(zs, dim=1)
        rep = self.readout_drop(self.readout_act(self.readout(z_fused)))
        if self.vix_fusion_mode == "film":
            if vix_ready is None:
                raise RuntimeError("vix tensor unexpectedly missing for film fusion")
            rep = self._apply_vix_film(rep, vix=vix_ready)
        if self.include_vix_image:
            if self.vix_img_encoder is None:
                raise RuntimeError("vix image encoder missing while include_vix_image=True")
            if vix_img_ready is None:
                raise RuntimeError("vix image tensor unexpectedly missing")
            rep = torch.cat([rep, self.vix_img_encoder(vix_img_ready)], dim=1)
        if self.vix_fusion_mode == "late_concat":
            if vix_ready is None:
                raise RuntimeError("vix tensor unexpectedly missing for late-concat fusion")
            if self.vix_embed is None:
                raise RuntimeError("vix_embed module missing while vix_fusion_mode='late_concat'")
            rep = torch.cat([rep, self.vix_embed(vix_ready)], dim=1)
        logits = self.cls_head(rep)
        kl = self._weighted_kl(mus=mus, logvars=logvars)
        return logits, kl


def run_epoch(
    model: nn.Module,
    store: ShardedImageStore,
    split_range: tuple[int, int],
    batch_size: int,
    device: torch.device,
    amp_enabled: bool,
    scaler: torch.amp.GradScaler | None,
    beta_kl: float,
    train: bool,
    stochastic_train_enabled: bool,
    optimizer: torch.optim.Optimizer | None,
    class_weights: torch.Tensor | None,
    seed: int,
    progress_label: str | None = None,
    vix_enabled: bool = False,
    vix_image_enabled: bool = False,
    distributed: bool = False,
    rank: int = 0,
) -> dict[str, float | int]:
    if train and optimizer is None:
        raise ValueError("optimizer is required for train epoch")
    model.train(mode=train)
    total_n = 0
    loss_sum = 0.0
    ce_sum = 0.0
    kl_sum = 0.0
    acc_sum = 0.0
    pred_pos_sum = 0
    true_pos_sum = 0
    tp_sum = 0
    tn_sum = 0
    fp_sum = 0
    fn_sum = 0
    autocast_enabled = bool(amp_enabled and device.type == "cuda")
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
    step_count = 0
    split_n_hint = max(0, int(split_range[1]) - int(split_range[0]))
    steps_total_hint = (
        int(math.ceil(float(split_n_hint) / float(max(1, int(batch_size)))))
        if split_n_hint > 0
        else 0
    )
    progress_enabled = (
        bool(progress_label)
        and bool(KERAS_INTRA_EPOCH_LOGGING_ENABLED)
        and (not bool(distributed) or int(rank) == 0)
    )
    progress_update_every = max(1, int(KERAS_PROGRESS_UPDATE_EVERY_STEPS))
    progress_line_len = 0
    progress_prefix = ""
    running_n_local = 0
    running_loss_local = 0.0
    running_acc_local = 0.0
    t_epoch_start = time.perf_counter()

    def write_progress_line(line: str) -> None:
        nonlocal progress_line_len
        pad_len = max(0, int(progress_line_len - len(line)))
        sys.stdout.write("\r" + line + (" " * pad_len))
        sys.stdout.flush()
        progress_line_len = len(line)

    if progress_enabled:
        label_txt = str(progress_label)
        progress_prefix = f"{label_txt} | "
        if " - train" in label_txt.lower():
            # Visually separate train bar from prior epoch summary lines.
            print("")

    batch_iter = store.iter_batches(
        split_range=split_range,
        batch_size=int(batch_size),
        shuffle=bool(train),
        seed=int(seed),
        return_vix=bool(vix_enabled),
        return_vix_img=bool(vix_image_enabled),
    )

    use_ddp_join = bool(train and isinstance(model, DDP))
    join_ctx = model.join if use_ddp_join else nullcontext
    with join_ctx():
        for batch in batch_iter:
            step_count += 1
            if bool(vix_enabled) and bool(vix_image_enabled):
                xb_np, yb_np, vixb_np, vix_imgb_np = batch
            elif bool(vix_enabled):
                xb_np, yb_np, vixb_np = batch
                vix_imgb_np = None
            elif bool(vix_image_enabled):
                xb_np, yb_np, vix_imgb_np = batch
                vixb_np = None
            else:
                xb_np, yb_np = batch
                vixb_np = None
                vix_imgb_np = None
            xb, yb, vixb, vix_imgb = to_device_batch(
                xb_np,
                yb_np,
                vixb_np,
                vix_imgb_np,
                device=device,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=autocast_device_type, enabled=autocast_enabled):
                    logits, kl = model(
                        xb,
                        vix=vixb,
                        vix_img=vix_imgb,
                        sample_latent=bool(stochastic_train_enabled),
                    )
                    if kl.ndim > 0:
                        kl = kl.mean()
                    ce = F.cross_entropy(logits, yb, weight=class_weights, reduction="mean")
                    loss = ce + float(beta_kl) * kl
                if autocast_enabled and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                with torch.no_grad():
                    with torch.amp.autocast(device_type=autocast_device_type, enabled=autocast_enabled):
                        logits, kl = model(xb, vix=vixb, vix_img=vix_imgb, sample_latent=False)
                        if kl.ndim > 0:
                            kl = kl.mean()
                        ce = F.cross_entropy(logits, yb, weight=class_weights, reduction="mean")
                        loss = ce + float(beta_kl) * kl

            bs = int(yb.shape[0])
            total_n += bs
            running_n_local += bs
            running_loss_local += float(loss.detach().item()) * bs
            pred = torch.argmax(logits.detach(), dim=1)
            running_acc_local += float((pred == yb).sum().item())
            pred_pos_sum += int((pred == 1).sum().item())
            true_pos_sum += int((yb == 1).sum().item())
            tp_sum += int(((pred == 1) & (yb == 1)).sum().item())
            tn_sum += int(((pred == 0) & (yb == 0)).sum().item())
            fp_sum += int(((pred == 1) & (yb == 0)).sum().item())
            fn_sum += int(((pred == 0) & (yb == 1)).sum().item())
            acc_sum += float((pred == yb).sum().item())
            loss_sum += float(loss.detach().item()) * bs
            ce_sum += float(ce.detach().item()) * bs
            kl_sum += float(kl.detach().item()) * bs
            if progress_enabled and (step_count % progress_update_every == 0 or step_count == 1):
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
                    line = progress_prefix + line
                    write_progress_line(line)

    if bool(distributed) and dist.is_available() and dist.is_initialized():
        stats_vec = torch.tensor(
            [
                float(loss_sum),
                float(ce_sum),
                float(kl_sum),
                float(acc_sum),
                float(pred_pos_sum),
                float(true_pos_sum),
                float(tp_sum),
                float(tn_sum),
                float(fp_sum),
                float(fn_sum),
                float(total_n),
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(stats_vec, op=dist.ReduceOp.SUM)
        loss_sum = float(stats_vec[0].item())
        ce_sum = float(stats_vec[1].item())
        kl_sum = float(stats_vec[2].item())
        acc_sum = float(stats_vec[3].item())
        pred_pos_sum = int(round(float(stats_vec[4].item())))
        true_pos_sum = int(round(float(stats_vec[5].item())))
        tp_sum = int(round(float(stats_vec[6].item())))
        tn_sum = int(round(float(stats_vec[7].item())))
        fp_sum = int(round(float(stats_vec[8].item())))
        fn_sum = int(round(float(stats_vec[9].item())))
        total_n = int(round(float(stats_vec[10].item())))

    if total_n <= 0:
        if progress_enabled:
            elapsed_total = float(max(1e-12, time.perf_counter() - t_epoch_start))
            total_display = max(1, int(steps_total_hint), int(step_count))
            line = format_keras_progbar_line(
                steps_done=int(step_count),
                total_steps=int(total_display),
                elapsed_sec=float(elapsed_total),
                metric_pairs=[("loss", float("nan")), ("acc", float("nan"))],
                final=True,
            )
            line = progress_prefix + line
            write_progress_line(line)
            sys.stdout.write("\n")
            sys.stdout.flush()
        return {
            "loss": float("nan"),
            "ce_loss": float("nan"),
            "kl_loss": float("nan"),
            "acc": float("nan"),
            "pred_pos_rate": float("nan"),
            "true_pos_rate": float("nan"),
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "n": 0.0,
        }

    out = {
        "loss": float(loss_sum / total_n),
        "ce_loss": float(ce_sum / total_n),
        "kl_loss": float(kl_sum / total_n),
        "acc": float(acc_sum / total_n),
        "pred_pos_rate": float(pred_pos_sum / total_n),
        "true_pos_rate": float(true_pos_sum / total_n),
        "tp": int(tp_sum),
        "tn": int(tn_sum),
        "fp": int(fp_sum),
        "fn": int(fn_sum),
        "n": float(total_n),
    }
    if progress_enabled:
        elapsed_total = float(max(1e-12, time.perf_counter() - t_epoch_start))
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
        line = progress_prefix + line
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
    seed: int,
    progress_label: str | None = None,
    vix_enabled: bool = False,
    vix_image_enabled: bool = False,
    distributed: bool = False,
    rank: int = 0,
) -> dict[str, np.ndarray]:
    model.eval()
    autocast_enabled = bool(amp_enabled and device.type == "cuda")
    autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
    step_count = 0
    split_n_hint = max(0, int(split_range[1]) - int(split_range[0]))
    steps_total_hint = (
        int(math.ceil(float(split_n_hint) / float(max(1, int(batch_size)))))
        if split_n_hint > 0
        else 0
    )
    progress_enabled = (
        bool(progress_label)
        and bool(KERAS_INTRA_EPOCH_LOGGING_ENABLED)
        and (not bool(distributed) or int(rank) == 0)
    )
    progress_update_every = max(1, int(KERAS_PROGRESS_UPDATE_EVERY_STEPS))
    progress_line_len = 0
    running_n_local = 0
    t_collect_start = time.perf_counter()

    def write_progress_line(line: str) -> None:
        nonlocal progress_line_len
        pad_len = max(0, int(progress_line_len - len(line)))
        sys.stdout.write("\r" + line + (" " * pad_len))
        sys.stdout.flush()
        progress_line_len = len(line)

    if progress_enabled:
        print(str(progress_label))

    batch_iter = store.iter_batches(
        split_range=split_range,
        batch_size=int(batch_size),
        shuffle=False,
        seed=int(seed),
        return_sample_indices=True,
        return_ret_pct=True,
        return_timestamps=True,
        return_ticker_ids=True,
        return_vix=bool(vix_enabled),
        return_vix_img=bool(vix_image_enabled),
    )

    all_sample_indices: list[np.ndarray] = []
    all_y_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_prob1: list[np.ndarray] = []
    all_ret_pct_true: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    all_ticker_ids: list[np.ndarray] = []

    with torch.no_grad():
        for batch in batch_iter:
            step_count += 1
            if bool(vix_enabled) and bool(vix_image_enabled):
                xb_np, yb_np, si_np, ret_pct_np, ts_np, tid_np, vixb_np, vix_imgb_np = batch
            elif bool(vix_enabled):
                xb_np, yb_np, si_np, ret_pct_np, ts_np, tid_np, vixb_np = batch
                vix_imgb_np = None
            elif bool(vix_image_enabled):
                xb_np, yb_np, si_np, ret_pct_np, ts_np, tid_np, vix_imgb_np = batch
                vixb_np = None
            else:
                xb_np, yb_np, si_np, ret_pct_np, ts_np, tid_np = batch
                vixb_np = None
                vix_imgb_np = None
            xb, yb, vixb, vix_imgb = to_device_batch(
                xb_np,
                yb_np,
                vixb_np,
                vix_imgb_np,
                device=device,
            )
            with torch.amp.autocast(device_type=autocast_device_type, enabled=autocast_enabled):
                logits, _ = model(xb, vix=vixb, vix_img=vix_imgb, sample_latent=False)
                prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)
            prob1 = prob[:, 1]

            y_cpu = yb.detach().to(device="cpu", dtype=torch.int64).numpy()
            pred_cpu = pred.detach().to(device="cpu", dtype=torch.int64).numpy()
            prob1_cpu = prob1.detach().to(device="cpu", dtype=torch.float32).numpy()
            si_cpu = np.asarray(si_np, dtype=np.int64)

            all_sample_indices.append(si_cpu)
            all_y_true.append(y_cpu)
            all_pred.append(pred_cpu)
            all_prob1.append(prob1_cpu)
            all_ret_pct_true.append(np.asarray(ret_pct_np, dtype=np.float32))
            all_timestamps.append(np.asarray(ts_np, dtype=object))
            all_ticker_ids.append(np.asarray(tid_np, dtype=np.int64))
            running_n_local += int(y_cpu.shape[0])

            if progress_enabled and (step_count % progress_update_every == 0 or step_count == 1):
                elapsed_now = float(max(1e-12, time.perf_counter() - t_collect_start))
                total_display = max(1, int(steps_total_hint), int(step_count))
                if int(step_count) < int(total_display):
                    metric_pairs = [("rows", float(running_n_local))]
                    line = format_keras_progbar_line(
                        steps_done=int(step_count),
                        total_steps=int(total_display),
                        elapsed_sec=float(elapsed_now),
                        metric_pairs=metric_pairs,
                        final=False,
                    )
                    write_progress_line(line)

    if all_sample_indices:
        sample_indices = np.ascontiguousarray(np.concatenate(all_sample_indices, axis=0), dtype=np.int64)
        y_true = np.ascontiguousarray(np.concatenate(all_y_true, axis=0), dtype=np.int64)
        pred = np.ascontiguousarray(np.concatenate(all_pred, axis=0), dtype=np.int64)
        prob1 = np.ascontiguousarray(np.concatenate(all_prob1, axis=0), dtype=np.float32)
        ret_pct_true = np.ascontiguousarray(
            np.concatenate(all_ret_pct_true, axis=0),
            dtype=np.float32,
        )
        timestamps = np.asarray(np.concatenate(all_timestamps, axis=0), dtype=object)
        ticker_ids = np.ascontiguousarray(np.concatenate(all_ticker_ids, axis=0), dtype=np.int64)
    else:
        sample_indices = np.empty((0,), dtype=np.int64)
        y_true = np.empty((0,), dtype=np.int64)
        pred = np.empty((0,), dtype=np.int64)
        prob1 = np.empty((0,), dtype=np.float32)
        ret_pct_true = np.empty((0,), dtype=np.float32)
        timestamps = np.empty((0,), dtype=object)
        ticker_ids = np.empty((0,), dtype=np.int64)

    tickers = np.asarray(
        [store._ticker_symbol_for_id(int(tid)) or "" for tid in ticker_ids],
        dtype=object,
    )

    if progress_enabled:
        elapsed_total = float(max(1e-12, time.perf_counter() - t_collect_start))
        total_display = max(1, int(steps_total_hint), int(step_count))
        line = format_keras_progbar_line(
            steps_done=int(step_count),
            total_steps=int(total_display),
            elapsed_sec=float(elapsed_total),
            metric_pairs=[("rows", float(sample_indices.shape[0]))],
            final=True,
        )
        write_progress_line(line)
        sys.stdout.write("\n")
        sys.stdout.flush()

    return {
        "sample_index": sample_indices,
        "y_true": y_true,
        "pred": pred,
        "prob_1": prob1,
        "ret_pct_true": ret_pct_true,
        "timestamp": timestamps,
        "ticker_id": ticker_ids,
        "ticker": tickers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--amp-enabled", type=int, default=AMP_ENABLED_DEFAULT)
    parser.add_argument("--stochastic-train-enabled", type=int, default=STOCHASTIC_MODE)
    parser.add_argument("--multi-gpu-enabled", type=int, default=MULTI_GPU_ENABLED_DEFAULT)
    parser.add_argument("--beta-kl", type=float, default=BETA_KL)
    parser.add_argument("--ret-atr-threshold", type=float, default=RET_ATR_THRESHOLD)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--latent-dim-per-scale", type=int, default=LATENT_DIM_PER_SCALE)
    parser.add_argument("--readout-dim", type=int, default=READOUT_DIM)
    parser.add_argument("--fc-dropout", type=float, default=FC_DROPOUT)
    parser.add_argument(
        "--vix-fusion-mode",
        type=str,
        default=VIX_FUSION_MODE,
        help="One of: none, film, late_concat",
    )
    parser.add_argument("--vix-embed-dim", type=int, default=VIX_EMBED_DIM_DEFAULT)
    parser.add_argument("--vix-daily-csv", type=str, default=VIX_DAILY_CSV_DEFAULT)
    parser.add_argument("--vix-date-col", type=str, default=VIX_DATE_COL_DEFAULT)
    parser.add_argument("--vix-value-col", type=str, default=VIX_VALUE_COL_DEFAULT)
    parser.add_argument("--vix-norm-method", type=str, default=VIX_NORM_METHOD_DEFAULT)
    parser.add_argument("--vix-norm-clip", type=float, default=VIX_NORM_CLIP_DEFAULT)
    parser.add_argument("--vix-log1p", type=int, default=int(VIX_LOG1P_DEFAULT))
    parser.add_argument("--vix-image-enabled", type=int, default=int(VIX_IMAGE_ENABLED))
    parser.add_argument("--vix-image-embed-dim", type=int, default=int(VIX_IMAGE_EMBED_DIM))
    parser.add_argument("--class-weighted-ce-enabled", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    distributed = False
    rank = 0
    world_size = 1
    local_rank = 0
    try:
        device = resolve_device(args.device)
        multi_gpu_enabled = bool(int(args.multi_gpu_enabled))
        distributed, rank, world_size, local_rank, device = init_distributed_if_needed(
            device=device,
            multi_gpu_enabled=multi_gpu_enabled,
        )
        main_process = is_main_process(rank)
        set_seed(int(args.seed) + int(rank))
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

        amp_enabled = bool(int(args.amp_enabled)) and device.type == "cuda"
        stochastic_train_enabled = bool(int(args.stochastic_train_enabled))
        vix_fusion_mode = resolve_vix_fusion_mode(str(args.vix_fusion_mode))
        vix_enabled = bool(vix_fusion_mode != "none")
        vix_image_enabled = bool(int(args.vix_image_enabled))
        if not bool(int(VIX_FEATURES_ENABLED)):
            # Global VIX kill switch: force-disable scalar fusion and VIX image path.
            vix_fusion_mode = "none"
            vix_enabled = False
            vix_image_enabled = False
        vix_daily_csv = str(args.vix_daily_csv).strip()
        args.vix_fusion_mode = str(vix_fusion_mode)
        args.vix_daily_csv = str(vix_daily_csv)
        args.vix_image_enabled = int(vix_image_enabled)
        args.multi_gpu_enabled = int(multi_gpu_enabled)
        args.distributed = int(distributed)
        args.rank = int(rank)
        args.world_size = int(world_size)
        args.local_rank = int(local_rank)
        if bool(vix_enabled) and not vix_daily_csv:
            raise ValueError("--vix-daily-csv is required when --vix-fusion-mode is not 'none'")

        store = ShardedImageStore(
            data_dir=Path(args.data_dir),
            ret_atr_threshold=float(args.ret_atr_threshold),
            vix_daily_csv=(vix_daily_csv if bool(vix_enabled) else ""),
            vix_date_col=str(args.vix_date_col),
            vix_value_col=str(args.vix_value_col),
        )
        splits = build_split_ranges(
            total=int(store.subset_count),
            val_fraction=float(args.val_fraction),
            test_fraction=float(args.test_fraction),
        )
        train_split = (
            split_range_for_rank(splits.train, rank=rank, world_size=world_size)
            if distributed
            else splits.train
        )
        val_split = (
            split_range_for_rank(splits.val, rank=rank, world_size=world_size)
            if distributed
            else splits.val
        )
        test_split = (
            split_range_for_rank(splits.test, rank=rank, world_size=world_size)
            if distributed
            else splits.test
        )
        if bool(vix_image_enabled) and not bool(store.include_vix_image):
            raise ValueError(
                "--vix-image-enabled=1 requested but dataset manifest does not include VIX images"
            )
        vix_norm_meta: dict[str, float | int | str] | None = None
        if bool(vix_enabled):
            local_vix_stats: VixNormStats | None = None
            if (not distributed) or main_process:
                vix_norm_meta = store.fit_vix_normalizer(
                    split_range=splits.train,
                    method=str(args.vix_norm_method),
                    clip=float(args.vix_norm_clip),
                    log1p=bool(int(args.vix_log1p)),
                )
                local_vix_stats = store.vix_norm_stats
            if distributed:
                vix_norm_meta = broadcast_object(vix_norm_meta, src=0)
                local_vix_stats = broadcast_object(local_vix_stats, src=0)
            store.vix_norm_stats = local_vix_stats

        model_core = VariationalDecompClassifier(
            scales=int(store.scales),
            input_height=int(store.height),
            input_width_per_scale=[int(v) for v in store.width_per_scale],
            latent_dim_per_scale=int(args.latent_dim_per_scale),
            readout_dim=int(args.readout_dim),
            head_dims=HEAD_DIMS,
            fc_dropout=float(args.fc_dropout),
            vix_fusion_mode=str(vix_fusion_mode),
            vix_embed_dim=int(args.vix_embed_dim),
            include_vix_image=bool(vix_image_enabled),
            vix_image_height=int(store.vix_image_height),
            vix_image_width=int(store.vix_image_width),
            vix_image_embed_dim=int(args.vix_image_embed_dim),
        ).to(device=device)
        cuda_device_count = int(torch.cuda.device_count()) if device.type == "cuda" else 0
        model: nn.Module
        if distributed:
            model = DDP(
                model_core,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            model = model_core

        class_weights = None
        if bool(args.class_weighted_ce_enabled):
            c0, c1 = store.compute_class_counts(splits.train)
            class_weights = build_balanced_class_weights_binary(c0, c1)
            if class_weights is not None:
                class_weights = class_weights.to(device=device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
        )
        scaler: torch.amp.GradScaler | None = None
        if amp_enabled:
            scaler = torch.amp.GradScaler("cuda", enabled=True)

        out_dir_raw = str(args.output_dir).strip()
        if main_process:
            if out_dir_raw:
                out_dir = Path(out_dir_raw)
                out_dir.mkdir(parents=True, exist_ok=True)
            else:
                runs_root = Path("runs")
                while True:
                    candidate = resolve_next_variational_run_dir(runs_root)
                    try:
                        candidate.mkdir(parents=True, exist_ok=False)
                    except FileExistsError:
                        continue
                    out_dir = candidate
                    break
        else:
            out_dir = Path(".")
        if distributed:
            out_dir = Path(str(broadcast_object(str(out_dir), src=0)))
        preds_dir = out_dir / "preds"
        best_path = out_dir / "best.pt"
        run_config_path = out_dir / "run_config.json"
        if main_process:
            out_dir.mkdir(parents=True, exist_ok=True)
            preds_dir.mkdir(parents=True, exist_ok=True)
        if distributed:
            dist.barrier()

        if main_process:
            run_config_payload = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "script_path": str(Path(__file__).resolve()),
                "argv": [str(v) for v in sys.argv],
                "config": vars(args),
                "runtime": {
                    "device": str(device),
                    "amp_enabled": bool(amp_enabled),
                    "distributed": bool(distributed),
                    "world_size": int(world_size),
                    "rank": int(rank),
                    "local_rank": int(local_rank),
                    "cuda_device_count": int(cuda_device_count),
                    "torch_version": str(torch.__version__),
                },
                "data": {
                    "data_dir": str(Path(args.data_dir).resolve()),
                    "subset_count": int(store.subset_count),
                    "scales": int(store.scales),
                    "height": int(store.height),
                    "width": int(store.width),
                    "width_per_scale": [int(v) for v in store.width_per_scale],
                    "split_train": splits.train,
                    "split_val": splits.val,
                    "split_test": splits.test,
                    "train_split_local": train_split,
                    "val_split_local": val_split,
                    "test_split_local": test_split,
                    "vix_enabled": bool(vix_enabled),
                    "vix_daily_csv": str(vix_daily_csv),
                    "dataset_include_vix_image": bool(store.include_vix_image),
                    "vix_image_enabled": bool(vix_image_enabled),
                    "vix_image_height": int(store.vix_image_height),
                    "vix_image_width": int(store.vix_image_width),
                    "vix_norm_train_split": vix_norm_meta,
                },
                "model": {
                    "msf_scale_weights": [float(v) for v in model_core.msf_scale_weights],
                    "active_scale_indices": [int(v) for v in model_core.active_scale_indices],
                    "active_scale_weights": [float(v) for v in model_core.active_scale_weights],
                    "disabled_scale_indices": [int(v) for v in model_core.disabled_scale_indices],
                    "latent_dim_per_scale": int(args.latent_dim_per_scale),
                    "head_dims": [int(v) for v in HEAD_DIMS],
                    "readout_dim": int(args.readout_dim),
                    "stochastic_train_enabled": bool(stochastic_train_enabled),
                    "vix_fusion_mode": str(vix_fusion_mode),
                    "vix_embed_dim": int(args.vix_embed_dim),
                    "vix_image_embed_dim": int(args.vix_image_embed_dim),
                    "vix_pooling": [int(v) for v in VIX_POOLING],
                    "param_count": count_model_parameters(model_core),
                },
            }
            run_config_path.write_text(json.dumps(run_config_payload, indent=2), encoding="utf-8")
        if distributed:
            dist.barrier()

        if main_process:
            print(
                "[info] data",
                {
                    "data_dir": str(Path(args.data_dir).resolve()),
                    "subset_count": int(store.subset_count),
                    "scales": int(store.scales),
                    "height": int(store.height),
                    "width": int(store.width),
                    "width_per_scale": [int(v) for v in store.width_per_scale],
                    "split_train": splits.train,
                    "split_val": splits.val,
                    "split_test": splits.test,
                    "train_split_local": train_split,
                    "val_split_local": val_split,
                    "test_split_local": test_split,
                    "vix_enabled": bool(vix_enabled),
                    "vix_daily_csv": str(vix_daily_csv),
                    "vix_norm_train_split": vix_norm_meta,
                    "dataset_include_vix_image": bool(store.include_vix_image),
                    "vix_image_enabled": bool(vix_image_enabled),
                    "vix_image_height": int(store.vix_image_height),
                    "vix_image_width": int(store.vix_image_width),
                },
            )
            print(
                "[info] model",
                {
                    "msf_scale_weights": [float(v) for v in model_core.msf_scale_weights],
                    "active_scale_indices": [int(v) for v in model_core.active_scale_indices],
                    "disabled_scale_indices": [int(v) for v in model_core.disabled_scale_indices],
                    "latent_dim_per_scale": int(args.latent_dim_per_scale),
                    "amp_enabled": bool(amp_enabled),
                    "stochastic_train_enabled": bool(stochastic_train_enabled),
                    "multi_gpu_enabled": bool(multi_gpu_enabled),
                    "cuda_device_count": int(cuda_device_count),
                    "distributed": bool(distributed),
                    "world_size": int(world_size),
                    "rank": int(rank),
                    "local_rank": int(local_rank),
                    "ddp": bool(distributed),
                    "vix_fusion_mode": str(vix_fusion_mode),
                    "vix_embed_dim": int(args.vix_embed_dim),
                    "vix_image_enabled": bool(vix_image_enabled),
                    "vix_image_embed_dim": int(args.vix_image_embed_dim),
                    "vix_pooling": [int(v) for v in VIX_POOLING],
                    "param_count": count_model_parameters(model_core),
                },
            )

        history: list[dict[str, float | int]] = []
        best_val_loss = float("inf")
        best_epoch = -1

        for epoch in range(1, int(args.epochs) + 1):
            train_metrics = run_epoch(
                model=model,
                store=store,
                split_range=train_split,
                batch_size=int(args.batch_size),
                device=device,
                amp_enabled=bool(amp_enabled),
                scaler=scaler,
                beta_kl=float(args.beta_kl),
                train=True,
                stochastic_train_enabled=bool(stochastic_train_enabled),
                optimizer=optimizer,
                class_weights=class_weights,
                seed=int(args.seed) + epoch * 101 + int(rank) * 100003,
                progress_label=f"Epoch {epoch}/{int(args.epochs)} - train",
                vix_enabled=bool(vix_enabled),
                vix_image_enabled=bool(vix_image_enabled),
                distributed=bool(distributed),
                rank=int(rank),
            )
            val_metrics = run_epoch(
                model=model,
                store=store,
                split_range=val_split,
                batch_size=int(args.batch_size),
                device=device,
                amp_enabled=bool(amp_enabled),
                scaler=None,
                beta_kl=float(args.beta_kl),
                train=False,
                stochastic_train_enabled=False,
                optimizer=None,
                class_weights=class_weights,
                seed=int(args.seed) + 1000 + epoch + int(rank) * 100003,
                progress_label=f"Epoch {epoch}/{int(args.epochs)} - val",
                vix_enabled=bool(vix_enabled),
                vix_image_enabled=bool(vix_image_enabled),
                distributed=bool(distributed),
                rank=int(rank),
            )
            row = {
                "epoch": int(epoch),
                "train_loss": float(train_metrics["loss"]),
                "train_ce_loss": float(train_metrics["ce_loss"]),
                "train_kl_loss": float(train_metrics["kl_loss"]),
                "train_acc": float(train_metrics["acc"]),
                "train_pred_pos_rate": float(train_metrics["pred_pos_rate"]),
                "train_true_pos_rate": float(train_metrics["true_pos_rate"]),
                "train_tp": int(train_metrics["tp"]),
                "train_tn": int(train_metrics["tn"]),
                "train_fp": int(train_metrics["fp"]),
                "train_fn": int(train_metrics["fn"]),
                "val_loss": float(val_metrics["loss"]),
                "val_ce_loss": float(val_metrics["ce_loss"]),
                "val_kl_loss": float(val_metrics["kl_loss"]),
                "val_acc": float(val_metrics["acc"]),
                "val_pred_pos_rate": float(val_metrics["pred_pos_rate"]),
                "val_true_pos_rate": float(val_metrics["true_pos_rate"]),
                "val_tp": int(val_metrics["tp"]),
                "val_tn": int(val_metrics["tn"]),
                "val_fp": int(val_metrics["fp"]),
                "val_fn": int(val_metrics["fn"]),
            }
            if main_process:
                history.append(row)
                print(
                    f"[epoch {epoch:03d}] "
                    f"train_loss={row['train_loss']:.6f} "
                    f"train_ce={row['train_ce_loss']:.6f} "
                    f"train_kl={row['train_kl_loss']:.6f} "
                    f"train_acc={row['train_acc']:.4f} "
                    f"train_p1={row['train_pred_pos_rate']:.4f} "
                    f"train_conf=(tp={row['train_tp']},tn={row['train_tn']},fp={row['train_fp']},fn={row['train_fn']}) "
                    f"val_loss={row['val_loss']:.6f} "
                    f"val_ce={row['val_ce_loss']:.6f} "
                    f"val_kl={row['val_kl_loss']:.6f} "
                    f"val_acc={row['val_acc']:.4f} "
                    f"val_p1={row['val_pred_pos_rate']:.4f} "
                    f"val_conf=(tp={row['val_tp']},tn={row['val_tn']},fp={row['val_fp']},fn={row['val_fn']})"
                )
            if row["val_loss"] < best_val_loss:
                best_val_loss = float(row["val_loss"])
                best_epoch = int(epoch)
                if main_process:
                    torch.save(
                        {
                            "epoch": int(epoch),
                            "model_state_dict": unwrap_model(model).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_val_loss": float(best_val_loss),
                        },
                        best_path,
                    )

        if distributed:
            dist.barrier()
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device)
            unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
        if distributed:
            dist.barrier()

        val_pred_path = preds_dir / "val_preds_best.npz"
        val_pred_ranked_csv_path = preds_dir / "val_preds_best_ranked_prob1.csv"
        if main_process:
            val_pred = collect_split_predictions(
                model=unwrap_model(model),
                store=store,
                split_range=splits.val,
                batch_size=int(args.batch_size),
                device=device,
                amp_enabled=bool(amp_enabled),
                seed=int(args.seed) + 3000,
                progress_label="Collect val preds",
                vix_enabled=bool(vix_enabled),
                vix_image_enabled=bool(vix_image_enabled),
                distributed=False,
                rank=0,
            )
            np.savez_compressed(
                val_pred_path,
                sample_index=val_pred["sample_index"],
                ticker_id=val_pred["ticker_id"],
                ticker=val_pred["ticker"],
                y_true=val_pred["y_true"],
                pred=val_pred["pred"],
                prob_1=val_pred["prob_1"],
                ret_pct_true=val_pred["ret_pct_true"],
                timestamp=val_pred["timestamp"],
            )
            rank_order = np.argsort(val_pred["prob_1"], kind="mergesort")[::-1]
            with val_pred_ranked_csv_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(
                    [
                        "sample_index",
                        "ticker_id",
                        "ticker",
                        "timestamp",
                        "y_true",
                        "pred",
                        "prob_1",
                        "ret_pct_true",
                    ]
                )
                for idx in rank_order.tolist():
                    writer.writerow(
                        [
                            int(val_pred["sample_index"][idx]),
                            int(val_pred["ticker_id"][idx]),
                            str(val_pred["ticker"][idx]),
                            str(val_pred["timestamp"][idx]),
                            int(val_pred["y_true"][idx]),
                            int(val_pred["pred"][idx]),
                            float(val_pred["prob_1"][idx]),
                            float(val_pred["ret_pct_true"][idx]),
                        ]
                    )
        if distributed:
            dist.barrier()

        test_metrics = run_epoch(
            model=model,
            store=store,
            split_range=test_split,
            batch_size=int(args.batch_size),
            device=device,
            amp_enabled=bool(amp_enabled),
            scaler=None,
            beta_kl=float(args.beta_kl),
            train=False,
            stochastic_train_enabled=False,
            optimizer=None,
            class_weights=class_weights,
            seed=int(args.seed) + 2000 + int(rank) * 100003,
            progress_label="Test",
            vix_enabled=bool(vix_enabled),
            vix_image_enabled=bool(vix_image_enabled),
            distributed=bool(distributed),
            rank=int(rank),
        )

        if main_process:
            summary = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": vars(args),
                "data": {
                    "data_dir": str(Path(args.data_dir).resolve()),
                    "subset_count": int(store.subset_count),
                    "scales": int(store.scales),
                    "height": int(store.height),
                    "width": int(store.width),
                    "width_per_scale": [int(v) for v in store.width_per_scale],
                    "ret_atr_threshold": float(args.ret_atr_threshold),
                    "vix_enabled": bool(vix_enabled),
                    "vix_daily_csv": str(vix_daily_csv),
                    "vix_norm_train_split": vix_norm_meta,
                    "dataset_include_vix_image": bool(store.include_vix_image),
                    "vix_image_enabled": bool(vix_image_enabled),
                    "vix_image_height": int(store.vix_image_height),
                    "vix_image_width": int(store.vix_image_width),
                },
                "splits": asdict(splits),
                "model": {
                    "msf_scale_weights": [float(v) for v in model_core.msf_scale_weights],
                    "active_scale_indices": [int(v) for v in model_core.active_scale_indices],
                    "active_scale_weights": [float(v) for v in model_core.active_scale_weights],
                    "disabled_scale_indices": [int(v) for v in model_core.disabled_scale_indices],
                    "latent_dim_per_scale": int(args.latent_dim_per_scale),
                    "head_dims": [int(v) for v in HEAD_DIMS],
                    "readout_dim": int(args.readout_dim),
                    "amp_enabled": bool(amp_enabled),
                    "stochastic_train_enabled": bool(stochastic_train_enabled),
                    "multi_gpu_enabled": bool(multi_gpu_enabled),
                    "cuda_device_count": int(cuda_device_count),
                    "distributed": bool(distributed),
                    "world_size": int(world_size),
                    "ddp": bool(distributed),
                    "vix_fusion_mode": str(vix_fusion_mode),
                    "vix_embed_dim": int(args.vix_embed_dim),
                    "vix_image_enabled": bool(vix_image_enabled),
                    "vix_image_embed_dim": int(args.vix_image_embed_dim),
                    "vix_pooling": [int(v) for v in VIX_POOLING],
                    "param_count": count_model_parameters(model_core),
                },
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val_loss),
                "history": history,
                "test_metrics": test_metrics,
                "artifacts": {
                    "run_config": str(run_config_path),
                    "best_checkpoint": str(best_path),
                    "val_preds_best": str(val_pred_path),
                    "val_preds_best_ranked_prob1_csv": str(val_pred_ranked_csv_path),
                },
            }
            summary_path = out_dir / "metrics.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[done] wrote {summary_path}")
        if distributed:
            dist.barrier()
    finally:
        if distributed:
            distributed_cleanup()


if __name__ == "__main__":
    main()
