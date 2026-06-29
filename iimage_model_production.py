#!/usr/bin/env python3
"""Minimal production inference for decomp-image checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

LRELU_SLOPE = 0.05
CONV_CHANNELS = (64, 128)
KERNEL_SIZE = (5, 3)
POOL_KERNEL = (2, 1)
POOL_STRIDE = (2, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=str,
        default="production_weights",
        help="Checkpoint path or directory. Directory mode prefers best.pt then last.pt.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="production_datasets",
        help=(
            "Root production dataset directory. When --dataset-folder is provided, "
            "the effective dataset path becomes <data-dir>/<dataset-folder>."
        ),
    )
    parser.add_argument(
        "--dataset-folder",
        type=str,
        default=None,
        help=(
            "Dataset subfolder name inside --data-dir to score "
            "(for example: 2026-03-13)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="production_predictions",
        help="Root directory for date-named prediction folders.",
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, cuda:0, ...",
    )
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    token = str(raw).strip().lower()
    if token in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(token)


def resolve_data_dir(raw_data_dir: str | Path, dataset_folder: str | None) -> tuple[Path, str | None]:
    data_root = Path(raw_data_dir).resolve()
    folder_raw = None if dataset_folder is None else str(dataset_folder).strip()
    if not folder_raw:
        return data_root, None

    folder_path = Path(folder_raw)
    if folder_path.is_absolute():
        raise ValueError("--dataset-folder must be relative to --data-dir")

    data_dir = (data_root / folder_path).resolve()
    try:
        data_dir.relative_to(data_root)
    except ValueError as exc:
        raise ValueError("--dataset-folder must stay inside --data-dir") from exc
    return data_dir, folder_raw


class MsfBlock(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        pad = (KERNEL_SIZE[0] // 2, KERNEL_SIZE[1] // 2)
        self.conv1 = nn.Conv2d(1, CONV_CHANNELS[0], kernel_size=KERNEL_SIZE, padding=pad)
        self.act1 = nn.LeakyReLU(LRELU_SLOPE)
        self.conv2 = nn.Conv2d(
            CONV_CHANNELS[0], CONV_CHANNELS[1], kernel_size=KERNEL_SIZE, padding=pad
        )
        self.act2 = nn.LeakyReLU(LRELU_SLOPE)
        self.pool1 = nn.MaxPool2d(kernel_size=POOL_KERNEL, stride=POOL_STRIDE)
        self.conv3 = nn.Conv2d(
            CONV_CHANNELS[1], int(out_channels), kernel_size=KERNEL_SIZE, padding=pad
        )
        self.act3 = nn.LeakyReLU(LRELU_SLOPE)
        self.pool2 = nn.MaxPool2d(kernel_size=POOL_KERNEL, stride=POOL_STRIDE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.pool1(x)
        x = self.act3(self.conv3(x))
        x = self.pool2(x)
        return x


class DecompImageClassifier(nn.Module):
    def __init__(
        self,
        *,
        scales: int,
        input_height: int,
        input_width: int,
        readout_dim: int,
        head_weight_indices: list[int],
        head_dims: list[int],
        fc_dropout: float,
        per_scale_out_channels: list[int],
    ) -> None:
        super().__init__()
        self.scales = int(scales)
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.msf_blocks = nn.ModuleList(
            [MsfBlock(out_channels=out_ch) for out_ch in per_scale_out_channels]
        )

        with torch.no_grad():
            reps = [
                block(torch.zeros(1, 1, self.input_height, self.input_width))
                for block in self.msf_blocks
            ]
            rep = torch.cat(reps, dim=1)
            flat_dim = int(np.prod(rep.shape[1:]))

        self.readout = nn.Linear(flat_dim, int(readout_dim))
        self.readout_act = nn.LeakyReLU(LRELU_SLOPE)
        self.readout_drop = nn.Dropout(float(fc_dropout)) if fc_dropout > 0 else nn.Identity()
        self.cls_head = self._build_cls_head(
            readout_dim=int(readout_dim),
            head_weight_indices=head_weight_indices,
            head_dims=head_dims,
            fc_dropout=float(fc_dropout),
        )

    @staticmethod
    def _build_cls_head(
        *,
        readout_dim: int,
        head_weight_indices: list[int],
        head_dims: list[int],
        fc_dropout: float,
    ) -> nn.Sequential:
        if not head_weight_indices:
            raise ValueError("checkpoint is missing cls_head linear weights")
        if head_weight_indices[0] != 0:
            raise ValueError(
                f"expected first cls_head linear index to be 0, got {head_weight_indices[0]}"
            )

        dims = [int(readout_dim)] + [int(v) for v in head_dims] + [2]
        modules: list[nn.Module] = []
        for layer_idx in range(len(dims) - 2):
            modules.append(nn.Linear(dims[layer_idx], dims[layer_idx + 1]))
            modules.append(nn.LeakyReLU(LRELU_SLOPE))
            gap = head_weight_indices[layer_idx + 1] - head_weight_indices[layer_idx]
            if gap == 3:
                modules.append(nn.Dropout(float(fc_dropout)))
            elif gap != 2:
                raise ValueError(
                    f"unsupported cls_head layout gap {gap} between linear indices "
                    f"{head_weight_indices[layer_idx]} and {head_weight_indices[layer_idx + 1]}"
                )
        modules.append(nn.Linear(dims[-2], dims[-1]))
        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected input shape (batch, scales, H, W), got {tuple(x.shape)}")
        if int(x.shape[1]) != self.scales:
            raise ValueError(f"expected {self.scales} scales, got {int(x.shape[1])}")
        reps = []
        for scale_idx, block in enumerate(self.msf_blocks):
            xi = x[:, scale_idx : scale_idx + 1, :, : self.input_width]
            reps.append(block(xi))
        rep = torch.cat(reps, dim=1)
        rep = rep.flatten(start_dim=1)
        rep = self.readout_drop(self.readout_act(self.readout(rep)))
        return self.cls_head(rep)


def load_checkpoint(weights_path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        config = {}
    else:
        raise TypeError(f"unsupported checkpoint type: {type(checkpoint).__name__}")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"checkpoint has no usable model state: {weights_path}")
    return state_dict, config if isinstance(config, dict) else {}


def resolve_weights_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).resolve()
    if path.is_file():
        return path
    if path.is_dir():
        for name in ("best.pt", "last.pt"):
            candidate = path / name
            if candidate.is_file():
                return candidate
        matches = sorted(path.glob("*.pt"))
        if len(matches) == 1:
            return matches[0].resolve()
        if matches:
            raise FileNotFoundError(
                f"weights directory has multiple .pt files and no best.pt/last.pt: {path}"
            )
    raise FileNotFoundError(f"weights not found: {path}")


def infer_model_spec(
    state_dict: dict[str, torch.Tensor],
    config: dict,
    input_height: int,
    input_width: int,
) -> dict:
    if any(key.startswith(("vix_film.", "vix_embed.", "vix_image_block.")) for key in state_dict):
        raise ValueError("VIX-enabled checkpoints are not supported by this minimal production script")

    block_indices = sorted(
        {
            int(match.group(1))
            for key in state_dict
            if (match := re.match(r"msf_blocks\.(\d+)\.conv1\.weight$", key))
        }
    )
    if not block_indices:
        raise ValueError("checkpoint is missing msf_blocks.*.conv1.weight entries")
    if block_indices != list(range(len(block_indices))):
        raise ValueError(f"unexpected msf block indices: {block_indices}")

    per_scale_out_channels = [
        int(state_dict[f"msf_blocks.{idx}.conv3.weight"].shape[0]) for idx in block_indices
    ]
    readout_dim = int(state_dict["readout.weight"].shape[0])
    head_weight_indices = sorted(
        int(match.group(1))
        for key in state_dict
        if (match := re.match(r"cls_head\.(\d+)\.weight$", key))
    )
    if not head_weight_indices:
        raise ValueError("checkpoint is missing cls_head.*.weight entries")

    head_dims = [
        int(state_dict[f"cls_head.{idx}.weight"].shape[0]) for idx in head_weight_indices[:-1]
    ]
    fc_dropout = float(config.get("fc_dropout", 0.5 if any(
        head_weight_indices[i + 1] - head_weight_indices[i] == 3
        for i in range(len(head_weight_indices) - 1)
    ) else 0.0))

    return {
        "scales": int(len(block_indices)),
        "input_height": int(input_height),
        "input_width": int(input_width),
        "readout_dim": int(readout_dim),
        "head_weight_indices": head_weight_indices,
        "head_dims": head_dims,
        "fc_dropout": float(fc_dropout),
        "per_scale_out_channels": per_scale_out_channels,
    }


def read_manifest(data_dir: Path) -> dict:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must be a JSON object: {manifest_path}")
    return manifest


def iter_shard_entries(manifest: dict) -> Iterable[dict]:
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("manifest is missing shard metadata")
    for entry in shards:
        if not isinstance(entry, dict) or "file" not in entry:
            raise ValueError(f"invalid shard entry in manifest: {entry!r}")
        yield entry


def latest_sample_date(data_dir: Path, manifest: dict) -> str:
    shards = list(iter_shard_entries(manifest))
    last_path = data_dir / str(shards[-1]["file"])
    with np.load(last_path, allow_pickle=False) as shard:
        timestamps = shard["timestamps"]
        if timestamps.size == 0:
            raise ValueError(f"last shard has no timestamps: {last_path}")
        return normalize_date(str(timestamps[-1]))


def normalize_date(raw: str) -> str:
    token = str(raw).strip()
    if not token:
        raise ValueError("timestamp is empty")
    if "T" in token:
        token = token.split("T", 1)[0]
    if " " in token:
        token = token.split(" ", 1)[0]
    return token


def build_prediction_write_order(
    timestamps: np.ndarray,
    prob1: np.ndarray,
) -> np.ndarray:
    ts_arr = np.asarray(timestamps).astype(str, copy=False).reshape(-1)
    prob_arr = np.asarray(prob1, dtype=np.float64).reshape(-1)
    if int(ts_arr.shape[0]) != int(prob_arr.shape[0]):
        raise ValueError(
            "timestamp/probability length mismatch: "
            f"timestamps={int(ts_arr.shape[0])} prob1={int(prob_arr.shape[0])}"
        )
    n = int(ts_arr.shape[0])
    if n <= 0:
        return np.empty((0,), dtype=np.int64)
    date_keys = np.asarray([normalize_date(ts) for ts in ts_arr], dtype=object)
    row_order = np.arange(n, dtype=np.int64)
    return np.lexsort((row_order, -prob_arr, date_keys))


def load_ticker_symbols(data_dir: Path) -> np.ndarray:
    tickers_path = data_dir / "tickers.npy"
    if tickers_path.is_file():
        tickers = np.load(tickers_path, allow_pickle=True)
        return np.asarray(tickers, dtype=object)
    return np.empty((0,), dtype=object)


def ticker_symbol(tickers: np.ndarray, ticker_id: int) -> str:
    if 0 <= int(ticker_id) < int(tickers.shape[0]):
        return str(tickers[int(ticker_id)])
    return ""


def build_model(weights_path: Path, data_dir: Path) -> tuple[DecompImageClassifier, dict, dict]:
    state_dict, config = load_checkpoint(weights_path)
    manifest = read_manifest(data_dir)
    input_height = int(manifest.get("image_height", 0) or 0)
    input_width = int(manifest.get("image_width", 0) or 0)
    if input_height < 1 or input_width < 1:
        first_entry = next(iter_shard_entries(manifest))
        with np.load(data_dir / str(first_entry["file"]), allow_pickle=False) as shard:
            x = shard["X_img"]
            if x.ndim != 4:
                raise ValueError(f"expected shard X_img shape (N,S,H,W), got {tuple(x.shape)}")
            input_height = int(x.shape[2])
            input_width = int(x.shape[3])

    spec = infer_model_spec(
        state_dict=state_dict,
        config=config,
        input_height=input_height,
        input_width=input_width,
    )
    model = DecompImageClassifier(**spec)
    filtered_state = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(("ts_block.", "ts_projection.", "regression_map_fusion.", "reg_readout.", "reg_head."))
    }
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    if missing:
        raise ValueError(f"checkpoint did not populate required model weights: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"unexpected checkpoint weights for minimal model: {sorted(unexpected)}")
    return model, manifest, spec


def run_inference(
    *,
    model: nn.Module,
    manifest: dict,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "production_preds.csv"
    summary_path = output_dir / "production_preds_summary.json"
    tickers = load_ticker_symbols(data_dir)

    model.eval()
    model.to(device)

    total_rows = 0
    prob1_sum = 0.0
    pred1_sum = 0
    label_cols_raw = manifest.get("label_cols")
    label_cols = [str(col) for col in label_cols_raw] if isinstance(label_cols_raw, list) else []
    ret_pct_index = label_cols.index("ret_pct") if "ret_pct" in label_cols else None
    all_sample_indices: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    all_ticker_ids: list[np.ndarray] = []
    all_pred_cls: list[np.ndarray] = []
    all_prob1: list[np.ndarray] = []
    all_ret_pct: list[np.ndarray] = []
    with torch.inference_mode():
        for entry in iter_shard_entries(manifest):
            shard_path = data_dir / str(entry["file"])
            with np.load(shard_path, allow_pickle=False) as shard:
                x = shard["X_img"]
                timestamps = shard["timestamps"].astype(str, copy=False)
                if "sample_indices" in shard.files:
                    sample_indices = shard["sample_indices"].astype(np.int64, copy=False)
                else:
                    start = int(entry.get("sample_start", 0))
                    sample_indices = np.arange(start, start + int(x.shape[0]), dtype=np.int64)
                if "ticker_ids" in shard.files:
                    ticker_ids = shard["ticker_ids"].astype(np.int64, copy=False)
                else:
                    ticker_ids = np.full((int(x.shape[0]),), -1, dtype=np.int64)
                ret_pct = None
                if ret_pct_index is not None and "y_raw" in shard.files:
                    y_raw = shard["y_raw"]
                    if y_raw.ndim >= 2 and ret_pct_index < int(y_raw.shape[1]):
                        ret_pct = y_raw[:, ret_pct_index].astype(np.float32, copy=False)

                for start in range(0, int(x.shape[0]), int(batch_size)):
                    end = min(start + int(batch_size), int(x.shape[0]))
                    xb = torch.from_numpy(x[start:end].astype(np.float32, copy=False)).to(device)
                    logits = model(xb)
                    probs = torch.softmax(logits, dim=1)
                    pred_cls = torch.argmax(probs, dim=1)

                    prob1_np = probs[:, 1].cpu().numpy().astype(np.float32, copy=False)
                    pred_cls_np = pred_cls.cpu().numpy().astype(np.int64, copy=False)
                    sample_indices_np = sample_indices[start:end]
                    timestamps_np = timestamps[start:end]
                    ticker_ids_np = ticker_ids[start:end]
                    ret_pct_np = (
                        ret_pct[start:end]
                        if ret_pct is not None
                        else np.full((int(end - start),), np.nan, dtype=np.float32)
                    )

                    total_rows += int(end - start)
                    prob1_sum += float(prob1_np.sum(dtype=np.float64))
                    pred1_sum += int(pred_cls_np.sum())

                    all_sample_indices.append(sample_indices_np)
                    all_timestamps.append(timestamps_np)
                    all_ticker_ids.append(ticker_ids_np)
                    all_pred_cls.append(pred_cls_np)
                    all_prob1.append(prob1_np)
                    all_ret_pct.append(ret_pct_np)

    sample_indices_all = np.concatenate(all_sample_indices, axis=0) if all_sample_indices else np.empty((0,), dtype=np.int64)
    timestamps_all = np.concatenate(all_timestamps, axis=0) if all_timestamps else np.empty((0,), dtype=object)
    ticker_ids_all = np.concatenate(all_ticker_ids, axis=0) if all_ticker_ids else np.empty((0,), dtype=np.int64)
    pred_cls_all = np.concatenate(all_pred_cls, axis=0) if all_pred_cls else np.empty((0,), dtype=np.int64)
    prob1_all = np.concatenate(all_prob1, axis=0) if all_prob1 else np.empty((0,), dtype=np.float32)
    ret_pct_all = np.concatenate(all_ret_pct, axis=0) if all_ret_pct else np.empty((0,), dtype=np.float32)
    order = build_prediction_write_order(timestamps=timestamps_all, prob1=prob1_all)
    include_ret_pct = ret_pct_index is not None and int(ret_pct_all.shape[0]) == int(total_rows)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        header = [
            "sample_index",
            "timestamp",
            "ticker_id",
            "ticker",
            "pred_class",
            "prob_1",
        ]
        if include_ret_pct:
            header.append("ret_pct")
        writer.writerow(header)
        writer.writerows(
            [
                ([
                    int(sample_indices_all[row_idx]),
                    str(timestamps_all[row_idx]),
                    int(ticker_ids_all[row_idx]),
                    ticker_symbol(tickers, int(ticker_ids_all[row_idx])),
                    int(pred_cls_all[row_idx]),
                    float(prob1_all[row_idx]),
                ] + ([float(ret_pct_all[row_idx])] if include_ret_pct else []))
                for row_idx in order
            ]
        )

    summary = {
        "predictions_csv": csv_path.name,
        "num_rows": int(total_rows),
        "device": str(device),
        "mean_prob_1": float(prob1_sum / total_rows) if total_rows else float("nan"),
        "pred_class_1_count": int(pred1_sum),
        "pred_class_1_rate": float(pred1_sum / total_rows) if total_rows else float("nan"),
        "ret_pct_in_predictions_csv": bool(include_ret_pct),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    weights_path = resolve_weights_path(args.weights)
    data_dir, dataset_folder = resolve_data_dir(args.data_dir, args.dataset_folder)
    output_root = Path(args.output_root).resolve()
    device = resolve_device(args.device)

    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found: {data_dir}")
    if int(args.batch_size) < 1:
        raise ValueError("--batch-size must be >= 1")

    model, manifest, spec = build_model(weights_path=weights_path, data_dir=data_dir)
    latest_date = latest_sample_date(data_dir=data_dir, manifest=manifest)
    output_dir = output_root / latest_date
    summary = run_inference(
        model=model,
        manifest=manifest,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
        batch_size=int(args.batch_size),
    )

    summary.update(
        {
            "weights": str(weights_path),
            "data_dir": str(data_dir),
            "data_root": str(Path(args.data_dir).resolve()),
            "dataset_folder": dataset_folder,
            "output_dir": str(output_dir),
            "latest_sample_date": latest_date,
            "model_spec": spec,
        }
    )
    (output_dir / "production_preds_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
