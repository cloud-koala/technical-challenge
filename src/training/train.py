from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.dataloader import (
    Part3Metadata,
    SampleMeta,
    SubjectWindowDataset,
    WindowedSignalDataset,
    build_label_mapping,
    balance_samples_by_condition,
    loso_folds,
    read_signal_csv,
    sensor_class_distribution,
    save_label_mapping,
    split_by_sensor_id,
    validate_no_sensor_leakage,
    standardize_orientation,
)
from src.model.conv import ConvNetClassifier, ConvNetConfig
from src.model.conv_lstm import ConvLSTMClassifier, ConvLSTMConfig
from src.model.linear import LinearClassifier, LinearConfig
from src.model.window_hier_lstm import WindowHierLSTMClassifier, WindowHierLSTMConfig
from src.preprocessing.preprocessor import Preprocessor
from src.training.config import load_config
from src.training.metrics import compute_metrics


def _apply_time_mask_(
    x: torch.Tensor,
    *,
    prob: float,
    ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """In-place time masking over the last dimension.

    Supports:
      - (B, C, L)
      - (B, Nw, C, L)
    """
    if prob <= 0.0 or ratio <= 0.0:
        return x
    if x.dim() not in (3, 4):
        return x

    length = int(x.shape[-1])
    mask_len = int(round(length * float(ratio)))
    if mask_len <= 0 or mask_len >= length:
        return x

    b = int(x.shape[0])
    # Generate randomness on CPU to avoid device-specific generator constraints (e.g. MPS).
    to_mask = torch.rand((b,), generator=generator) < float(prob)
    idxs = to_mask.nonzero(as_tuple=False).view(-1).tolist()
    if not idxs:
        return x

    for i in idxs:
        start = int(torch.randint(0, length - mask_len + 1, (1,), generator=generator).item())
        x[i, ... , start : start + mask_len] = 0.0
    return x


def _mixup(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """Return mixup(x, y) and (y_a, y_b, lam, perm)."""
    if alpha <= 0.0:
        perm = torch.arange(x.shape[0], device=x.device)
        return x, y, y, 1.0, perm
    # Sample lam from Beta(alpha, alpha).
    beta = torch.distributions.Beta(concentration1=float(alpha), concentration0=float(alpha))
    lam = float(beta.sample((1,)).item())

    # Generate permutation on CPU for broad device compatibility, then move indices.
    perm = torch.randperm(x.shape[0], generator=generator).to(device=x.device)
    x2 = x.index_select(0, perm)
    y2 = y.index_select(0, perm)
    x_mix = x.mul(lam).add(x2, alpha=(1.0 - lam))
    return x_mix, y, y2, lam, perm


def _select_device(device: str) -> torch.device:
    device = device.lower()
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda")
    if device == "mps":
        return torch.device("mps")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _WithIndex(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        item = self.base[index]
        if not isinstance(item, (tuple, list)) or len(item) not in (2, 3):
            raise ValueError("Base dataset must return (x, y) or (x, y, rpm) to be wrapped with indices")
        if len(item) == 2:
            x, y = item
            return x, y, index
        x, y, r = item
        return x, y, r, index


@torch.no_grad()
def _predict_probs_with_index(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[List[int], List[int], np.ndarray]:
    model.eval()
    y_true: List[int] = []
    indices: List[int] = []
    probs_all: List[List[float]] = []
    for batch in loader:
        if len(batch) == 3:
            xb, yb, idxb = batch
            rb = None
        else:
            xb, yb, rb, idxb = batch

        xb = xb.to(device)
        if rb is not None:
            rb = rb.to(device)
        logits = model(xb, rpm=rb)
        probs = torch.softmax(logits, dim=1).cpu().tolist()
        probs_all.extend(probs)
        y_true.extend(yb.cpu().tolist())
        # idxb will be a tensor after collation
        indices.extend(idxb.cpu().tolist())
    return y_true, indices, np.asarray(probs_all, dtype=np.float32)


def _per_sample_predictions_rows(
    *,
    split_name: str,
    dataset: Dataset,
    label_to_index: Dict[str, int],
    y_true: List[int],
    indices: List[int],
    probs: np.ndarray,
) -> List[Dict[str, Any]]:
    index_to_label = {v: k for k, v in label_to_index.items()}

    # Map base-dataset index -> sample_idx
    sample_indices: List[int] = []
    if hasattr(dataset, "_windows") and hasattr(dataset, "samples"):
        # WindowedSignalDataset: index refers to a window; map to (sample_idx, window_idx)
        windows = getattr(dataset, "_windows")
        for wi in indices:
            sample_indices.append(int(windows[int(wi)][0]))
    elif hasattr(dataset, "samples"):
        # SubjectWindowDataset: one item per sample
        sample_indices = [int(i) for i in indices]
    else:
        raise ValueError("Unsupported dataset type for per-sample prediction logging")

    num_classes = int(probs.shape[1])
    sum_probs: Dict[int, np.ndarray] = {}
    counts: Dict[int, int] = {}
    for si, p in zip(sample_indices, probs):
        if si not in sum_probs:
            sum_probs[si] = np.zeros((num_classes,), dtype=np.float32)
            counts[si] = 0
        sum_probs[si] += p
        counts[si] += 1

    # Choose stable output order: by sample_id
    samples = getattr(dataset, "samples")
    rows: List[Dict[str, Any]] = []
    for si in sorted(sum_probs.keys(), key=lambda i: str(samples[int(i)].sample_id)):
        meta = samples[int(si)]
        avg = sum_probs[int(si)] / float(counts[int(si)])
        pred_idx = int(np.argmax(avg))
        rows.append(
            {
                "split": str(split_name),
                "sample_id": str(meta.sample_id),
                "sensor_id": None if meta.sensor_id is None else str(meta.sensor_id),
                "rpm": None if meta.rpm is None else float(meta.rpm),
                "orientation": json.dumps(meta.orientation, sort_keys=True),
                "true_label": None if meta.condition is None else str(meta.condition),
                "pred_label": str(index_to_label[pred_idx]),
            }
        )
    return rows


def _write_predictions_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "sample_id", "sensor_id", "rpm", "orientation", "true_label", "pred_label"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, None) for k in fieldnames})


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    for batch in loader:
        if len(batch) == 2:
            xb, yb = batch
            rb = None
        else:
            xb, yb, rb = batch

        xb = xb.to(device)
        if rb is not None:
            rb = rb.to(device)
        logits = model(xb, rpm=rb)
        pred = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(pred)
        # Avoid yb.numpy() for environments where PyTorch NumPy bridge is unavailable.
        y_true.extend(yb.cpu().tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def _train_one(
    *,
    fold_name: str,
    samples_split: Dict[str, List[SampleMeta]],
    label_to_index: Dict[str, int],
    data_dir: Path,
    output_order: List[str],
    cfg,
    out_dir: Path,
    log_root: Optional[Path] = None,
    epoch_callback: Optional[Callable[[int, float, float], None]] = None,
) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)

    if log_root is None:
        # Fall back to logs/<run_name>/ when not provided explicitly.
        artifacts_cfg = getattr(cfg, "artifacts", {})
        run_name = str(artifacts_cfg.get("run_name", "run"))
        log_root = Path("logs") / run_name
    log_dir = Path(log_root) / str(fold_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = log_dir / "debug_plots"
    debug_dir.mkdir(parents=True, exist_ok=True)

    def _log_line(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with (log_dir / "train.log").open("a") as f:
            f.write(f"[{ts}] {msg}\n")

    seed = int(cfg.training.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    pre_cfg = cfg.preprocessing
    preprocessor = Preprocessor(
        downsample_hz=float(pre_cfg.get("downsample_hz", 100.0)),
        window_seconds=float(pre_cfg.get("window_seconds", 0.05)),
        step_seconds=pre_cfg.get("step_seconds", None),
        z_norm=bool(pre_cfg.get("z_norm", True)),
        feature_mode=str(pre_cfg.get("feature_mode", "time")),
        rpm_policy=str(pre_cfg.get("rpm_policy", "auto")),
        rpm_min=float(pre_cfg.get("rpm_min", 300.0)),
        rpm_max=float(pre_cfg.get("rpm_max", 6000.0)),
        rpm_discrepancy_tol=float(pre_cfg.get("rpm_discrepancy_tol", 0.2)),
        rpm_top_k=int(pre_cfg.get("rpm_top_k", 8)),
        rpm_harmonics=int(pre_cfg.get("rpm_harmonics", 5)),
        order_max=float(pre_cfg.get("order_max", 20.0)),
        order_bins=int(pre_cfg.get("order_bins", 128)),
        order_log_power=bool(pre_cfg.get("order_log_power", True)),
        order_per_window_standardize=bool(pre_cfg.get("order_per_window_standardize", True)),
    )

    _log_line(
        "Preprocessor: "
        + json.dumps(
            {
                "feature_mode": str(pre_cfg.get("feature_mode", "time")),
                "downsample_hz": float(pre_cfg.get("downsample_hz", 100.0)),
                "window_seconds": float(pre_cfg.get("window_seconds", 0.05)),
                "step_seconds": pre_cfg.get("step_seconds", None),
                "z_norm": bool(pre_cfg.get("z_norm", True)),
                "rpm_policy": str(pre_cfg.get("rpm_policy", "auto")),
                "rpm_min": float(pre_cfg.get("rpm_min", 300.0)),
                "rpm_max": float(pre_cfg.get("rpm_max", 6000.0)),
                "rpm_discrepancy_tol": float(pre_cfg.get("rpm_discrepancy_tol", 0.2)),
                "order_max": float(pre_cfg.get("order_max", 20.0)),
                "order_bins": int(pre_cfg.get("order_bins", 128)),
            },
            sort_keys=True,
        )
    )

    # Fit normalizer on train samples only (streaming, low-RAM)
    def iter_train_times_and_signals():
        for meta in samples_split["train"]:
            t, sig = read_signal_csv(data_dir / f"{meta.sample_id}.csv")
            sig = standardize_orientation(sig, meta.orientation, output_order=output_order)
            yield t, sig, meta.rpm

    preprocessor.fit(times_and_signals=iter_train_times_and_signals())

    # Debug plots + RPM validation rows (small, fixed number of samples).
    train_cfg = cfg.training
    debug_plots = bool(train_cfg.get("debug_plots", False))
    debug_n_samples = int(train_cfg.get("debug_n_samples", 3))
    debug_max_windows = int(train_cfg.get("debug_max_windows", 200))
    if debug_plots and debug_n_samples > 0:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            rpm_rows: List[Dict[str, object]] = []
            for meta in list(samples_split["train"])[:debug_n_samples]:
                t, sig = read_signal_csv(data_dir / f"{meta.sample_id}.csv")
                sig = standardize_orientation(sig, meta.orientation, output_order=output_order)

                rpm_prov = None if meta.rpm is None else float(meta.rpm)
                rpm_est = preprocessor.estimate_rpm(time=np.asarray(t), signals=np.asarray(sig))
                windows, rpm_used = preprocessor.transform_with_rpm(
                    time=np.asarray(t),
                    signals=np.asarray(sig),
                    rpm=rpm_prov,
                )

                w = windows
                if w.shape[0] > debug_max_windows:
                    w = w[:debug_max_windows]

                # Aggregate for visualization: mean over windows -> (L, C)
                if w.shape[0] == 0:
                    mean_feat = np.zeros((int(getattr(preprocessor, "order_bins", 1)), int(windows.shape[-1])), dtype=np.float32)
                    first_feat = mean_feat
                else:
                    mean_feat = w.mean(axis=0)
                    first_feat = w[0]

                # Save a compact figure describing the actual features fed to the classifier.
                # (Line plots only; heatmap removed for readability.)
                fig = plt.figure(figsize=(12, 7))

                feature_mode = str(getattr(preprocessor, "feature_mode", "time"))
                if feature_mode == "order_spectrum":
                    x_vals = np.linspace(0.0, float(getattr(preprocessor, "order_max", 1.0)), mean_feat.shape[0])
                    x_label = "Order (× shaft speed)"
                    feat_label = "feat (order)"
                else:
                    x_vals = np.arange(mean_feat.shape[0], dtype=np.float64)
                    x_label = "Time index (samples)"
                    feat_label = "feat (time)"

                channel_names = [str(x) for x in (output_order or [])]
                if len(channel_names) != mean_feat.shape[1]:
                    channel_names = [f"ch{c}" for c in range(mean_feat.shape[1])]
                ax1 = fig.add_subplot(2, 1, 1)
                ax1.set_title("Mean feature across windows")
                for c in range(mean_feat.shape[1]):
                    ax1.plot(x_vals, mean_feat[:, c], label=channel_names[c])
                ax1.legend(loc="best", fontsize=8)
                ax1.set_xlabel(x_label)
                ax1.set_ylabel(feat_label)

                ax3 = fig.add_subplot(2, 1, 2)
                ax3.set_title("First window feature")
                for c in range(first_feat.shape[1]):
                    ax3.plot(x_vals, first_feat[:, c], label=channel_names[c])
                ax3.legend(loc="best", fontsize=8)
                ax3.set_xlabel(x_label)
                ax3.set_ylabel(feat_label)

                # Put metadata as a figure footer to avoid axis overlap.
                rel_err = None
                if rpm_prov is not None and rpm_est is not None and rpm_prov > 0:
                    rel_err = abs(float(rpm_est) - float(rpm_prov)) / max(abs(float(rpm_prov)), 1e-6)

                rpm_est_str = "None" if rpm_est is None else f"{float(rpm_est):.3f}"
                true_label = None if meta.condition is None else str(meta.condition)
                txt = (
                    f"sample_id: {meta.sample_id}\n"
                    f"true_label: {true_label}\n"
                    f"rpm_provided: {rpm_prov}\n"
                    f"rpm_estimated: {rpm_est_str}\n"
                    f"rpm_used: {float(rpm_used):.3f}\n"
                    f"feature_mode: {feature_mode}\n"
                    f"windows: {int(windows.shape[0])} (shown up to {debug_max_windows})\n"
                    f"rel_err(prov vs est): {None if rel_err is None else float(rel_err):.3f}"
                )

                # Increase bottom margin for footer text.
                fig.subplots_adjust(bottom=0.22, hspace=0.35)
                fig.text(0.01, 0.01, txt, va="bottom", ha="left", family="monospace", fontsize=9)

                fig.tight_layout(rect=(0, 0.18, 1, 1))
                out_png = debug_dir / f"features_sample_{meta.sample_id}.png"
                fig.savefig(out_png, dpi=150)
                plt.close(fig)

                rpm_rows.append(
                    {
                        "sample_id": str(meta.sample_id),
                        "rpm_provided": rpm_prov,
                        "rpm_estimated": None if rpm_est is None else float(rpm_est),
                        "rpm_used": float(rpm_used),
                        "rel_err": None if rel_err is None else float(rel_err),
                        "feature_mode": str(getattr(preprocessor, "feature_mode", "time")),
                        "n_windows": int(windows.shape[0]),
                    }
                )

            # Write validation table
            import csv as _csv

            with (log_dir / "rpm_validation.csv").open("w", newline="") as f:
                w = _csv.DictWriter(
                    f,
                    fieldnames=[
                        "sample_id",
                        "rpm_provided",
                        "rpm_estimated",
                        "rpm_used",
                        "rel_err",
                        "feature_mode",
                        "n_windows",
                    ],
                )
                w.writeheader()
                for r in rpm_rows:
                    w.writerow(r)

            _log_line(f"Wrote debug plots to {debug_dir}")
        except Exception as e:
            _log_line(f"[WARN] Debug plotting failed: {e}")

    # Sanity: no sensor_id leakage across splits (sample_id is fine to repeat only within a split)
    # Only enforce for protocols that are *supposed* to be group-disjoint.
    protocol_name = str(getattr(cfg, "protocol", {}).get("name", "split")).lower()
    if protocol_name in {"split", "loso"}:
        validate_no_sensor_leakage(samples_split)  # type: ignore[arg-type]

    # Optional: balance splits at the *sample_id* level (undersample majority class).
    # Default is to balance TRAIN only, and keep val/test as-is for honest evaluation.
    train_cfg = cfg.training
    balance_train = bool(train_cfg.get("balance_train", False))
    balance_val = bool(train_cfg.get("balance_val", False))
    balance_test = bool(train_cfg.get("balance_test", False))
    balance_seed = int(train_cfg.get("balance_seed", seed))

    def _label_counts(ss: List[SampleMeta]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for s in ss:
            if s.condition is None:
                continue
            out[str(s.condition)] = out.get(str(s.condition), 0) + 1
        return out

    split_before = {
        k: {
            "n": len(v),
            "labels": {s.condition for s in v if s.condition is not None},
            "label_counts": _label_counts(list(v)),
        }
        for k, v in samples_split.items()
    }

    train_samples = (
        balance_samples_by_condition(samples_split["train"], seed=balance_seed) if balance_train else list(samples_split["train"])
    )
    val_samples = (
        balance_samples_by_condition(samples_split["val"], seed=balance_seed + 1) if balance_val else list(samples_split["val"])
    )
    test_samples = (
        balance_samples_by_condition(samples_split["test"], seed=balance_seed + 2) if balance_test else list(samples_split["test"])
    )

    split_after = {
        "train": {"n": len(train_samples), "label_counts": _label_counts(train_samples)},
        "val": {"n": len(val_samples), "label_counts": _label_counts(val_samples)},
        "test": {"n": len(test_samples), "label_counts": _label_counts(test_samples)},
    }

    (out_dir / "balancing.json").write_text(
        json.dumps(
            {
                "enabled": {
                    "train": balance_train,
                    "val": balance_val,
                    "test": balance_test,
                },
                "seed": balance_seed,
                "before": {
                    k: {
                        "n": int(v["n"]),
                        "labels": sorted([str(x) for x in v["labels"]]),
                        "label_counts": {str(lbl): int(cnt) for lbl, cnt in dict(v["label_counts"]).items()},
                    }
                    for k, v in split_before.items()
                },
                "after": split_after,
                "note": "Balancing is sample-level undersampling (drops sample_id rows), not window-level.",
            },
            indent=2,
        )
    )

    model_cfg = cfg.model
    model_name = str(model_cfg.get("name", "conv_lstm")).lower()
    rpm_conditioning = bool(model_cfg.get("rpm_conditioning", False))
    rpm_embed_dim = int(model_cfg.get("rpm_embed_dim", 16))

    # Determine expected input length (L) for simple models.
    if getattr(preprocessor, "feature_mode", "time") == "order_spectrum":
        feature_length = int(getattr(preprocessor, "order_bins", preprocessor.window_size()))
    else:
        feature_length = int(preprocessor.window_size())

    if model_name == "conv_lstm":
        train_ds = WindowedSignalDataset(
            data_dir=data_dir,
            samples=train_samples,
            label_to_index=label_to_index,
            preprocessor=preprocessor,
            output_order=output_order,
            return_rpm=rpm_conditioning,
        )
        val_ds = WindowedSignalDataset(
            data_dir=data_dir,
            samples=val_samples,
            label_to_index=label_to_index,
            preprocessor=preprocessor,
            output_order=output_order,
            return_rpm=rpm_conditioning,
        )
        test_ds = WindowedSignalDataset(
            data_dir=data_dir,
            samples=test_samples,
            label_to_index=label_to_index,
            preprocessor=preprocessor,
            output_order=output_order,
            return_rpm=rpm_conditioning,
        )

        model = ConvLSTMClassifier(
            ConvLSTMConfig(
                in_channels=int(model_cfg.get("in_channels", 3)),
                conv_channels=int(model_cfg.get("conv_channels", 128)),
                lstm_hidden=int(model_cfg.get("lstm_hidden", 64)),
                lstm_layers=int(model_cfg.get("lstm_layers", 2)),
                num_classes=len(label_to_index),
                rpm_conditioning=rpm_conditioning,
                rpm_embed_dim=rpm_embed_dim,
            )
        )

    elif model_name in {"conv_net", "conv"}:
        train_ds = WindowedSignalDataset(
            data_dir=data_dir,
            samples=train_samples,
            label_to_index=label_to_index,
            preprocessor=preprocessor,
            output_order=output_order,
            return_rpm=rpm_conditioning,
        )
        val_ds = WindowedSignalDataset(
            data_dir=data_dir,
            samples=val_samples,
            label_to_index=label_to_index,
            preprocessor=preprocessor,
            output_order=output_order,
            return_rpm=rpm_conditioning,
        )
        test_ds = WindowedSignalDataset(
            data_dir=data_dir,
            samples=test_samples,
            label_to_index=label_to_index,
            preprocessor=preprocessor,
            output_order=output_order,
            return_rpm=rpm_conditioning,
        )

        model = ConvNetClassifier(
            ConvNetConfig(
                in_channels=int(model_cfg.get("in_channels", 3)),
                stem_channels=int(model_cfg.get("stem_channels", 64)),
                block_channels=tuple(int(x) for x in model_cfg.get("block_channels", [128, 128, 256, 256])),
                kernel_sizes=tuple(int(x) for x in model_cfg.get("kernel_sizes", [3, 5, 7])),
                dropout=float(model_cfg.get("dropout", 0.2)),
                num_classes=len(label_to_index),
                rpm_conditioning=rpm_conditioning,
                rpm_embed_dim=rpm_embed_dim,
            )
        )

    elif model_name in {"linear", "logreg", "baseline"}:
        # Very simple baseline (probabilities via softmax of logits).
        train_ds = WindowedSignalDataset(
            data_dir=data_dir,
            samples=train_samples,
            label_to_index=label_to_index,
            preprocessor=preprocessor,
            output_order=output_order,
            return_rpm=rpm_conditioning,
        )
        val_ds = WindowedSignalDataset(
            data_dir=data_dir,
            samples=val_samples,
            label_to_index=label_to_index,
            preprocessor=preprocessor,
            output_order=output_order,
            return_rpm=rpm_conditioning,
        )
        test_ds = WindowedSignalDataset(
            data_dir=data_dir,
            samples=test_samples,
            label_to_index=label_to_index,
            preprocessor=preprocessor,
            output_order=output_order,
            return_rpm=rpm_conditioning,
        )

        model = LinearClassifier(
            LinearConfig(
                in_channels=int(model_cfg.get("in_channels", 3)),
                input_length=int(model_cfg.get("input_length", feature_length)),
                dropout=float(model_cfg.get("dropout", 0.0)),
                num_classes=len(label_to_index),
                rpm_conditioning=rpm_conditioning,
                rpm_embed_dim=rpm_embed_dim,
            )
        )

    elif model_name == "window_hier_lstm":
        n_windows = int(model_cfg.get("n_windows", 10))
        train_ds = SubjectWindowDataset(
            data_dir=data_dir,
            samples=train_samples,
            label_to_index=label_to_index,
            preprocessor=preprocessor,
            n_windows=n_windows,
            output_order=output_order,
            random_start=True,
            seed=seed,
            return_rpm=rpm_conditioning,
        )
        val_ds = SubjectWindowDataset(
            data_dir=data_dir,
            samples=val_samples,
            label_to_index=label_to_index,
            preprocessor=preprocessor,
            n_windows=n_windows,
            output_order=output_order,
            random_start=False,
            seed=seed,
            return_rpm=rpm_conditioning,
        )
        test_ds = SubjectWindowDataset(
            data_dir=data_dir,
            samples=test_samples,
            label_to_index=label_to_index,
            preprocessor=preprocessor,
            n_windows=n_windows,
            output_order=output_order,
            random_start=False,
            seed=seed,
            return_rpm=rpm_conditioning,
        )

        model = WindowHierLSTMClassifier(
            WindowHierLSTMConfig(
                in_channels=int(model_cfg.get("in_channels", 3)),
                fc_dim=int(model_cfg.get("fc_dim", 64)),
                fc_layers=int(model_cfg.get("fc_layers", 4)),
                lstm_hidden=int(model_cfg.get("lstm_hidden", 64)),
                lstm_layers=int(model_cfg.get("lstm_layers", 2)),
                num_classes=len(label_to_index),
                rpm_conditioning=rpm_conditioning,
                rpm_embed_dim=rpm_embed_dim,
            )
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = _select_device(str(train_cfg.get("device", "auto")))
    model.to(device)

    lr = float(train_cfg.get("lr", 1e-3))
    optimizer_name = str(train_cfg.get("optimizer", "adam")).lower()
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    if optimizer_name == "adamw" or weight_decay > 0.0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    grad_clip_norm: Optional[float] = train_cfg.get("grad_clip_norm", None)
    mixup_alpha = float(train_cfg.get("mixup_alpha", 0.0))
    time_mask_prob = float(train_cfg.get("time_mask_prob", 0.0))
    time_mask_ratio = float(train_cfg.get("time_mask_ratio", 0.0))
    torch_rng = torch.Generator(device="cpu")
    torch_rng.manual_seed(seed)

    epochs = int(train_cfg.get("epochs", 20))
    best_val_f1 = -1.0
    best_epoch = 0

    early_patience = int(train_cfg.get("early_stopping_patience", 0))
    early_min_epochs = int(train_cfg.get("early_stopping_min_epochs", 0))
    early_min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    bad_epochs = 0

    (out_dir / "fold.json").write_text(json.dumps({"fold": fold_name}, indent=2))

    last_epoch = 0

    for epoch in range(1, epochs + 1):
        last_epoch = epoch
        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_n = 0
        for batch in train_loader:
            if len(batch) == 2:
                xb, yb = batch
                rb = None
            else:
                xb, yb, rb = batch

            xb = xb.to(device)
            yb = yb.to(device)
            if rb is not None:
                rb = rb.to(device)

            # Regularization/augmentation (train only)
            _apply_time_mask_(
                xb,
                prob=time_mask_prob,
                ratio=time_mask_ratio,
                generator=torch_rng,
            )

            if mixup_alpha > 0.0 and xb.shape[0] >= 2:
                xb, y_a, y_b, lam, perm = _mixup(xb, yb, alpha=mixup_alpha, generator=torch_rng)
                if rb is not None and lam != 1.0:
                    rb2 = rb.index_select(0, perm)
                    rb = rb.mul(lam).add(rb2, alpha=(1.0 - lam))
            else:
                y_a, y_b, lam = yb, yb, 1.0

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, rpm=rb)
            if lam == 1.0:
                loss = criterion(logits, y_a)
            else:
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            loss.backward()

            epoch_loss_sum += float(loss.detach().cpu().item())
            epoch_loss_n += 1

            if grad_clip_norm is not None:
                gcn = float(grad_clip_norm)
                if gcn > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gcn)
            optimizer.step()

        y_true, y_pred = _predict(model, val_loader, device)
        val_metrics = compute_metrics(y_true, y_pred)

        improved = val_metrics.f1_macro > (best_val_f1 + early_min_delta)
        if improved:
            best_val_f1 = val_metrics.f1_macro
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model": model_cfg,
                    "preprocessing": pre_cfg,
                    "label_to_index": label_to_index,
                    "output_order": output_order,
                },
                out_dir / "model.pkl",
            )

            (out_dir / "metrics_val_best.json").write_text(
                json.dumps({"epoch": epoch, **val_metrics.as_dict()}, indent=2)
            )
        else:
            bad_epochs += 1

        train_loss_mean = float(epoch_loss_sum / max(1, epoch_loss_n))
        _log_line(
            "Epoch "
            + json.dumps(
                {
                    "epoch": int(epoch),
                    "train_loss": train_loss_mean,
                    "val_accuracy": float(val_metrics.accuracy),
                    "val_f1_macro": float(val_metrics.f1_macro),
                    "best_val_f1_macro": float(best_val_f1),
                    "improved": bool(improved),
                },
                sort_keys=True,
            )
        )

        with (log_dir / "metrics.jsonl").open("a") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": int(epoch),
                        "train_loss": train_loss_mean,
                        "val": val_metrics.as_dict(),
                        "best_val_f1_macro": float(best_val_f1),
                        "best_epoch": int(best_epoch),
                    }
                )
                + "\n"
            )

        (out_dir / "metrics_val_latest.json").write_text(
            json.dumps({"epoch": epoch, **val_metrics.as_dict()}, indent=2)
        )

        if epoch_callback is not None:
            # signature: (epoch, val_f1_macro, best_val_f1_macro)
            epoch_callback(epoch, float(val_metrics.f1_macro), float(best_val_f1))

        if early_patience > 0 and epoch >= max(early_min_epochs, 1) and bad_epochs >= early_patience:
            (out_dir / "early_stopping.json").write_text(
                json.dumps(
                    {
                        "stopped": True,
                        "epoch": epoch,
                        "best_epoch": best_epoch,
                        "best_val_f1_macro": float(best_val_f1),
                        "patience": early_patience,
                        "min_epochs": early_min_epochs,
                        "min_delta": early_min_delta,
                    },
                    indent=2,
                )
            )
            break

    # Test metrics for the *latest* model state (end of training loop)
    if len(test_ds) > 0:
        y_true_t_latest, y_pred_t_latest = _predict(model, test_loader, device)
        test_metrics_latest = compute_metrics(y_true_t_latest, y_pred_t_latest)
        (out_dir / "metrics_test_latest.json").write_text(
            json.dumps({"epoch": int(last_epoch), **test_metrics_latest.as_dict()}, indent=2)
        )

    # Load best checkpoint
    ckpt = torch.load(out_dir / "model.pkl", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    if len(test_ds) > 0:
        y_true_t, y_pred_t = _predict(model, test_loader, device)
        test_metrics = compute_metrics(y_true_t, y_pred_t)
        # Best test metrics (best-by-validation checkpoint)
        (out_dir / "metrics_test_best.json").write_text(
            json.dumps({"epoch": int(best_epoch), **test_metrics.as_dict()}, indent=2)
        )
        # Backward-compatible alias
        (out_dir / "metrics_test.json").write_text(json.dumps(test_metrics.as_dict(), indent=2))
    else:
        test_metrics = None
        (out_dir / "metrics_test_skipped.json").write_text(
            json.dumps(
                {
                    "skipped": True,
                    "reason": "No test samples in this protocol.",
                },
                indent=2,
            )
        )

    # Per-sample prediction log (best-by-validation checkpoint), aggregated over windows when needed.
    try:
        pred_rows: List[Dict[str, Any]] = []
        index_val_loader = DataLoader(_WithIndex(val_ds), batch_size=batch_size, shuffle=False, num_workers=num_workers)
        index_test_loader = None
        if len(test_ds) > 0:
            index_test_loader = DataLoader(_WithIndex(test_ds), batch_size=batch_size, shuffle=False, num_workers=num_workers)

        y_true_v, idx_v, probs_v = _predict_probs_with_index(model, index_val_loader, device)
        y_true_t2, idx_t2, probs_t2 = _predict_probs_with_index(model, index_test_loader, device)
        pred_rows.extend(
            _per_sample_predictions_rows(
                split_name="val",
                dataset=val_ds,
                label_to_index=label_to_index,
                y_true=y_true_v,
                indices=idx_v,
                probs=probs_v,
            )
        )
        if index_test_loader is not None:
            y_true_t2, idx_t2, probs_t2 = _predict_probs_with_index(model, index_test_loader, device)
            pred_rows.extend(
                _per_sample_predictions_rows(
                    split_name="test",
                    dataset=test_ds,
                    label_to_index=label_to_index,
                    y_true=y_true_t2,
                    indices=idx_t2,
                    probs=probs_t2,
                )
            )
        _write_predictions_csv(out_dir / "predictions_per_sample.csv", pred_rows)
    except Exception as e:
        (out_dir / "predictions_per_sample_error.txt").write_text(str(e))

    # Save normalizer + label map
    if bool(pre_cfg.get("z_norm", True)):
        preprocessor.save_normalizer(out_dir / "normalizer.pkl")
    save_label_mapping(label_to_index, out_dir / "label_mapping.json")

    if test_metrics is None:
        test_accuracy = float("nan")
        test_f1_macro = float("nan")
    else:
        test_accuracy = float(test_metrics.accuracy)
        test_f1_macro = float(test_metrics.f1_macro)

    return {
        "val_f1_macro_best": float(best_val_f1),
        "test_accuracy": test_accuracy,
        "test_f1_macro": test_f1_macro,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    meta = Part3Metadata(cfg.train_metadata_path)
    # Preflight: check metadata <-> filesystem consistency.
    meta_samples_all = list(meta.iter_samples())
    meta_ids_all = {s.sample_id for s in meta_samples_all}
    data_dir = cfg.train_data_dir
    data_ids_all = {p.stem for p in data_dir.glob("*.csv")}

    extra_on_disk = sorted(data_ids_all - meta_ids_all)
    missing_on_disk = sorted(meta_ids_all - data_ids_all)

    if extra_on_disk:
        print(
            f"[WARN] Found {len(extra_on_disk)} CSVs in {data_dir} not present in metadata; they will be ignored."
        )
    if missing_on_disk:
        raise FileNotFoundError(
            f"Metadata references {len(missing_on_disk)} CSVs not found in {data_dir}. Example: {missing_on_disk[:3]}"
        )

    # Training samples are defined ONLY by metadata rows with labels.
    samples = [s for s in meta_samples_all if s.condition is not None]

    artifacts_cfg = cfg.artifacts
    run_name = str(artifacts_cfg.get("run_name", "run"))
    out_root = Path(artifacts_cfg.get("output_dir", "artifacts")) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    logging_cfg = {}
    if hasattr(cfg, "raw") and isinstance(getattr(cfg, "raw"), dict):
        logging_cfg = dict(getattr(cfg, "raw").get("logging", {}) or {})
    logs_root = Path(str(logging_cfg.get("output_dir", "logs"))) / run_name
    logs_root.mkdir(parents=True, exist_ok=True)

    # Diagnostics: per-sensor label distribution
    per_sensor = sensor_class_distribution(samples)
    (out_root / "sensor_class_distribution.json").write_text(json.dumps(per_sensor, indent=2, sort_keys=True))
    single_class_sensors = [sid for sid, d in per_sensor.items() if len(d.keys()) < 2]
    if single_class_sensors:
        print(
            f"[WARN] Found sensor_id values with only one class in metadata: {single_class_sensors}. "
            "This can make group splits and LOSO unstable."
        )

    label_to_index = build_label_mapping([s.condition for s in samples if s.condition is not None])
    (out_root / "config_used.json").write_text(json.dumps(cfg.raw, indent=2))
    (out_root / "data_preflight.json").write_text(
        json.dumps(
            {
                "data_dir": str(data_dir),
                "metadata_rows": len(meta_samples_all),
                "labeled_rows": len(samples),
                "csv_files": len(data_ids_all),
                "extra_csvs_ignored": len(extra_on_disk),
            },
            indent=2,
        )
    )

    protocol = cfg.protocol
    protocol_name = str(protocol.get("name", "split")).lower()
    output_order = cfg.output_order

    def _split_summary(split_dict: Dict[str, List[SampleMeta]]) -> Dict[str, object]:
        def _summ(split_name: str) -> Dict[str, object]:
            ss = split_dict[split_name]
            sensors = sorted({s.sensor_id for s in ss if s.sensor_id is not None})
            labels = sorted({s.condition for s in ss if s.condition is not None})
            label_counts: Dict[str, int] = {}
            for s in ss:
                if s.condition is None:
                    continue
                label_counts[s.condition] = label_counts.get(s.condition, 0) + 1
            return {
                "n_samples": len(ss),
                "n_sensors": len(sensors),
                "sensors": sensors,
                "labels": labels,
                "label_counts": label_counts,
            }

        return {
            "train": _summ("train"),
            "val": _summ("val"),
            "test": _summ("test"),
        }

    if protocol_name == "split":
        split_cfg = protocol.get("split", {})
        splits = split_by_sensor_id(
            samples,
            train_ratio=float(split_cfg.get("train_ratio", 0.7)),
            val_ratio=float(split_cfg.get("val_ratio", 0.15)),
            test_ratio=float(split_cfg.get("test_ratio", 0.15)),
            seed=int(cfg.training.get("seed", 42)),
        )

        (out_root / "split_summary.json").write_text(json.dumps(_split_summary(splits), indent=2))
        metrics = _train_one(
            fold_name="split",
            samples_split=splits,
            label_to_index=label_to_index,
            data_dir=cfg.train_data_dir,
            output_order=output_order,
            cfg=cfg,
            out_dir=out_root,
            log_root=logs_root,
        )
        print(json.dumps(metrics, indent=2))
        return

    if protocol_name == "loso":
        loso_cfg = protocol.get("loso", {})
        folds = loso_folds(
            samples,
            val_ratio_within_train=float(loso_cfg.get("val_ratio_within_train", 0.15)),
            seed=int(cfg.training.get("seed", 42)),
        )

        all_metrics = []
        for heldout_sensor, split_dict in folds:
            fold_dir = out_root / "loso" / heldout_sensor
            fold_dir.mkdir(parents=True, exist_ok=True)
            (fold_dir / "split_summary.json").write_text(json.dumps(_split_summary(split_dict), indent=2))
            m = _train_one(
                fold_name=f"loso_{heldout_sensor}",
                samples_split=split_dict,
                label_to_index=label_to_index,
                data_dir=cfg.train_data_dir,
                output_order=output_order,
                cfg=cfg,
                out_dir=fold_dir,
                log_root=logs_root,
            )
            all_metrics.append({"heldout_sensor": heldout_sensor, **m})

        (out_root / "loso_summary.json").write_text(json.dumps(all_metrics, indent=2))
        mean_test_f1 = float(np.mean([m["test_f1_macro"] for m in all_metrics]))
        print(json.dumps({"folds": len(all_metrics), "mean_test_f1_macro": mean_test_f1}, indent=2))
        return

    if protocol_name in {"train_all", "all"}:
        all_cfg = protocol.get("train_all", protocol.get("all", {}))
        val_ratio = float(all_cfg.get("val_ratio", 0.15))
        seed = int(cfg.training.get("seed", 42))

        # Stratified split by condition over sample_id (keeps all sensors represented in training).
        by_label: Dict[str, List[SampleMeta]] = {}
        for s in samples:
            if s.condition is None:
                continue
            by_label.setdefault(str(s.condition), []).append(s)

        rng = np.random.default_rng(seed)
        train_samples: List[SampleMeta] = []
        val_samples: List[SampleMeta] = []
        for label, items in sorted(by_label.items()):
            items = list(items)
            rng.shuffle(items)
            n_val = int(round(len(items) * val_ratio))
            n_val = max(1, min(len(items) - 1, n_val)) if len(items) >= 2 else 0
            val_samples.extend(items[:n_val])
            train_samples.extend(items[n_val:])

        splits = {
            "train": train_samples,
            "val": val_samples,
            "test": [],
        }

        (out_root / "split_summary.json").write_text(json.dumps(_split_summary(splits), indent=2))
        metrics = _train_one(
            fold_name="train_all",
            samples_split=splits,
            label_to_index=label_to_index,
            data_dir=cfg.train_data_dir,
            output_order=output_order,
            cfg=cfg,
            out_dir=out_root,
            log_root=logs_root,
        )
        print(json.dumps(metrics, indent=2))
        return

    raise ValueError(f"Unknown protocol: {protocol_name}")


if __name__ == "__main__":
    main()
