from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from src.data.dataloader import Part3Metadata, read_signal_csv, standardize_orientation
from src.model.conv import ConvNetClassifier, ConvNetConfig
from src.model.linear import LinearClassifier, LinearConfig
from src.preprocessing.preprocessor import Preprocessor
from src.training.config import load_config


def _select_device(device: str) -> torch.device:
    device = device.lower()
    if device in {"cpu", "cuda", "mps"}:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _predict_sample_windows(model: torch.nn.Module, windows: np.ndarray, device: torch.device) -> np.ndarray:
    """windows: (N, L, C) -> probs (N, K)."""
    if windows.shape[0] == 0:
        return np.zeros((0, model.cfg.num_classes), dtype=np.float32)  # type: ignore[attr-defined]
    # Avoid torch.from_numpy for environments where PyTorch NumPy bridge is unavailable.
    x = torch.tensor(windows.tolist(), dtype=torch.float32).transpose(1, 2)  # (N, C, L)
    x = x.to(device)
    logits = model(x)
    probs = np.asarray(torch.softmax(logits, dim=1).cpu().tolist(), dtype=np.float32)
    return probs


@torch.no_grad()
def _predict_sample_windows_with_rpm(
    model: torch.nn.Module,
    windows: np.ndarray,
    *,
    rpm_used: float,
    device: torch.device,
) -> np.ndarray:
    """windows: (N, L, C) -> probs (N, K), with RPM conditioning."""
    if windows.shape[0] == 0:
        return np.zeros((0, model.cfg.num_classes), dtype=np.float32)  # type: ignore[attr-defined]
    x = torch.tensor(windows.tolist(), dtype=torch.float32).transpose(1, 2)  # (N, C, L)
    x = x.to(device)
    r = torch.full((x.shape[0],), float(rpm_used), dtype=torch.float32, device=device)
    logits = model(x, rpm=r)
    probs = np.asarray(torch.softmax(logits, dim=1).cpu().tolist(), dtype=np.float32)
    return probs


@torch.no_grad()
def _predict_subject(
    model: torch.nn.Module,
    windows: np.ndarray,
    *,
    n_windows: int,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    """windows: (N, L, C) -> probs (K,) aggregated for the subject."""
    if windows.shape[0] == 0:
        return np.zeros((num_classes,), dtype=np.float32)

    n = int(windows.shape[0])
    if n >= n_windows:
        w = windows[:n_windows]
    else:
        pad = np.zeros((n_windows - n, windows.shape[1], windows.shape[2]), dtype=np.float32)
        w = np.concatenate([windows, pad], axis=0)

    x = torch.tensor(w.tolist(), dtype=torch.float32).transpose(1, 2).contiguous()  # (Nw, C, L)
    x = x.unsqueeze(0).to(device)  # (1, Nw, C, L)
    logits = model(x)
    probs = np.asarray(torch.softmax(logits, dim=1).cpu().tolist(), dtype=np.float32)[0]
    return probs


@torch.no_grad()
def _predict_subject_with_rpm(
    model: torch.nn.Module,
    windows: np.ndarray,
    *,
    rpm_used: float,
    n_windows: int,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    """windows: (N, L, C) -> probs (K,), with RPM conditioning."""
    if windows.shape[0] == 0:
        return np.zeros((num_classes,), dtype=np.float32)

    n = int(windows.shape[0])
    if n >= n_windows:
        w = windows[:n_windows]
    else:
        pad = np.zeros((n_windows - n, windows.shape[1], windows.shape[2]), dtype=np.float32)
        w = np.concatenate([windows, pad], axis=0)

    x = torch.tensor(w.tolist(), dtype=torch.float32).transpose(1, 2).contiguous()  # (Nw, C, L)
    x = x.unsqueeze(0).to(device)  # (1, Nw, C, L)
    r = torch.tensor([float(rpm_used)], dtype=torch.float32, device=device)
    logits = model(x, rpm=r)
    probs = np.asarray(torch.softmax(logits, dim=1).cpu().tolist(), dtype=np.float32)[0]
    return probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument(
        "--artifacts",
        type=str,
        required=True,
        help="Path to an artifacts folder containing model.pkl, normalizer.pkl, label_mapping.json",
    )
    ap.add_argument("--out", type=str, default="external_test_predictions.csv")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    artifacts_dir = Path(args.artifacts)

    label_to_index: Dict[str, int] = json.loads((artifacts_dir / "label_mapping.json").read_text())
    index_to_label = {v: k for k, v in label_to_index.items()}

    ckpt = torch.load(artifacts_dir / "model.pkl", map_location="cpu")

    model_cfg = ckpt.get("model", {})
    model_name = str(model_cfg.get("name", "conv_lstm")).lower()
    rpm_conditioning = bool(model_cfg.get("rpm_conditioning", False))

    # Load preprocessing config early so simple models can infer input shapes.
    pre_cfg = ckpt.get("preprocessing", cfg.preprocessing)
    feature_mode = str(pre_cfg.get("feature_mode", "time")).lower().strip()
    if feature_mode == "order_spectrum":
        inferred_input_length = int(pre_cfg.get("order_bins", 128))
    else:
        inferred_input_length = int(
            round(float(pre_cfg.get("downsample_hz", 100.0)) * float(pre_cfg.get("window_seconds", 0.05)))
        )
        inferred_input_length = max(1, inferred_input_length)

    if model_name in {"conv_net", "conv"}:
        model = ConvNetClassifier(
            ConvNetConfig(
                in_channels=int(model_cfg.get("in_channels", 3)),
                stem_channels=int(model_cfg.get("stem_channels", 64)),
                block_channels=tuple(int(x) for x in model_cfg.get("block_channels", [128, 128, 256, 256])),
                kernel_sizes=tuple(int(x) for x in model_cfg.get("kernel_sizes", [3, 5, 7])),
                dropout=float(model_cfg.get("dropout", 0.2)),
                num_classes=len(label_to_index),
            )
        )
    elif model_name in {"linear", "logreg", "baseline"}:
        model = LinearClassifier(
            LinearConfig(
                in_channels=int(model_cfg.get("in_channels", 3)),
                input_length=int(model_cfg.get("input_length", inferred_input_length)),
                dropout=float(model_cfg.get("dropout", 0.0)),
                num_classes=len(label_to_index),
                rpm_conditioning=bool(model_cfg.get("rpm_conditioning", False)),
                rpm_embed_dim=int(model_cfg.get("rpm_embed_dim", 16)),
            )
        )
    else:
        raise ValueError(f"Unsupported model in checkpoint: {model_name}")

    model.load_state_dict(ckpt["model_state_dict"])

    output_order = list(ckpt.get("output_order", cfg.output_order))

    # Build preprocessor and load fitted normalizer
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
    if bool(pre_cfg.get("z_norm", True)) and (artifacts_dir / "normalizer.pkl").exists():
        preprocessor.load_normalizer(artifacts_dir / "normalizer.pkl")

    device = _select_device(str(cfg.training.get("device", "auto")))
    model.to(device)
    model.eval()

    # External test data
    test_meta_path = cfg.part3_root / "test_metadata.csv"
    test_data_dir = cfg.part3_root / "test_data"

    meta = Part3Metadata(test_meta_path)

    rows: List[Dict[str, object]] = []
    for s in meta.iter_samples():
        csv_path = test_data_dir / f"{s.sample_id}.csv"
        if not csv_path.exists():
            # allow missing files without crashing the whole run
            continue

        time, signals = read_signal_csv(csv_path)
        signals = standardize_orientation(signals, s.orientation, output_order=output_order)
        windows, rpm_used = preprocessor.transform_with_rpm(time=time, signals=signals, rpm=s.rpm)  # (N, L, C)

        if model_name in {"conv_lstm", "conv_net", "conv", "linear", "logreg", "baseline"}:
            if rpm_conditioning:
                probs_w = _predict_sample_windows_with_rpm(model, windows, rpm_used=rpm_used, device=device)
            else:
                probs_w = _predict_sample_windows(model, windows, device)
            if probs_w.shape[0] == 0:
                agg = np.zeros((len(label_to_index),), dtype=np.float32)
            else:
                agg = probs_w.mean(axis=0)
        else:
            if rpm_conditioning:
                agg = _predict_subject_with_rpm(
                    model,
                    windows,
                    rpm_used=rpm_used,
                    n_windows=int(model_cfg.get("n_windows", 10)),
                    num_classes=len(label_to_index),
                    device=device,
                )
            else:
                agg = _predict_subject(
                    model,
                    windows,
                    n_windows=int(model_cfg.get("n_windows", 10)),
                    num_classes=len(label_to_index),
                    device=device,
                )

        pred_idx = int(np.argmax(agg))
        row: Dict[str, object] = {
            "sample_id": s.sample_id,
            "sensor_id": s.sensor_id,
            "rpm": s.rpm,
            "orientation": json.dumps(s.orientation, sort_keys=True),
            "pred_label": index_to_label[pred_idx],
        }
        for k in range(len(label_to_index)):
            row[f"prob_{index_to_label[k]}"] = float(agg[k])
        rows.append(row)

    out_path = Path(args.out)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} predictions to {out_path}")


if __name__ == "__main__":
    main()
