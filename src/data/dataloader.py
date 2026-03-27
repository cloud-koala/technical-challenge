from __future__ import annotations

import ast
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


ConditionLabel = str
SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SampleMeta:
    sample_id: str
    condition: Optional[ConditionLabel]
    sensor_id: Optional[str]
    rpm: Optional[float]
    orientation: Dict[str, str]


def _safe_parse_orientation(value: str) -> Dict[str, str]:
    """Parse orientation column values like "{'axisX': 'horizontal', ...}"."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        return {str(k): str(v) for k, v in parsed.items()}
    return {}


def _read_signal_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (time, signals) where signals has shape (T, 3).

    Supports two schemas:
    - Training schema: "X-Axis" (time) + 3 channel columns
    - External test schema: "t" (time) + "x","y","z" channels
    """
    df = pd.read_csv(csv_path)

    # Time column
    if "X-Axis" in df.columns:
        time_col = "X-Axis"
    elif "t" in df.columns:
        time_col = "t"
    else:
        raise ValueError(f"Missing time column ('X-Axis' or 't') in {csv_path}")

    # Channel columns
    if {"x", "y", "z"}.issubset(set(df.columns)):
        channel_cols = ["x", "y", "z"]
    else:
        # Heuristic: treat the remaining columns as channels (keep their order).
        channel_cols = [c for c in df.columns if c != time_col]
        if len(channel_cols) < 3:
            raise ValueError(f"Expected >=3 channel columns in {csv_path}, got {channel_cols}")
        channel_cols = channel_cols[:3]

    time = df[time_col].to_numpy(dtype=np.float64)
    signals = df[channel_cols].to_numpy(dtype=np.float32)
    return time, signals


def read_signal_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Public wrapper around internal CSV reader."""
    return _read_signal_csv(Path(csv_path))


def _standardize_orientation(
    signals: np.ndarray,
    orientation: Dict[str, str],
    *,
    output_order: Sequence[str] = ("vertical", "axial", "horizontal"),
) -> np.ndarray:
    """Reorder channels so output is in a consistent physical orientation.

    Assumptions (based on provided metadata schema):
    - CSV channels correspond to sensor axes in order: axisX, axisY, axisZ.
    - orientation maps sensor axes (axisX/axisY/axisZ) -> physical directions
      (vertical/horizontal/axial).

    Returns array shaped (T, 3) ordered as output_order.

    Note: if orientation values include a leading sign (e.g. "-horizontal"),
    we apply that sign flip.
    """
    if signals.shape[1] < 3:
        raise ValueError("signals must have at least 3 columns")

    axis_to_idx = {"axisX": 0, "axisY": 1, "axisZ": 2}

    # Build physical -> (source index, sign) mapping
    physical_to_src: Dict[str, Tuple[int, float]] = {}
    for axis, physical_raw in orientation.items():
        if axis not in axis_to_idx:
            continue
        physical_s = str(physical_raw).strip().lower()
        sign = 1.0
        if physical_s.startswith("-"):
            sign = -1.0
            physical_s = physical_s[1:].strip()
        elif physical_s.startswith("+"):
            physical_s = physical_s[1:].strip()

        if physical_s in {"vertical", "horizontal", "axial"}:
            physical_to_src[physical_s] = (axis_to_idx[axis], sign)

    # If incomplete, fallback to identity mapping.
    if not all(p in physical_to_src for p in output_order):
        return signals[:, :3]

    return np.stack([signals[:, physical_to_src[p][0]] * physical_to_src[p][1] for p in output_order], axis=1)


def standardize_orientation(
    signals: np.ndarray,
    orientation: Dict[str, str],
    *,
    output_order: Sequence[str] = ("vertical", "axial", "horizontal"),
) -> np.ndarray:
    """Public wrapper around internal orientation standardization."""
    return _standardize_orientation(signals, orientation, output_order=output_order)


class Part3Metadata:
    """Loads metadata for part_3 training data."""

    def __init__(self, metadata_csv: Path):
        self.metadata_csv = Path(metadata_csv)
        self.df = pd.read_csv(self.metadata_csv)

        required = {"sample_id", "orientation"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Metadata missing columns: {sorted(missing)}")

        # condition + sensor_id may be absent in test_metadata
        if "condition" not in self.df.columns:
            self.df["condition"] = None
        if "sensor_id" not in self.df.columns:
            self.df["sensor_id"] = None

        self.df["orientation"] = self.df["orientation"].apply(_safe_parse_orientation)

    def iter_samples(self) -> Iterable[SampleMeta]:
        for row in self.df.itertuples(index=False):
            cond = getattr(row, "condition")
            if cond is None or (isinstance(cond, float) and np.isnan(cond)):
                cond_s = None
            else:
                cond_s = str(cond)

            sid = getattr(row, "sensor_id")
            if sid is None or (isinstance(sid, float) and np.isnan(sid)):
                sid_s = None
            else:
                sid_s = str(sid)

            rpm_v: Optional[float]
            if hasattr(row, "rpm"):
                rpm_raw = getattr(row, "rpm")
                if rpm_raw is None or (isinstance(rpm_raw, float) and np.isnan(rpm_raw)):
                    rpm_v = None
                else:
                    try:
                        rpm_v = float(rpm_raw)
                    except Exception:
                        rpm_v = None
            else:
                rpm_v = None

            yield SampleMeta(
                sample_id=str(getattr(row, "sample_id")),
                condition=cond_s,
                sensor_id=sid_s,
                rpm=rpm_v,
                orientation=getattr(row, "orientation"),
            )


class WindowedSignalDataset(Dataset):
    """Loads files, applies orientation standardization, and yields windows.

    This dataset yields raw arrays; preprocessing (downsampling, windowing, z-norm)
    is delegated to a Preprocessor object to keep responsibilities clean.

    Each item is (window_tensor[C, L], label_int).
    """

    def __init__(
        self,
        *,
        data_dir: Path,
        samples: Sequence[SampleMeta],
        label_to_index: Dict[ConditionLabel, int],
        preprocessor,
        output_order: Sequence[str] = ("vertical", "axial", "horizontal"),
        cache_max_samples: int = 8,
        return_rpm: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.samples = list(samples)
        self.label_to_index = dict(label_to_index)
        self.preprocessor = preprocessor
        self.output_order = tuple(output_order)

        self.cache_max_samples = int(cache_max_samples)
        self.return_rpm = bool(return_rpm)

        # Cache per-sample: (windows, label_idx, rpm_used)
        self._cache: "OrderedDict[int, Tuple[np.ndarray, int, float]]" = OrderedDict()

        # Materialize only the index (sample_idx, window_idx). Windows are computed lazily.
        self._windows: List[Tuple[int, int]] = []

        for sample_idx in range(len(self.samples)):
            windows, _, _ = self._load_and_preprocess_sample(sample_idx, use_cache=False)
            for w_idx in range(int(windows.shape[0])):
                self._windows.append((sample_idx, w_idx))

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int):
        sample_idx, w_idx = self._windows[index]
        windows, label_idx, rpm_used = self._load_and_preprocess_sample(sample_idx, use_cache=True)
        x = windows[w_idx]  # (L, C)    
        # Some PyTorch builds can be compiled without NumPy support; avoid torch.from_numpy.
        x_t = torch.tensor(x.tolist(), dtype=torch.float32).transpose(0, 1).contiguous()  # (C, L)
        y_t = torch.tensor(label_idx, dtype=torch.long)
        if self.return_rpm:
            r_t = torch.tensor(float(rpm_used), dtype=torch.float32)
            return x_t, y_t, r_t
        return x_t, y_t

    def _load_and_preprocess_sample(self, sample_idx: int, *, use_cache: bool) -> Tuple[np.ndarray, int, float]:
        if use_cache and sample_idx in self._cache:
            self._cache.move_to_end(sample_idx)
            return self._cache[sample_idx]

        meta = self.samples[sample_idx]
        if meta.condition is None:
            raise ValueError("Dataset requires labels; got condition=None")
        if meta.condition not in self.label_to_index:
            raise KeyError(f"Unknown condition: {meta.condition}")

        csv_path = self.data_dir / f"{meta.sample_id}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing sample file: {csv_path}")

        time, signals = _read_signal_csv(csv_path)
        signals = _standardize_orientation(signals, meta.orientation, output_order=self.output_order)
        windows, rpm_used = self.preprocessor.transform_with_rpm(time=time, signals=signals, rpm=meta.rpm)
        label_idx = self.label_to_index[meta.condition]

        if use_cache and self.cache_max_samples > 0:
            self._cache[sample_idx] = (windows, label_idx, float(rpm_used))
            self._cache.move_to_end(sample_idx)
            while len(self._cache) > self.cache_max_samples:
                self._cache.popitem(last=False)
        return windows, label_idx, float(rpm_used)


class SubjectWindowDataset(Dataset):
    """One item per CSV (subject/sample), represented as a fixed window sequence.

    Returns:
      x: (Nw, C, L) float32
      y: scalar long

    Notes:
    - Windows are produced by the configured preprocessor.
    - If a sample produces fewer than n_windows windows, it is zero-padded.
    - If it produces more, a contiguous chunk is selected (optionally random).
    """

    def __init__(
        self,
        *,
        data_dir: Path,
        samples: Sequence[SampleMeta],
        label_to_index: Dict[ConditionLabel, int],
        preprocessor,
        n_windows: int,
        output_order: Sequence[str] = ("vertical", "axial", "horizontal"),
        random_start: bool = False,
        seed: int = 42,
        cache_max_samples: int = 8,
        return_rpm: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.samples = list(samples)
        self.label_to_index = dict(label_to_index)
        self.preprocessor = preprocessor
        self.output_order = tuple(output_order)
        self.n_windows = int(n_windows)
        if self.n_windows <= 0:
            raise ValueError("n_windows must be > 0")

        self.random_start = bool(random_start)
        self.rng = np.random.default_rng(int(seed))

        self.cache_max_samples = int(cache_max_samples)
        self.return_rpm = bool(return_rpm)

        # Cache per-sample: (windows, label_idx, rpm_used)
        self._cache: "OrderedDict[int, Tuple[np.ndarray, int, float]]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        windows, label_idx, rpm_used = self._load_and_preprocess_sample(index, use_cache=True)
        # windows: (N, L, C)
        n = int(windows.shape[0])
        if n >= self.n_windows:
            if self.random_start and n > self.n_windows:
                start = int(self.rng.integers(0, n - self.n_windows + 1))
            else:
                start = 0
            w = windows[start : start + self.n_windows]
        else:
            pad = np.zeros((self.n_windows - n, windows.shape[1], windows.shape[2]), dtype=np.float32)
            w = np.concatenate([windows, pad], axis=0)

        # Convert to torch without torch.from_numpy.
        # (Nw, L, C) -> (Nw, C, L)
        x_t = torch.tensor(w.tolist(), dtype=torch.float32).transpose(1, 2).contiguous()
        y_t = torch.tensor(label_idx, dtype=torch.long)
        if self.return_rpm:
            r_t = torch.tensor(float(rpm_used), dtype=torch.float32)
            return x_t, y_t, r_t
        return x_t, y_t

    def _load_and_preprocess_sample(self, sample_idx: int, *, use_cache: bool) -> Tuple[np.ndarray, int, float]:
        if use_cache and sample_idx in self._cache:
            self._cache.move_to_end(sample_idx)
            return self._cache[sample_idx]

        meta = self.samples[sample_idx]
        if meta.condition is None:
            raise ValueError("Dataset requires labels; got condition=None")
        if meta.condition not in self.label_to_index:
            raise KeyError(f"Unknown condition: {meta.condition}")

        csv_path = self.data_dir / f"{meta.sample_id}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing sample file: {csv_path}")

        time, signals = _read_signal_csv(csv_path)
        signals = _standardize_orientation(signals, meta.orientation, output_order=self.output_order)
        windows, rpm_used = self.preprocessor.transform_with_rpm(time=time, signals=signals, rpm=meta.rpm)
        label_idx = self.label_to_index[meta.condition]

        if use_cache and self.cache_max_samples > 0:
            self._cache[sample_idx] = (windows, label_idx, float(rpm_used))
            self._cache.move_to_end(sample_idx)
            while len(self._cache) > self.cache_max_samples:
                self._cache.popitem(last=False)
        return windows, label_idx, float(rpm_used)


def build_label_mapping(conditions: Sequence[str]) -> Dict[str, int]:
    unique = sorted(set(conditions))
    return {c: i for i, c in enumerate(unique)}


def validate_no_sensor_leakage(splits: Dict[SplitName, Sequence[SampleMeta]]) -> None:
    """Ensure sensor_id groups are disjoint across splits."""
    sensors_by_split: Dict[SplitName, set[str]] = {}
    for split_name in ("train", "val", "test"):
        sensors_by_split[split_name] = {s.sensor_id for s in splits[split_name] if s.sensor_id is not None}

    inter_tv = sensors_by_split["train"].intersection(sensors_by_split["val"])
    inter_tt = sensors_by_split["train"].intersection(sensors_by_split["test"])
    inter_vt = sensors_by_split["val"].intersection(sensors_by_split["test"])
    if inter_tv or inter_tt or inter_vt:
        raise ValueError(
            "sensor_id leakage across splits detected: "
            f"train∩val={sorted(inter_tv)}, train∩test={sorted(inter_tt)}, val∩test={sorted(inter_vt)}"
        )


def sensor_class_distribution(samples: Sequence[SampleMeta]) -> Dict[str, Dict[str, int]]:
    """Return {sensor_id: {condition: count}} for quick diagnostics."""
    out: Dict[str, Dict[str, int]] = {}
    for s in samples:
        if s.sensor_id is None or s.condition is None:
            continue
        per = out.setdefault(str(s.sensor_id), {})
        per[str(s.condition)] = per.get(str(s.condition), 0) + 1
    return out


def balance_samples_by_condition(
    samples: Sequence[SampleMeta],
    *,
    seed: int = 42,
    min_per_class: int = 1,
) -> List[SampleMeta]:
    """Undersample at the sample_id level to balance class counts.

    This is *sample-level* balancing (drops sample_ids), not window-level.
    It is usually appropriate for the training split only.
    """
    by_label: Dict[str, List[SampleMeta]] = {}
    for s in samples:
        if s.condition is None:
            continue
        by_label.setdefault(str(s.condition), []).append(s)

    if len(by_label) == 0:
        return list(samples)

    # Determine target count = min class count (classic undersampling).
    counts = {k: len(v) for k, v in by_label.items()}
    target = min(counts.values())
    if target < int(min_per_class):
        raise ValueError(f"Cannot balance: a class has only {target} samples (<{min_per_class}). Counts={counts}")

    rng = np.random.default_rng(int(seed))
    out: List[SampleMeta] = []
    for label, items in sorted(by_label.items()):
        if len(items) == target:
            chosen = list(items)
        else:
            idxs = rng.choice(len(items), size=target, replace=False).tolist()
            chosen = [items[i] for i in idxs]
        out.extend(chosen)

    rng.shuffle(out)
    return out


def split_by_sensor_id(
    samples: Sequence[SampleMeta],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    ensure_train_has_all_labels: bool = True,
    ensure_eval_labels_in_train: bool = True,
    max_tries: int = 512,
) -> Dict[SplitName, List[SampleMeta]]:
    """Split samples by unique sensor_id so sensors don't leak across splits.

    Notes:
    - This splitter groups by sensor_id (no leakage).
    - By default it also retries shuffles to ensure the training split contains
      all labels present in the dataset. This avoids a common failure mode where
      the test split contains a class that never appears in training.
    """

    ratios_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratios_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios_sum}")

    sensor_to_samples: Dict[str, List[SampleMeta]] = {}
    for s in samples:
        if not s.sensor_id:
            raise ValueError("sensor_id is required for train/val/test split")
        sensor_to_samples.setdefault(s.sensor_id, []).append(s)

    sensors = sorted(sensor_to_samples.keys())
    n = len(sensors)
    if n < 3:
        raise ValueError(
            f"Need at least 3 unique sensor_id values for a non-empty train/val/test split; got {n}"  # noqa: E501
        )

    labels_all = {s.condition for s in samples if s.condition is not None}

    def _labels(split_samples: Sequence[SampleMeta]) -> set[str]:
        return {s.condition for s in split_samples if s.condition is not None}

    def collect(sensor_ids: Sequence[str]) -> List[SampleMeta]:
        out: List[SampleMeta] = []
        for sid in sensor_ids:
            out.extend(sensor_to_samples[sid])
        return out

    def _sensor_counts(shuffled: Sequence[str]) -> Tuple[int, int, int]:
        # Choose counts that (a) respect ratios approximately, and (b) guarantee
        # all splits are non-empty.
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        # Ensure room for at least 1 test sensor
        while n_train + n_val > n - 1:
            if n_train >= n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break
        n_test = n - n_train - n_val
        if n_test < 1:
            # Should not happen given the while loop, but keep safe.
            n_test = 1
            if n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
        return n_train, n_val, n_test

    def _make_split(shuffled_sensors: Sequence[str]) -> Dict[SplitName, List[SampleMeta]]:
        n_train, n_val, n_test = _sensor_counts(shuffled_sensors)
        train_sensors = list(shuffled_sensors[:n_train])
        val_sensors = list(shuffled_sensors[n_train : n_train + n_val])
        test_sensors = list(shuffled_sensors[n_train + n_val : n_train + n_val + n_test])
        return {
            "train": collect(train_sensors),
            "val": collect(val_sensors),
            "test": collect(test_sensors),
        }

    # Retry shuffles to satisfy label coverage constraints.
    rng = np.random.default_rng(seed)
    tries = max(1, int(max_tries))

    best_split: Optional[Dict[SplitName, List[SampleMeta]]] = None
    best_score: Tuple[int, int, int] = (-1, -1, -1)

    for _ in range(tries):
        shuffled = list(sensors)
        rng.shuffle(shuffled)
        split = _make_split(shuffled)

        labels_train = _labels(split["train"])
        labels_val = _labels(split["val"])
        labels_test = _labels(split["test"])

        missing_in_train = len(labels_all - labels_train)
        unseen_in_val = len(labels_val - labels_train)
        unseen_in_test = len(labels_test - labels_train)

        ok = True
        if ensure_train_has_all_labels and missing_in_train != 0:
            ok = False
        if ensure_eval_labels_in_train and (unseen_in_val != 0 or unseen_in_test != 0):
            ok = False

        # Prefer: all labels in train, then minimize unseen eval labels, then maximize train labels.
        score = (-missing_in_train, -(unseen_in_val + unseen_in_test), len(labels_train))
        if score > best_score:
            best_score = score
            best_split = split

        if ok:
            return split

    # Fall back to the best we found.
    if best_split is None:
        # Should never happen.
        return _make_split(sensors)
    return best_split


def loso_folds(
    samples: Sequence[SampleMeta],
    *,
    val_ratio_within_train: float = 0.15,
    seed: int = 42,
) -> List[Tuple[str, Dict[SplitName, List[SampleMeta]]]]:
    """Create Leave-One-Sensor-Out folds.

    For each sensor_id S:
      - test: all samples from S
      - train/val: remaining sensors split by sensor_id

    Returns: list of (heldout_sensor_id, split_dict)
    """

    sensor_to_samples: Dict[str, List[SampleMeta]] = {}
    for s in samples:
        if not s.sensor_id:
            raise ValueError("sensor_id is required for LOSO")
        sensor_to_samples.setdefault(s.sensor_id, []).append(s)

    sensors = sorted(sensor_to_samples.keys())
    folds: List[Tuple[str, Dict[SplitName, List[SampleMeta]]]] = []

    # Use one RNG so each fold's train/val split differs (but stays deterministic).
    rng = np.random.default_rng(seed)

    for held_out in sensors:
        test_samples = list(sensor_to_samples[held_out])
        remaining = [s for sid in sensors if sid != held_out for s in sensor_to_samples[sid]]

        # Split remaining sensors into train/val (still grouped by sensor)
        rem_sensors = sorted({s.sensor_id for s in remaining if s.sensor_id})
        rng.shuffle(rem_sensors)

        n_val = max(1, int(round(len(rem_sensors) * val_ratio_within_train)))
        val_sensors = rem_sensors[:n_val]
        train_sensors = rem_sensors[n_val:]

        train_samples = [s for s in remaining if s.sensor_id in set(train_sensors)]
        val_samples = [s for s in remaining if s.sensor_id in set(val_sensors)]

        folds.append(
            (
                held_out,
                {
                    "train": train_samples,
                    "val": val_samples,
                    "test": test_samples,
                },
            )
        )

    return folds


def save_label_mapping(mapping: Dict[str, int], path: Path) -> None:
    path = Path(path)
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True))
