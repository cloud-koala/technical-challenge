from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple
import warnings

import numpy as np


def resample_to_frequency(
    time: np.ndarray,
    signals: np.ndarray,
    *,
    target_hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Public wrapper around internal resampler."""
    return _resample_to_frequency(time, signals, target_hz=target_hz)


def _estimate_rpm_from_resampled(
    sig_ds: np.ndarray,
    *,
    fs_hz: float,
    min_rpm: float,
    max_rpm: float,
    top_k: int = 8,
    harmonics: int = 5,
) -> Optional[float]:
    """Estimate RPM from a resampled multi-channel signal.

    Heuristic approach:
    - Compute spectrum of the vector magnitude.
    - Search candidate fundamentals in the band [min_rpm/60, max_rpm/60].
    - Score candidates by harmonic support.

    Returns RPM, or None if estimation fails.
    """
    if sig_ds.ndim != 2 or sig_ds.shape[0] < 16:
        return None
    fs_hz = float(fs_hz)
    if fs_hz <= 0:
        return None

    min_rpm = float(min_rpm)
    max_rpm = float(max_rpm)
    if min_rpm <= 0 or max_rpm <= 0 or max_rpm <= min_rpm:
        return None

    # Use vector magnitude to reduce axis dependence.
    x = np.linalg.norm(sig_ds.astype(np.float64, copy=False), axis=1)
    x = x - float(np.mean(x))
    n = int(x.shape[0])
    if n < 16:
        return None

    # Window to reduce spectral leakage.
    w = np.hanning(n)
    xw = x * w

    spec = np.fft.rfft(xw)
    power = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float64)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)

    f_min = min_rpm / 60.0
    f_max = max_rpm / 60.0
    if f_max <= 0:
        return None

    band = np.where((freqs >= f_min) & (freqs <= f_max))[0]
    if band.size < 5:
        return None

    p_band = power[band]
    # Drop DC-ish region just in case.
    if np.all(p_band <= 0):
        return None

    k = int(max(1, min(int(top_k), int(band.size))))
    # Take top-k indices in the band as candidates.
    cand_rel = np.argpartition(p_band, -k)[-k:]
    cand_idx = band[cand_rel]

    def p_at(freq: float) -> float:
        if freq <= 0:
            return 0.0
        # nearest bin
        i = int(np.clip(np.searchsorted(freqs, freq), 1, len(freqs) - 1))
        # choose closer of i-1 and i
        if abs(freqs[i] - freq) > abs(freqs[i - 1] - freq):
            i = i - 1
        return float(power[i])

    best_score = -1.0
    best_f = None
    for idx in cand_idx.tolist():
        f0 = float(freqs[int(idx)])
        if f0 <= 0:
            continue
        score = 0.0
        for h in range(1, int(harmonics) + 1):
            fh = f0 * float(h)
            if fh > f_max:
                break
            score += p_at(fh) / float(h)
        if score > best_score:
            best_score = score
            best_f = f0

    if best_f is None or best_f <= 0:
        return None
    rpm = 60.0 * best_f
    if rpm < min_rpm or rpm > max_rpm:
        return None
    return float(rpm)


def _order_spectrum_windows(
    windows: np.ndarray,
    *,
    fs_hz: float,
    rpm: float,
    order_max: float,
    n_bins: int,
    log_power: bool = True,
    per_window_standardize: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """Convert time-domain windows (N, L, C) into order-spectrum windows (N, B, C)."""
    if windows.ndim != 3:
        raise ValueError("windows must be (N, L, C)")
    if windows.shape[0] == 0:
        return np.zeros((0, int(n_bins), int(windows.shape[2])), dtype=np.float32)

    fs_hz = float(fs_hz)
    rpm = float(rpm)
    if fs_hz <= 0 or rpm <= 0:
        raise ValueError("fs_hz and rpm must be > 0")

    order_max = float(order_max)
    if order_max <= 0:
        raise ValueError("order_max must be > 0")
    n_bins = int(n_bins)
    if n_bins <= 4:
        raise ValueError("n_bins must be > 4")

    # Convert frequency axis to orders: order = f / (rpm/60)
    shaft_hz = rpm / 60.0

    n = int(windows.shape[1])
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz).astype(np.float64)
    orders = freqs / shaft_hz
    # Desired order bins
    order_bins = np.linspace(0.0, order_max, n_bins, dtype=np.float64)

    out = np.zeros((int(windows.shape[0]), n_bins, int(windows.shape[2])), dtype=np.float32)

    # Precompute Hann window
    hann = np.hanning(n).astype(np.float64)

    for i in range(int(windows.shape[0])):
        w = windows[i].astype(np.float64, copy=False)  # (L, C)
        # Demean per channel to reduce DC
        w = w - np.mean(w, axis=0, keepdims=True)
        w = w * hann[:, None]
        spec = np.fft.rfft(w, axis=0)
        power = (spec.real * spec.real + spec.imag * spec.imag)

        # Interpolate power to order bins for each channel.
        for c in range(int(windows.shape[2])):
            oc = np.interp(order_bins, orders, power[:, c]).astype(np.float32)
            if log_power:
                oc = np.log1p(oc)
            out[i, :, c] = oc

        if per_window_standardize:
            mu = out[i].mean(axis=0, keepdims=True)
            sd = out[i].std(axis=0, keepdims=True)
            sd = np.where(sd < float(eps), 1.0, sd)
            out[i] = (out[i] - mu) / sd

    return out


def _resample_to_frequency(
    time: np.ndarray,
    signals: np.ndarray,
    *,
    target_hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample multi-channel signals to a fixed frequency using linear interpolation."""
    if target_hz <= 0:
        raise ValueError("target_hz must be > 0")
    if time.ndim != 1:
        raise ValueError("time must be 1D")
    if signals.ndim != 2:
        raise ValueError("signals must be 2D (T, C)")
    if len(time) != signals.shape[0]:
        raise ValueError("time and signals length mismatch")

    # Ensure increasing time for interpolation.
    order = np.argsort(time)
    time_sorted = time[order]
    sig_sorted = signals[order]

    t0 = float(time_sorted[0])
    t1 = float(time_sorted[-1])
    if t1 <= t0:
        raise ValueError("Invalid time range")

    dt = 1.0 / float(target_hz)
    new_time = np.arange(t0, t1 + 1e-12, dt, dtype=np.float64)

    new_signals = np.zeros((len(new_time), sig_sorted.shape[1]), dtype=np.float32)
    for c in range(sig_sorted.shape[1]):
        new_signals[:, c] = np.interp(new_time, time_sorted, sig_sorted[:, c]).astype(np.float32)

    return new_time, new_signals


def _window_signals(
    signals: np.ndarray,
    *,
    window_size: int,
    step_size: Optional[int] = None,
) -> np.ndarray:
    """Create (N, L, C) windows from (T, C)."""
    if step_size is None:
        step_size = window_size
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be > 0")
    if signals.ndim != 2:
        raise ValueError("signals must be 2D")

    t, c = signals.shape
    if t < window_size:
        return np.zeros((0, window_size, c), dtype=np.float32)

    starts = range(0, t - window_size + 1, step_size)
    windows = np.stack([signals[s : s + window_size] for s in starts], axis=0).astype(np.float32)
    return windows


@dataclass
class ZNormalizer:
    mean_: Optional[np.ndarray] = None  # (C,)
    std_: Optional[np.ndarray] = None  # (C,)
    eps: float = 1e-8

    def fit_streaming(self, flat_batches: Iterable[np.ndarray]) -> "ZNormalizer":
        """Fit mean/std from an iterable of 2D arrays shaped (N, C).

        Uses a numerically-stable parallel/streaming variance update.
        """
        count = 0
        mean = None
        m2 = None

        for batch in flat_batches:  # type: ignore[assignment]
            if batch is None:
                continue
            batch = np.asarray(batch)
            if batch.ndim != 2 or batch.shape[0] == 0:
                continue

            x = batch.astype(np.float64, copy=False)
            b_count = int(x.shape[0])
            b_mean = x.mean(axis=0)
            b_m2 = ((x - b_mean) ** 2).sum(axis=0)

            if count == 0:
                count = b_count
                mean = b_mean
                m2 = b_m2
            else:
                assert mean is not None and m2 is not None
                total = count + b_count
                delta = b_mean - mean
                mean = mean + delta * (b_count / total)
                m2 = m2 + b_m2 + (delta**2) * (count * b_count / total)
                count = total

        if count == 0 or mean is None or m2 is None:
            raise ValueError("Cannot fit normalizer on empty data")

        var = m2 / float(count)
        std = np.sqrt(var)
        std = np.where(std < self.eps, 1.0, std)

        self.mean_ = mean.astype(np.float32)
        self.std_ = std.astype(np.float32)
        return self

    def fit(self, windows: np.ndarray) -> "ZNormalizer":
        """Fit on windows shaped (N, L, C)."""
        if windows.ndim != 3:
            raise ValueError("windows must be (N, L, C)")
        if windows.shape[0] == 0:
            raise ValueError("Cannot fit normalizer on empty windows")

        flat = windows.reshape(-1, windows.shape[-1]).astype(np.float64)
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        std = np.where(std < self.eps, 1.0, std)

        self.mean_ = mean.astype(np.float32)
        self.std_ = std.astype(np.float32)
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Normalizer is not fitted")
        return ((windows - self.mean_) / (self.std_ + self.eps)).astype(np.float32)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"mean": self.mean_, "std": self.std_, "eps": self.eps}, f)

    @classmethod
    def load(cls, path: Path) -> "ZNormalizer":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        obj = cls(eps=float(payload.get("eps", 1e-8)))
        obj.mean_ = payload["mean"]
        obj.std_ = payload["std"]
        return obj


class Preprocessor:
    """Preprocessing pipeline: downsample -> window -> z-norm."""

    def __init__(
        self,
        *,
        downsample_hz: float = 100.0,
        window_seconds: float = 0.05,
        step_seconds: Optional[float] = None,
        z_norm: bool = True,
        feature_mode: str = "time",
        # RPM / order-spectrum settings (used when feature_mode == 'order_spectrum')
        rpm_policy: str = "auto",
        rpm_min: float = 300.0,
        rpm_max: float = 6000.0,
        rpm_discrepancy_tol: float = 0.2,
        rpm_top_k: int = 8,
        rpm_harmonics: int = 5,
        order_max: float = 20.0,
        order_bins: int = 128,
        order_log_power: bool = True,
        order_per_window_standardize: bool = True,
    ):
        self.downsample_hz = float(downsample_hz)
        self.window_seconds = float(window_seconds)
        self.step_seconds = float(step_seconds) if step_seconds is not None else None
        self.z_norm = bool(z_norm)

        self.feature_mode = str(feature_mode).lower().strip()
        if self.feature_mode not in {"time", "order_spectrum"}:
            raise ValueError(f"Unsupported feature_mode: {self.feature_mode}")

        self.rpm_policy = str(rpm_policy).lower().strip()
        if self.rpm_policy not in {"auto", "validate", "trust", "estimate"}:
            raise ValueError(
                "Unsupported rpm_policy. Expected one of: auto, validate, trust, estimate"
            )
        self._rpm_warning_count = 0
        self._rpm_warning_limit = 5

        self.rpm_min = float(rpm_min)
        self.rpm_max = float(rpm_max)
        self.rpm_discrepancy_tol = float(rpm_discrepancy_tol)
        self.rpm_top_k = int(rpm_top_k)
        self.rpm_harmonics = int(rpm_harmonics)

        self.order_max = float(order_max)
        self.order_bins = int(order_bins)
        self.order_log_power = bool(order_log_power)
        self.order_per_window_standardize = bool(order_per_window_standardize)

        self.normalizer = ZNormalizer() if self.z_norm else None

    def _warn_rpm(self, message: str) -> None:
        if self._rpm_warning_count >= self._rpm_warning_limit:
            return
        self._rpm_warning_count += 1
        if self._rpm_warning_count == self._rpm_warning_limit:
            message = message + " (further RPM warnings suppressed)"
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    def _choose_rpm(self, *, provided_rpm: Optional[float], sig_ds: np.ndarray) -> float:
        """Choose an RPM value based on provided metadata and/or a signal-derived estimate.

        Policies:
        - auto: current behavior (estimate; override metadata if discrepancy is large)
        - validate: trust metadata when present, but estimate to validate and warn (never override)
        - trust: trust metadata when present; only estimate if metadata is missing/invalid
        - estimate: prefer estimate; fall back to metadata if estimation fails
        """

        rpm_prov = None
        if provided_rpm is not None:
            try:
                rpm_prov = float(provided_rpm)
            except Exception:
                rpm_prov = None

        def is_valid_rpm(r: Optional[float]) -> bool:
            if r is None:
                return False
            if not np.isfinite(r):
                return False
            if r <= 0:
                return False
            return self.rpm_min <= float(r) <= self.rpm_max

        rpm_prov_valid = is_valid_rpm(rpm_prov)

        # "trust" can skip estimation entirely when metadata is present.
        if self.rpm_policy == "trust" and rpm_prov_valid:
            return float(rpm_prov)

        rpm_est = _estimate_rpm_from_resampled(
            sig_ds,
            fs_hz=self.downsample_hz,
            min_rpm=self.rpm_min,
            max_rpm=self.rpm_max,
            top_k=self.rpm_top_k,
            harmonics=self.rpm_harmonics,
        )
        rpm_est_valid = is_valid_rpm(rpm_est)

        # estimate: use estimator if possible.
        if self.rpm_policy == "estimate":
            if rpm_est_valid:
                return float(rpm_est)
            if rpm_prov_valid:
                return float(rpm_prov)
            raise ValueError("Cannot determine RPM: estimation failed and no valid provided rpm")

        # validate: trust metadata when present, but warn if discrepancy is too large.
        if self.rpm_policy == "validate":
            if rpm_prov_valid:
                if rpm_est_valid:
                    rel_err = abs(float(rpm_est) - float(rpm_prov)) / max(abs(float(rpm_prov)), 1e-6)
                    if rel_err > max(0.0, self.rpm_discrepancy_tol):
                        self._warn_rpm(
                            f"Provided rpm={float(rpm_prov):.3f} disagrees with estimated rpm={float(rpm_est):.3f} "
                            f"(rel_err={rel_err:.3f} > tol={float(self.rpm_discrepancy_tol):.3f}); using provided rpm"
                        )
                else:
                    self._warn_rpm(
                        f"RPM validation skipped (estimation failed); using provided rpm={float(rpm_prov):.3f}"
                    )
                return float(rpm_prov)

            # No valid provided rpm: must estimate.
            if rpm_est_valid:
                return float(rpm_est)
            raise ValueError("Cannot determine RPM: no valid provided rpm and estimation failed")

        # auto: preserve previous behavior.
        if rpm_prov is None:
            if not rpm_est_valid:
                raise ValueError("Cannot determine RPM: no provided rpm and estimation failed")
            return float(rpm_est)

        # Metadata exists: if estimate fails, trust metadata (within range).
        if not rpm_est_valid:
            if not rpm_prov_valid:
                raise ValueError(f"Provided rpm={rpm_prov} invalid/out of range and estimation failed")
            return float(rpm_prov)

        # Both exist: if disagree too much, use estimate.
        if not rpm_prov_valid:
            return float(rpm_est)

        rel_err = abs(float(rpm_est) - float(rpm_prov)) / max(abs(float(rpm_est)), 1e-6)
        if rel_err > max(0.0, self.rpm_discrepancy_tol):
            return float(rpm_est)
        return float(rpm_prov)

    def window_size(self) -> int:
        return max(1, int(round(self.downsample_hz * self.window_seconds)))

    def step_size(self) -> int:
        if self.step_seconds is None:
            return self.window_size()
        return max(1, int(round(self.downsample_hz * self.step_seconds)))

    def estimate_rpm(self, *, time: np.ndarray, signals: np.ndarray) -> Optional[float]:
        """Estimate RPM from the signal (after resampling).

        This is intended for debugging/validation and does not apply rpm_policy.
        Returns None if estimation fails.
        """
        _, sig_ds = _resample_to_frequency(np.asarray(time), np.asarray(signals), target_hz=self.downsample_hz)
        return _estimate_rpm_from_resampled(
            sig_ds,
            fs_hz=self.downsample_hz,
            min_rpm=self.rpm_min,
            max_rpm=self.rpm_max,
            top_k=self.rpm_top_k,
            harmonics=self.rpm_harmonics,
        )

    def fit(self, *, times_and_signals) -> "Preprocessor":
        """Fit normalizer using training data only.

        Accepts any iterable of (time, signals).
        """
        if not self.z_norm:
            return self

        def flat_batches():
            produced = False
            for item in times_and_signals:
                # Accept either (time, signals) or (time, signals, rpm)
                if not isinstance(item, (tuple, list)):
                    raise ValueError("times_and_signals must yield tuples")
                if len(item) == 2:
                    time, signals = item
                    rpm = None
                elif len(item) == 3:
                    time, signals, rpm = item
                else:
                    raise ValueError("times_and_signals must yield (time, signals) or (time, signals, rpm)")

                _, sig_ds = _resample_to_frequency(np.asarray(time), np.asarray(signals), target_hz=self.downsample_hz)
                windows_t = _window_signals(sig_ds, window_size=self.window_size(), step_size=self.step_size())
                if windows_t.shape[0] == 0:
                    continue

                if self.feature_mode == "time":
                    windows = windows_t
                else:
                    rpm_used = self._choose_rpm(provided_rpm=None if rpm is None else float(rpm), sig_ds=sig_ds)
                    windows = _order_spectrum_windows(
                        windows_t,
                        fs_hz=self.downsample_hz,
                        rpm=rpm_used,
                        order_max=self.order_max,
                        n_bins=self.order_bins,
                        log_power=self.order_log_power,
                        per_window_standardize=self.order_per_window_standardize,
                    )

                if windows.shape[0] == 0:
                    continue
                produced = True
                yield windows.reshape(-1, windows.shape[-1])
            if not produced:
                return

        if self.normalizer is None:
            self.normalizer = ZNormalizer()
        try:
            self.normalizer.fit_streaming(flat_batches())
        except ValueError as e:
            msg = str(e)
            if "empty data" in msg.lower():
                raise ValueError(
                    "Cannot fit normalizer because preprocessing produced zero windows. "
                    f"Check your settings: downsample_hz={self.downsample_hz}, "
                    f"window_seconds={self.window_seconds} (window_size={self.window_size()} samples), "
                    f"step_seconds={self.step_seconds} (step_size={self.step_size()} samples). "
                    "Fix: reduce window_seconds and/or downsample_hz, or verify your time column units."
                ) from e
            raise
        return self

    def transform_with_rpm(
        self,
        *,
        time: np.ndarray,
        signals: np.ndarray,
        rpm: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """Like transform(), but also returns the RPM actually used.

        This is useful for RPM-conditioned models, because it applies the same
        metadata-vs-estimate resolution logic used for order-spectrum features.
        """
        _, sig_ds = _resample_to_frequency(time, signals, target_hz=self.downsample_hz)
        windows_t = _window_signals(sig_ds, window_size=self.window_size(), step_size=self.step_size())

        rpm_used = self._choose_rpm(provided_rpm=rpm, sig_ds=sig_ds)

        if self.feature_mode == "time" or windows_t.shape[0] == 0:
            windows = windows_t
        else:
            windows = _order_spectrum_windows(
                windows_t,
                fs_hz=self.downsample_hz,
                rpm=rpm_used,
                order_max=self.order_max,
                n_bins=self.order_bins,
                log_power=self.order_log_power,
                per_window_standardize=self.order_per_window_standardize,
            )

        if self.z_norm and self.normalizer is not None:
            windows = self.normalizer.transform(windows)
        return windows, float(rpm_used)

    def transform(self, *, time: np.ndarray, signals: np.ndarray, rpm: Optional[float] = None) -> np.ndarray:
        """Return windows shaped (N, L, C).

        If feature_mode == 'order_spectrum', L is the number of order bins.
        """
        windows, _ = self.transform_with_rpm(time=time, signals=signals, rpm=rpm)
        return windows

    def save_normalizer(self, path: Path) -> None:
        if not self.z_norm or self.normalizer is None:
            raise RuntimeError("No normalizer to save")
        self.normalizer.save(path)

    def load_normalizer(self, path: Path) -> None:
        if not self.z_norm:
            raise RuntimeError("Preprocessor configured with z_norm=False")
        self.normalizer = ZNormalizer.load(path)
