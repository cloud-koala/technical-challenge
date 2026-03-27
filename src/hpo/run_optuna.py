from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import optuna
import yaml
from optuna.trial import TrialState

from src.data.dataloader import (
    Part3Metadata,
    SampleMeta,
    build_label_mapping,
    loso_folds,
    sensor_class_distribution,
    split_by_sensor_id,
)
from src.training.config import LoadedConfig, load_config
from src.training.train import _train_one


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return None
    if isinstance(raw, dict):
        return raw
    return None


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.write_text(src.read_text())


def _update_study_artifacts_after_trial(
    *,
    study_dir: Path,
    trial_number: int,
    val_f1_best: float,
    trial_dir: Path,
    cfg_raw: Dict[str, Any],
    sampled: Dict[str, Any],
) -> None:
    """Update parent-study artifacts incrementally.

    This runs after every completed trial so that early-interrupted studies still
    have (a) a running leaderboard and (b) best-so-far config + metrics copied to
    the study folder.
    """

    study_dir.mkdir(parents=True, exist_ok=True)

    # Update running leaderboard (top trials so far)
    leaderboard_path = study_dir / "best_configs_partial.json"
    leaderboard: List[Dict[str, object]] = []
    if leaderboard_path.exists():
        try:
            prev = json.loads(leaderboard_path.read_text())
            if isinstance(prev, list):
                leaderboard = [x for x in prev if isinstance(x, dict)]
        except Exception:
            leaderboard = []

    entry = {
        "trial_number": int(trial_number),
        "value": float(val_f1_best),
        "params": dict(sampled),
        "trial_dir": str(trial_dir),
        "config_path": str(trial_dir / "config_trial.json"),
        "result_path": str(trial_dir / "hpo_result.json"),
        "params_path": str(trial_dir / "hpo_params.json"),
    }

    # Replace existing trial entry if present
    leaderboard = [e for e in leaderboard if int(e.get("trial_number", -1)) != int(trial_number)]
    leaderboard.append(entry)
    leaderboard.sort(key=lambda e: float(e.get("value", float("-inf"))), reverse=True)
    leaderboard_path.write_text(json.dumps(leaderboard, indent=2))

    # Track best-so-far
    best_so_far_path = study_dir / "best_so_far.json"
    best_so_far = _load_json_if_exists(best_so_far_path) or {}
    prev_best_val = float(best_so_far.get("best_value", float("-inf")))
    prev_best_trial = int(best_so_far.get("best_trial_number", -1))

    if float(val_f1_best) >= prev_best_val:
        best_so_far_path.write_text(
            json.dumps(
                {
                    "best_value": float(val_f1_best),
                    "best_trial_number": int(trial_number),
                    "previous_best_value": prev_best_val,
                    "previous_best_trial_number": prev_best_trial,
                },
                indent=2,
            )
        )

        # Copy best config up to study folder
        (study_dir / "best_config.json").write_text(json.dumps(cfg_raw, indent=2))

        # Copy key metrics up to study folder (fallback to metrics_test.json for older trials)
        _copy_if_exists(trial_dir / "metrics_val_best.json", study_dir / "metrics_val_best.json")
        _copy_if_exists(trial_dir / "metrics_val_latest.json", study_dir / "metrics_val_latest.json")

        if (trial_dir / "metrics_test_best.json").exists() or (trial_dir / "metrics_test_latest.json").exists():
            _copy_if_exists(trial_dir / "metrics_test_best.json", study_dir / "metrics_test_best.json")
            _copy_if_exists(trial_dir / "metrics_test_latest.json", study_dir / "metrics_test_latest.json")
        else:
            # Back-compat for trials created before metrics_test_{best,latest}.json existed
            _copy_if_exists(trial_dir / "metrics_test.json", study_dir / "metrics_test_best.json")
            _copy_if_exists(trial_dir / "metrics_test.json", study_dir / "metrics_test_latest.json")

        _copy_if_exists(trial_dir / "hpo_result.json", study_dir / "best_hpo_result.json")



def _deep_set(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: Any = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _deep_get(d: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    parts = dotted_key.split(".")
    cur: Any = d
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


@dataclass(frozen=True)
class SpaceParam:
    name: str
    type: str
    spec: Dict[str, Any]


def _load_space(path: Path) -> Dict[str, Any]:
    with Path(path).open("r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("search space YAML must be a mapping")
    if "base_config" not in raw:
        raise ValueError("search space must define base_config")
    if "search_space" not in raw:
        raise ValueError("search space must define search_space")
    return raw


def _make_sampler(name: str, seed: int) -> optuna.samplers.BaseSampler:
    name = name.lower()
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed, multivariate=True)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"Unknown sampler: {name}")


def _make_pruner(name: str) -> optuna.pruners.BasePruner:
    name = name.lower()
    if name in {"none", "off", "disabled"}:
        return optuna.pruners.NopPruner()
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner()
    if name in {"asha", "successivehalving"}:
        return optuna.pruners.SuccessiveHalvingPruner()
    raise ValueError(f"Unknown pruner: {name}")


def _suggest(trial: optuna.Trial, p: SpaceParam) -> Any:
    t = p.type.lower()
    if t == "float":
        low = float(p.spec["low"])
        high = float(p.spec["high"])
        log = bool(p.spec.get("log", False))
        return trial.suggest_float(p.name, low, high, log=log)
    if t == "int":
        low = int(p.spec["low"])
        high = int(p.spec["high"])
        step = int(p.spec.get("step", 1))
        log = bool(p.spec.get("log", False))
        return trial.suggest_int(p.name, low, high, step=step, log=log)
    if t == "categorical":
        choices = p.spec["choices"]
        if not isinstance(choices, list) or len(choices) == 0:
            raise ValueError(f"categorical param {p.name} must have non-empty choices")
        return trial.suggest_categorical(p.name, choices)
    raise ValueError(f"Unknown param type: {p.type}")


def _build_conv_net_arch(params: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    """Return (block_channels, kernel_sizes) for conv_net based on HPO params."""
    n_blocks = int(params.get("hpo.num_blocks", 4))
    width = int(params.get("hpo.block_width", 32))
    mult = int(params.get("hpo.width_multiplier", 1))
    kernel_set = params.get("hpo.kernel_set", [1, 3, 7, 21])
    if not isinstance(kernel_set, list) or len(kernel_set) == 0:
        raise ValueError("hpo.kernel_set must be a non-empty list")

    # Simple, robust pattern: half of blocks at width, half at width*mult.
    split = int(math.ceil(n_blocks / 2))
    chans = [width] * split + [width * mult] * (n_blocks - split)
    kernels = [int(k) for k in kernel_set]
    return chans, kernels


def _baseline_params_from_base_config(
    *,
    base_cfg_raw: Dict[str, Any],
    space_raw: Dict[str, Any],
) -> Dict[str, Any]:
    """Create an enqueue_trial() param dict that reproduces base_config as closely as possible.

    Only includes keys that exist in the declared search_space.
    """
    search_space = space_raw.get("search_space", {})
    if not isinstance(search_space, dict):
        raise ValueError("search_space must be a mapping")

    out: Dict[str, Any] = {}
    for k in search_space.keys():
        k = str(k)
        if k.startswith("hpo."):
            continue
        val = _deep_get(base_cfg_raw, k, None)
        if val is not None:
            out[k] = val

    # Derived hpo.* knobs (if present in the space)
    model_name = str(_deep_get(base_cfg_raw, "model.name", "")).lower()
    if model_name in {"conv_net", "conv"}:
        base_block_channels = _deep_get(base_cfg_raw, "model.block_channels", None)
        base_kernel_sizes = _deep_get(base_cfg_raw, "model.kernel_sizes", None)

        if "hpo.num_blocks" in search_space and isinstance(base_block_channels, list) and len(base_block_channels) > 0:
            out["hpo.num_blocks"] = int(len(base_block_channels))

        if "hpo.block_width" in search_space and isinstance(base_block_channels, list) and len(base_block_channels) > 0:
            out["hpo.block_width"] = int(base_block_channels[0])

        if "hpo.width_multiplier" in search_space and isinstance(base_block_channels, list) and len(base_block_channels) > 0:
            # If widths are all equal, multiplier=1. Otherwise approximate using max/first.
            first = int(base_block_channels[0])
            mx = int(max(int(x) for x in base_block_channels))
            out["hpo.width_multiplier"] = 1 if first <= 0 else max(1, int(round(mx / first)))

        if "hpo.kernel_set" in search_space and isinstance(base_kernel_sizes, list) and len(base_kernel_sizes) > 0:
            out["hpo.kernel_set"] = [int(x) for x in base_kernel_sizes]

    return out


def _apply_fixed_overrides(cfg_raw: Dict[str, Any], fixed: Dict[str, Any]) -> None:
    for k, v in fixed.items():
        _deep_set(cfg_raw, k, v)


def _objective_factory(
    *,
    base_cfg_path: Path,
    space_raw: Dict[str, Any],
    study_name: str,
    out_root: Path,
) -> Any:
    base_cfg_loaded = load_config(base_cfg_path)

    search_space = space_raw.get("search_space", {})
    if not isinstance(search_space, dict):
        raise ValueError("search_space must be a mapping")

    params: List[SpaceParam] = []
    for k, spec in search_space.items():
        if not isinstance(spec, dict) or "type" not in spec:
            raise ValueError(f"search_space entry {k} must be a mapping with 'type'")
        params.append(SpaceParam(name=str(k), type=str(spec["type"]), spec=spec))

    fixed = space_raw.get("fixed_overrides", {})
    if fixed is None:
        fixed = {}
    if not isinstance(fixed, dict):
        raise ValueError("fixed_overrides must be a mapping")

    # Cache metadata + splits once, so trials only change model/training knobs.
    meta = Part3Metadata(base_cfg_loaded.train_metadata_path)
    meta_samples_all = list(meta.iter_samples())
    samples = [s for s in meta_samples_all if s.condition is not None]
    label_to_index = build_label_mapping([s.condition for s in samples if s.condition is not None])

    # Diagnostics: per-sensor label distribution (helps catch metadata oddities)
    (out_root / study_name).mkdir(parents=True, exist_ok=True)
    (out_root / study_name / "sensor_class_distribution.json").write_text(
        json.dumps(sensor_class_distribution(samples), indent=2, sort_keys=True)
    )

    protocol = base_cfg_loaded.protocol
    protocol_name = str(protocol.get("name", "split")).lower()

    seed = int(base_cfg_loaded.training.get("seed", 42))

    splits: Optional[Dict[str, List[SampleMeta]]] = None
    folds: Optional[List[Tuple[str, Dict[str, List[SampleMeta]]]]] = None

    if protocol_name == "split":
        split_cfg = protocol.get("split", {})
        splits = split_by_sensor_id(
            samples,
            train_ratio=float(split_cfg.get("train_ratio", 0.7)),
            val_ratio=float(split_cfg.get("val_ratio", 0.15)),
            test_ratio=float(split_cfg.get("test_ratio", 0.15)),
            seed=seed,
        )

    elif protocol_name == "loso":
        loso_cfg = protocol.get("loso", {})
        folds = loso_folds(
            samples,
            val_ratio_within_train=float(loso_cfg.get("val_ratio_within_train", 0.15)),
            seed=seed,
        )

    else:
        raise ValueError(f"HPO runner supports protocol.name=split or loso; got {protocol_name}")

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

    (out_root / study_name).mkdir(parents=True, exist_ok=True)
    if splits is not None:
        (out_root / study_name / "split_summary.json").write_text(json.dumps(_split_summary(splits), indent=2))
    if folds is not None:
        (out_root / study_name / "loso_folds.json").write_text(
            json.dumps(
                {
                    "n_folds": len(folds),
                    "heldout_sensors": [held for held, _ in folds],
                    "note": "Each fold holds out one sensor as the evaluation set; HPO objective is mean fold test_f1_macro.",
                },
                indent=2,
            )
        )

    output_order = base_cfg_loaded.output_order
    data_dir = base_cfg_loaded.train_data_dir

    def objective(trial: optuna.Trial) -> float:
        # Build trial config
        cfg_raw = deepcopy(base_cfg_loaded.raw)
        _apply_fixed_overrides(cfg_raw, fixed)

        sampled: Dict[str, Any] = {}
        for p in params:
            sampled[p.name] = _suggest(trial, p)

        # Apply standard dotted-path params
        for k, v in sampled.items():
            if k.startswith("hpo."):
                continue
            _deep_set(cfg_raw, k, v)

        # Derived architecture fields for conv_net
        if str(_deep_get(cfg_raw, "model.name", "")).lower() in {"conv_net", "conv"}:
            block_channels, kernel_sizes = _build_conv_net_arch(sampled)
            _deep_set(cfg_raw, "model.block_channels", block_channels)
            _deep_set(cfg_raw, "model.kernel_sizes", kernel_sizes)

        # Ensure optimizer defaults are consistent
        if _deep_get(cfg_raw, "training.optimizer", None) is None:
            _deep_set(cfg_raw, "training.optimizer", "adamw")

        # Per-trial output
        save_per_trial = bool(space_raw.get("artifacts", {}).get("save_per_trial", True))
        if save_per_trial:
            trial_dir = out_root / study_name / f"trial_{trial.number:05d}"
            _deep_set(cfg_raw, "artifacts.output_dir", str(trial_dir.parent))
            _deep_set(cfg_raw, "artifacts.run_name", trial_dir.name)
        else:
            _deep_set(cfg_raw, "artifacts.output_dir", str(out_root / study_name))
            _deep_set(cfg_raw, "artifacts.run_name", f"trial_{trial.number:05d}")

        cfg = LoadedConfig(raw=cfg_raw)

        trial_root = Path(cfg.artifacts.get("output_dir", "artifacts")) / str(cfg.artifacts.get("run_name", "run"))

        if splits is not None:
            def epoch_cb(epoch: int, val_f1: float, best_val_f1: float) -> None:
                # Report intermediate values so Hyperband/ASHA can prune.
                trial.report(val_f1, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            metrics = _train_one(
                fold_name=f"hpo_trial_{trial.number}",
                samples_split=splits,
                label_to_index=label_to_index,
                data_dir=data_dir,
                output_order=output_order,
                cfg=cfg,
                out_dir=trial_root,
                epoch_callback=epoch_cb,
            )

        else:
            assert folds is not None

            fold_results: List[Dict[str, object]] = []
            fold_test_f1: List[float] = []
            fold_test_acc: List[float] = []
            fold_val_best: List[float] = []

            for i, (heldout_sensor, split_dict) in enumerate(folds, start=1):
                fold_dir = trial_root / "loso" / heldout_sensor
                m = _train_one(
                    fold_name=f"hpo_loso_{heldout_sensor}",
                    samples_split=split_dict,
                    label_to_index=label_to_index,
                    data_dir=data_dir,
                    output_order=output_order,
                    cfg=cfg,
                    out_dir=fold_dir,
                    epoch_callback=None,
                )
                fold_results.append({"heldout_sensor": heldout_sensor, **m})
                fold_test_f1.append(float(m["test_f1_macro"]))
                fold_test_acc.append(float(m["test_accuracy"]))
                fold_val_best.append(float(m["val_f1_macro_best"]))

                running_mean = float(sum(fold_test_f1) / len(fold_test_f1))
                trial.report(running_mean, step=i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            mean_test_f1 = float(sum(fold_test_f1) / max(1, len(fold_test_f1)))
            mean_test_acc = float(sum(fold_test_acc) / max(1, len(fold_test_acc)))
            mean_val_best = float(sum(fold_val_best) / max(1, len(fold_val_best)))

            (trial_root / "loso_summary.json").parent.mkdir(parents=True, exist_ok=True)
            (trial_root / "loso_summary.json").write_text(json.dumps(fold_results, indent=2))

            # Emit top-level metrics files so parent-study copying works uniformly.
            (trial_root / "metrics_test_best.json").write_text(
                json.dumps({"epoch": 0, "accuracy": mean_test_acc, "f1_macro": mean_test_f1}, indent=2)
            )
            (trial_root / "metrics_test_latest.json").write_text(
                json.dumps({"epoch": 0, "accuracy": mean_test_acc, "f1_macro": mean_test_f1}, indent=2)
            )
            (trial_root / "metrics_val_best.json").write_text(
                json.dumps({"epoch": 0, "accuracy": 0.0, "f1_macro": mean_val_best}, indent=2)
            )
            (trial_root / "metrics_val_latest.json").write_text(
                json.dumps({"epoch": 0, "accuracy": 0.0, "f1_macro": mean_val_best}, indent=2)
            )

            metrics = {
                # For compatibility with the rest of the runner, treat this as the objective value.
                "val_f1_macro_best": mean_test_f1,
                "test_accuracy": mean_test_acc,
                "test_f1_macro": mean_test_f1,
                "loso_val_f1_macro_best_mean": mean_val_best,
            }

        # Persist trial params + result alongside artifacts
        if save_per_trial:
            trial_dir = out_root / study_name / f"trial_{trial.number:05d}"
            (trial_dir / "config_trial.json").write_text(json.dumps(cfg_raw, indent=2))
            (trial_dir / "hpo_params.json").write_text(json.dumps(sampled, indent=2))
            (trial_dir / "hpo_result.json").write_text(json.dumps(metrics, indent=2))
            
            # Update parent study artifacts incrementally (works even if you Ctrl+C later)
            _update_study_artifacts_after_trial(
                study_dir=out_root / study_name,
                trial_number=int(trial.number),
                val_f1_best=float(metrics["val_f1_macro_best"]),
                trial_dir=trial_dir,
                cfg_raw=cfg_raw,
                sampled=sampled,
            )

        return float(metrics["val_f1_macro_best"])

    return objective


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", type=str, required=True, help="Path to a search space YAML under search_spaces/")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=0, help="Timeout in seconds (0 = no timeout)")
    ap.add_argument("--study-name", type=str, default="conv_net_hpo")
    ap.add_argument(
        "--storage",
        type=str,
        default="sqlite:///artifacts/hpo/optuna_study.db",
        help="Optuna storage URL (sqlite:///...)",
    )
    ap.add_argument("--sampler", type=str, default=None, help="Override sampler (tpe|random)")
    ap.add_argument("--pruner", type=str, default=None, help="Override pruner (hyperband|asha|none)")
    ap.add_argument(
        "--enqueue-baseline",
        action="store_true",
        help="Enqueue a first trial matching base_config values (warm start).",
    )
    args = ap.parse_args()

    space_path = Path(args.space)
    space_raw = _load_space(space_path)

    base_cfg_path = Path(space_raw["base_config"])
    if not base_cfg_path.exists():
        raise FileNotFoundError(f"base_config not found: {base_cfg_path}")

    engine = space_raw.get("engine", {})
    if not isinstance(engine, dict):
        raise ValueError("engine must be a mapping")

    seed = int(engine.get("seed", 42))
    sampler_name = str(args.sampler or engine.get("sampler", "tpe"))
    pruner_name = str(args.pruner or engine.get("pruner", "hyperband"))

    sampler = _make_sampler(sampler_name, seed=seed)
    pruner = _make_pruner(pruner_name)

    out_root = Path(space_raw.get("artifacts", {}).get("output_root", "artifacts/hpo"))
    out_root.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=str(args.study_name),
        direction=str(space_raw.get("study", {}).get("direction", "maximize")),
        sampler=sampler,
        pruner=pruner,
        storage=str(args.storage),
        load_if_exists=True,
    )

    # Warm start: enqueue baseline trial derived from base_config.
    # If the study is empty, do this automatically; otherwise require the flag.
    should_enqueue = bool(args.enqueue_baseline) or len(study.trials) == 0
    if should_enqueue:
        base_cfg_loaded = load_config(base_cfg_path)
        fixed = space_raw.get("fixed_overrides", {}) or {}
        if not isinstance(fixed, dict):
            raise ValueError("fixed_overrides must be a mapping")

        base_raw = deepcopy(base_cfg_loaded.raw)
        _apply_fixed_overrides(base_raw, fixed)
        baseline_params = _baseline_params_from_base_config(base_cfg_raw=base_raw, space_raw=space_raw)

        # Only enqueue if it has at least one param (Optuna requires dict).
        if len(baseline_params) > 0:
            study.enqueue_trial(baseline_params)

    objective = _objective_factory(
        base_cfg_path=base_cfg_path,
        space_raw=space_raw,
        study_name=str(args.study_name),
        out_root=out_root,
    )

    study.optimize(objective, n_trials=int(args.n_trials), timeout=None if int(args.timeout) <= 0 else int(args.timeout))

    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
    }
    (out_root / str(args.study_name) / "best.json").parent.mkdir(parents=True, exist_ok=True)
    (out_root / str(args.study_name) / "best.json").write_text(json.dumps(best, indent=2))

    # Dump ranked best config list into the study folder for easy inspection.
    study_dir = out_root / str(args.study_name)
    direction = study.direction
    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    reverse = True if str(direction).lower().endswith("maximize") else False
    complete_trials.sort(key=lambda t: float(t.value), reverse=reverse)

    ranked: List[Dict[str, object]] = []
    for rank, t in enumerate(complete_trials, start=1):
        trial_dir = study_dir / f"trial_{t.number:05d}"
        ranked.append(
            {
                "rank": int(rank),
                "trial_number": int(t.number),
                "value": float(t.value),
                "params": dict(t.params),
                "trial_dir": str(trial_dir),
                "config_path": str(trial_dir / "config_trial.json"),
                "result_path": str(trial_dir / "hpo_result.json"),
                "params_path": str(trial_dir / "hpo_params.json"),
            }
        )

    (study_dir / "best_configs.json").write_text(json.dumps(ranked, indent=2))

    # Convenience: copy the best trial config to best_config.json if it exists.
    best_trial_dir = study_dir / f"trial_{study.best_trial.number:05d}"
    best_cfg_path = best_trial_dir / "config_trial.json"
    if best_cfg_path.exists():
        (study_dir / "best_config.json").write_text(best_cfg_path.read_text())

    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
