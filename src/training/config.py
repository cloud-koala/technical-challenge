from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class LoadedConfig:
    raw: Dict[str, Any]

    @property
    def part3_root(self) -> Path:
        return Path(self.raw["data"]["part3_root"])

    @property
    def train_metadata_path(self) -> Path:
        return self.part3_root / self.raw["data"]["train_metadata"]

    @property
    def train_data_dir(self) -> Path:
        return self.part3_root / self.raw["data"]["train_data_dir"]

    @property
    def output_order(self) -> List[str]:
        return list(self.raw.get("orientation", {}).get("output_order", ["vertical", "axial", "horizontal"]))

    @property
    def preprocessing(self) -> Dict[str, Any]:
        return dict(self.raw.get("preprocessing", {}))

    @property
    def model(self) -> Dict[str, Any]:
        return dict(self.raw.get("model", {}))

    @property
    def training(self) -> Dict[str, Any]:
        return dict(self.raw.get("training", {}))

    @property
    def protocol(self) -> Dict[str, Any]:
        return dict(self.raw.get("protocol", {}))

    @property
    def artifacts(self) -> Dict[str, Any]:
        return dict(self.raw.get("artifacts", {}))


def load_config(path: Path) -> LoadedConfig:
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping")
    return LoadedConfig(raw=raw)
