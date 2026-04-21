from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    seed: int
    data_dir: str
    image_size: int
    val_size: float
    test_size: float
    batch_size: int
    num_workers: int
    model_name: str
    pretrained: bool
    learning_rate: float
    weight_decay: float
    epochs: int
    output_dir: str
    device: str


def load_config(config_path: str | Path) -> TrainConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    return TrainConfig(
        seed=int(raw.get("seed", 42)),
        data_dir=str(raw["data_dir"]),
        image_size=int(raw.get("image_size", 224)),
        val_size=float(raw.get("val_size", 0.1)),
        test_size=float(raw.get("test_size", 0.1)),
        batch_size=int(raw.get("batch_size", 32)),
        num_workers=int(raw.get("num_workers", 2)),
        model_name=str(raw.get("model_name", "resnet18")),
        pretrained=bool(raw.get("pretrained", True)),
        learning_rate=float(raw.get("learning_rate", 1e-3)),
        weight_decay=float(raw.get("weight_decay", 1e-4)),
        epochs=int(raw.get("epochs", 10)),
        output_dir=str(raw.get("output_dir", "models")),
        device=str(raw.get("device", "auto")),
    )
