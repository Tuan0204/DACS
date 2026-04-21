from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def save_labels(path: str | Path, class_names: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)


def load_labels(path: str | Path) -> list[str]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
