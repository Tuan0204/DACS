from __future__ import annotations

import argparse
from pathlib import Path

from src.leaf_disease.config import load_config
from src.leaf_disease.data import create_dataloaders
from src.leaf_disease.engine import evaluate_model
from src.leaf_disease.io import load_checkpoint
from src.leaf_disease.modeling import build_model, count_model_size_mb
from src.leaf_disease.utils import resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate leaf disease classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.device)

    _, _, test_loader, stats = create_dataloaders(
        data_dir=cfg.data_dir,
        image_size=cfg.image_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))
    model_name = checkpoint.get("model_name", cfg.model_name)
    num_classes = checkpoint.get("num_classes", stats["num_classes"])

    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    metrics = evaluate_model(model, test_loader, device)
    metrics["model_name"] = model_name
    metrics["model_size_mb"] = round(count_model_size_mb(model), 2)
    metrics["device"] = str(device)

    out_path = Path("reports") / "metrics_test.json"
    save_json(metrics, out_path)

    print("Test metrics:")
    for k, v in metrics.items():
        print(f"- {k}: {v}")
    print("Saved to", out_path)


if __name__ == "__main__":
    main()
