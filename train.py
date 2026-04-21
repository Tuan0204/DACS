from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.leaf_disease.config import load_config
from src.leaf_disease.data import create_dataloaders
from src.leaf_disease.engine import evaluate_model, train_one_epoch
from src.leaf_disease.io import save_checkpoint, save_labels
from src.leaf_disease.modeling import build_model, count_model_size_mb
from src.leaf_disease.utils import current_timestamp, resolve_device, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train leaf disease classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    train_loader, val_loader, _, stats = create_dataloaders(
        data_dir=cfg.data_dir,
        image_size=cfg.image_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    model = build_model(cfg.model_name, num_classes=stats["num_classes"], pretrained=cfg.pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            save_checkpoint(
                output_dir / "best_model.pt",
                {
                    "model_state": model.state_dict(),
                    "model_name": cfg.model_name,
                    "num_classes": stats["num_classes"],
                    "class_names": stats["class_names"],
                    "image_size": cfg.image_size,
                },
            )

    save_labels(output_dir / "class_names.json", stats["class_names"])
    save_json({"history": history}, Path("reports") / f"train_history_{current_timestamp()}.json")

    summary = {
        "model_name": cfg.model_name,
        "num_classes": stats["num_classes"],
        "model_size_mb": round(count_model_size_mb(model), 2),
        "best_val_macro_f1": round(best_val_f1, 4),
        "train_size": stats["train_size"],
        "val_size": stats["val_size"],
        "test_size": stats["test_size"],
        "device": str(device),
    }
    save_json(summary, Path("reports") / "train_summary.json")
    print("Saved checkpoint to", output_dir / "best_model.pt")


if __name__ == "__main__":
    main()
