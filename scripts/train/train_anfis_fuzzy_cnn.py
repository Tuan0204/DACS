from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.leaf_disease.io import save_checkpoint, save_labels
from src.leaf_disease.paper_anfis_fuzzy_cnn import (
    PaperAnfisFuzzyCNN,
    create_paper_dataloaders,
    evaluate_model,
    train_one_epoch,
)
from src.leaf_disease.utils import current_timestamp, resolve_device, save_json, seed_everything


@dataclass
class PaperAnfisConfig:
    seed: int
    data_dir: str
    image_size: int
    val_size: float
    test_size: float
    batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    epochs: int
    output_dir: str
    device: str
    num_rules: int
    dropout: float


def load_config(config_path: str | Path) -> PaperAnfisConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return PaperAnfisConfig(
        seed=int(raw.get("seed", 42)),
        data_dir=str(raw["data_dir"]),
        image_size=int(raw.get("image_size", 224)),
        val_size=float(raw.get("val_size", 0.1)),
        test_size=float(raw.get("test_size", 0.1)),
        batch_size=int(raw.get("batch_size", 16)),
        num_workers=int(raw.get("num_workers", 2)),
        learning_rate=float(raw.get("learning_rate", 2e-4)),
        weight_decay=float(raw.get("weight_decay", 1e-4)),
        epochs=int(raw.get("epochs", 12)),
        output_dir=str(raw.get("output_dir", "models/anfis_fuzzy_cnn")),
        device=str(raw.get("device", "auto")),
        num_rules=int(raw.get("num_rules", 8)),
        dropout=float(raw.get("dropout", 0.3)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train paper-style ANFIS-Fuzzy-CNN model")
    parser.add_argument("--config", type=str, default="configs/anfis_fuzzy_cnn.yaml")
    return parser.parse_args()


def _gather_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    train_loader, val_loader, test_loader, stats = create_paper_dataloaders(
        data_dir=cfg.data_dir,
        image_size=cfg.image_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    model = PaperAnfisFuzzyCNN(
        num_classes=stats["num_classes"],
        num_rules=cfg.num_rules,
        image_size=cfg.image_size,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

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
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            save_checkpoint(
                output_dir / "best_model.pt",
                {
                    "model_state": model.state_dict(),
                    "model_name": "paper_anfis_fuzzy_cnn",
                    "num_classes": stats["num_classes"],
                    "class_names": stats["class_names"],
                    "image_size": cfg.image_size,
                    "num_rules": cfg.num_rules,
                    "dropout": cfg.dropout,
                },
            )

    test_metrics = evaluate_model(model, test_loader, device, return_predictions=True)
    test_summary = _gather_metrics(test_metrics["y_true"], test_metrics["y_pred"])

    save_labels(output_dir / "class_names.json", stats["class_names"])
    save_json({"history": history}, Path("reports") / f"train_history_paper_anfis_{current_timestamp()}.json")

    summary = {
        "model_name": "paper_anfis_fuzzy_cnn",
        "num_classes": stats["num_classes"],
        "best_val_macro_f1": round(best_val_f1, 4),
        "test_accuracy": round(test_summary["accuracy"], 4),
        "test_precision": round(test_summary["precision"], 4),
        "test_recall": round(test_summary["recall"], 4),
        "test_macro_f1": round(test_summary["macro_f1"], 4),
        "train_size": stats["train_size"],
        "val_size": stats["val_size"],
        "test_size": stats["test_size"],
        "device": str(device),
    }
    save_json(summary, Path("reports") / "train_summary_paper_anfis.json")
    print("Saved checkpoint to", output_dir / "best_model.pt")


if __name__ == "__main__":
    main()