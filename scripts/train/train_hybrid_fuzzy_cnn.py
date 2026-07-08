#!/usr/bin/env python
"""Train a standalone hybrid fuzzy CNN model on the corn leaf disease dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.optim as optim
import torchvision.models as models
import yaml

from src.leaf_disease.data import create_dataloaders
from src.leaf_disease.engine import evaluate_model, train_one_epoch
from src.leaf_disease.io import save_checkpoint, save_labels
from src.leaf_disease.utils import current_timestamp, resolve_device, save_json, seed_everything


@dataclass
class HybridFuzzyConfig:
    seed: int
    data_dir: str
    image_size: int
    val_size: float
    test_size: float
    batch_size: int
    num_workers: int
    backbone_name: str
    pretrained: bool
    learning_rate: float
    weight_decay: float
    epochs: int
    output_dir: str
    device: str
    fuzzy_rules: int
    latent_dim: int
    dropout: float


def load_config(config_path: str | Path) -> HybridFuzzyConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return HybridFuzzyConfig(
        seed=int(raw.get("seed", 42)),
        data_dir=str(raw["data_dir"]),
        image_size=int(raw.get("image_size", 224)),
        val_size=float(raw.get("val_size", 0.1)),
        test_size=float(raw.get("test_size", 0.1)),
        batch_size=int(raw.get("batch_size", 16)),
        num_workers=int(raw.get("num_workers", 2)),
        backbone_name=str(raw.get("backbone_name", "mobilenet_v3_small")),
        pretrained=bool(raw.get("pretrained", True)),
        learning_rate=float(raw.get("learning_rate", 2e-4)),
        weight_decay=float(raw.get("weight_decay", 1e-4)),
        epochs=int(raw.get("epochs", 12)),
        output_dir=str(raw.get("output_dir", "models/hybrid_fuzzy_cnn")),
        device=str(raw.get("device", "auto")),
        fuzzy_rules=int(raw.get("fuzzy_rules", 8)),
        latent_dim=int(raw.get("latent_dim", 64)),
        dropout=float(raw.get("dropout", 0.3)),
    )


def _build_backbone(backbone_name: str, pretrained: bool) -> tuple[nn.Module, int]:
    name = backbone_name.lower()

    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        feature_dim = backbone.classifier[-1].in_features
        backbone.classifier = nn.Identity()
        return backbone, feature_dim

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim

    raise ValueError("Unsupported backbone_name. Use one of: mobilenet_v3_small, resnet18")


class HybridFuzzyCNN(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, fuzzy_rules: int, latent_dim: int, dropout: float, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone, feature_dim = _build_backbone(backbone_name, pretrained=pretrained)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.rule_centers = nn.Parameter(torch.randn(fuzzy_rules, latent_dim) * 0.1)
        self.rule_log_sigmas = nn.Parameter(torch.zeros(fuzzy_rules, latent_dim))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fuzzy_rules, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.ndim > 2:
            features = torch.flatten(features, 1)

        latent = self.projector(features)
        centers = self.rule_centers.unsqueeze(0)
        sigmas = torch.exp(self.rule_log_sigmas).unsqueeze(0).clamp(min=1e-3)

        membership = torch.exp(-((latent.unsqueeze(1) - centers) ** 2) / (2.0 * sigmas**2))
        rule_strength = membership.mean(dim=2)
        rule_strength = rule_strength / (rule_strength.sum(dim=1, keepdim=True) + 1e-8)
        return self.classifier(rule_strength)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hybrid fuzzy CNN model")
    parser.add_argument("--config", type=str, default="configs/hybrid_fuzzy_cnn.yaml")
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

    model = HybridFuzzyCNN(
        backbone_name=cfg.backbone_name,
        num_classes=stats["num_classes"],
        fuzzy_rules=cfg.fuzzy_rules,
        latent_dim=cfg.latent_dim,
        dropout=cfg.dropout,
        pretrained=cfg.pretrained,
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
                    "model_name": "hybrid_fuzzy_cnn",
                    "backbone_name": cfg.backbone_name,
                    "num_classes": stats["num_classes"],
                    "class_names": stats["class_names"],
                    "image_size": cfg.image_size,
                    "fuzzy_rules": cfg.fuzzy_rules,
                    "latent_dim": cfg.latent_dim,
                    "dropout": cfg.dropout,
                },
            )

    save_labels(output_dir / "class_names.json", stats["class_names"])
    save_json({"history": history}, Path("reports") / f"train_history_hybrid_{current_timestamp()}.json")

    summary = {
        "model_name": "hybrid_fuzzy_cnn",
        "backbone_name": cfg.backbone_name,
        "num_classes": stats["num_classes"],
        "best_val_macro_f1": round(best_val_f1, 4),
        "train_size": stats["train_size"],
        "val_size": stats["val_size"],
        "test_size": stats["test_size"],
        "device": str(device),
    }
    save_json(summary, Path("reports") / "train_summary_hybrid.json")
    print("Saved checkpoint to", output_dir / "best_model.pt")


if __name__ == "__main__":
    main()