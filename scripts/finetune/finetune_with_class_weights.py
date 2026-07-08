#!/usr/bin/env python
"""Quick finetune using class weights and stronger augmentation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
import torch.optim as optim
import yaml
from torchvision import transforms

from src.leaf_disease.io import load_checkpoint, save_checkpoint
from src.leaf_disease.paper_anfis_fuzzy_cnn import (
    PaperAnfisFuzzyCNN,
    create_paper_dataloaders,
    evaluate_model,
    train_one_epoch,
)
from src.leaf_disease.utils import resolve_device, seed_everything


def load_config(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/anfis_fuzzy_cnn.yaml")
    p.add_argument("--ckpt", default="models/anfis_fuzzy_cnn/best_model.pt")
    p.add_argument("--epochs", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed_everything(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "auto"))

    train_loader, val_loader, test_loader, stats = create_paper_dataloaders(
        data_dir=cfg["data_dir"],
        image_size=int(cfg.get("image_size", 224)),
        val_size=float(cfg.get("val_size", 0.1)),
        test_size=float(cfg.get("test_size", 0.1)),
        batch_size=int(cfg.get("batch_size", 16)),
        num_workers=int(cfg.get("num_workers", 2)),
        seed=int(cfg.get("seed", 42)),
    )

    # Stronger augmentation for training
    aug = transforms.Compose(
        [
            transforms.RandomResizedCrop(cfg.get("image_size", 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
        ]
    )

    # apply augmentation to the dataset (train_loader.dataset is PaperAnfisLeafDataset)
    try:
        train_loader.dataset.rgb_transform = aug
    except Exception:
        pass

    num_classes = stats["num_classes"]
    class_counts = stats.get("train_distribution", {})
    counts = [class_counts.get(i, 0) for i in range(num_classes)]
    total = sum(counts) if counts else 1
    weights = [total / (num_classes * max(1, c)) for c in counts]
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    model = PaperAnfisFuzzyCNN(
        num_classes=num_classes,
        num_rules=int(cfg.get("num_rules", 8)),
        image_size=int(cfg.get("image_size", 224)),
        dropout=float(cfg.get("dropout", 0.3)),
    ).to(device)

    # load checkpoint if available
    ckpt_path = Path(args.ckpt)
    if ckpt_path.exists():
        ck = load_checkpoint(ckpt_path, map_location="cpu")
        model_state = ck.get("model_state") or ck
        model.load_state_dict(model_state, strict=False)
        print("Loaded checkpoint:", ckpt_path)
    else:
        print("Checkpoint not found, training from scratch")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 2e-4)) * 0.25, weight_decay=float(cfg.get("weight_decay", 1e-4)))

    out_dir = Path(cfg.get("output_dir", "models/anfis_fuzzy_cnn"))
    out_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} loss={loss:.4f} acc={acc:.4f} val_f1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            save_checkpoint(
                out_dir / "finetune_best.pt",
                {
                    "model_state": model.state_dict(),
                    "num_classes": num_classes,
                    "class_names": stats.get("class_names"),
                    "image_size": cfg.get("image_size"),
                    "num_rules": cfg.get("num_rules"),
                },
            )

    # final eval on test
    test_metrics = evaluate_model(model, test_loader, device, return_predictions=False)
    print("Finetune finished. Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
