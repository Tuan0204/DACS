"""
Fine-tune EfficientNetB4 using masked segmentation data (10 epochs).
Loads from existing checkpoint and applies lower learning rate.
Saves best checkpoint to models/efficientnetb4/finetune_masked.pt
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.leaf_disease.config import load_config
from src.leaf_disease.data import create_dataloaders
from src.leaf_disease.engine import evaluate_model, train_one_epoch
from src.leaf_disease.io import load_checkpoint, save_checkpoint, save_labels
from src.leaf_disease.modeling import build_model
from src.leaf_disease.utils import current_timestamp, resolve_device, save_json, seed_everything


def main() -> None:
    # Configuration
    config_path = "configs/efficientnetb4.yaml"
    pretrained_checkpoint = Path("models/efficientnetb4/best_model.pt")
    output_checkpoint = Path("models/efficientnetb4/finetune_masked.pt")
    finetune_epochs = 10
    finetune_lr = 0.00001  # Lower LR for fine-tuning vs initial training
    
    cfg = load_config(config_path)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    print("[INFO] Loading EfficientNetB4 checkpoint from:", pretrained_checkpoint)
    if not pretrained_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_checkpoint}")

    checkpoint = load_checkpoint(pretrained_checkpoint, map_location=str(device))
    num_classes = checkpoint["num_classes"]
    class_names = checkpoint["class_names"]

    # Create dataloaders (will auto-detect and apply segmented masks if they exist)
    print("[INFO] Creating dataloaders with auto-detected segmented masks...")
    train_loader, val_loader, _, stats = create_dataloaders(
        data_dir=cfg.data_dir,
        image_size=cfg.image_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    print(f"[INFO] Train: {stats['train_size']}, Val: {stats['val_size']}, Classes: {num_classes}")
    print(f"[INFO] Train distribution: {stats['train_distribution']}")

    # Build and load model
    model = build_model(cfg.model_name, num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"[INFO] Model loaded: {cfg.model_name}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr, weight_decay=cfg.weight_decay)

    best_val_f1 = -1.0
    history = []

    print(f"\n[INFO] Starting fine-tuning for {finetune_epochs} epochs (LR={finetune_lr})...")
    for epoch in range(1, finetune_epochs + 1):
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
            f"Epoch {epoch}/{finetune_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                output_checkpoint,
                {
                    "model_state": model.state_dict(),
                    "model_name": cfg.model_name,
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "image_size": cfg.image_size,
                },
            )
            print(f"  ✓ New best checkpoint saved (F1={best_val_f1:.4f})")

    # Save history
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    save_json(
        {"history": history, "best_val_f1": best_val_f1, "finetune_lr": finetune_lr},
        reports_dir / f"finetune_efficientnetb4_masked_{current_timestamp()}.json",
    )

    print(f"\n[DONE] Fine-tuning complete!")
    print(f"  Best checkpoint: {output_checkpoint} (F1={best_val_f1:.4f})")
    print(f"  History saved to: reports/finetune_efficientnetb4_masked_*.json")


if __name__ == "__main__":
    main()
