"""
Fine-tune a binary Healthy vs Diseased classifier using EfficientNetB4.
This script reuses the existing dataloader and maps multi-class labels to binary:
  0 = healthy, 1 = diseased

Saves best checkpoint to `models/efficientnetb4/binary_finetune_masked.pt`.
"""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from src.leaf_disease.config import load_config
from src.leaf_disease.data import create_dataloaders
from src.leaf_disease.engine import evaluate_model, train_one_epoch
from src.leaf_disease.io import load_checkpoint, save_checkpoint
from src.leaf_disease.modeling import build_model
from src.leaf_disease.utils import current_timestamp, resolve_device, save_json, seed_everything


def remap_to_binary(dataset, healthy_idx: int):
    # mutate dataset.labels in-place
    dataset.labels = [0 if l == healthy_idx else 1 for l in dataset.labels]


def main() -> None:
    # Resolve paths relative to script directory when given as relative paths
    base_dir = Path(__file__).resolve().parents[2]
    config_path = Path("configs/efficientnetb4.yaml")
    if not config_path.is_absolute():
        config_path = base_dir / config_path

    pretrained_checkpoint = Path("models/efficientnetb4/best_model.pt")
    output_checkpoint = Path("models/efficientnetb4/binary_finetune_masked.pt")
    if not pretrained_checkpoint.is_absolute():
        pretrained_checkpoint = base_dir / pretrained_checkpoint
    if not output_checkpoint.is_absolute():
        output_checkpoint = base_dir / output_checkpoint
    epochs = 8
    lr = 1e-5

    cfg = load_config(config_path)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Using CPU")

    print("[INFO] Creating dataloaders...")
    # Resolve data_dir relative to script folder if needed
    data_dir_path = Path(cfg.data_dir)
    if not data_dir_path.is_absolute():
        data_dir_path = base_dir / data_dir_path

    train_loader, val_loader, test_loader, stats = create_dataloaders(
        data_dir=str(data_dir_path),
        image_size=cfg.image_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    print(f"[INFO] Dataloader pin_memory={train_loader.pin_memory if hasattr(train_loader, 'pin_memory') else 'n/a'}")

    # find healthy class index
    try:
        healthy_idx = stats["class_names"].index("Corn_(maize)___healthy")
    except ValueError:
        # fallback: try any class containing 'healthy'
        healthy_idx = next(i for i, n in enumerate(stats["class_names"]) if "healthy" in n.lower())

    print(f"[INFO] Healthy class index: {healthy_idx} ({stats['class_names'][healthy_idx]})")

    # Remap labels in datasets to binary
    remap_to_binary(train_loader.dataset, healthy_idx)
    remap_to_binary(val_loader.dataset, healthy_idx)
    remap_to_binary(test_loader.dataset, healthy_idx)

    binary_class_names = [stats["class_names"][healthy_idx], "diseased"]
    num_classes = 2

    # Build model and load pretrained weights if available (strict=False to adapt head)
    model = build_model(cfg.model_name, num_classes=num_classes, pretrained=False).to(device)
    if pretrained_checkpoint.exists():
        print("[INFO] Loading pretrained checkpoint and adapting head (strict=False)...")
        ckpt = load_checkpoint(pretrained_checkpoint, map_location=str(device))
        try:
            model.load_state_dict(ckpt["model_state"], strict=False)
        except Exception as e:
            print("[WARN] Could not load some weights:", e)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)

    best_val_f1 = -1.0
    history = []

    print(f"\n[INFO] Starting binary fine-tune for {epochs} epochs (LR={lr})...")
    for epoch in range(1, epochs + 1):
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
            f"Epoch {epoch}/{epochs} | "
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
                    "class_names": binary_class_names,
                    "image_size": cfg.image_size,
                },
            )
            print(f"  ✓ New best checkpoint saved (F1={best_val_f1:.4f})")

    # Save history
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    save_json({"history": history, "best_val_f1": best_val_f1}, reports_dir / f"finetune_binary_{current_timestamp()}.json")

    print("\n[DONE] Binary fine-tuning complete")
    print(f"  Best checkpoint: {output_checkpoint} (F1={best_val_f1:.4f})")


if __name__ == "__main__":
    main()
