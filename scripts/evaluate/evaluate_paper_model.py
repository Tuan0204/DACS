#!/usr/bin/env python
"""Evaluate paper-style ANFIS-Fuzzy-CNN: confusion matrix and per-class report."""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import torch
from sklearn.metrics import confusion_matrix, classification_report

from src.leaf_disease.paper_anfis_fuzzy_cnn import create_paper_dataloaders, PaperAnfisFuzzyCNN, evaluate_model
from src.leaf_disease.io import load_checkpoint, save_labels
from src.leaf_disease.utils import current_timestamp, resolve_device, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/anfis_fuzzy_cnn.yaml")
    p.add_argument("--ckpt", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    device = resolve_device(cfg.get("device", "auto"))

    train_loader, val_loader, test_loader, stats = create_paper_dataloaders(
        data_dir=cfg["data_dir"],
        image_size=cfg.get("image_size", 224),
        val_size=cfg.get("val_size", 0.1),
        test_size=cfg.get("test_size", 0.1),
        batch_size=cfg.get("batch_size", 16),
        num_workers=cfg.get("num_workers", 2),
        seed=cfg.get("seed", 42),
    )

    model = PaperAnfisFuzzyCNN(
        num_classes=stats["num_classes"],
        num_rules=int(cfg.get("num_rules", 8)),
        image_size=int(cfg.get("image_size", 224)),
        dropout=float(cfg.get("dropout", 0.3)),
    )

    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Evaluate on TEST set (not val set)
    metrics = evaluate_model(model, test_loader, device, return_predictions=True)

    y_true = metrics.pop("y_true")
    y_pred = metrics.pop("y_pred")

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=stats["class_names"], output_dict=True, zero_division=0)

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    ts = current_timestamp()
    save_json({"confusion_matrix": cm.tolist()}, out_dir / f"confusion_{ts}.json")
    save_json({"classification_report": report}, out_dir / f"classification_report_{ts}.json")

    print("Confusion matrix saved to", out_dir / f"confusion_{ts}.json")
    print("Classification report saved to", out_dir / f"classification_report_{ts}.json")
    print("Summary metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
