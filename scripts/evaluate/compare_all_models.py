#!/usr/bin/env python
"""Compare all models on TEST set: ANFIS, ResNet18, MobileNetV3, EfficientNetB4."""

from __future__ import annotations

from pathlib import Path
import yaml
import torch
from sklearn.metrics import confusion_matrix, classification_report

from src.leaf_disease.paper_anfis_fuzzy_cnn import create_paper_dataloaders, PaperAnfisFuzzyCNN, evaluate_model as eval_anfis
from src.leaf_disease.data import create_dataloaders
from src.leaf_disease.engine import evaluate_model
from src.leaf_disease.io import load_checkpoint
from src.leaf_disease.modeling import build_model
from src.leaf_disease.utils import resolve_device, save_json


MODEL_CONFIGS = {
    "ResNet18": {
        "config": "configs/resnet18.yaml",
        "checkpoint": "models/resnet18/best_model.pt",
        "type": "standard",
    },
    "MobileNetV3-Small": {
        "config": "configs/mobilenetv3.yaml",
        "checkpoint": "models/mobilenetv3/best_model.pt",
        "type": "standard",
        "model_name": "mobilenet_v3_small",
    },
    "ANFIS-Fuzzy-CNN (baseline)": {
        "config": "configs/anfis_fuzzy_cnn.yaml",
        "checkpoint": "models/anfis_fuzzy_cnn/best_model.pt",
        "type": "anfis",
    },
    "ANFIS-Fuzzy-CNN (finetune-weights)": {
        "config": "configs/anfis_fuzzy_cnn.yaml",
        "checkpoint": "models/anfis_fuzzy_cnn/finetune_best.pt",
        "type": "anfis",
    },
    "ANFIS-Fuzzy-CNN (finetune-oversample)": {
        "config": "configs/anfis_fuzzy_cnn.yaml",
        "checkpoint": "models/anfis_fuzzy_cnn/finetune_oversample_best.pt",
        "type": "anfis",
    },
}


def eval_standard_model(model_name, config_path, checkpoint_path, model_name_override=None):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    device = resolve_device(cfg.get("device", "auto"))

    _, _, test_loader, stats = create_dataloaders(
        data_dir=cfg["data_dir"],
        image_size=cfg.get("image_size", 224),
        val_size=cfg.get("val_size", 0.1),
        test_size=cfg.get("test_size", 0.1),
        batch_size=cfg.get("batch_size", 16),
        num_workers=cfg.get("num_workers", 2),
        seed=cfg.get("seed", 42),
    )

    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    # Use override model_name if provided, else extract from checkpoint or config
    actual_model_name = model_name_override or model_name.replace("-", "_")
    model = build_model(actual_model_name, num_classes=stats["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    metrics = evaluate_model(model, test_loader, device, return_predictions=True)
    return metrics, stats


def eval_anfis_model(config_path, checkpoint_path):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    device = resolve_device(cfg.get("device", "auto"))

    _, _, test_loader, stats = create_paper_dataloaders(
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

    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    metrics = eval_anfis(model, test_loader, device, return_predictions=True)
    return metrics, stats


def main():
    results = []

    for model_name, cfg_dict in MODEL_CONFIGS.items():
        config_path = cfg_dict["config"]
        checkpoint_path = cfg_dict["checkpoint"]
        model_type = cfg_dict["type"]

        print(f"\nEvaluating {model_name}...")

        if not Path(checkpoint_path).exists():
            print(f"  ⚠️  Checkpoint not found: {checkpoint_path}")
            continue

        try:
            if model_type == "standard":
                model_name_override = cfg_dict.get("model_name", None)
                metrics, stats = eval_standard_model(model_name, config_path, checkpoint_path, model_name_override)
            elif model_type == "anfis":
                metrics, stats = eval_anfis_model(config_path, checkpoint_path)

            macro_f1 = metrics.get("macro_f1", 0)
            accuracy = metrics.get("accuracy", 0)
            print(f"  ✅ Accuracy: {accuracy:.4f}, Macro-F1: {macro_f1:.4f}")

            results.append({
                "model": model_name,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "latency_ms": metrics.get("latency_ms_per_image", 0),
            })

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("\n" + "="*70)
    print("SUMMARY - TEST SET RESULTS")
    print("="*70)
    print(f"{'Model':<20} {'Accuracy':<12} {'Macro-F1':<12} {'Precision':<12} {'Recall':<12}")
    print("-"*70)
    for r in sorted(results, key=lambda x: x["macro_f1"], reverse=True):
        print(f"{r['model']:<20} {r['accuracy']:<12.4f} {r['macro_f1']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f}")

    # Save summary
    out_path = Path("reports") / "model_comparison_test.json"
    save_json({"results": results}, out_path)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
