from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from src.leaf_disease.config import load_config
from src.leaf_disease.data import create_dataloaders
from src.leaf_disease.io import load_checkpoint
from src.leaf_disease.modeling import build_model
from src.leaf_disease.utils import resolve_device
from sklearn.metrics import accuracy_score, f1_score


def evaluate_ensemble(checkpoint_paths: list[str], config_path: str) -> dict:
    """Evaluate ensemble on test set."""
    cfg = load_config(config_path)
    device = resolve_device(cfg.device)

    # Load data
    _, _, test_loader, stats = create_dataloaders(
        data_dir=cfg.data_dir,
        image_size=cfg.image_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    # Load models
    models = []
    for cp in checkpoint_paths:
        ckpt = load_checkpoint(cp, map_location=str(device))
        model = build_model(
            ckpt.get("model_name", cfg.model_name),
            ckpt["num_classes"],
            pretrained=False,
        )
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device).eval()
        models.append(model)

    # Evaluate
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.cpu().numpy()

            probs_sum = None
            for model in models:
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs

            probs_avg = probs_sum / len(models)
            preds = torch.argmax(probs_avg, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_models": len(models),
        "checkpoint_paths": checkpoint_paths,
    }


if __name__ == "__main__":
    result = evaluate_ensemble(
        checkpoint_paths=[
            "models/best_model_resnet.pt",
            "models/best_model_mobilenet.pt",
        ],
        config_path="configs/default.yaml",
    )

    print("Ensemble Evaluation Results:")
    print(f"- Accuracy: {result['accuracy']:.4f}")
    print(f"- Macro F1: {result['macro_f1']:.4f}")
    print(f"- Models: {result['num_models']}")

    output_path = Path("reports/metrics_ensemble.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output_path}")
