"""
Evaluate the binary Healthy vs Diseased EfficientNetB4 checkpoint on the held-out test set.

Expected checkpoint:
  models/efficientnetb4/binary_finetune_masked.pt

Outputs:
  reports/binary_metrics_test.json
  reports/binary_efficientnetb4_test_report.md
"""
from __future__ import annotations

from pathlib import Path

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.leaf_disease.config import load_config
from src.leaf_disease.data import create_dataloaders
from src.leaf_disease.engine import evaluate_model
from src.leaf_disease.io import load_checkpoint
from src.leaf_disease.modeling import build_model, count_model_size_mb
from src.leaf_disease.utils import resolve_device, save_json


def remap_to_binary(dataset, healthy_idx: int):
    dataset.labels = [0 if l == healthy_idx else 1 for l in dataset.labels]


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    config_path = base_dir / "configs/efficientnetb4.yaml"
    checkpoint_path = base_dir / "models/efficientnetb4/binary_finetune_masked.pt"

    cfg = load_config(config_path)
    device = resolve_device(cfg.device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data_dir_path = Path(cfg.data_dir)
    if not data_dir_path.is_absolute():
        data_dir_path = base_dir / data_dir_path

    _, _, test_loader, stats = create_dataloaders(
        data_dir=str(data_dir_path),
        image_size=cfg.image_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    try:
        healthy_idx = stats["class_names"].index("Corn_(maize)___healthy")
    except ValueError:
        healthy_idx = next(i for i, n in enumerate(stats["class_names"]) if "healthy" in n.lower())

    remap_to_binary(test_loader.dataset, healthy_idx)

    checkpoint = load_checkpoint(checkpoint_path, map_location=str(device))
    num_classes = checkpoint.get("num_classes", 2)
    model = build_model(checkpoint.get("model_name", cfg.model_name), num_classes=num_classes, pretrained=False)
    try:
        model.load_state_dict(checkpoint["model_state"])
    except Exception:
        model.load_state_dict(checkpoint["model_state"], strict=False)
    model = model.to(device)

    metrics = evaluate_model(model, test_loader, device, return_predictions=True)
    y_true = metrics.pop("y_true")
    y_pred = metrics.pop("y_pred")

    class_names = checkpoint.get("class_names", ["healthy", "diseased"])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    metrics.update(
        {
            "model_name": checkpoint.get("model_name", cfg.model_name),
            "model_size_mb": round(count_model_size_mb(model), 2),
            "device": str(device),
            "class_names": class_names,
            "healthy_class_index": healthy_idx,
            "confusion_matrix": cm.tolist(),
            "per_class": [
                {
                    "class_name": class_name,
                    "precision": float(p),
                    "recall": float(r),
                    "f1": float(f),
                    "support": int(s),
                }
                for class_name, p, r, f, s in zip(class_names, precision, recall, f1, support)
            ],
        }
    )

    out_json = base_dir / "reports/binary_metrics_test.json"
    save_json(metrics, out_json)

    report_path = base_dir / "reports/binary_efficientnetb4_test_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Binary EfficientNetB4 Test Evaluation",
        "",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Macro F1: {metrics['macro_f1']:.4f}",
        f"- Latency: {metrics['latency_ms_per_image']:.4f} ms/img",
        f"- Model size: {metrics['model_size_mb']:.2f} MB",
        "",
        "## Per-class metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|-------|-----------|--------|----|---------|",
    ]
    for row in metrics["per_class"]:
        lines.append(
            f"| {row['class_name']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['support']} |"
        )

    lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            "| Actual \\ Predicted | healthy | diseased |",
            "|--------------------|---------|----------|",
            f"| healthy | {cm[0][0]} | {cm[0][1]} |",
            f"| diseased | {cm[1][0]} | {cm[1][1]} |",
            "",
            "## Interpretation",
            "",
            "- This binary model is intended to reduce false healthy predictions before the multiclass step.",
            "- If healthy recall is high and diseased recall is also high, it is suitable as stage 1 of the pipeline.",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("Test metrics:")
    for key, value in metrics.items():
        if key not in {"y_true", "y_pred"}:
            print(f"- {key}: {value}")
    print("Saved to", out_json)
    print("Saved report to", report_path)


if __name__ == "__main__":
    main()
