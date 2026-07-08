from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.leaf_disease.config import load_config
from src.leaf_disease.data import create_dataloaders
from src.leaf_disease.engine import evaluate_model
from src.leaf_disease.io import load_checkpoint
from src.leaf_disease.modeling import build_model, count_model_size_mb
from src.leaf_disease.utils import resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate leaf disease classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.device)

    _, _, test_loader, stats = create_dataloaders(
        data_dir=cfg.data_dir,
        image_size=cfg.image_size,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))
    model_name = checkpoint.get("model_name", cfg.model_name)
    num_classes = checkpoint.get("num_classes", stats["num_classes"])

    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)

    metrics = evaluate_model(model, test_loader, device, return_predictions=True)
    metrics["model_name"] = model_name
    metrics["model_size_mb"] = round(count_model_size_mb(model), 2)
    metrics["device"] = str(device)

    y_true = metrics.pop("y_true")
    y_pred = metrics.pop("y_pred")
    class_names = checkpoint.get("class_names", stats["class_names"])

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    metrics["class_names"] = class_names
    metrics["confusion_matrix"] = cm.tolist()
    metrics["per_class"] = [
        {
            "class_name": class_name,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }
        for class_name, precision, recall, f1, support in zip(
            class_names,
            per_class_precision,
            per_class_recall,
            per_class_f1,
            per_class_support,
        )
    ]

    out_path = Path("reports") / "metrics_test.json"
    save_json(metrics, out_path)

    report_path = Path("reports") / "mobilenetv3_test_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# MobileNetV3-Small Test Evaluation",
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
        report_lines.append(
            f"| {row['class_name']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['support']} |"
        )

    report_lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            "| Actual \\ Predicted | " + " | ".join(class_names) + " |",
            "|--------------------|" + "|".join(["---------"] * len(class_names)) + "|",
        ]
    )

    for class_name, row in zip(class_names, cm):
        report_lines.append("| " + class_name + " | " + " | ".join(str(int(value)) for value in row) + " |")

    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- MobileNetV3-Small is the final model because it is the strongest in the comparison table.",
            "- The class-level table and confusion matrix expose which classes remain harder to separate.",
        ]
    )

    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("Test metrics:")
    for k, v in metrics.items():
        print(f"- {k}: {v}")
    print("Saved to", out_path)
    print("Saved report to", report_path)


if __name__ == "__main__":
    main()
