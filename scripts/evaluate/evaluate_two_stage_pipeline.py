"""
Evaluate the two-stage pipeline:
  stage 1: binary healthy vs diseased
  stage 2: multiclass disease classifier for diseased samples

Outputs:
  reports/two_stage_metrics_test.json
  reports/two_stage_pipeline_test_report.md
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score

from src.leaf_disease.config import load_config
from src.leaf_disease.data import create_dataloaders
from src.leaf_disease.io import load_checkpoint
from src.leaf_disease.modeling import build_model, count_model_size_mb
from src.leaf_disease.utils import resolve_device, save_json


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    config_path = base_dir / "configs/efficientnetb4.yaml"
    binary_checkpoint_path = base_dir / "models/efficientnetb4/binary_finetune_masked.pt"
    multiclass_checkpoint_path = base_dir / "models/efficientnetb4/finetune_masked.pt"

    cfg = load_config(config_path)
    device = resolve_device(cfg.device)

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

    if not binary_checkpoint_path.exists():
        raise FileNotFoundError(f"Binary checkpoint not found: {binary_checkpoint_path}")
    if not multiclass_checkpoint_path.exists():
        raise FileNotFoundError(f"Multiclass checkpoint not found: {multiclass_checkpoint_path}")

    binary_ckpt = load_checkpoint(binary_checkpoint_path, map_location=str(device))
    binary_model = build_model(binary_ckpt.get("model_name", cfg.model_name), num_classes=binary_ckpt.get("num_classes", 2), pretrained=False)
    try:
        binary_model.load_state_dict(binary_ckpt["model_state"])
    except Exception:
        binary_model.load_state_dict(binary_ckpt["model_state"], strict=False)
    binary_model = binary_model.to(device).eval()

    multiclass_ckpt = load_checkpoint(multiclass_checkpoint_path, map_location=str(device))
    multiclass_model = build_model(multiclass_ckpt.get("model_name", cfg.model_name), num_classes=multiclass_ckpt["num_classes"], pretrained=False)
    try:
        multiclass_model.load_state_dict(multiclass_ckpt["model_state"])
    except Exception:
        multiclass_model.load_state_dict(multiclass_ckpt["model_state"], strict=False)
    multiclass_model = multiclass_model.to(device).eval()

    y_true = []
    y_pred = []
    stage1_true = []
    stage1_pred = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            b_logits = binary_model(images)
            b_probs = F.softmax(b_logits, dim=1)
            b_stage_pred = torch.argmax(b_probs, dim=1)

            m_logits = multiclass_model(images)
            m_pred = torch.argmax(m_logits, dim=1)

            for i in range(labels.size(0)):
                true_label = int(labels[i].item())
                stage1_true_label = 0 if true_label == healthy_idx else 1
                stage1_pred_label = int(b_stage_pred[i].item())

                if stage1_pred_label == 0:
                    final_pred = healthy_idx
                else:
                    final_pred = int(m_pred[i].item())

                y_true.append(true_label)
                y_pred.append(final_pred)
                stage1_true.append(stage1_true_label)
                stage1_pred.append(stage1_pred_label)

    class_names = multiclass_ckpt.get("class_names", stats["class_names"])
    binary_class_names = [class_names[healthy_idx], "diseased"]

    final_acc = accuracy_score(y_true, y_pred)
    final_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    final_cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    final_precision, final_recall, final_f1_per_class, final_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    stage1_acc = accuracy_score(stage1_true, stage1_pred)
    stage1_f1 = f1_score(stage1_true, stage1_pred, average="macro", zero_division=0)
    stage1_cm = confusion_matrix(stage1_true, stage1_pred, labels=[0, 1])

    metrics = {
        "final_accuracy": float(final_acc),
        "final_macro_f1": float(final_f1),
        "stage1_accuracy": float(stage1_acc),
        "stage1_macro_f1": float(stage1_f1),
        "class_names": class_names,
        "binary_class_names": binary_class_names,
        "healthy_class_index": healthy_idx,
        "final_confusion_matrix": final_cm.tolist(),
        "stage1_confusion_matrix": stage1_cm.tolist(),
        "final_per_class": [
            {
                "class_name": class_name,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1v),
                "support": int(s),
            }
            for class_name, p, r, f1v, s in zip(class_names, final_precision, final_recall, final_f1_per_class, final_support)
        ],
        "binary_model_size_mb": round(count_model_size_mb(binary_model), 2),
        "multiclass_model_size_mb": round(count_model_size_mb(multiclass_model), 2),
        "device": str(device),
    }

    out_json = base_dir / "reports/two_stage_metrics_test.json"
    save_json(metrics, out_json)

    report_path = base_dir / "reports/two_stage_pipeline_test_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Two-Stage Pipeline Test Evaluation",
        "",
        f"- Final accuracy: {metrics['final_accuracy']:.4f}",
        f"- Final macro F1: {metrics['final_macro_f1']:.4f}",
        f"- Stage 1 accuracy: {metrics['stage1_accuracy']:.4f}",
        f"- Stage 1 macro F1: {metrics['stage1_macro_f1']:.4f}",
        "",
        "## Final per-class metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|-------|-----------|--------|----|---------|",
    ]
    for row in metrics["final_per_class"]:
        lines.append(
            f"| {row['class_name']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['support']} |"
        )

    lines.extend(
        [
            "",
            "## Stage 1 confusion matrix",
            "",
            "| Actual \\ Predicted | healthy | diseased |",
            "|--------------------|---------|----------|",
            f"| healthy | {stage1_cm[0][0]} | {stage1_cm[0][1]} |",
            f"| diseased | {stage1_cm[1][0]} | {stage1_cm[1][1]} |",
            "",
            "## Final confusion matrix",
            "",
            "| Actual \\ Predicted | " + " | ".join(class_names) + " |",
            "|--------------------|" + "|".join(["---------"] * len(class_names)) + "|",
        ]
    )
    for class_name, row in zip(class_names, final_cm):
        lines.append("| " + class_name + " | " + " | ".join(str(int(v)) for v in row) + " |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage 1 is the gate that reduces false healthy predictions.",
            "- Stage 2 only runs when the binary model thinks the leaf is diseased.",
            "- If the final macro F1 is at least as good as the multiclass-only model, this two-stage setup is the safer default.",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("Two-stage pipeline metrics:")
    for key, value in metrics.items():
        if key not in {"final_per_class"}:
            print(f"- {key}: {value}")
    print("Saved to", out_json)
    print("Saved report to", report_path)


if __name__ == "__main__":
    main()
