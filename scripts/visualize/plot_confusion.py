# scripts/visualize/plot_confusion.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

labels = [
    "Cercospora",
    "Common_rust",
    "Northern_Leaf_Blight",
    "healthy",
]

cm = np.array([
    [47, 0, 4, 0],
    [0, 120, 0, 0],
    [2, 0, 96, 1],
    [0, 0, 0, 116],
])

out_dir = Path("reports")
out_dir.mkdir(parents=True, exist_ok=True)

# Absolute counts heatmap
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — MobileNetV3-Small (test)")
plt.tight_layout()
plt.savefig(out_dir / "mobilenetv3_confusion_counts.png", dpi=300)
plt.close()

# Normalized by true class (row) — show % 
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
plt.figure(figsize=(7,6))
sns.heatmap(cm_norm * 100, annot=True, fmt='.1f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (%) — MobileNetV3-Small (test) — normalized by actual class")
plt.tight_layout()
plt.savefig(out_dir / "mobilenetv3_confusion_percent.png", dpi=300)
plt.close()

print("Saved:", out_dir / "mobilenetv3_confusion_counts.png", "and", out_dir / "mobilenetv3_confusion_percent.png")