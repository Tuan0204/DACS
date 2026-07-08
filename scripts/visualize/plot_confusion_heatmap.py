#!/usr/bin/env python
"""Plot confusion matrix heatmap from reports JSON."""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    # choose latest confusion JSON in reports/
    rpt_dir = Path("reports")
    files = sorted(rpt_dir.glob("confusion_*.json"))
    if not files:
        raise FileNotFoundError("No confusion_*.json found in reports/")
    rpt = files[-1]
    data = json.loads(rpt.read_text(encoding="utf-8"))
    cm = np.array(data["confusion_matrix"])

    # try to load human-readable class names from models/class_names.json
    labels = None
    fallback = Path("models") / "class_names.json"
    if fallback.exists():
        try:
            labels = json.loads(fallback.read_text(encoding="utf-8"))
        except Exception:
            labels = None
    if not labels:
        labels = [
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "Corn_(maize)___Common_rust_",
            "Corn_(maize)___Northern_Leaf_Blight",
            "Corn_(maize)___healthy",
        ]

    def short_label(name: str) -> str:
        n = name.lower()
        if "cercospora" in n or "cercospora_leaf" in n:
            return "Cercospora"
        if "common_rust" in n or "rust" in n:
            return "Rust"
        if "northern_leaf_blight" in n or "nlb" in n:
            return "NLB"
        if "healthy" in n or "healthy" in name.lower():
            return "Healthy"
        # fallback: keep last path part trimmed
        return name.split("___")[-1].replace("_", " ")[:20]

    labels = [short_label(l) for l in labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=labels, yticklabels=labels, ylabel="True label", xlabel="Predicted label")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    stamp = rpt.stem.replace("confusion_", "")
    out = Path("reports") / f"confusion_heatmap_{stamp}.png"
    plt.savefig(out, dpi=150)
    print("Saved heatmap to", out)


if __name__ == "__main__":
    main()
