#!/usr/bin/env python
"""Plot two latest confusion matrices side-by-side for comparison."""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_confusion(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    cm = np.array(data.get("confusion_matrix") or data.get("confusion", []))
    class_names = data.get("class_names") or data.get("labels")
    return cm, class_names


def format_cell_text(abs_val: int, pct: float) -> str:
    return f"{abs_val}\n({pct:.0%})"


def main():
    rpt_dir = Path("reports")
    files = sorted(rpt_dir.glob("confusion_*.json"))
    if len(files) < 2:
        raise SystemExit("Need at least two confusion_*.json files in reports/ to compare")

    left_path, right_path = files[-2], files[-1]
    cm1, names1 = load_confusion(left_path)
    cm2, names2 = load_confusion(right_path)

    if cm1.size == 0 or cm2.size == 0:
        raise SystemExit("One of the confusion files is empty or malformed")

    # try to get human-readable class names from the confusion JSONs, else fallback to models/class_names.json
    labels = names1 or names2
    if not labels:
        fallback = Path("models") / "class_names.json"
        if fallback.exists():
            try:
                labels = json.loads(fallback.read_text(encoding="utf-8"))
            except Exception:
                labels = None
    if not labels:
        labels = [f"C{i}" for i in range(cm1.shape[0])]

    def short_label(name: str) -> str:
        n = name.lower()
        if "cercospora" in n or "cercospora_leaf" in n:
            return "Cercospora"
        if "common_rust" in n or "rust" in n:
            return "Rust"
        if "northern_leaf_blight" in n or "nlb" in n:
            return "NLB"
        if "healthy" in n:
            return "Healthy"
        return name.split("___")[-1].replace("_", " ")[:20]

    labels = [short_label(l) for l in labels]

    # compute per-row (true-class) normalized matrices for percentages
    cm1_pct = cm1.astype(float) / (cm1.sum(axis=1, keepdims=True) + 1e-9)
    cm2_pct = cm2.astype(float) / (cm2.sum(axis=1, keepdims=True) + 1e-9)

    vmax = max(cm1.max(), cm2.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    titles = [left_path.stem, right_path.stem]
    cms = [(cm1, cm1_pct), (cm2, cm2_pct)]

    for ax, (cm, cm_pct), title in zip(axes, cms, titles):
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=vmax)
        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_yticks(np.arange(cm.shape[0]))
        ax.set_xticklabels(labels, rotation=0)
        ax.set_yticklabels(labels)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.set_title(title)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                text = format_cell_text(int(cm[i, j]), float(cm_pct[i, j]))
                color = "white" if cm[i, j] > vmax / 2.0 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)

    # add a single colorbar to the right
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.ax.set_ylabel("Count", rotation=270, labelpad=15)

    out = rpt_dir / f"confusion_comparison_clean_{left_path.stem}_{right_path.stem}.png"
    plt.savefig(out, dpi=200)
    print("Saved clean comparison to", out)


if __name__ == "__main__":
    main()
