from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_history(history_path: str, output_dir: str = "reports") -> None:
    """Plot training history (loss, accuracy, f1) over epochs."""
    history_path = Path(history_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with history_path.open() as f:
        data = json.load(f)
    
    # Handle both nested and flat history formats
    history = data.get("history", data) if isinstance(data, dict) else data
    if isinstance(history, dict):
        history = data if "epoch" in data else history

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_accuracy"] for row in history]
    val_f1 = [row["val_macro_f1"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].set_title("Training Loss", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Accuracy & F1 plot
    axes[1].plot(epochs, train_acc, marker="o", label="Train Acc", linewidth=2)
    axes[1].plot(epochs, val_acc, marker="s", label="Val Acc", linewidth=2)
    axes[1].plot(epochs, val_f1, marker="^", label="Val Macro F1", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Score", fontsize=11)
    axes[1].set_title("Training Metrics", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    output_path = output_dir / "training_curves.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved training curves to {output_path}")


if __name__ == "__main__":
    history_files = list(Path("reports").glob("train_history_*.json"))
    history_file = history_files[0] if history_files else "reports/train_history.json"
    plot_training_history(str(history_file))
