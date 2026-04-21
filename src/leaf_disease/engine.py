from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from tqdm import tqdm


@torch.inference_mode()
def evaluate_model(model: nn.Module, loader, device: torch.device) -> dict:
    model.eval()
    y_true = []
    y_pred = []

    total_time = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        start = time.perf_counter()
        logits = model(images)
        elapsed = time.perf_counter() - start

        preds = torch.argmax(logits, dim=1)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

        total_time += elapsed
        total_samples += labels.size(0)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    latency_ms = (total_time / max(total_samples, 1)) * 1000.0

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "latency_ms_per_image": float(latency_ms),
    }


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    y_true = []
    y_pred = []

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        running_loss += loss.item() * labels.size(0)

        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    epoch_acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    return float(epoch_loss), float(epoch_acc)
