from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def _discover_image_samples(data_dir: str | Path):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    class_names = sorted([p.name for p in data_path.iterdir() if p.is_dir()])
    if not class_names:
        raise ValueError(f"No class folders found in {data_path}")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    image_paths = []
    labels = []

    for class_name in class_names:
        class_dir = data_path / class_name
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for image_path in class_dir.glob(ext):
                image_paths.append(image_path)
                labels.append(class_to_idx[class_name])

    if not image_paths:
        raise ValueError(f"No images found in {data_path}")

    return image_paths, labels, class_names


def _compute_lbp_histogram(image: Image.Image, image_size: int) -> torch.Tensor:
    gray = image.convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.uint8)

    center = pixels[1:-1, 1:-1]
    codes = np.zeros_like(center, dtype=np.uint8)

    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ]

    for bit_index, (dy, dx) in enumerate(neighbors):
        neighbor = pixels[1 + dy : image_size - 1 + dy, 1 + dx : image_size - 1 + dx]
        codes |= ((neighbor >= center).astype(np.uint8) << bit_index)

    histogram = np.bincount(codes.ravel(), minlength=256).astype(np.float32)
    histogram /= histogram.sum() + 1e-8
    return torch.from_numpy(histogram)


class PaperAnfisLeafDataset(Dataset):
    def __init__(self, paths: Sequence[Path], labels: Sequence[int], image_size: int, train: bool) -> None:
        self.paths = list(paths)
        self.labels = list(labels)
        self.image_size = image_size
        self.train = train
        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image_path = self.paths[index]
        label = int(self.labels[index])
        image = Image.open(image_path).convert("RGB")
        rgb = self.rgb_transform(image)
        lbp = _compute_lbp_histogram(image, self.image_size)
        return rgb, lbp, label


def create_paper_dataloaders(
    data_dir: str | Path,
    image_size: int,
    val_size: float,
    test_size: float,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    image_paths, labels, class_names = _discover_image_samples(data_dir)
    indices = np.arange(len(image_paths))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_size + test_size),
        random_state=seed,
        stratify=labels,
    )

    temp_labels = [labels[i] for i in temp_idx]
    val_ratio_in_temp = val_size / (val_size + test_size)

    val_idx_rel, test_idx_rel = train_test_split(
        np.arange(len(temp_idx)),
        test_size=(1.0 - val_ratio_in_temp),
        random_state=seed,
        stratify=temp_labels,
    )
    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]

    train_ds = PaperAnfisLeafDataset([image_paths[i] for i in train_idx], [labels[i] for i in train_idx], image_size, train=True)
    val_ds = PaperAnfisLeafDataset([image_paths[i] for i in val_idx], [labels[i] for i in val_idx], image_size, train=False)
    test_ds = PaperAnfisLeafDataset([image_paths[i] for i in test_idx], [labels[i] for i in test_idx], image_size, train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    split_stats = {
        "num_classes": len(class_names),
        "class_names": class_names,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "train_distribution": dict(Counter([labels[i] for i in train_idx])),
    }

    return train_loader, val_loader, test_loader, split_stats


class FuzzyActivation2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.centers = nn.Parameter(torch.zeros(channels))
        self.log_sigmas = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centers = self.centers.view(1, -1, 1, 1)
        sigmas = F.softplus(self.log_sigmas).view(1, -1, 1, 1) + 1e-3
        return torch.exp(-0.5 * ((x - centers) / sigmas) ** 2)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            FuzzyActivation2d(out_channels),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TSKFuzzyRuleLayer(nn.Module):
    def __init__(self, input_dim: int, num_rules: int, num_classes: int) -> None:
        super().__init__()
        self.num_rules = num_rules
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.randn(num_rules, input_dim) * 0.1)
        self.log_sigmas = nn.Parameter(torch.zeros(num_rules, input_dim))
        self.consequents = nn.Parameter(torch.randn(num_rules, num_classes, input_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(num_rules, num_classes))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        sigmas = F.softplus(self.log_sigmas).unsqueeze(0) + 1e-3
        membership = torch.exp(-0.5 * (diff / sigmas) ** 2)
        log_strength = torch.log(membership.clamp_min(1e-6)).sum(dim=2)
        rule_strength = torch.softmax(log_strength, dim=1)

        consequent_logits = torch.einsum("bd,rcd->brc", x, self.consequents) + self.bias.unsqueeze(0)
        logits = (rule_strength.unsqueeze(-1) * consequent_logits).sum(dim=1)
        return logits, rule_strength


class PaperAnfisFuzzyCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_rules: int,
        image_size: int,
        lbp_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.rgb_branch = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.rgb_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.lbp_encoder = nn.Sequential(
            nn.Linear(lbp_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.fuzzy_rules = TSKFuzzyRuleLayer(128, num_rules, num_classes)
        self.image_size = image_size

    def forward(self, rgb: torch.Tensor, lbp: torch.Tensor) -> torch.Tensor:
        rgb_features = self.rgb_projection(self.rgb_branch(rgb))
        lbp_features = self.lbp_encoder(lbp)
        fused = self.fusion(torch.cat([rgb_features, lbp_features], dim=1))
        logits, _ = self.fuzzy_rules(fused)
        return logits


@torch.inference_mode()
def evaluate_model(model: nn.Module, loader, device: torch.device, return_predictions: bool = False) -> dict:
    model.eval()
    y_true = []
    y_pred = []

    for rgb, lbp, labels in loader:
        rgb = rgb.to(device)
        lbp = lbp.to(device)
        labels = labels.to(device)

        logits = model(rgb, lbp)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    result = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(macro_f1),
    }
    if return_predictions:
        result.update({"y_true": y_true, "y_pred": y_pred})
    return result


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    y_true = []
    y_pred = []

    for rgb, lbp, labels in loader:
        rgb = rgb.to(device)
        lbp = lbp.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(rgb, lbp)
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