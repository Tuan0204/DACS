from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class LeafDataset(Dataset):
    def __init__(self, paths: Sequence[Path], labels: Sequence[int], transform=None) -> None:
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image_path = self.paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


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


def create_dataloaders(
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

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = LeafDataset([image_paths[i] for i in train_idx], [labels[i] for i in train_idx], train_transform)
    val_ds = LeafDataset([image_paths[i] for i in val_idx], [labels[i] for i in val_idx], eval_transform)
    test_ds = LeafDataset([image_paths[i] for i in test_idx], [labels[i] for i in test_idx], eval_transform)

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


def preprocess_for_inference(image: Image.Image, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)
