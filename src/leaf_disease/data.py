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
    def __init__(
        self,
        paths: Sequence[Path],
        labels: Sequence[int],
        transform=None,
        segmented_root: Path | None = None,
        grayscale_root: Path | None = None,
    ) -> None:
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform
        self.segmented_root = Path(segmented_root) if segmented_root is not None else None
        self.grayscale_root = Path(grayscale_root) if grayscale_root is not None else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image_path = self.paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert("RGB")

        # If segmented masks are available, try to apply mask with same relative path
        if self.segmented_root is not None:
            # Compose mask path: segmented_root / class_name / image_name
            try:
                seg_path = self.segmented_root.joinpath(*image_path.parts[-2:])
            except Exception:
                seg_path = self.segmented_root / image_path.name

            if seg_path.exists():
                try:
                    mask = Image.open(seg_path).convert("L")
                    # resize mask to image size if needed
                    if mask.size != image.size:
                        mask = mask.resize(image.size, Image.Resampling.NEAREST)
                    mask_arr = np.array(mask).astype(np.float32) / 255.0
                    img_arr = np.array(image).astype(np.float32) / 255.0
                    img_arr = (img_arr * mask_arr[..., None])
                    image = Image.fromarray((img_arr * 255).astype(np.uint8))
                except Exception:
                    pass

        # grayscale variants can be used as augmentation elsewhere; keep color here
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
    # detect expected layout: data_dir may be root containing 'color', 'segmented', 'grayscale'
    base = Path(data_dir)
    if (base / "color").exists():
        color_root = base / "color"
    else:
        color_root = base

    segmented_root = base / "segmented" if (base / "segmented").exists() else None
    grayscale_root = base / "grayscale" if (base / "grayscale").exists() else None
    use_cuda = torch.cuda.is_available()

    image_paths, labels, class_names = _discover_image_samples(color_root)
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

    train_ds = LeafDataset(
        [image_paths[i] for i in train_idx],
        [labels[i] for i in train_idx],
        train_transform,
        segmented_root=segmented_root,
        grayscale_root=grayscale_root,
    )
    val_ds = LeafDataset(
        [image_paths[i] for i in val_idx],
        [labels[i] for i in val_idx],
        eval_transform,
        segmented_root=segmented_root,
        grayscale_root=grayscale_root,
    )
    test_ds = LeafDataset(
        [image_paths[i] for i in test_idx],
        [labels[i] for i in test_idx],
        eval_transform,
        segmented_root=segmented_root,
        grayscale_root=grayscale_root,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "persistent_workers": bool(use_cuda and num_workers > 0),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

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
    """
    Preprocess image for inference using letterboxing (preserve aspect ratio).
    
    Why letterbox vs direct resize?
    - Direct Resize(224,224) distorts aspect ratio → hurts disease detection
    - Letterbox (pad + center) preserves aspect ratio → better for web images
    - This matches how models should handle diverse input ratios
    """
    # Step 1: Fit image into image_size×image_size while preserving aspect ratio
    image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    
    # Step 2: Create square canvas and center image (letterbox)
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    offset_x = (image_size - image.width) // 2
    offset_y = (image_size - image.height) // 2
    canvas.paste(image, (offset_x, offset_y))
    
    # Step 3: Normalize
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(canvas).unsqueeze(0)
