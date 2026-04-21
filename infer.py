from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import torch

from src.leaf_disease.config import load_config
from src.leaf_disease.data import preprocess_for_inference
from src.leaf_disease.io import load_checkpoint
from src.leaf_disease.modeling import build_model
from src.leaf_disease.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one image")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.device)

    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))
    model = build_model(
        checkpoint.get("model_name", cfg.model_name),
        num_classes=checkpoint["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    class_names = checkpoint["class_names"]

    image = Image.open(args.image).convert("RGB")
    x = preprocess_for_inference(image, image_size=checkpoint.get("image_size", cfg.image_size)).to(device)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    label = class_names[int(pred.item())]
    confidence = float(conf.item())

    print(f"Predicted class: {label}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
