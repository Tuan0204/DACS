from __future__ import annotations

import argparse

from PIL import Image
import torch

from src.leaf_disease.config import load_config
from src.leaf_disease.data import preprocess_for_inference
from src.leaf_disease.io import load_checkpoint
from src.leaf_disease.modeling import build_model
from src.leaf_disease.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Soft-voting ensemble inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--image", required=True)
    return parser.parse_args()


def load_model_from_checkpoint(path: str, cfg, device: torch.device):
    ckpt = load_checkpoint(path, map_location=str(device))
    model = build_model(ckpt.get("model_name", cfg.model_name), ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    return model, ckpt


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.device)

    models = []
    class_names = None
    image_size = cfg.image_size

    for cp in args.checkpoints:
        model, ckpt = load_model_from_checkpoint(cp, cfg, device)
        models.append(model)
        if class_names is None:
            class_names = ckpt["class_names"]
            image_size = ckpt.get("image_size", cfg.image_size)

    image = Image.open(args.image).convert("RGB")
    x = preprocess_for_inference(image, image_size=image_size).to(device)

    probs_sum = None
    with torch.inference_mode():
        for model in models:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs

    probs_avg = probs_sum / len(models)
    conf, pred = torch.max(probs_avg, dim=1)

    print(f"Predicted class: {class_names[int(pred.item())]}")
    print(f"Confidence: {float(conf.item()):.4f}")


if __name__ == "__main__":
    main()
