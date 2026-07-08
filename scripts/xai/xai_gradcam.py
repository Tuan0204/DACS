from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.leaf_disease.config import load_config
from src.leaf_disease.data import preprocess_for_inference
from src.leaf_disease.io import load_checkpoint
from src.leaf_disease.modeling import build_model
from src.leaf_disease.utils import resolve_device


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_hook = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_hook = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _input, output):
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self.fwd_hook.remove()
        self.bwd_hook.remove()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        logits = self.model(x)
        score = logits[:, class_idx].sum()

        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for one image")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="reports/gradcam.png")
    return parser.parse_args()


def pick_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError("Cannot infer target layer for Grad-CAM")


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

    image = Image.open(args.image).convert("RGB")
    x = preprocess_for_inference(image, image_size=checkpoint.get("image_size", cfg.image_size)).to(device)

    with torch.inference_mode():
        logits = model(x)
        pred_idx = int(torch.argmax(logits, dim=1).item())

    target_layer = pick_target_layer(model)
    cam_engine = GradCAM(model, target_layer)
    cam = cam_engine(x, pred_idx)
    cam_engine.remove_hooks()

    image_np = np.array(image.resize((x.shape[-1], x.shape[-2]))).astype(np.float32) / 255.0
    heatmap = plt.cm.jet(cam)[..., :3]
    overlay = np.clip(0.55 * image_np + 0.45 * heatmap, 0, 1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print("Saved Grad-CAM to", output_path)


if __name__ == "__main__":
    main()
