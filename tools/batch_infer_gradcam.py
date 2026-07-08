"""
Batch inference + Grad-CAM overlays for debugging domain-shifted images.
Usage examples:
  # quick check (only load model + print info)
  python tools/batch_infer_gradcam.py --check

  # run on folder of images
  python tools/batch_infer_gradcam.py --input web_test_images --output reports/web_test_results

Outputs:
  - CSV: <output>/results.csv (filename, pred_label, pred_conf, top3)
  - Grad-CAM overlays: <output>/overlays/<filename>.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ensure project root is on sys.path so `src` package is importable when running from tools/
project_root = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(project_root))

from src.leaf_disease.config import load_config
from src.leaf_disease.data import preprocess_for_inference
from src.leaf_disease.io import load_checkpoint
from src.leaf_disease.modeling import build_model
from src.leaf_disease.utils import resolve_device


def _pick_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError("Cannot infer target layer for Grad-CAM")


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


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = load_checkpoint(checkpoint_path, map_location=str(device))
    
    # Handle both old (just state_dict) and new (wrapped with metadata) formats
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        # New format with metadata
        model = build_model(ckpt.get("model_name", "mobilenet_v3_small"), num_classes=ckpt["num_classes"], pretrained=False)
        model.load_state_dict(ckpt["model_state"])
    else:
        # Old format or just state_dict - infer num_classes from state_dict
        # Assume mobilenet_v3_small and get num_classes from classifier layer
        if isinstance(ckpt, dict) and any("classifier" in k for k in ckpt.keys()):
            # This is likely a state_dict from mobilenet_v3_small
            classifier_weight_key = None
            for k in ckpt.keys():
                if "classifier.3.weight" in k:  # Last linear layer
                    classifier_weight_key = k
                    break
            if classifier_weight_key:
                num_classes = ckpt[classifier_weight_key].shape[0]
            else:
                num_classes = 4  # Default fallback
        else:
            num_classes = 4  # Default fallback
        
        model = build_model("mobilenet_v3_small", num_classes=num_classes, pretrained=False)
        model.load_state_dict(ckpt)
        ckpt = {"model_name": "mobilenet_v3_small", "num_classes": num_classes}
    
    model = model.to(device).eval()
    return model, ckpt


def infer_image(model: torch.nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, int, float, torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    top_probs, top_indices = torch.topk(probs, k=min(3, probs.numel()))
    return probs, int(torch.argmax(probs).item()), float(torch.max(probs).item()), top_probs, top_indices


def make_overlay(image: Image.Image, cam: np.ndarray, x_shape: Tuple[int, int]) -> np.ndarray:
    image_np = np.array(image.resize((x_shape[1], x_shape[0]))).astype(np.float32) / 255.0
    heatmap = plt.cm.jet(cam)[..., :3]
    overlay = np.clip(0.55 * image_np + 0.45 * heatmap, 0, 1)
    return (overlay * 255).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="web_test_images", help="Folder with test images")
    parser.add_argument("--output", type=str, default="reports/web_test_results", help="Output folder")
    parser.add_argument("--checkpoint", type=str, default="models/mobilenetv3/best_model.pt")
    parser.add_argument("--config", type=str, default="configs/mobilenetv3.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--extensions", type=str, default="jpg,jpeg,png,bmp")
    parser.add_argument("--check", action="store_true", help="Only load model and print info")
    parser.add_argument("--max", type=int, default=0, help="Max images to process (0=all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    out_dir = Path(args.output)
    out_overlays = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_overlays.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    base_dir = project_root
    checkpoint_path = base_dir / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    config_path = base_dir / args.config if not Path(args.config).is_absolute() else Path(args.config)

    if not checkpoint_path.exists():
        print("Checkpoint not found:", checkpoint_path)
        sys.exit(1)

    # load config to get image size
    cfg = load_config(config_path)

    model, ckpt = load_model(checkpoint_path, device)
    print("Model loaded:", ckpt.get("model_name", "unknown"))
    print("Num classes:", ckpt["num_classes"]) if "num_classes" in ckpt else None
    print("Class names:", ckpt.get("class_names", []))

    if args.check:
        print("Check mode: loaded model and config successfully. Exiting.")
        return

    exts = tuple(ext.strip().lower() for ext in args.extensions.split(","))
    images = [p for p in (Path(args.input).glob("**/*")) if p.suffix.lower().lstrip(".") in exts]
    if not images:
        print("No images found in", input_dir)
        return

    target_layer = _pick_target_layer(model)
    cam_engine = GradCAM(model, target_layer)

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "pred_label", "pred_conf", "top3"])

        processed = 0
        for img_path in images:
            if args.max and processed >= args.max:
                break
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print("Failed to open image", img_path, e)
                continue

            x = preprocess_for_inference(image, image_size=cfg.image_size).to(device)
            probs, pred_idx, conf, top_probs, top_indices = infer_image(model, x)
            class_names = ckpt.get("class_names", [str(i) for i in range(ckpt.get("num_classes", 4))])
            pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
            top3 = ";".join([f"{class_names[int(idx)]}:{float(p):.4f}" for p, idx in zip(top_probs.tolist(), top_indices.tolist())])

            # Grad-CAM
            cam = cam_engine(x, pred_idx)
            overlay = make_overlay(image, cam, x.shape[-2:])
            overlay_path = out_overlays / f"{img_path.stem}_overlay.png"
            Image.fromarray(overlay).save(overlay_path)

            writer.writerow([str(img_path), pred_label, f"{conf:.4f}", top3])

            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed} images")

    cam_engine.remove_hooks()
    print("Done. Results:", csv_path)


if __name__ == "__main__":
    main()
