from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from src.leaf_disease.config import load_config
from src.leaf_disease.data import preprocess_for_inference
from src.leaf_disease.io import load_checkpoint
from src.leaf_disease.modeling import build_model
from src.leaf_disease.utils import resolve_device


@st.cache_resource
def load_artifacts(config_path: str, checkpoint_path: str):
    cfg = load_config(config_path)
    device = resolve_device(cfg.device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=str(device))

    model = build_model(
        checkpoint.get("model_name", cfg.model_name),
        num_classes=checkpoint["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    return cfg, device, model, checkpoint["class_names"], checkpoint.get("image_size", cfg.image_size)


def main() -> None:
    st.set_page_config(page_title="Leaf Disease Demo", layout="centered")
    st.title("Leaf Disease Detection Demo")
    st.caption("Do an co so - Tomato disease classification")

    config_path = st.text_input("Config path", value="configs/default.yaml")
    checkpoint_path = st.text_input("Checkpoint path", value="models/best_model.pt")

    if not Path(checkpoint_path).exists():
        st.warning("Checkpoint not found. Please train model first.")
        return

    cfg, device, model, class_names, image_size = load_artifacts(config_path, checkpoint_path)
    st.write(f"Running on device: {device}")

    uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input image", use_container_width=True)

    x = preprocess_for_inference(image, image_size=image_size).to(device)
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    label = class_names[int(pred.item())]
    confidence = float(conf.item())

    st.subheader("Prediction")
    st.write(f"Label: {label}")
    st.write(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
