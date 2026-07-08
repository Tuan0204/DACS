from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from src.leaf_disease.config import load_config
import yaml
from src.leaf_disease.data import preprocess_for_inference
from src.leaf_disease.paper_anfis_fuzzy_cnn import _compute_lbp_histogram
from src.leaf_disease.io import load_checkpoint
from src.leaf_disease.modeling import build_model
from src.leaf_disease.utils import resolve_device


MODEL_OPTIONS = {
    "MobileNetV3-Small": {
        "config": Path("configs/mobilenetv3.yaml"),
        "checkpoint": Path("models/mobilenetv3/best_model.pt"),
    },
    "ResNet18": {
        "config": Path("configs/resnet18.yaml"),
        "checkpoint": Path("models/resnet18/best_model.pt"),
    },
    "EfficientNetB4": {
        "config": Path("configs/efficientnetb4.yaml"),
        "checkpoint": Path("models/efficientnetb4/finetune_masked.pt"),
    },
    "Paper-ANFIS": {
        "config": Path("configs/anfis_fuzzy_cnn.yaml"),
        "checkpoint": Path("models/anfis_fuzzy_cnn/best_model.pt"),
    },
}

SHORT_LABELS = {
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Cercospora Leaf Spot",
    "Corn_(maize)___Common_rust_": "Common Rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Northern Leaf Blight",
    "Corn_(maize)___healthy": "Healthy",
}


def _short_label(label: str) -> str:
    return SHORT_LABELS.get(label, label)


def _pick_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    # standard torchvision backbones
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "features"):
        return model.features[-1]
    # PaperAnfisFuzzyCNN: use last conv in rgb_branch if present
    if hasattr(model, "rgb_branch"):
        try:
            # rgb_branch is Sequential of ConvBlock; ConvBlock.block[0] is Conv2d
            last_block = model.rgb_branch[-1]
            if hasattr(last_block, "block") and isinstance(last_block.block, torch.nn.Sequential):
                # find Conv2d module inside
                for module in last_block.block:
                    if isinstance(module, torch.nn.Conv2d):
                        target = module
            else:
                target = last_block
            return target
        except Exception:
            pass
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

        # some model wrappers may provide zero_grad; delegate if available
        if hasattr(self.model, "zero_grad"):
            try:
                self.model.zero_grad(set_to_none=True)
            except TypeError:
                # fallback if signature differs
                self.model.zero_grad()
        else:
            # best-effort: zero grads on contained nn.Modules if present
            try:
                for p in getattr(self.model, "parameters", lambda: [])():
                    if p.grad is not None:
                        p.grad = None
            except Exception:
                pass

        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


@st.cache_resource
def load_artifacts(model_key: str):
    base_dir = Path(__file__).resolve().parent
    model_cfg = MODEL_OPTIONS[model_key]
    config_path = base_dir / model_cfg["config"]
    checkpoint_path = base_dir / model_cfg["checkpoint"]

    cfg = load_config(config_path)
    # also load raw dict for model-specific keys (e.g., num_rules, dropout)
    try:
        raw_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        raw_cfg = {}
    device = resolve_device(cfg.device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=str(device))
    model_name = checkpoint.get("model_name", getattr(cfg, "model_name", None))
    # support paper_anfis_fuzzy_cnn specially
    # Heuristic: some checkpoints may not record model_name correctly; detect ANFIS by saved state keys.
    state_keys = list(checkpoint.get("model_state", {}).keys()) if isinstance(checkpoint.get("model_state", {}), dict) else []
    seems_anfis = any(k.startswith("rgb_branch") or "num_rules" in k for k in state_keys) or (model_name and "anfis" in str(model_name).lower())
    if seems_anfis:
        from src.leaf_disease.paper_anfis_fuzzy_cnn import PaperAnfisFuzzyCNN

        model = PaperAnfisFuzzyCNN(
            num_classes=checkpoint["num_classes"],
            num_rules=checkpoint.get("num_rules", raw_cfg.get("num_rules", 8)),
            image_size=checkpoint.get("image_size", raw_cfg.get("image_size", cfg.image_size)),
            dropout=checkpoint.get("dropout", raw_cfg.get("dropout", 0.3)),
        )
        model.load_state_dict(checkpoint["model_state"], strict=False)
    else:
        model = build_model(
            checkpoint.get("model_name", getattr(cfg, "model_name", None)),
            num_classes=checkpoint["num_classes"],
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state"])

    model = model.to(device)
    model.eval()

    return {
        "cfg": cfg,
        "device": device,
        "model": model,
        "class_names": checkpoint["class_names"],
        "image_size": checkpoint.get("image_size", cfg.image_size),
        "checkpoint_path": checkpoint_path,
        "model_name": checkpoint.get("model_name", cfg.model_name),
    }


def _predict(model: torch.nn.Module, x: torch.Tensor, lbp: torch.Tensor | None = None):
    with torch.inference_mode():
        if lbp is None:
            logits = model(x)
        else:
            logits = model(x, lbp)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    top_probs, top_indices = torch.topk(probs, k=min(3, probs.numel()))
    return probs, int(torch.argmax(probs).item()), float(torch.max(probs).item()), top_probs, top_indices


def _render_prob_bar(class_names: list[str], probs: torch.Tensor) -> None:
    short_names = [_short_label(name) for name in class_names]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    values = probs.detach().cpu().numpy()
    colors = ["#64748b", "#2563eb", "#0ea5e9", "#14b8a6"]
    bars = ax.barh(short_names, values, color=colors[: len(short_names)])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Prediction Confidence")
    ax.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(min(value + 0.01, 0.99), bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    st.pyplot(fig, clear_figure=True)


def _apply_mask_if_available(image: Image.Image, uploaded_name: str) -> Image.Image:
    """Try to apply a segmented mask from data/raw/segmented using the uploaded filename.

    If the same basename exists under segmented/*/, the image is masked before inference.
    This helps keep demo preprocessing closer to training when using train-set images.
    """
    base_dir = Path(__file__).resolve().parent
    segmented_root = base_dir / "data" / "raw" / "segmented"
    if not segmented_root.exists():
        return image

    matches = list(segmented_root.glob(f"**/{uploaded_name}"))
    if not matches:
        return image

    seg_path = matches[0]
    try:
        mask = Image.open(seg_path).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        mask_arr = np.array(mask).astype(np.float32) / 255.0
        img_arr = np.array(image).astype(np.float32) / 255.0
        img_arr = img_arr * mask_arr[..., None]
        return Image.fromarray((img_arr * 255).astype(np.uint8))
    except Exception:
        return image


def _find_mask_path_for_upload(uploaded_name: str):
    base_dir = Path(__file__).resolve().parent
    segmented_root = base_dir / "data" / "raw" / "segmented"
    if not segmented_root.exists():
        return None
    matches = list(segmented_root.glob(f"**/{uploaded_name}"))
    return matches[0] if matches else None


def _make_gradcam(model: torch.nn.Module, x: torch.Tensor, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    target_layer = _pick_target_layer(model)
    cam_engine = GradCAM(model, target_layer)
    try:
        # handle models that require lbp (PaperAnfisFuzzyCNN)
        # Only compute LBP for ANFIS models (models with rgb_branch)
        lbp = None
        if hasattr(model, "rgb_branch"):
            try:
                # compute LBP histogram for the image
                lbp_vec = _compute_lbp_histogram(image, x.shape[-1])
                lbp = lbp_vec.unsqueeze(0).to(x.device)
            except Exception:
                lbp = None

        with torch.inference_mode():
            if lbp is None:
                logits = model(x)
            else:
                logits = model(x, lbp)
            pred_idx = int(torch.argmax(logits, dim=1).item())
        # Grad-CAM call: ensure cam engine forward uses same signature.
        # Create a small wrapper that provides __call__ and zero_grad delegating to original model.
        class CamModelWrapper:
            def __init__(self, orig_model, lbp_arg):
                self._orig = orig_model
                self._lbp = lbp_arg

            def __call__(self, t: torch.Tensor):
                if self._lbp is None:
                    return self._orig(t)
                return self._orig(t, self._lbp)

            def zero_grad(self, set_to_none: bool = True):
                # delegate if orig has zero_grad
                if hasattr(self._orig, "zero_grad"):
                    try:
                        self._orig.zero_grad(set_to_none=set_to_none)
                        return
                    except TypeError:
                        try:
                            self._orig.zero_grad()
                            return
                        except Exception:
                            pass
                # fallback: clear grads on parameters
                for p in getattr(self._orig, "parameters", lambda: [])():
                    if p.grad is not None:
                        p.grad = None

        orig_model = cam_engine.model
        cam_engine.model = CamModelWrapper(model, lbp)
        cam = cam_engine(x, pred_idx)
        cam_engine.model = orig_model
    finally:
        cam_engine.remove_hooks()

    image_np = np.array(image.resize((x.shape[-1], x.shape[-2]))).astype(np.float32) / 255.0
    heatmap = plt.cm.jet(cam)[..., :3]
    overlay = np.clip(0.55 * image_np + 0.45 * heatmap, 0, 1)
    return image_np, overlay


def main() -> None:
    st.set_page_config(page_title="Leaf Disease Demo", page_icon="🌿", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f7faf7 0%, #eef6ef 100%);
        }
        .block-container {
            padding-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Leaf Disease Detection Demo")
    st.caption("Corn leaf disease classification with optional Grad-CAM explanation")

    with st.sidebar:
        st.header("Demo Settings")
        model_key = st.selectbox("Choose model", list(MODEL_OPTIONS.keys()), index=0)
        two_stage = st.checkbox("Enable binary gate for EfficientNetB4", value=(model_key == "EfficientNetB4"))
        binary_threshold = st.slider("Binary diseased threshold", 0.50, 0.95, 0.60, 0.05)
        show_gradcam = st.checkbox("Show Grad-CAM", value=True)
        debug_info = st.checkbox("Show developer debug info", value=True)
        st.markdown("---")
        st.write("**Quick tips**")
        st.write("- Upload one corn leaf image")
        st.write("- Review top predictions")
        st.write("- Enable Grad-CAM to see the highlighted region")

    artifacts = load_artifacts(model_key)
    cfg = artifacts["cfg"]
    device = artifacts["device"]
    model = artifacts["model"]
    class_names = artifacts["class_names"]
    image_size = artifacts["image_size"]
    base_dir = Path(__file__).resolve().parent

    # Optionally load binary healthy vs diseased model for two-stage inference
    binary_model = None
    binary_class_names = None
    if two_stage and model_key == "EfficientNetB4":
        base_dir = Path(__file__).resolve().parent
        binary_cp = base_dir / "models/efficientnetb4/binary_finetune_masked.pt"
        if binary_cp.exists():
            try:
                b_ckpt = load_checkpoint(binary_cp, map_location=str(device))
                binary_model = build_model(b_ckpt.get("model_name", cfg.model_name), num_classes=b_ckpt.get("num_classes", 2), pretrained=False)
                try:
                    binary_model.load_state_dict(b_ckpt["model_state"])
                except Exception:
                    binary_model.load_state_dict(b_ckpt["model_state"], strict=False)
                binary_model = binary_model.to(device).eval()
                binary_class_names = b_ckpt.get("class_names", ["healthy", "diseased"])[:2]
            except Exception:
                st.warning("Binary checkpoint found but failed to load. Two-stage disabled.")
                binary_model = None
        else:
            st.info("Binary checkpoint not found: two-stage mode unavailable.")
    elif two_stage:
        st.info("Binary gate is only available for EfficientNetB4.")

    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.subheader("Upload image")
        uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded is None:
            st.info("Upload an image to start prediction.")
            st.stop()

        image = Image.open(uploaded).convert("RGB")
        # find mask path (if any) for debug display
        seg_path = _find_mask_path_for_upload(uploaded.name)
        # apply mask if available (keeps existing behavior)
        image = _apply_mask_if_available(image, uploaded.name)
        st.image(image, caption="Input image", use_container_width=True)

        if debug_info:
            if seg_path is not None:
                try:
                    mask_img = Image.open(seg_path).convert("L")
                    if mask_img.size != image.size:
                        mask_img = mask_img.resize(image.size, Image.Resampling.NEAREST)
                    st.caption(f"Applied mask: {seg_path.as_posix()}")
                    st.image(mask_img.convert("RGB"), caption="Applied mask (grayscale)", use_container_width=True)
                except Exception:
                    st.caption(f"Mask found but failed to load: {seg_path.as_posix()}")
            else:
                st.caption("No segmented mask found for this uploaded filename.")

    x = preprocess_for_inference(image, image_size=image_size).to(device)

    # Some models (PaperAnfisFuzzyCNN) require an LBP vector in addition to the RGB tensor.
    lbp = None
    if hasattr(model, "rgb_branch"):
        try:
            lbp_vec = _compute_lbp_histogram(image, image_size)
            lbp = lbp_vec.unsqueeze(0).to(device)
        except Exception:
            lbp = None

    # If two-stage enabled and binary model loaded, run binary first
    binary_result = None
    if binary_model is not None:
        with torch.inference_mode():
            b_logits = binary_model(x)
            b_probs = torch.softmax(b_logits, dim=1).squeeze(0)
            b_diseased_prob = float(b_probs[1].item()) if b_probs.numel() > 1 else 0.0
            b_pred = 1 if b_diseased_prob >= binary_threshold else 0
            b_conf = float(torch.max(b_probs).item())
        binary_result = (binary_class_names, b_probs, b_pred, b_conf)
        # If binary predicts healthy (index 0), skip multiclass and report healthy
        if b_pred == 0:
            # Map binary "healthy" to the multiclass index for the healthy class
            healthy_idx = None
            for i, nm in enumerate(class_names):
                if "healthy" in str(nm).lower():
                    healthy_idx = i
                    break
            if healthy_idx is None:
                healthy_idx = 0
            probs = torch.zeros(len(class_names), device=device)
            probs[healthy_idx] = 1.0
            pred_idx = int(healthy_idx)
            confidence = float(b_probs[0].item()) if b_probs.numel() > 0 else b_conf
            top_probs = torch.tensor([confidence])
            top_indices = torch.tensor([pred_idx])
            predicted_label = class_names[pred_idx]
            binary_used = True
        else:
            probs, pred_idx, confidence, top_probs, top_indices = _predict(model, x, lbp)
            # If binary predicted diseased, ensure multiclass does not return the 'healthy' class
            if b_pred == 1:
                # find healthy index in class_names
                healthy_idx = None
                for i, nm in enumerate(class_names):
                    if "healthy" in str(nm).lower():
                        healthy_idx = i
                        break
                if healthy_idx is not None and healthy_idx < probs.numel():
                    probs = probs.clone()
                    probs[healthy_idx] = 0.0
                    # renormalize
                    s = float(probs.sum().item())
                    if s > 0:
                        probs = probs / s
                    # recompute top-k
                    top_probs, top_indices = torch.topk(probs, k=min(3, probs.numel()))
                    pred_idx = int(top_indices[0].item())
                    confidence = float(top_probs[0].item())
            predicted_label = class_names[pred_idx]
            binary_used = True
    else:
        probs, pred_idx, confidence, top_probs, top_indices = _predict(model, x, lbp)
        predicted_label = class_names[pred_idx]
        binary_used = False

    with col_right:
        st.subheader("Prediction result")
        if binary_result is not None:
            b_class_names, b_probs, b_pred, b_conf = binary_result
            st.metric("Binary stage", _short_label(b_class_names[b_pred]))
            st.caption(f"P(healthy)={b_probs[0].item() * 100:.2f}% | P(diseased)={b_probs[1].item() * 100:.2f}%")
            st.caption(f"Binary threshold: {binary_threshold:.2f}")
            st.caption("Binary gate: active" if binary_used else "Binary gate: inactive")
        st.metric("Predicted disease", _short_label(predicted_label))
        st.metric("Confidence", f"{confidence * 100:.2f}%")
        st.caption(f"Device: {device} | Image size: {image_size}px | Checkpoint: {artifacts['checkpoint_path'].as_posix()}")

        st.markdown("#### Top-3 classes")
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            st.write(f"- {_short_label(class_names[idx])}: {prob:.4f}")

        st.markdown("#### Confidence chart")
        _render_prob_bar(class_names, probs)

    if show_gradcam:
        st.subheader("Grad-CAM explanation")
        image_np, overlay = _make_gradcam(model, x, image)
        cam_col1, cam_col2 = st.columns(2)
        with cam_col1:
            st.image(image_np, caption="Original resized image", clamp=True, use_container_width=True)
        with cam_col2:
            st.image(overlay, caption=f"Grad-CAM for {_short_label(predicted_label)}", clamp=True, use_container_width=True)

    if debug_info:
        with st.expander("Developer debug info (logits & probs)", expanded=True):
            try:
                with torch.inference_mode():
                    if hasattr(model, "rgb_branch") and lbp is not None:
                        mc_logits = model(x, lbp)
                    else:
                        mc_logits = model(x)
                    mc_probs = torch.softmax(mc_logits, dim=1).squeeze(0)
                # prepare table
                probs_table = { _short_label(name): float(p) for name, p in zip(class_names, mc_probs.tolist()) }
                st.write("**Multiclass probabilities**")
                st.table(probs_table)
                st.write("**Multiclass raw logits**")
                try:
                    logits_np = mc_logits.detach().cpu().numpy().squeeze(0)
                    logits_table = { _short_label(name): float(v) for name, v in zip(class_names, logits_np.tolist()) }
                    st.table(logits_table)
                except Exception:
                    st.write(mc_logits)

                if binary_result is not None:
                    st.write("**Binary stage logits/probs**")
                    try:
                        b_logits_np = b_logits.detach().cpu().numpy().squeeze(0)
                        b_probs_np = b_probs.detach().cpu().numpy().tolist()
                        st.write({ 'binary_logits': b_logits_np.tolist(), 'binary_probs': b_probs_np })
                    except Exception:
                        st.write({'binary_probs': [float(p) for p in b_probs.tolist()]})
            except Exception as e:
                st.write(f"Failed to compute debug logits: {e}")

    with st.expander("Model information", expanded=False):
        st.write(f"- Model name: {artifacts['model_name']}")
        st.write(f"- Number of classes: {len(class_names)}")
        st.write(f"- Config file: {MODEL_OPTIONS[model_key]['config'].as_posix()}")
        st.write(f"- Checkpoint: {MODEL_OPTIONS[model_key]['checkpoint'].as_posix()}")
        st.write("- Classes: " + ", ".join(_short_label(name) for name in class_names))


if __name__ == "__main__":
    main()
