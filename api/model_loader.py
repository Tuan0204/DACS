from pathlib import Path
import torch
import yaml

from src.leaf_disease.config import load_config
from src.leaf_disease.data import preprocess_for_inference
from src.leaf_disease.io import load_checkpoint
from src.leaf_disease.modeling import build_model
from src.leaf_disease.utils import resolve_device

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "mobilenetv3.yaml"
CHECKPOINT_PATH = BASE_DIR / "models" / "mobilenetv3" / "best_model.pt"

def load_model():
    cfg = load_config(CONFIG_PATH)
    device = resolve_device(cfg.device)
    ckpt = load_checkpoint(CHECKPOINT_PATH, map_location=str(device))
    model = build_model(
        ckpt.get("model_name", cfg.model_name),
        num_classes=ckpt["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    return model, device, ckpt["class_names"], ckpt.get("image_size", cfg.image_size)