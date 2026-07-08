from io import BytesIO

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from api.schemas import PredictResponse
from api.model_loader import load_model
from src.leaf_disease.data import preprocess_for_inference

app = FastAPI(title="Leaf Disease API")

model, device, class_names, image_size = load_model()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(BytesIO(content)).convert("RGB")
    x = preprocess_for_inference(image, image_size=image_size).to(device)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probs, k=min(3, probs.numel()))
    pred_idx = int(torch.argmax(probs).item())

    top3 = [
        {"label": class_names[i], "confidence": float(p)}
        for p, i in zip(top_probs.tolist(), top_indices.tolist())
    ]

    return PredictResponse(
        predicted_label=class_names[pred_idx],
        confidence=float(probs[pred_idx].item()),
        top3=top3,
    )