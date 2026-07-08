import torch
from PIL import Image
import sys
sys.path.insert(0, '.')

from src.leaf_disease.data import preprocess_for_inference
from src.leaf_disease.io import load_checkpoint
from src.leaf_disease.modeling import build_model
from src.leaf_disease.utils import resolve_device

device = resolve_device('cpu')

# Load original model
print("=" * 60)
print("ORIGINAL MODEL (base)")
print("=" * 60)
ckpt_base = load_checkpoint('models/mobilenetv3/best_model.pt')
model_base = build_model('mobilenet_v3_small', num_classes=ckpt_base['num_classes'], pretrained=False)
model_base.load_state_dict(ckpt_base['model_state'])
model_base = model_base.to(device).eval()

# Load fine-tuned model
print("\nLOADING FINE-TUNED MODEL...")
ckpt_finetuned = load_checkpoint('models/mobilenetv3/best_model_finetuned_20260511_004244.pt')
model_finetuned = build_model('mobilenet_v3_small', num_classes=ckpt_finetuned['num_classes'], pretrained=False)
model_finetuned.load_state_dict(ckpt_finetuned['model_state'])
model_finetuned = model_finetuned.to(device).eval()
print("✓ Fine-tuned model loaded\n")

class_names = ckpt_base['class_names']

# Test on Northern Leaf Blight image
img_path = 'web_test_images/250px-Sporulation_on_leaf.jpg'
image = Image.open(img_path).convert('RGB')
x = preprocess_for_inference(image, 224).to(device)

with torch.no_grad():
    # Original
    logits_base = model_base(x)
    probs_base = torch.softmax(logits_base, dim=1).squeeze(0)
    
    # Fine-tuned
    logits_finetuned = model_finetuned(x)
    probs_finetuned = torch.softmax(logits_finetuned, dim=1).squeeze(0)

print(f"Image: {img_path}")
print(f"Ground truth: Northern Leaf Blight\n")

print("ORIGINAL MODEL predictions:")
for i, (prob, name) in enumerate(zip(probs_base, class_names)):
    marker = "← WRONG" if i == 3 else ""  # Was predicting Healthy
    print(f"  {i}: {name:45s} {prob:.1%} {marker}")

print("\nFINE-TUNED MODEL predictions:")
for i, (prob, name) in enumerate(zip(probs_finetuned, class_names)):
    marker = "← CORRECT!" if i == 2 else ""  # Should predict NLB
    print(f"  {i}: {name:45s} {prob:.1%} {marker}")

pred_base = torch.argmax(probs_base).item()
conf_base = float(torch.max(probs_base).item())
pred_ft = torch.argmax(probs_finetuned).item()
conf_ft = float(torch.max(probs_finetuned).item())

print(f"\n{'='*60}")
print(f"COMPARISON:")
print(f"  Original:    {class_names[pred_base]:45s} {conf_base:.1%} {'❌' if pred_base != 2 else '✓'}")
print(f"  Fine-tuned:  {class_names[pred_ft]:45s} {conf_ft:.1%} {'✓' if pred_ft == 2 else '❌'}")
print(f"{'='*60}")
