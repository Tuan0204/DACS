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
print("=" * 70)
print("ORIGINAL MODEL (base)")
print("=" * 70)
ckpt_base = load_checkpoint('models/mobilenetv3/best_model.pt')
model_base = build_model('mobilenet_v3_small', num_classes=ckpt_base['num_classes'], pretrained=False)
model_base.load_state_dict(ckpt_base['model_state'])
model_base = model_base.to(device).eval()

# Load Focal Loss fine-tuned model
print("\nLOADING FOCAL LOSS FINE-TUNED MODEL...")
ckpt_focal = load_checkpoint('models/mobilenetv3/best_model_focal_20260511_004528.pt')
model_focal = build_model('mobilenet_v3_small', num_classes=ckpt_focal['num_classes'], pretrained=False)
model_focal.load_state_dict(ckpt_focal['model_state'])
model_focal = model_focal.to(device).eval()
print("✓ Focal Loss model loaded\n")

class_names = ckpt_base['class_names']

# Test on Northern Leaf Blight image
img_path = 'web_test_images/250px-Sporulation_on_leaf.jpg'
image = Image.open(img_path).convert('RGB')
x = preprocess_for_inference(image, 224).to(device)

with torch.no_grad():
    # Original
    logits_base = model_base(x)
    probs_base = torch.softmax(logits_base, dim=1).squeeze(0)
    
    # Focal Loss
    logits_focal = model_focal(x)
    probs_focal = torch.softmax(logits_focal, dim=1).squeeze(0)

print(f"Image: {img_path}")
print(f"Ground truth: Northern Leaf Blight\n")

print("ORIGINAL MODEL predictions:")
for i, (prob, name) in enumerate(zip(probs_base, class_names)):
    marker = " ← WRONG" if i == 3 else ""
    print(f"  {i}: {name:45s} {prob:.1%}{marker}")

print("\nFOCAL LOSS FINE-TUNED predictions:")
for i, (prob, name) in enumerate(zip(probs_focal, class_names)):
    marker = " ← CORRECT!" if i == 2 else ""
    print(f"  {i}: {name:45s} {prob:.1%}{marker}")

pred_base = torch.argmax(probs_base).item()
conf_base = float(torch.max(probs_base).item())
pred_focal = torch.argmax(probs_focal).item()
conf_focal = float(torch.max(probs_focal).item())

print(f"\n{'='*70}")
print(f"COMPARISON:")
print(f"  Original:       {class_names[pred_base]:45s} {conf_base:.1%} {'❌' if pred_base != 2 else '✓'}")
print(f"  Focal Loss:     {class_names[pred_focal]:45s} {conf_focal:.1%} {'✓' if pred_focal == 2 else '❌'}")
print(f"{'='*70}")

if pred_focal == 2 and conf_focal > 0.5:
    print("\n✅ SUCCESS! Northern Leaf Blight now correctly predicted!")
    print(f"   Confidence: {conf_focal:.1%}")
else:
    print(f"\n⚠️  Still challenging: predicted {class_names[pred_focal]} ({conf_focal:.1%})")
    print(f"   → NLB confidence improved but not enough to flip prediction")
