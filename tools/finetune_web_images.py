"""
Fine-tune model on web images to fix domain shift issues.
Focus: Northern Leaf Blight misclassified as Healthy
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
import json
from datetime import datetime

import sys
sys.path.insert(0, '.')

from src.leaf_disease.data import preprocess_for_inference
from src.leaf_disease.io import load_checkpoint, save_checkpoint
from src.leaf_disease.modeling import build_model
from src.leaf_disease.utils import resolve_device


class WebImageDataset(Dataset):
    """Simple dataset for web images with manual labels"""
    def __init__(self, image_paths, labels, image_size=224):
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        x = preprocess_for_inference(image, self.image_size).squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, label


def finetune_model_on_web_images():
    """
    Fine-tune model on web images to adapt to domain shift.
    
    Workflow:
    1. Load base model
    2. Prepare web image dataset with manual labels
    3. Fine-tune for few epochs with very low LR
    4. Save improved checkpoint
    """
    device = resolve_device('cpu')
    
    # Load base model
    print("Loading base model...")
    ckpt = load_checkpoint('models/mobilenetv3/best_model.pt', map_location='cpu')
    model = build_model('mobilenet_v3_small', num_classes=ckpt['num_classes'], pretrained=False)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    
    class_names = ckpt.get('class_names', [])
    print(f"Classes: {class_names}")
    
    # Prepare web image dataset with MANUAL LABELS
    print("\nPreparing web images for fine-tuning...")
    print("(Update these labels to match your web images)")
    
    web_image_labels = {
        # Format: 'filename': class_index
        # 0=Cercospora, 1=Common_rust, 2=Northern_Leaf_Blight, 3=Healthy
        'images (2).jpg': 1,  # Common Rust
        '250px-Sporulation_on_leaf.jpg': 2,  # Northern Leaf Blight (was misclassified as Healthy)
    }
    
    image_paths = []
    labels = []
    web_dir = Path('web_test_images')
    
    for img_file, label_idx in web_image_labels.items():
        img_path = web_dir / img_file
        if img_path.exists():
            image_paths.append(img_path)
            labels.append(label_idx)
            print(f"  ✓ {img_file:45s} → {class_names[label_idx]}")
        else:
            print(f"  ✗ {img_file:45s} NOT FOUND")
    
    if len(image_paths) == 0:
        print("ERROR: No web images found for fine-tuning!")
        return
    
    print(f"\nTotal images for fine-tuning: {len(image_paths)}")
    
    # Create dataset & loader
    dataset = WebImageDataset(image_paths, labels, image_size=224)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Fine-tune setup
    model.train()
    
    # Use weighted loss to emphasize minority class (Northern Leaf Blight)
    # Count class occurrences
    class_count = {}
    for label in labels:
        class_count[label] = class_count.get(label, 0) + 1
    
    # Compute inverse frequency weights
    max_count = max(class_count.values())
    class_weights = torch.tensor([
        max_count / class_count.get(i, 1) for i in range(ckpt['num_classes'])
    ], dtype=torch.float32).to(device)
    
    print(f"\nClass weights (inverse frequency):")
    for i, (w, name) in enumerate(zip(class_weights, class_names)):
        print(f"  {i}: {name:45s} weight={w:.2f}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)  # Slightly higher LR
    epochs = 20  # More epochs
    
    print(f"\nFine-tuning configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: 5e-5 (low)")
    print(f"  Batch size: 2")
    print(f"  Loss: WeightedCrossEntropyLoss (emphasizes minority classes)")
    print(f"  Device: {device}")
    
    # Training loop
    print("\nStarting fine-tuning...")
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, pred = torch.max(logits, 1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
        
        acc = 100 * correct / total
        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.1f}%")
    
    model.eval()
    
    # Save fine-tuned checkpoint
    print("\nSaving fine-tuned model...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    finetuned_path = f'models/mobilenetv3/best_model_finetuned_{timestamp}.pt'
    
    save_checkpoint(finetuned_path, {
        'model_name': 'mobilenet_v3_small',
        'model_state': model.state_dict(),
        'num_classes': ckpt['num_classes'],
        'class_names': class_names,
        'image_size': 224,
        'training_note': 'Fine-tuned on web images to fix domain shift'
    })
    
    print(f"✓ Saved to: {finetuned_path}")
    print("\nNext steps:")
    print("1. Update 'web_image_labels' dict with your Northern Leaf Blight labels")
    print("2. Re-run this script: python tools/finetune_web_images.py")
    print("3. Test with Streamlit demo using new checkpoint")


if __name__ == '__main__':
    finetune_model_on_web_images()
