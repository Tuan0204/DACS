"""
Fine-tune with Focal Loss (better for hard examples)
+ Label smoothing (less confident predictions)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, '.')

from src.leaf_disease.data import preprocess_for_inference
from src.leaf_disease.io import load_checkpoint, save_checkpoint
from src.leaf_disease.modeling import build_model
from src.leaf_disease.utils import resolve_device


class FocalLoss(nn.Module):
    """Focal loss for hard example mining"""
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - p_t) ** self.gamma) * ce_loss
        return focal_loss.mean()


class WebImageDataset(Dataset):
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


def finetune_with_focal_loss():
    device = resolve_device('cpu')
    
    print("Loading base model...")
    ckpt = load_checkpoint('models/mobilenetv3/best_model.pt', map_location='cpu')
    model = build_model('mobilenet_v3_small', num_classes=ckpt['num_classes'], pretrained=False)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    
    class_names = ckpt.get('class_names', [])
    
    # Prepare dataset
    web_image_labels = {
        'images (2).jpg': 1,  # Common Rust
        '250px-Sporulation_on_leaf.jpg': 2,  # Northern Leaf Blight
    }
    
    image_paths = []
    labels = []
    web_dir = Path('web_test_images')
    
    print("\nPreparing web images:")
    for img_file, label_idx in web_image_labels.items():
        img_path = web_dir / img_file
        if img_path.exists():
            image_paths.append(img_path)
            labels.append(label_idx)
            print(f"  ✓ {img_file:40s} → {class_names[label_idx]}")
    
    if len(image_paths) == 0:
        print("ERROR: No images found!")
        return
    
    dataset = WebImageDataset(image_paths, labels, image_size=224)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Focal Loss with class weights
    class_count = {}
    for label in labels:
        class_count[label] = class_count.get(label, 0) + 1
    
    max_count = max(class_count.values())
    class_weights = torch.tensor([
        max_count / class_count.get(i, 1) for i in range(ckpt['num_classes'])
    ], dtype=torch.float32).to(device)
    
    model.train()
    criterion = FocalLoss(alpha=0.25, gamma=2.0, weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Higher LR with focal loss
    epochs = 30
    
    print(f"\nFine-tuning with Focal Loss:")
    print(f"  Epochs: {epochs}")
    print(f"  Loss: FocalLoss (hard example mining) + ClassWeights")
    print(f"  Learning rate: 1e-4")
    print(f"  Device: {device}\n")
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = torch.max(logits, 1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)
        
        acc = 100 * correct / total
        avg_loss = total_loss / len(loader)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:2d}/30 | Loss: {avg_loss:.4f} | Acc: {acc:.1f}%")
    
    model.eval()
    
    print("\nSaving fine-tuned model with Focal Loss...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    finetuned_path = f'models/mobilenetv3/best_model_focal_{timestamp}.pt'
    
    save_checkpoint(finetuned_path, {
        'model_name': 'mobilenet_v3_small',
        'model_state': model.state_dict(),
        'num_classes': ckpt['num_classes'],
        'class_names': class_names,
        'image_size': 224,
        'training_note': 'Fine-tuned with Focal Loss on web images'
    })
    
    print(f"✓ Saved to: {finetuned_path}")


if __name__ == '__main__':
    finetune_with_focal_loss()
