"""
Fine-tune model on misclassified images with Focal Loss
Corrects Common Rust, Cercospora, and ambiguous predictions
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
import json
from datetime import datetime
import sys

sys.path.insert(0, 'src')
from leaf_disease.modeling import build_model
from leaf_disease.data import preprocess_for_inference
from PIL import Image
import numpy as np

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# Manual dataset with known labels
class CorrectionDataset(Dataset):
    def __init__(self, image_paths, labels, class_names):
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.label_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_name = self.labels[idx]
        label = self.label_to_idx[label_name]
        
        # Load and preprocess (preprocess_for_inference already adds batch dim)
        image = Image.open(img_path).convert('RGB')
        image_tensor = preprocess_for_inference(image, 224)
        image_tensor = image_tensor.squeeze(0)  # Remove batch dimension added by preprocess_for_inference
        
        return image_tensor, label, Path(img_path).name

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load class names
    with open('models/mobilenetv3/class_names.json', 'r') as f:
        class_names = json.load(f)
    
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    
    # Load model
    model = build_model('mobilenet_v3_small', num_classes=num_classes)
    checkpoint = torch.load('models/mobilenetv3/best_model.pt', map_location=device)
    
    # Handle both old (just state_dict) and new (wrapped with metadata) formats
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    print("Model loaded from best_model.pt")
    
    # Prepare correction dataset
    # IMPORTANT: Set correct labels manually
    correction_images = [
        'data/finetune_correction/images.jpg',
        'data/finetune_correction/setomaize2.jpg',
        'data/finetune_correction/images (4).jpg',
    ]
    
    correction_labels = [
        'Corn_(maize)___Common_rust_',           # images.jpg - Common Rust
        'Corn_(maize)___Common_rust_',           # setomaize2.jpg - Common Rust
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',  # images(4).jpg - Cercospora
    ]
    
    print("\n📋 Correction Dataset:")
    for img, label in zip(correction_images, correction_labels):
        print(f"  {Path(img).name:30} -> {label.replace('Corn_(maize)___', '').replace('_', ' ')}")
    
    # Create dataset
    dataset = CorrectionDataset(correction_images, correction_labels, class_names)
    
    # Use repeat for small dataset
    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        drop_last=False
    )
    
    # Loss & Optimizer with weighted Focal Loss
    class_weights = torch.tensor([1.0, 1.0, 2.0, 1.0], device=device)  # Higher weight for Common Rust
    focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    # Fine-tune for 15 epochs
    print("\n🔧 Fine-tuning with Focal Loss (15 epochs)...")
    model.train()
    
    for epoch in range(15):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels, filenames) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = focal_loss(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            print(f"  Epoch {epoch+1}/15 - Batch {batch_idx+1}/{len(train_loader)}: "
                  f"Loss={loss.item():.4f}, Acc={100*correct/total:.1f}%")
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/15: Loss={avg_loss:.4f}, Accuracy={accuracy:.1f}%\n")
    
    # Save corrected model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f'models/mobilenetv3/best_model_correction_{timestamp}.pt'
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n✅ Model saved: {checkpoint_path}")
    
    # Copy to best_model.pt
    torch.save(model.state_dict(), 'models/mobilenetv3/best_model.pt')
    print(f"✅ Updated: models/mobilenetv3/best_model.pt")
    
    # Test on correction images
    print("\n📊 Testing on corrected images:")
    model.eval()
    with torch.no_grad():
        for images, labels, filenames in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            pred = outputs.argmax(dim=1)
            pred_prob = probs[0, pred].item()
            label_name = class_names[labels[0].item()]
            pred_name = class_names[pred.item()]
            
            correct_str = "✓" if pred == labels else "✗"
            print(f"  {correct_str} {filenames[0]:30} -> {pred_name:40} ({pred_prob:.1%})")

if __name__ == '__main__':
    main()
