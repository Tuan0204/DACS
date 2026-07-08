# ResNet18 Training Report

## Model Configuration
- **Architecture**: ResNet18 (18 residual blocks)
- **Pretrained**: ImageNet weights
- **Optimizer**: AdamW (lr=0.0003, weight_decay=0.0001)
- **Loss**: CrossEntropyLoss
- **Epochs**: 10
- **Batch Size**: 32

## Training Metrics
- **Final Train Loss**: 0.0321
- **Final Train Accuracy**: 0.9886
- **Best Val Accuracy**: 0.9870 (epoch 6)
- **Best Val F1**: 0.9827 (epoch 6)

## Test Set Performance
- **Test Accuracy**: 0.9585 (96/1007 errors)
- **Test Macro F1**: 0.9385
- **Model Size**: 42.68 MB
- **Inference Latency**: 1.26 ms/image

## Key Observations
- Steady improvement over epochs with minor overfitting
- Strong performance on validation set suggests good generalization
- Traditional CNN architecture baseline for comparison

## Checkpoint Location
- `models/resnet18/best_model.pt`
- Config: `configs/resnet18.yaml`
