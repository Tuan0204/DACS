# MobileNetV3-Small Training Report

## Model Configuration
- **Architecture**: MobileNetV3-Small (efficient modern architecture)
- **Key Features**: Depthwise separable convolutions, squeeze-and-excitation blocks
- **Pretrained**: ImageNet weights
- **Optimizer**: AdamW (lr=0.0003, weight_decay=0.0001)
- **Loss**: CrossEntropyLoss
- **Epochs**: 10
- **Batch Size**: 32

## Training Metrics
- **Final Train Loss**: 0.0112 (lower than ResNet)
- **Final Train Accuracy**: 0.9958
- **Best Val Accuracy**: 0.9922 (epoch 9)
- **Best Val F1**: 0.9904 (epoch 9) — 0.77% higher than ResNet18

## Test Set Performance
- **Test Accuracy**: 0.9819 (17/1007 errors) — 79% fewer errors than ResNet18
- **Test Macro F1**: 0.9751
- **Model Size**: 5.85 MB — **7x smaller than ResNet18**
- **Inference Latency**: 1.09 ms/image — **faster than ResNet18**

## Key Observations
- Exceptional training efficiency (lower loss, higher accuracy)
- **Outperforms ResNet18 on all metrics**
- Modern architecture advantages evident
- Excellent candidate for production deployment

## Checkpoint Location
- `models/mobilenetv3/best_model.pt`
- Config: `configs/mobilenetv3.yaml`
