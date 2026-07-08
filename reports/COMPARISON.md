# Model Performance Comparison - Organized Training

## Test Set Results

| Model | Test Accuracy | Test Macro F1 | Model Size (MB) | Latency (ms/img) |
|-------|---------------|---------------|-----------------|-----------------|
| **ResNet18** | 0.9585 | 0.9385 | 42.68 | 1.26 |
| **MobileNetV3-Small** | **0.9819** ✓ | **0.9751** ✓ | **5.85** ✓ | **1.09** ✓ |

## Summary

### ResNet18 (Baseline - Traditional CNN)
- Standard residual blocks, deeper architecture
- Good accuracy but computationally expensive
- **Suitable for**: Research baseline, comparison reference

### MobileNetV3-Small (Modern Architecture) 🏆
- Depthwise separable convolutions, efficiency-optimized
- **Higher accuracy** (0.9819 vs 0.9585)
- **7x smaller** (5.85 MB vs 42.68 MB)
- **Faster inference** (1.09 ms vs 1.26 ms)
- **Suitable for**: Production, mobile/edge deployment

## Key Insights

1. **MobileNetV3 wins on all metrics** — better accuracy with smaller footprint
2. **Modern efficient architectures** outperform traditional CNNs on this task
3. **Training differences**: 
   - ResNet18 peak val F1: 0.9702 (epoch 10)
   - MobileNetV3 peak val F1: 0.9904 (epoch 9) — 2% higher
4. **Generalization**: Both models generalize well from validation to test set

## Recommendation

**Use MobileNetV3-Small for production** — best balance of accuracy, speed, and size for real-world deployment on resource-constrained devices.

## Detailed Evaluation

The detailed test report for the final model is available in [reports/mobilenetv3_test_report.md](reports/mobilenetv3_test_report.md). The corresponding JSON metrics are stored in [reports/metrics_test.json](reports/metrics_test.json). Note that the detailed evaluation was generated on CPU, so its latency is not directly comparable to the earlier benchmark table.

### Per-class results

- Corn_(maize)___Common_rust_: precision 1.0000, recall 1.0000, F1 1.0000
- Corn_(maize)___healthy: precision 0.9915, recall 1.0000, F1 0.9957
- Corn_(maize)___Northern_Leaf_Blight: precision 0.9600, recall 0.9697, F1 0.9648
- Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot: precision 0.9592, recall 0.9216, F1 0.9400

### Confusion matrix insights

- The model classifies Common rust and healthy leaves almost perfectly.
- The main confusion is between Cercospora leaf spot and Northern Leaf Blight, where 4 Cercospora samples were predicted as Northern Leaf Blight.
- Northern Leaf Blight also has a small number of samples confused with Cercospora leaf spot and healthy.
