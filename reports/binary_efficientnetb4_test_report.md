# Binary EfficientNetB4 Test Evaluation

- Accuracy: 0.9948
- Macro F1: 0.9939
- Latency: 50.6897 ms/img
- Model size: 67.43 MB

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Corn_(maize)___healthy | 0.9831 | 1.0000 | 0.9915 | 116 |
| diseased | 1.0000 | 0.9926 | 0.9963 | 270 |

## Confusion Matrix

| Actual \ Predicted | healthy | diseased |
|--------------------|---------|----------|
| healthy | 116 | 0 |
| diseased | 2 | 268 |

## Interpretation

- This binary model is intended to reduce false healthy predictions before the multiclass step.
- If healthy recall is high and diseased recall is also high, it is suitable as stage 1 of the pipeline.