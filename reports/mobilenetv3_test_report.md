# MobileNetV3-Small Test Evaluation

- Accuracy: 0.9819
- Macro F1: 0.9751
- Latency: 4.8998 ms/img
- Model size: 5.85 MB

## Per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 0.9592 | 0.9216 | 0.9400 | 51 |
| Corn_(maize)___Common_rust_ | 1.0000 | 1.0000 | 1.0000 | 120 |
| Corn_(maize)___Northern_Leaf_Blight | 0.9600 | 0.9697 | 0.9648 | 99 |
| Corn_(maize)___healthy | 0.9915 | 1.0000 | 0.9957 | 116 |

## Confusion Matrix

| Actual \ Predicted | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | Corn_(maize)___Common_rust_ | Corn_(maize)___Northern_Leaf_Blight | Corn_(maize)___healthy |
|--------------------|---------|---------|---------|---------|
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 47 | 0 | 4 | 0 |
| Corn_(maize)___Common_rust_ | 0 | 120 | 0 | 0 |
| Corn_(maize)___Northern_Leaf_Blight | 2 | 0 | 96 | 1 |
| Corn_(maize)___healthy | 0 | 0 | 0 | 116 |

## Interpretation

- MobileNetV3-Small is the final model because it is the strongest in the comparison table.
- The class-level table and confusion matrix expose which classes remain harder to separate.