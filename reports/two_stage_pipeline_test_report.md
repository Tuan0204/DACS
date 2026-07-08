# Two-Stage Pipeline Test Evaluation

- Final accuracy: 0.9767
- Final macro F1: 0.9689
- Stage 1 accuracy: 0.9948
- Stage 1 macro F1: 0.9939

## Final per-class metrics

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 0.9583 | 0.9020 | 0.9293 | 51 |
| Corn_(maize)___Common_rust_ | 1.0000 | 1.0000 | 1.0000 | 120 |
| Corn_(maize)___Northern_Leaf_Blight | 0.9500 | 0.9596 | 0.9548 | 99 |
| Corn_(maize)___healthy | 0.9831 | 1.0000 | 0.9915 | 116 |

## Stage 1 confusion matrix

| Actual \ Predicted | healthy | diseased |
|--------------------|---------|----------|
| healthy | 116 | 0 |
| diseased | 2 | 268 |

## Final confusion matrix

| Actual \ Predicted | Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | Corn_(maize)___Common_rust_ | Corn_(maize)___Northern_Leaf_Blight | Corn_(maize)___healthy |
|--------------------|---------|---------|---------|---------|
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 46 | 0 | 5 | 0 |
| Corn_(maize)___Common_rust_ | 0 | 120 | 0 | 0 |
| Corn_(maize)___Northern_Leaf_Blight | 2 | 0 | 95 | 2 |
| Corn_(maize)___healthy | 0 | 0 | 0 | 116 |

## Interpretation

- Stage 1 is the gate that reduces false healthy predictions.
- Stage 2 only runs when the binary model thinks the leaf is diseased.
- If the final macro F1 is at least as good as the multiclass-only model, this two-stage setup is the safer default.