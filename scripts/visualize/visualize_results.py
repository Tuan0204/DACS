"""
Comprehensive visualization for model evaluation results
- Confusion Matrix (heatmap)
- Per-class Metrics (precision, recall, F1)
- Overall Statistics
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 6)
plt.rcParams['font.size'] = 11

# Load metrics
metrics_file = Path("reports/metrics_test.json")
if not metrics_file.exists():
    print(f"Error: {metrics_file} not found!")
    exit(1)

with open(metrics_file, 'r', encoding='utf-8') as f:
    metrics = json.load(f)

# Mapping from long class names to short names
class_name_mapping = {
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Cercospora Leaf Spot",
    "Corn_(maize)___Common_rust_": "Common Rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Northern Leaf Blight",
    "Corn_(maize)___healthy": "Healthy"
}

# Extract data
confusion_matrix = np.array(metrics['confusion_matrix'])
per_class_metrics = metrics['per_class']
overall_accuracy = metrics['accuracy']
overall_macro_f1 = metrics['macro_f1']

class_names = [class_name_mapping.get(item['class_name'], item['class_name']) for item in per_class_metrics]
precisions = [item['precision'] for item in per_class_metrics]
recalls = [item['recall'] for item in per_class_metrics]
f1_scores = [item['f1'] for item in per_class_metrics]

# Create figure with subplots
fig = plt.figure(figsize=(16, 6))

# ===== SUBPLOT 1: Confusion Matrix =====
ax1 = plt.subplot(1, 2, 1)
sns.heatmap(
    confusion_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar_kws={'label': 'Count'},
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax1,
    linewidths=0.5,
    linecolor='gray'
)
ax1.set_title('Confusion Matrix - Test Set Predictions', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')

# Rotate labels for readability
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=10)
plt.setp(ax1.get_yticklabels(), rotation=0, fontsize=10)

# ===== SUBPLOT 2: Per-class Metrics =====
ax2 = plt.subplot(1, 2, 2)

x = np.arange(len(class_names))
width = 0.25

bars1 = ax2.bar(x - width, precisions, width, label='Precision', color='#2ecc71', alpha=0.8)
bars2 = ax2.bar(x, recalls, width, label='Recall', color='#3498db', alpha=0.8)
bars3 = ax2.bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c', alpha=0.8)

ax2.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel('Disease Class', fontsize=12, fontweight='bold')
ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
ax2.set_ylim([0.85, 1.02])
ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.3)
ax2.legend(loc='lower right', fontsize=11)
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.003,
            f'{height:.4f}',
            ha='center',
            va='bottom',
            fontsize=9
        )

# Overall metrics text box
textstr = f'Overall Test Accuracy: {overall_accuracy:.4f}\nMacro F1-Score: {overall_macro_f1:.4f}'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
fig.text(0.5, 0.02, textstr, ha='center', fontsize=11, bbox=props, fontweight='bold')

plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save figure
output_path = Path("reports/evaluation_visualization.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Visualization saved to: {output_path}")

# Also create a detailed metrics table image
fig2, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = [['Disease Class', 'Precision', 'Recall', 'F1-Score', 'Support']]
for item in per_class_metrics:
    short_name = class_name_mapping.get(item['class_name'], item['class_name'])
    table_data.append([
        short_name,
        f"{item['precision']:.4f}",
        f"{item['recall']:.4f}",
        f"{item['f1']:.4f}",
        f"{item['support']}"
    ])

# Add totals row
total_support = sum(item['support'] for item in per_class_metrics)
avg_precision = np.mean(precisions)
avg_recall = np.mean(recalls)
avg_f1 = np.mean(f1_scores)
table_data.append(['Average', f'{avg_precision:.4f}', f'{avg_recall:.4f}', f'{avg_f1:.4f}', f'{total_support}'])

table = ax.table(
    cellText=table_data,
    cellLoc='center',
    loc='center',
    colWidths=[0.35, 0.15, 0.15, 0.15, 0.15]
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(5):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style average row
for i in range(5):
    table[(len(table_data)-1, i)].set_facecolor('#ecf0f1')
    table[(len(table_data)-1, i)].set_text_props(weight='bold')

# Alternate row colors
for i in range(1, len(table_data)-1):
    color = '#f8f9fa' if i % 2 == 0 else '#ffffff'
    for j in range(5):
        table[(i, j)].set_facecolor(color)

plt.title('Per-Class Metrics Summary Table', fontsize=14, fontweight='bold', pad=20)
table_path = Path("reports/metrics_table.png")
plt.savefig(table_path, dpi=150, bbox_inches='tight')
print(f"✓ Metrics table saved to: {table_path}")

print("\n" + "="*60)
print("EVALUATION RESULTS SUMMARY")
print("="*60)
print(f"Overall Test Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
print(f"Macro F1-Score: {overall_macro_f1:.4f}")
print("\nPer-Class Results:")
for item in per_class_metrics:
    print(f"  {item['class_name']}")
    print(f"    - Precision: {item['precision']:.4f}")
    print(f"    - Recall: {item['recall']:.4f}")
    print(f"    - F1-Score: {item['f1']:.4f}")
    print(f"    - Support: {item['support']}")
print("="*60)

plt.show()
