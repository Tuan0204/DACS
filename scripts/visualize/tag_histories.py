# scripts/visualize/tag_histories.py
import json
from pathlib import Path

# mapping: filename_stem -> model label (chỉnh theo file của bạn)
mapping = {
  "train_history_20260510_215317": "ResNet18",
  "train_history_20260510_215753": "EfficientNetB4",
  "train_history_20260513_105049": "MobileNetV3",
  "train_history_20260513_140333": "MobileNetV3",
  "train_history_20260513_151121": "ANFIS-Fuzzy-CNN",
}

for p in sorted(Path("reports").glob("train_history*.json")):
    stem = p.stem
    if stem in mapping:
        data = json.load(p.open(encoding="utf-8"))
        data["model_name"] = mapping[stem]
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Tagged", p.name, "as", mapping[stem])
    else:
        print("Skipped (no mapping):", p.name)