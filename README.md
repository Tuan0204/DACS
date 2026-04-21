# Leaf Disease Detection - Do an co so

Project khoi tao cho de tai phat hien benh la cay (pham vi tomato/PlantVillage subset), uu tien mo hinh gon + danh gia kha nang trien khai.

## 1) Cau truc thu muc

- `configs/default.yaml`: Cau hinh train/eval
- `src/leaf_disease/`: Thu vien xu ly chinh
- `train.py`: Huan luyen mo hinh
- `evaluate.py`: Danh gia va xuat metrics
- `infer.py`: Du doan 1 anh
- `demo_streamlit.py`: Demo web don gian

## 2) Chuan bi du lieu

Dat dataset dang `ImageFolder` vao:

- `data/raw/tomato/`
  - `ClassA/*.jpg`
  - `ClassB/*.jpg`
  - ...

Vi du voi PlantVillage subset tomato, moi thu muc lop la 1 loai benh.

## 3) Cai dat

```powershell
cd leaf-disease-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4) Huan luyen

```powershell
python train.py --config configs/default.yaml
```

Checkpoint se duoc luu vao `models/best_model.pt`.

## 5) Danh gia

```powershell
python evaluate.py --config configs/default.yaml --checkpoint models/best_model.pt
```

Bao cao metrics duoc luu vao `reports/metrics_test.json`.

## 6) Du doan 1 anh

```powershell
python infer.py --config configs/default.yaml --checkpoint models/best_model.pt --image path/to/leaf.jpg
```

## 7) Chay demo

```powershell
streamlit run demo_streamlit.py
```

## 8) KPI khuyen nghi cho do an co so

- Accuracy, Macro F1
- Kich thuoc model (MB)
- Thoi gian suy luan trung binh/anh (ms)

## 9) Huong phat trien tiep

- Ensemble nhe (2-3 backbone)
- Grad-CAM/LIME cho explainability
- Quantization cho trien khai mobile/edge
