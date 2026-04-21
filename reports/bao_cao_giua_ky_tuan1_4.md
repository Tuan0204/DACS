# BAO CAO GIUA KY (TUAN 1 - TUAN 4)

## 1. Thong tin de tai
- Ten de tai: Phat hien benh la cay bang hoc may (Leaf Disease Detection)
- Huong thuc hien: Phan loai anh la ca chua theo tung loai benh
- Bo du lieu su dung: PlantVillage (tomato, anh mau)
- So lop hien tai: 10 lop
- Tong so anh da dua vao he thong: 18,160 anh

## 2. Ly do chon de tai
- Nhu cau phat hien benh som trong nong nghiep la rat lon.
- Thi giac may tinh co the ho tro nong dan phan loai nhanh benh la.
- De tai co tinh ung dung cao va phu hop voi huong hoc may ung dung.

## 3. Muc tieu giai doan giua ky (tuan 1 - 4)
- Xay dung duoc pipeline xu ly du lieu va huan luyen co ban.
- Khao sat tong quan cac huong nghien cuu lien quan (CNN, Ensemble, XAI).
- Chuan bi bo du lieu theo dung dinh dang train/evaluate.
- Khoi tao codebase co kha nang mo rong cho giai doan cuoi ky.

## 4. Noi dung da thuc hien theo tien do

### 4.1. Tuan 1 - Xac dinh bai toan va tong quan tai lieu
- Lam ro bai toan: phan loai benh la ca chua tu anh.
- Nghien cuu 2 huong lien quan tu tai lieu:
  - Huong 1: Hybrid feature + DL (LBP/ANFIS-CNN).
  - Huong 2: Ensemble Learning + Explainable AI.
- Rut ra huong thuc hien phu hop 8-9 tuan:
  - Uu tien mo hinh gon de de train va de trien khai.
  - Bo sung danh gia latency va kich thuoc model.

### 4.2. Tuan 2 - Chuan bi moi truong va cau truc du an
- Tao cau truc project theo huong tai hien duoc:
  - configs, src, reports, data/raw, data/processed.
- Tao file cau hinh huan luyen ban dau.
- Tao bo script train/evaluate/infer/demo.
- Cai dat va kiem tra moi truong Python trong workspace.

### 4.3. Tuan 3 - Chuan hoa du lieu va ket noi pipeline
- Lua chon bo anh mau (color) thay vi grayscale/segmented.
- Dua du lieu vao thu muc raw theo dinh dang ImageFolder.
- Kiem tra va xac nhan bo du lieu hop le:
  - 10 lop
  - 18,160 anh
- Dong bo cau hinh data_dir voi du lieu thuc te.

### 4.4. Tuan 4 - Hoan thien phien ban baseline
- Hoan thien module data split (train/val/test) va augmentation co ban.
- Hoan thien model factory cho cac backbone:
  - resnet18
  - resnet50
  - mobilenet_v3_small
- Hoan thien vong train/evaluate va luu checkpoint.
- Hoan thien infer anh don va demo Streamlit ban dau.
- Them huong mo rong:
  - ensemble_predict (soft-voting)
  - xai_gradcam (giai thich vung chu y)
- Kiem tra compile/syntax toan bo cac script chinh: dat.

## 5. San pham giua ky da dat duoc
- Co codebase hoat dong duoc cho bai toan phan loai benh la.
- Co bo du lieu da dat dung cau truc train.
- Co script train/evaluate/infer co the chay lai.
- Co demo co ban de trinh bay ket qua du doan.
- Co dinh huong phat trien ro rang cho giai doan tiep theo.

## 6. Danh gia giua ky

### 6.1. Diem dat duoc
- Tien do dung ke hoach 4 tuan dau.
- Da hoan thanh phan nen tang ky thuat (du lieu + code).
- Da xac dinh ro KPI cho giai doan thuc nghiem:
  - Accuracy
  - Macro F1
  - Latency/anh
  - Kich thuoc model

### 6.2. Han che hien tai
- Chua train day du de co bang ket qua cuoi cung giua cac backbone.
- Chua co bang so sanh dinh luong chi tiet giua model don va ensemble.
- Chua danh gia tren du lieu thuc dia (ngoai PlantVillage).

## 7. Ke hoach tuan 5 den tuan 9 (tom tat)
- Tuan 5: Train baseline day du, chot 1-2 model tot nhat.
- Tuan 6: Danh gia test va lap bang so sanh metric.
- Tuan 7: Tich hop ensemble nhe, do latency, model size.
- Tuan 8: Trien khai Grad-CAM/LIME va phan tich loi.
- Tuan 9: Hoan thien bao cao cuoi ky + slide + demo.

## 8. De xuat huong nang cap len do an chuyen nganh
- Bo sung du lieu thuc te tai Viet Nam de kiem thu domain shift.
- Toi uu mo hinh cho mobile/edge (quantization, distillation).
- Mo rong he thong thanh ung dung canh bao thuc te.

## 9. Ket luan giua ky
Trong 4 tuan dau, de tai da hoan thanh phan nen quan trong: xac dinh bai toan, chuan bi du lieu, khoi tao pipeline huan luyen va xay dung bo cong cu danh gia/co so trinh dien. Giai doan tiep theo se tap trung vao thuc nghiem, toi uu hieu nang va hoan thien tinh ung dung cua he thong.

---

## Phu luc A - Danh sach file code chinh
- train.py
- evaluate.py
- infer.py
- demo_streamlit.py
- ensemble_predict.py
- xai_gradcam.py
- src/leaf_disease/data.py
- src/leaf_disease/modeling.py
- src/leaf_disease/engine.py
- configs/default.yaml

## Phu luc B - Minh chung can chen khi nop
- Anh chup cau truc thu muc du an
- Anh chup cau truc dataset trong data/raw
- Log train/evaluate (neu da chay)
- Anh giao dien demo va ket qua du doan
