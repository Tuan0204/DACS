## Project Structure

Các script đã được gom theo nhóm trong thư mục `scripts/`:

- `scripts/train/` — các file huấn luyện
- `scripts/finetune/` — các file fine-tune
- `scripts/evaluate/` — các file đánh giá
- `scripts/infer/` — các file suy luận
- `scripts/test/` — các file test
- `scripts/visualize/` — các file vẽ biểu đồ / hiển thị kết quả
- `scripts/xai/` — các file giải thích mô hình

`demo_streamlit.py` được giữ ở thư mục gốc để chạy demo nhanh.

Ví dụ:

```bash
python scripts/train/train_efficientnetb4.py
python scripts/evaluate/evaluate_binary_efficientnetb4.py
python scripts/finetune/finetune_binary_efficientnetb4.py
```
