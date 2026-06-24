# 🔍 Anomaly Detection API

Hệ thống phát hiện bất thường trong ảnh sử dụng **Anomalib**, được đóng gói dưới dạng REST API với **FastAPI**. Hỗ trợ hai model: **PatchCore** và **CFA**.

---

## 📋 Mục lục

- [Tổng quan](#tổng-quan)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cấu trúc Dataset](#cấu-trúc-dataset)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [API Reference](#api-reference)
- [Models](#models)

---

## 📌 Tổng quan

Project này cung cấp một API để:
1. **Train** model phát hiện bất thường trên tập dữ liệu của bạn
2. **Inference** — dự đoán ảnh mới và lưu ảnh overlay kết quả

Kết quả inference bao gồm ảnh gốc được overlay với **anomaly map** để trực quan hóa vùng bất thường.

---

## ⚙️ Yêu cầu hệ thống

- Python >= 3.9
- CUDA (khuyến nghị, không bắt buộc)
- RAM >= 8GB

---

## 🛠 Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Tạo môi trường ảo

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Chạy server

```bash
python app.py
```

Server sẽ chạy tại: `http://127.0.0.1:8002`

Truy cập Swagger UI tại: `http://127.0.0.1:8002/docs`

---

## 📁 Cấu trúc thư mục

```
project/
│
├── app.py                  # FastAPI entrypoint
├── requirements.txt        # Dependencies
├── README.md
│
├── src/                    # Source code
│   ├── __init__.py
│   ├── dataset.py          # Custom dataset class
│   ├── datamodule.py       # LightningDataModule
│   ├── train.py            # Training logic
│   └── test.py             # Inference logic
│
├── data/                   # Dataset (xem cấu trúc bên dưới)
│   ├── train/
│   ├── val/
│   └── test/
│
├── model/                  # Checkpoint được lưu tại đây
│   └── best.ckpt           # (tự động tạo sau khi train)
│
├── overlay_results/        # Ảnh kết quả inference
└── results/                # Log kết quả test metrics
```

---

## 🗂 Cấu trúc Dataset

Dataset **bắt buộc** phải có cấu trúc như sau:

```
data/
├── train/
│   └── normal/             ← Chỉ chứa ảnh BÌNH THƯỜNG để train
│       ├── img001.png
│       ├── img002.png
│       └── ...
│
├── val/
│   ├── normal/             ← Ảnh bình thường để validate
│   │   ├── img001.png
│   │   └── ...
│   └── anomaly/            ← Ảnh bất thường để validate
│       ├── img001.png
│       └── ...
│
└── test/
    ├── normal/             ← Ảnh bình thường để test
    │   ├── img001.png
    │   └── ...
    └── anomaly/            ← Ảnh bất thường để test
        ├── img001.png
        └── ...
```

> **Lưu ý quan trọng:**
> - Thư mục `train/` **chỉ** chứa ảnh bình thường (unsupervised anomaly detection).
> - Thư mục `val/` và `test/` chứa **cả hai loại**: `normal` và `anomaly`.
> - Định dạng ảnh hỗ trợ: `.png`, `.jpg`, `.jpeg`, `.bmp`

---

## 🚀 Hướng dẫn sử dụng

### Bước 1 — Kiểm tra trạng thái

```bash
GET http://127.0.0.1:8002/status
```

Trả về trạng thái server, thiết bị (CPU/GPU), và tình trạng model.

---

### Bước 2 — Set đường dẫn dataset

```bash
POST http://127.0.0.1:8002/set_dataset_path?root=./data
```

| Parameter | Type   | Mô tả                              |
|-----------|--------|------------------------------------|
| `root`    | string | Đường dẫn đến thư mục chứa dataset |

---

### Bước 3 — Train model

```bash
POST http://127.0.0.1:8002/train
```

| Parameter          | Type    | Default      | Mô tả                              |
|--------------------|---------|--------------|-------------------------------------|
| `category`         | string  | `my_data`    | Tên category (không ảnh hưởng data load) |
| `type_model`       | string  | `Patchcore`  | Loại model: `Patchcore` hoặc `Cfa` |
| `backbone`         | string  | `resnet18`   | Backbone (chỉ dùng cho CFA)         |
| `lr`               | float   | `0.01`       | Learning rate (chỉ dùng cho CFA)    |
| `train_batch_size` | int     | `2`          | Batch size khi train                |
| `eval_batch_size`  | int     | `2`          | Batch size khi eval/test            |
| `num_workers`      | int     | `2`          | Số worker cho DataLoader            |
| `w`                | int     | `256`        | Chiều rộng ảnh resize               |
| `h`                | int     | `256`        | Chiều cao ảnh resize                |

Sau khi train xong, checkpoint được lưu tại `./model/best.ckpt`.

---

### Bước 4 — Inference

```bash
POST http://127.0.0.1:8002/inference?testPath=./data/test
```

| Parameter  | Type   | Mô tả                                     |
|------------|--------|-------------------------------------------|
| `testPath` | string | Đường dẫn đến thư mục ảnh cần dự đoán    |

Kết quả ảnh overlay được lưu vào thư mục `./overlay_results/`.

---

## 📡 API Reference

| Method | Endpoint              | Mô tả                          |
|--------|-----------------------|--------------------------------|
| GET    | `/status`             | Kiểm tra trạng thái server     |
| POST   | `/set_dataset_path`   | Cấu hình đường dẫn dataset     |
| POST   | `/train`              | Huấn luyện model               |
| POST   | `/inference`          | Dự đoán trên ảnh mới           |

Swagger UI đầy đủ: `http://127.0.0.1:8002/docs`

---

## 🤖 Models

### PatchCore (Khuyến nghị)

- **Nguyên lý:** So sánh patch features của ảnh test với memory bank tích lũy từ ảnh train bình thường.
- **Ưu điểm:** Không cần fine-tune, train nhanh, kết quả tốt trên nhiều bộ dữ liệu.
- **Phù hợp:** Khi bạn có ít ảnh train và muốn kết quả nhanh.

### CFA (Coupled-hypersphere-based Feature Adaptation)

- **Nguyên lý:** Học feature embedding trong không gian hypersphere để phân biệt normal/anomaly.
- **Ưu điểm:** Linh hoạt hơn, có thể fine-tune learning rate và backbone.
- **Phù hợp:** Khi muốn kiểm soát quá trình huấn luyện chi tiết hơn.

---

## 📊 Metrics

Sau khi train, hệ thống tự động đánh giá trên tập test với các metrics:

- **F1 Score** (image-level)
- **Precision** (image-level)
- **Recall** (image-level)

---

## 📝 Ví dụ với cURL

```bash
# 1. Kiểm tra status
curl -X GET http://127.0.0.1:8002/status

# 2. Set dataset path
curl -X POST "http://127.0.0.1:8002/set_dataset_path?root=./data"

# 3. Train với PatchCore
curl -X POST "http://127.0.0.1:8002/train?type_model=Patchcore&w=256&h=256&train_batch_size=4"

# 4. Inference
curl -X POST "http://127.0.0.1:8002/inference?testPath=./data/test"
```

---

## 🔧 Lưu ý

- Model checkpoint cũ sẽ bị **ghi đè** mỗi lần train (lưu tại `model/best.ckpt`).
- Trong khi đang train (`training: true`), không thể thay đổi dataset path.
- Nếu không có GPU, model sẽ tự động chạy trên CPU (chậm hơn đáng kể với CFA).
# Anomaly-detection-using-the-Anomalib-library-for-industrial-images
