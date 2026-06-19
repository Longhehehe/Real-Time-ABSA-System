# Real-Time ABSA System for Lazada Reviews

Hệ thống phân tích cảm xúc theo khía cạnh (Aspect-Based Sentiment Analysis - ABSA)
cho đánh giá sản phẩm Lazada. Dự án gồm frontend React để tìm kiếm/so sánh sản
phẩm, FastAPI backend, pipeline Airflow/Kafka/Spark, crawler Lazada và model
PhoBERT multi-polarity cho tiếng Việt.

## Tính năng chính

- Tìm kiếm sản phẩm Lazada từ frontend `lazada-product-explorer`.
- Trigger pipeline crawl review và phân tích ABSA theo sản phẩm.
- Dự đoán sentiment cho 9 khía cạnh thương mại điện tử:
  - Chất lượng sản phẩm
  - Hiệu năng & Trải nghiệm
  - Đúng mô tả
  - Giá cả & Khuyến mãi
  - Vận chuyển
  - Đóng gói
  - Dịch vụ & Thái độ Shop
  - Bảo hành & Đổi trả
  - Tính xác thực
- Hiển thị biểu đồ so sánh, phân bố sentiment và danh sách review.
- Hỗ trợ training/evaluation model qua Airflow.
- Có inference service riêng trong `serving/` để chuẩn bị tách model serving khỏi API gateway.

## Kiến trúc tổng quan

```text
React frontend
  -> FastAPI backend
      -> Airflow realtime DAG
          -> Lazada crawler
          -> Kafka raw_reviews
          -> Spark/Kafka consumer
          -> PhoBERT ABSA inference
          -> data/predictions
      -> React polling/status/result APIs
```

Định hướng tối ưu dài hạn:

```text
React
  -> FastAPI gateway
      -> Redis cache
      -> PostgreSQL jobs/reviews/predictions
      -> Kafka event queue

Crawler workers
  -> Kafka + PostgreSQL

Dedicated inference service
  -> dynamic batching
  -> distilled/ONNX/quantized PhoBERT model
```

## Cấu trúc thư mục

- `api/`: FastAPI backend cho frontend và prediction APIs.
- `app/`: Core Python modules: predictor, crawler, Kafka consumer, Streamlit, Airflow client.
- `serving/`: Inference service tách riêng, có `/predict`, `/predict-batch`, `/health`.
- `airflow/dags/`: DAG training, simulation và realtime ABSA.
- `lazada-product-explorer/`: React frontend chính.
- `lazada_crawler/`: Crawler app độc lập bằng Streamlit.
- `kafka/`: Producer/consumer demo cũ cho Kafka simulation.
- `model/`: Code training và định nghĩa model PhoBERT/NB dùng bởi Airflow, API evaluation và Spark.
- `labeled/`: Dữ liệu gán nhãn đang dùng để train PhoBERT ABSA.
- `models/`: Model runtime được API/consumer/serving load.
- `prepro/`: Code preprocessing/training cũ còn được `model/train_pipeline.py` import.
- `data/`: Runtime data, prediction output, trigger fallback, URL input và crawl output.
- `scripts/`: Script benchmark và tiện ích kỹ thuật.
- `docker/`: Dockerfile cho API, Airflow, Spark, Streamlit và inference service.
- `archive/`: Artifact cũ được giữ lại nhưng không nằm trên đường chạy chính.

## `prepro/` có cần thiết không?

Có, nhưng chỉ một phần.

`model/train_pipeline.py` đang import trực tiếp:

```python
prepro/23520932_23520903_20520692_src/23520932_23520903_20520692_src/Src
```

Vì vậy không nên xóa toàn bộ `prepro/` nếu còn muốn chạy `model/train_pipeline.py` hoặc các DAG training cũ. Các file notebook `.ipynb` đã bị xóa theo yêu cầu vì không nằm trên pipeline runtime.

## Training code

Các file training/model-definition đã được gom vào `model/`:

- `model/phobert_trainer_multipolarity.py`: training/evaluation/inference helper cho PhoBERT ABSA multi-polarity, được DAG `phobert_absa_training` gọi.
- `model/train_pipeline.py`: training pipeline cũ cho DAG `sentiment_model_training`.
- `model/phobert_trainer.py`: trainer/definition PhoBERT cũ, vẫn được Spark inference import lớp `PhoBERTForABSA`.

## Chạy bằng Docker Compose

Tạo file `.env` ở root nếu chưa có:

```bash
echo "AIRFLOW_UID=$(id -u)" > .env
```

Khởi tạo Airflow:

```bash
docker-compose up airflow-init
```

Chạy toàn bộ hệ thống:

```bash
docker-compose up -d
```

Các service chính:

- FastAPI: `http://localhost:8000`
- Airflow: `http://localhost:8080`
- Spark UI: `http://localhost:8081`
- Streamlit: `http://localhost:8501`
- React frontend: chạy riêng trong `lazada-product-explorer`

## Chạy frontend Lazada

```bash
cd lazada-product-explorer
npm install
npm run dev
```

Build frontend:

```bash
cd lazada-product-explorer
npm run build
```

## Benchmark inference

```bash
python3 scripts/benchmark_inference.py --batch-size 32 --iterations 20
```

Các biến môi trường hữu ích:

- `ABSA_BATCH_SIZE`: batch size cho inference.
- `ABSA_MAX_LENGTH`: max token length cho PhoBERT.
- `ABSA_CONSUMER_BATCH_SIZE`: batch size của Kafka consumer.
- `ABSA_CONSUMER_BATCH_TIMEOUT`: timeout gom batch consumer.
- `SPARK_TARGET_ROWS_PER_PARTITION`: số dòng mục tiêu mỗi Spark partition.

## Ghi chú vận hành

- `data/predictions/` và `data/triggers/` là output runtime, có thể xóa khi cần chạy lại.
- `models/` chứa artifact lớn, cần giữ nếu muốn API/consumer/serving load model offline.
- `node_modules/`, `dist/`, `build/`, `__pycache__/` là generated/dependency artifacts và không nên commit.
- `Đánh nhãn/`, `docs/` và toàn bộ `.ipynb` đã bị xóa để giảm nhiễu source tree.
