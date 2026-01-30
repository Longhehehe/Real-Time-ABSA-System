# Cấu trúc thư mục dự án Real-Time ABSA System

```
SE363 (1)/
│
├── 📁 airflow/                             # Apache Airflow
│   ├── 📁 dags/                            # DAG definitions
│   │   ├── phobert_training_dag.py         # DAG huấn luyện PhoBERT
│   │   ├── realtime_absa_dag.py            # DAG chính xử lý ABSA realtime
│   │   ├── simulation_dag.py               # DAG simulation test
│   │   └── training_dag.py                 # DAG huấn luyện khác
│   ├── 📁 logs/                            # Airflow logs
│   └── 📁 plugins/                         # Custom plugins
│
├── 📁 api/                                 # FastAPI Backend
│   ├── main.py                             # API endpoints (search, predictions, trigger)
│   └── lazada_cookies.json                 # Cookies cho API
│
├── 📁 app/                                 # Core Application Logic
│   ├── absa_predictor.py                   # PhoBERT ABSA model inference
│   ├── kafka_absa_consumer.py              # Kafka consumer với Spark processing
│   ├── lazada_producer.py                  # Kafka producer gửi reviews
│   ├── lazada_crawler.py                   # Full Selenium crawler
│   ├── lazada_crawler_simple.py            # Simple requests-based crawler
│   ├── lazada_browser.py                   # Browser automation
│   ├── lazada_search.py                    # Lazada product search API
│   ├── selenium_login.py                   # Selenium login automation
│   ├── dashboard.py                        # Streamlit dashboard
│   ├── utils.py                            # Utility functions
│   ├── spark_inference.py                  # Spark-based inference
│   ├── spark_processor.py                  # Spark data processor
│   ├── airflow_client.py                   # Airflow API client
│   ├── ollama_predictor.py                 # Ollama LLM predictor
│   ├── product_manager.py                  # Product management
│   ├── debug_model.py                      # Model debugging
│   └── 📁 pages/                           # Streamlit pages
│
├── 📁 lazada-product-explorer/             # React Frontend
│   ├── 📁 src/
│   │   ├── 📁 api/
│   │   │   └── absaApi.ts                  # API client functions
│   │   ├── 📁 components/
│   │   │   ├── RadarChartComparison.tsx    # Biểu đồ Radar so sánh
│   │   │   ├── AspectBarChart.tsx          # Biểu đồ cột (histogram)
│   │   │   ├── AspectPieCharts.tsx         # Biểu đồ tròn với chọn khía cạnh
│   │   │   ├── CommentsList.tsx            # Hiển thị bình luận
│   │   │   ├── ProductCard.tsx             # Card sản phẩm
│   │   │   ├── ProductComparisonDetail.tsx # Chi tiết so sánh
│   │   │   ├── AnalyzeProductDialog.tsx    # Dialog phân tích
│   │   │   ├── AspectLabels.tsx            # Labels khía cạnh
│   │   │   ├── AspectScoreChart.tsx        # Biểu đồ điểm khía cạnh
│   │   │   ├── SearchBar.tsx               # Thanh tìm kiếm
│   │   │   ├── SentimentSummary.tsx        # Tóm tắt cảm xúc
│   │   │   ├── NavLink.tsx                 # Navigation link
│   │   │   └── 📁 ui/                      # shadcn/ui components (49 files)
│   │   ├── 📁 pages/
│   │   │   ├── Index.tsx                   # Trang chính so sánh sản phẩm
│   │   │   ├── Comparison.tsx              # Trang chi tiết so sánh
│   │   │   └── NotFound.tsx                # Trang 404
│   │   ├── 📁 hooks/
│   │   │   ├── useAbsaPredictions.ts       # Hook lấy predictions
│   │   │   ├── useTriggerPipeline.ts       # Hook trigger pipeline
│   │   │   ├── use-toast.ts                # Toast notifications
│   │   │   └── use-mobile.tsx              # Mobile detection
│   │   ├── 📁 types/
│   │   │   └── product.ts                  # TypeScript types
│   │   ├── 📁 data/
│   │   │   └── mockData.ts                 # Mock data cho demo
│   │   ├── 📁 lib/
│   │   │   └── utils.ts                    # Utility functions
│   │   ├── App.tsx                         # React app entry
│   │   ├── App.css                         # Styles
│   │   ├── main.tsx                        # Vite entry
│   │   └── index.css                       # Global styles
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
│
├── 📁 lazada_crawler/                      # Streamlit Crawler App
│   ├── app.py                              # Streamlit app entry
│   ├── crawler.py                          # Crawler logic chính
│   ├── bulk_crawler.py                     # Bulk crawling nhiều sản phẩm
│   ├── keywords.py                         # Danh sách keywords
│   ├── utils.py                            # Utility functions
│   ├── requirements.txt                    # Dependencies
│   ├── run_lazada_crawler.bat              # Windows batch script
│   ├── HUONG_DAN.md                        # Hướng dẫn sử dụng
│   ├── lazada_cookies.json                 # Browser cookies
│   └── 📁 bulk_crawl_results/              # Kết quả crawl
│
├── 📁 models/                              # Trained Models
│   ├── 📁 phobert_absa_multipolarity/      # Multi-polarity model (đang dùng)
│   │   ├── config.json
│   │   └── phobert_absa_multipolarity.pt
│   ├── 📁 phobert_absa/                    # Legacy model
│   ├── 📁 phobert_absa_backup/             # Backup models
│   └── 📁 best_model/                      # Best checkpoint
│
├── 📁 data/                                # Data Files
│   ├── 📁 predictions/                     # Prediction JSON files
│   │   ├── {product_id}.json               # Predictions cho mỗi sản phẩm
│   │   ├── {product_id}.done               # Marker file hoàn tất
│   │   └── {product_id}_summary.json       # Summary statistics
│   ├── 📁 triggers/                        # Trigger status files
│   └── crawled_reviews_buffer.csv          # Buffer reviews đã crawl
│
├── 📁 prepro/                              # Preprocessing & Labeling
│   ├── labeling.py                         # Script gán nhãn
│   ├── labeling.ipynb                      # Notebook gán nhãn
│   ├── preprocess.ipynb                    # Notebook tiền xử lý
│   └── 23520903.py                         # Script xử lý dữ liệu
│
├── 📁 labeled/                             # Labeled training data (10 files)
│
├── 📁 cookie/                              # Browser cookies cho crawler
│   └── lazada_cookies.pkl
│
├── 📁 kafka/                               # Kafka config files
│
├── 📁 url/                                 # URL lists (10 files)
│
├── 📁 model/                               # Additional model files (6 files)
│
│
├── 📄 docker-compose.yaml                  # Docker services config
├── 📄 Dockerfile.airflow                   # Airflow container
├── 📄 Dockerfile.api                       # FastAPI container
├── 📄 Dockerfile.spark                     # Spark + Consumer container
├── 📄 Dockerfile.streamlit                 # Streamlit container
│
├── 📄 phobert_trainer.py                   # Script huấn luyện PhoBERT
├── 📄 phobert_trainer_multipolarity.py     # Script huấn luyện multi-polarity
├── 📄 train_pipeline.py                    # Training pipeline
├── 📄 merge_training_data.py               # Merge labeled data
│
├── 📄 airflow_requirements.txt             # Airflow dependencies
├── 📄 app_requirements.txt                 # App dependencies
├── 📄 spark_requirements.txt               # Spark dependencies
│
├── 📄 deployment_guide.md                  # Hướng dẫn deploy
├── 📄 PROJECT_STRUCTURE.md                 # File này
├── 📄 REPORT_PROJECT.docx                  # Báo cáo dự án
│
├── 📄 .gitignore                           # Git ignore rules
├── 📄 .env                                 # Environment variables
└── 📄 .dockerignore                        # Docker ignore rules
```

## Luồng dữ liệu (Data Flow)

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  React Frontend │────▶│   FastAPI       │────▶│    Airflow      │
│  (Port 3000)    │◀────│   (Port 8000)   │◀────│    (Port 8080)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                        ┌───────────────────────────────┼───────────────────────────────┐
                        │                               ▼                               │
                        │  ┌─────────────┐    ┌─────────────────┐    ┌─────────────┐   │
                        │  │   Crawler   │───▶│      Kafka      │───▶│  Consumer   │   │
                        │  │ (Lazada API)│    │   (Port 9092)   │    │   (Spark)   │   │
                        │  └─────────────┘    └─────────────────┘    └─────────────┘   │
                        │                                                    │         │
                        │                               ┌────────────────────┘         │
                        │                               ▼                              │
                        │                     ┌─────────────────┐                      │
                        │                     │    PhoBERT      │                      │
                        │                     │  ABSA Model     │                      │
                        │                     └─────────────────┘                      │
                        │                               │                              │
                        │                               ▼                              │
                        │                     ┌─────────────────┐                      │
                        │                     │   Predictions   │                      │
                        │                     │   (JSON Files)  │                      │
                        │                     └─────────────────┘                      │
                        └──────────────────────────────────────────────────────────────┘
```

## Các cổng dịch vụ (Service Ports)

| Service          | Port  | URL                        |
|------------------|-------|----------------------------|
| React Frontend   | 3000  | http://localhost:3000      |
| FastAPI Backend  | 8000  | http://localhost:8000      |
| Airflow Web UI   | 8080  | http://localhost:8080      |
| Spark Master UI  | 8081  | http://localhost:8081      |
| Kafka            | 9092  | localhost:9092             |
| Streamlit        | 8501  | http://localhost:8501      |

## 9 Khía cạnh phân tích (Aspects)

| #  | Khía cạnh                    | Mô tả                              |
|----|------------------------------|------------------------------------|
| 1  | Chất lượng sản phẩm          | Quality, durability, materials     |
| 2  | Hiệu năng & Trải nghiệm      | Performance, user experience       |
| 3  | Đúng mô tả                   | Accuracy of description            |
| 4  | Giá cả & Khuyến mãi          | Price, discounts, value            |
| 5  | Vận chuyển                   | Shipping speed, delivery           |
| 6  | Đóng gói                     | Packaging quality                  |
| 7  | Dịch vụ & Thái độ Shop       | Customer service, seller attitude  |
| 8  | Bảo hành & Đổi trả           | Warranty, returns                  |
| 9  | Tính xác thực                | Authenticity (fake/genuine)        |
