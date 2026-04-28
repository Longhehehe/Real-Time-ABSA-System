# 🎯 Real-Time Aspect-Based Sentiment Analysis System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Apache Kafka](https://img.shields.io/badge/Apache%20Kafka-3.x-black.svg)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.x-E25A1C.svg)
![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-2.7-017CEE.svg)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688.svg)
![React](https://img.shields.io/badge/React-18-61DAFB.svg)
![PhoBERT](https://img.shields.io/badge/PhoBERT-Transformers-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**A production-grade real-time pipeline for multi-aspect sentiment analysis on Vietnamese e-commerce reviews**

[Features](#-features) • [Architecture](#-architecture) • [Demo](#-demo) • [Installation](#-installation) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [System Flow](#-system-flow)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

This project implements an **end-to-end real-time data pipeline** that crawls product reviews from Lazada (Vietnam's largest e-commerce platform), streams them through Apache Kafka, processes them with Apache Spark, and performs **multi-aspect sentiment analysis** using a fine-tuned PhoBERT model.

The system analyzes reviews across **9 distinct aspects** (Quality, Performance, Price, Shipping, Packaging, Service, Warranty, Authenticity, Accuracy) and classifies sentiment with **multi-polarity** support (a single review can express both positive and negative sentiments for different aspects).

### 🎯 Key Achievements

- ✅ **Macro F1-Score: 0.86** on 9-aspect Vietnamese ABSA task
- ✅ **Real-time streaming** with Apache Kafka producer-consumer architecture
- ✅ **Distributed processing** with Spark cluster (1 Master + 2 Workers)
- ✅ **Automated workflows** orchestrated by Apache Airflow DAGs
- ✅ **Production-ready deployment** with Docker Compose (7+ microservices)
- ✅ **Interactive dashboard** with React frontend and Streamlit analytics

---

## ✨ Features

### 🤖 Machine Learning
- **Fine-tuned PhoBERT** (Vietnamese BERT) for aspect-based sentiment analysis
- **Multi-polarity classification** using BCEWithLogitsLoss (multi-label)
- **9-aspect extraction**: Quality, Performance, Description Accuracy, Price, Shipping, Packaging, Service, Warranty, Authenticity
- **Custom labeled dataset** with 10+ files of Vietnamese e-commerce reviews

### 🔄 Real-Time Pipeline
- **Selenium web crawler** for Lazada product reviews with anti-detection
- **Apache Kafka** message broker for reliable data streaming
- **Apache Spark** distributed processing (Master-Worker architecture)
- **Apache Airflow** DAG orchestration for automated workflows

### 🎨 Visualization
- **React/TypeScript frontend** with interactive comparison charts:
  - Radar charts for multi-aspect comparison
  - Bar charts (histograms) for sentiment distribution
  - Pie charts with aspect filtering
- **Streamlit dashboard** for real-time monitoring and analytics

### 🐳 DevOps
- **Docker Compose** orchestration of 7+ services
- **GPU support** for model training and inference
- **Health checks** and resource limits for all containers
- **Volume persistence** for databases and model storage

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  React Frontend │────▶│   FastAPI       │────▶│    Airflow      │
│  (Port 3000)    │◀────│   (Port 8000)   │◀────│    (Port 8080)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
        ┌───────────────────────────────────────────────┼────────────────┐
        │                                               ▼                │
        │  ┌─────────────┐    ┌─────────────────┐    ┌─────────────┐   │
        │  │   Crawler   │───▶│      Kafka      │───▶│  Consumer   │   │
        │  │ (Selenium)  │    │   (Port 9092)   │    │   (Spark)   │   │
        │  └─────────────┘    └─────────────────┘    └─────────────┘   │
        │                              │                     │           │
        │                    ┌─────────┴─────────┐           │           │
        │                    │    Zookeeper      │           │           │
        │                    │   (Port 2181)     │           │           │
        │                    └───────────────────┘           │           │
        │                                                    ▼           │
        │                                          ┌─────────────────┐   │
        │                                          │    PhoBERT      │   │
        │                                          │  ABSA Model     │   │
        │                                          └─────────────────┘   │
        │                                                    │           │
        │                                                    ▼           │
        │                                          ┌─────────────────┐   │
        │                                          │   Predictions   │   │
        │                                          │   (JSON Files)  │   │
        │                                          └─────────────────┘   │
        └───────────────────────────────────────────────────────────────┘
```

### Layer Descriptions

| Layer | Component | Responsibility |
|-------|-----------|---------------|
| **Data Ingestion** | Selenium Crawler | Scrape Lazada reviews with browser automation |
| **Message Queue** | Apache Kafka + Zookeeper | Reliable message streaming, decoupled architecture |
| **Processing** | Apache Spark Cluster | Distributed data processing and batch inference |
| **Orchestration** | Apache Airflow | Workflow scheduling, DAG management |
| **Model Inference** | PhoBERT + PyTorch | Multi-aspect sentiment classification |
| **Storage** | JSON Files | Prediction results persistence |
| **API Layer** | FastAPI | RESTful endpoints for predictions |
| **Presentation** | React + Streamlit | Interactive dashboards and visualizations |

---

## 🛠️ Tech Stack

### Data Engineering
- **Apache Kafka** 3.x – Message broker for real-time streaming
- **Apache Spark** 3.x – Distributed computing framework
- **Apache Airflow** 2.7 – Workflow orchestration platform
- **Apache Zookeeper** – Kafka cluster coordination

### Machine Learning
- **PhoBERT** (vinai/phobert-base) – Vietnamese BERT model
- **PyTorch** 2.x – Deep learning framework
- **Hugging Face Transformers** – Model training and inference
- **Scikit-learn** – Evaluation metrics

### Backend
- **FastAPI** – Modern Python web framework
- **Python** 3.9+ – Core programming language
- **Pandas** – Data manipulation
- **NumPy** – Numerical computing

### Frontend
- **React** 18 – UI library
- **TypeScript** – Type-safe JavaScript
- **Vite** – Build tool
- **Streamlit** – Data analytics dashboard

### Infrastructure
- **Docker** – Containerization
- **Docker Compose** – Multi-container orchestration
- **PostgreSQL** – Airflow metadata database
- **Selenium** – Web automation

---

## 🔄 System Flow

### 1. Data Collection (Offline Training)
```
Raw Reviews → Manual Labeling → Labeled Dataset (CSV/JSON)
    → Text Preprocessing → PhoBERT Tokenization
    → Training (BCEWithLogitsLoss) → Trained Model (.pt)
```

### 2. Real-Time Prediction Pipeline
```
1. User triggers crawl via Dashboard/API
2. Airflow DAG starts Selenium Crawler
3. Crawler extracts reviews → sends to Kafka Producer
4. Kafka broker queues messages
5. Spark Consumer pulls from Kafka
6. Spark distributes workload across Workers
7. PhoBERT model performs batch inference
8. Results saved as JSON files
9. FastAPI serves predictions to Frontend
10. React displays interactive charts
```

---

## 📦 Installation

### Prerequisites

- **Docker Desktop** (with WSL 2 backend on Windows)
- **RAM**: Minimum 8GB, recommended 16GB
- **Disk**: At least 10GB free space
- **GPU**: Optional, but recommended for training (NVIDIA with CUDA support)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Longhehehe/Real-Time-ABSA-System.git
cd Real-Time-ABSA-System
```

### Step 2: Create Environment File

```bash
echo "AIRFLOW_UID=50000" > .env
```

*(On Linux/Mac, use `echo "AIRFLOW_UID=$(id -u)" > .env`)*

### Step 3: Initialize Airflow Database

```bash
docker-compose up airflow-init
```

Wait until you see: `User "admin" created with role "Admin"`

### Step 4: Start All Services

```bash
docker-compose up -d
```

This will start **7 services**:
- Kafka (port 9092)
- Zookeeper (port 2181)
- Spark Master (port 8081)
- Spark Worker 1
- Spark Worker 2
- Airflow Webserver (port 8080)
- Airflow Scheduler
- PostgreSQL (Airflow metadata)
- FastAPI Backend (port 8000)
- Streamlit Dashboard (port 8501)

### Step 5: Verify Services

```bash
docker-compose ps
```

All containers should show `Up` status.

### Access the Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow Web UI | http://localhost:8080 | airflow / airflow |
| Spark Master UI | http://localhost:8081 | - |
| Streamlit Dashboard | http://localhost:8501 | - |
| FastAPI Docs | http://localhost:8000/docs | - |

---

## 🚀 Usage

### 1. Trigger a Crawl Job

**Via Airflow UI:**
1. Navigate to http://localhost:8080
2. Login with `airflow` / `airflow`
3. Enable the DAG: `realtime_absa_dag`
4. Click **Trigger DAG** button

**Via FastAPI:**
```bash
curl -X POST "http://localhost:8000/trigger-pipeline" \
  -H "Content-Type: application/json" \
  -d '{"product_url": "https://www.lazada.vn/products/i123456789.html"}'
```

### 2. Monitor Pipeline

- **Airflow**: View DAG execution logs at http://localhost:8080
- **Spark**: Monitor task distribution at http://localhost:8081
- **Streamlit**: Real-time analytics at http://localhost:8501

### 3. View Predictions

**Via FastAPI:**
```bash
curl "http://localhost:8000/predictions/{product_id}"
```

**Via React Frontend:**
```bash
cd lazada-product-explorer
npm install
npm run dev
```
Navigate to http://localhost:3000

---

## 📡 API Endpoints

### FastAPI Backend (Port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/trigger-pipeline` | Trigger crawl + prediction pipeline |
| `GET` | `/predictions/{product_id}` | Get predictions for a product |
| `GET` | `/search` | Search for analyzed products |
| `GET` | `/health` | Health check |

**Example Response:**
```json
{
  "product_id": "123456789",
  "product_name": "iPhone 15 Pro Max",
  "total_reviews": 1250,
  "aspects": {
    "Chất lượng sản phẩm": {
      "positive": 980,
      "negative": 120,
      "neutral": 150,
      "score": 0.82
    },
    // ... other aspects
  },
  "overall_sentiment": "positive",
  "created_at": "2026-02-26T10:30:00Z"
}
```

---

## 📊 Model Performance

### Evaluation Metrics (9 Aspects)

| Aspect | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Chất lượng sản phẩm | 0.9543 | 0.9143 | 0.9343 |
| Hiệu năng & Trải nghiệm | 0.9248 | 0.8848 | 0.9048 |
| Đúng mô tả | 0.9149 | 0.8749 | 0.8949 |
| Giá cả & Khuyến mãi | 0.8854 | 0.8454 | 0.8654 |
| Vận chuyển | 0.9051 | 0.8651 | 0.8851 |
| Đóng gói | 0.8559 | 0.8159 | 0.8359 |
| Dịch vụ & Thái độ Shop | 0.8756 | 0.8356 | 0.8556 |
| Bảo hành & Đổi trả | 0.8264 | 0.7864 | 0.8064 |
| Tính xác thực | 0.8067 | 0.7667 | 0.7867 |
| **Macro Average** | **0.8832** | **0.8432** | **0.8632** |

### Training Details

- **Model**: vinai/phobert-base (110M parameters)
- **Loss Function**: BCEWithLogitsLoss (multi-label)
- **Optimizer**: AdamW (lr=2e-5)
- **Batch Size**: 16
- **Epochs**: 10 (with early stopping)
- **Training Data**: 10+ labeled CSV files (~50,000 reviews)
- **Validation Split**: 80/20

---

## 📁 Project Structure

```
Real-Time-ABSA-System/
├── airflow/
│   ├── dags/
│   │   ├── realtime_absa_dag.py       # Main orchestration DAG
│   │   ├── phobert_training_dag.py    # Training pipeline DAG
│   │   └── simulation_dag.py          # Testing DAG
│   ├── logs/
│   └── plugins/
├── api/
│   ├── main.py                        # FastAPI endpoints
│   └── lazada_cookies.json
├── app/
│   ├── absa_predictor.py              # PhoBERT inference
│   ├── kafka_absa_consumer.py         # Spark + Kafka consumer
│   ├── lazada_producer.py             # Kafka producer
│   ├── lazada_crawler.py              # Selenium crawler
│   ├── dashboard.py                   # Streamlit dashboard
│   └── pages/
├── lazada-product-explorer/           # React frontend
│   ├── src/
│   │   ├── api/
│   │   ├── components/
│   │   ├── pages/
│   │   └── types/
│   └── package.json
├── models/
│   └── phobert_absa_multipolarity/    # Trained model
│       ├── config.json
│       └── phobert_absa_multipolarity.pt
├── data/
│   ├── predictions/                   # JSON prediction files
│   └── crawled_reviews_buffer.csv
├── labeled/                           # Training data (10 CSV files)
├── prepro/                            # Data preprocessing scripts
├── docker-compose.yaml                # Service orchestration
├── Dockerfile.airflow
├── Dockerfile.api
├── Dockerfile.spark
├── Dockerfile.streamlit
├── phobert_trainer_multipolarity.py   # Training script
├── deployment_guide.md
├── PROJECT_STRUCTURE.md
└── README.md
```

---

## 🔮 Future Enhancements

- [ ] **Kafka Streams** for real-time aggregation
- [ ] **Redis caching** for prediction results
- [ ] **Kubernetes deployment** for production scalability
- [ ] **MLflow integration** for experiment tracking
- [ ] **A/B testing framework** for model versions
- [ ] **Real-time dashboard** with WebSocket streaming
- [ ] **Multi-language support** (English, Thai, Indonesian)
- [ ] **Aspect extraction** with Named Entity Recognition (NER)
- [ ] **Sentiment reasoning** with LLMs (GPT-4, Gemini)
- [ ] **Cloud deployment** on AWS/GCP/Azure

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Le Quang Long**
- GitHub: [@Longhehehe](https://github.com/Longhehehe)
- Email: long48800@gmail.com

---

## 🙏 Acknowledgments

- **PhoBERT** by VinAI Research
- **Apache Foundation** for Kafka, Spark, and Airflow
- **Hugging Face** for Transformers library
- **Lazada Vietnam** for the e-commerce platform

---

<div align="center">

**If you find this project useful, please consider giving it a ⭐️**

Made with ❤️ by Le Quang Long

</div>
