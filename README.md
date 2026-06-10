# Real-Time ABSA System

Real-Time ABSA System is an end-to-end Aspect-Based Sentiment Analysis platform for Vietnamese e-commerce reviews. It can crawl Lazada product reviews, stream them through Kafka, process them with Spark, run ABSA inference with trained models, and display live product comparison analytics in a Streamlit web application.

ABSA means the system does not only classify a review as positive or negative. It extracts sentiment for specific product aspects, such as product quality, shipping, packaging, price, shop service, warranty, authenticity, and description correctness.

## Table of Contents

1. [System Overview](#system-overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Required Runtime Folders](#required-runtime-folders)
5. [File Access and Permission Setup](#file-access-and-permission-setup)
6. [Run the Full System with Docker](#run-the-full-system-with-docker)
7. [Web URLs and Service Ports](#web-urls-and-service-ports)
8. [How to Use the Web Application](#how-to-use-the-web-application)
9. [Run Individual Components Locally](#run-individual-components-locally)
10. [FastAPI Usage](#fastapi-usage)
11. [Airflow Usage](#airflow-usage)
12. [Kafka and Spark Flow](#kafka-and-spark-flow)
13. [Lazada Cookies](#lazada-cookies)
14. [Models and Training](#models-and-training)
15. [Runtime Output Files](#runtime-output-files)
16. [Known Implementation Notes](#known-implementation-notes)
17. [Troubleshooting](#troubleshooting)
18. [Security Notes](#security-notes)

## System Overview

The system is composed of these main services:

| Component | Main files | Purpose |
| --- | --- | --- |
| Streamlit web app | `app/dashboard.py`, `app/pages/` | Main web UI for analytics, product search, product list, comparison, and live prediction display. |
| FastAPI backend | `api/main.py` | REST API for prediction files, Lazada search, single-text prediction, and pipeline trigger. |
| Lazada crawler | `app/lazada_crawler_simple.py`, `app/lazada_crawler.py`, `lazada_crawler/` | Crawls Lazada products and reviews using requests or Selenium. |
| Airflow | `airflow/dags/` | Orchestrates the real-time pipeline. |
| Kafka | `app/lazada_producer.py`, `kafka/` | Streams raw reviews through the `raw_reviews` topic. |
| Spark consumer | `app/kafka_absa_consumer.py` | Consumes Kafka messages in batches and runs distributed inference. |
| ABSA models | `methods/`, `models/`, `models_agm/` | ML, deep learning, and transformer-based ABSA models. |
| Training scripts | `train_all_methods.py`, `absa_dataset.py`, `scripts/` | Train and evaluate ABSA models. |

The real-time pipeline is:

```text
Streamlit UI
  -> Airflow DAG realtime_absa_pipeline
  -> Lazada crawler
  -> Kafka topic raw_reviews
  -> Spark/Kafka consumer
  -> ABSA predictor
  -> data/predictions/*.json
  -> Streamlit live charts and comparison tables
```

## Repository Structure

```text
.
|-- api/                         FastAPI application
|-- app/                         Streamlit app, crawler, predictor, Kafka consumer
|-- app/pages/                   Streamlit pages for product list and comparison
|-- airflow/dags/                Airflow DAG definitions
|-- kafka/                       Kafka producer/consumer simulation scripts
|-- lazada_crawler/              Standalone Lazada crawler Streamlit app
|-- methods/                     Model definitions
|-- models/                      Main trained model artifacts
|-- models_agm/                  Alternative trained model artifacts
|-- scripts/                     Evaluation and diagnostic scripts
|-- test_results/                Evaluation outputs
|-- visualizations/              Confusion matrix images
|-- docker-compose.yaml          Full Docker Compose stack
|-- Dockerfile.streamlit         Streamlit image
|-- Dockerfile.api               FastAPI image
|-- Dockerfile.spark             Spark/Kafka consumer image
|-- Dockerfile.airflow           Airflow image
|-- app_requirements.txt         Streamlit dependencies
|-- airflow_requirements.txt     Airflow dependencies
|-- spark_requirements.txt       Spark consumer dependencies
|-- requirements_training.txt    Training/evaluation dependencies
```

The folders below are runtime folders. They may not exist after a fresh clone and should be created before running the system:

```text
data/
data/predictions/
data/triggers/
data/label/
airflow/logs/
airflow/plugins/
cookie/
```

## Prerequisites

### Required

- Windows 10/11, Linux, or WSL2.
- Python 3.9 or newer.
- Docker Desktop with Docker Compose v2.
- Git.
- Google Chrome or Chromium if you use Selenium login/crawling.
- Internet connection for Docker images, Python packages, HuggingFace tokenizers/models, and ChromeDriver.

### Recommended

- NVIDIA GPU for transformer inference/training.
- NVIDIA driver with `nvidia-smi` working.
- NVIDIA Container Toolkit or Docker Desktop GPU support through WSL2.

Check GPU:

```powershell
nvidia-smi
```

If the machine has no NVIDIA GPU, see [Run Docker without GPU](#run-docker-without-gpu).

## Required Runtime Folders

Create required folders from PowerShell:

```powershell
New-Item -ItemType Directory -Force `
  data\predictions, `
  data\triggers, `
  data\label, `
  airflow\logs, `
  airflow\plugins, `
  cookie
```

Create required folders from Linux/WSL:

```bash
mkdir -p data/predictions data/triggers data/label airflow/logs airflow/plugins cookie
```

The main dashboard simulation mode expects this file:

```text
data/label/absa_grouped_vietnamese.xlsx
```

If that file is missing, use `Live Predictions` mode or copy the dataset into `data/label/`.

## File Access and Permission Setup

### Windows and Docker Desktop

Docker must be allowed to mount the project directory.

1. Open Docker Desktop.
2. Go to `Settings`.
3. Open `Resources`.
4. Open `File Sharing`.
5. Make sure this directory or its drive is shared:

```text
C:\Users\Luc\Real-Time-ABSA-System
```

6. If you use WSL2, enable integration for your WSL distribution.
7. Make sure PowerShell is running as a user that can read and write inside the repository.

Windows usually does not need `chmod`. If containers cannot write `data/predictions` or `airflow/logs`, the most common causes are:

- Docker Desktop does not have file sharing access.
- The runtime folders do not exist.
- Files are locked by another process.

### Linux or WSL

Make shell scripts executable:

```bash
chmod +x setup.sh entrypoint.airflow.sh
```

Create a `.env` file for Airflow UID mapping:

```bash
echo "AIRFLOW_UID=$(id -u)" > .env
echo "_AIRFLOW_WWW_USER_USERNAME=admin" >> .env
echo "_AIRFLOW_WWW_USER_PASSWORD=admin" >> .env
```

Grant write permission for runtime folders:

```bash
mkdir -p data/predictions data/triggers data/label airflow/logs airflow/plugins cookie
chmod -R u+rwX,g+rwX data airflow/logs airflow/plugins cookie
```

For local development only, if Airflow still cannot write logs:

```bash
chmod -R 777 data airflow/logs airflow/plugins cookie
```

If you get `/bin/bash^M` errors, convert shell scripts to Linux line endings:

```bash
dos2unix setup.sh entrypoint.airflow.sh
```

## Run the Full System with Docker

Docker Compose is the recommended way to run the complete system because it starts Streamlit, FastAPI, Airflow, Kafka, Spark master, Spark worker, and the Kafka consumer together.

### 1. Build images

From the repository root:

```powershell
docker compose build
```

If the build is slow, build one service at a time:

```powershell
docker compose build streamlit-app
docker compose build absa-api
docker compose build kafka-consumer
docker compose build airflow-webserver
```

### 2. Initialize Airflow

Run Airflow initialization once:

```powershell
docker compose up airflow-init
```

This initializes the Airflow database and creates the default web user.

Default Airflow account:

```text
username: admin
password: admin
```

### 3. Start all services

```powershell
docker compose up -d
```

### 4. Check running containers

```powershell
docker compose ps
```

### 5. Watch logs

Useful logs:

```powershell
docker compose logs -f streamlit-app
docker compose logs -f absa-api
docker compose logs -f airflow-webserver airflow-scheduler
docker compose logs -f kafka zookeeper
docker compose logs -f spark-master spark-worker
docker compose logs -f kafka-consumer
```

### 6. Stop services

Stop containers but keep volumes:

```powershell
docker compose down
```

Stop containers and delete Docker volumes, including the Airflow/Postgres database:

```powershell
docker compose down -v
```

### 7. Rebuild after code changes

Because the Compose file mounts the repository into containers, Python code changes are usually visible immediately. If dependencies or Dockerfiles change, rebuild:

```powershell
docker compose up -d --build
```

### Run Docker without GPU

`docker-compose.yaml` currently requests NVIDIA GPU devices for Airflow and `kafka-consumer`.

If you see an error similar to:

```text
could not select device driver "nvidia" with capabilities: [[gpu]]
```

edit `docker-compose.yaml` and remove or comment the `deploy.resources.reservations.devices` blocks from:

- `airflow-webserver`
- `airflow-scheduler`
- `airflow-init`
- `kafka-consumer`

Then rebuild:

```powershell
docker compose up -d --build
```

### Important Docker note about model and data files

`.dockerignore` excludes large/runtime files such as:

```text
*.pt
*.pkl
data/
cookie/
```

That means model files, runtime data, and cookies are not copied into Docker images during build. This project relies on volume mounts in `docker-compose.yaml`, for example:

```yaml
volumes:
  - ./:/app
```

and:

```yaml
volumes:
  - ./:/opt/airflow/project
```

Do not remove these mounts unless you also change how models, cookies, and runtime data are copied into the images.

## Web URLs and Service Ports

After `docker compose up -d`, open these URLs:

| Service | URL | Description |
| --- | --- | --- |
| Streamlit dashboard | `http://localhost:8501` | Main web application. |
| FastAPI root | `http://localhost:8000` | API health/root endpoint. |
| FastAPI Swagger docs | `http://localhost:8000/docs` | Interactive API documentation. |
| Airflow UI | `http://localhost:8080` | DAG management and logs. |
| Spark Master UI | `http://localhost:8081` | Spark cluster status. |
| Kafka host port | `localhost:9092` | Kafka from the host machine. |
| Kafka internal port | `kafka:29092` | Kafka from other Docker containers. |
| Zookeeper | `localhost:2181` | Zookeeper port. |

## How to Use the Web Application

### Open the main dashboard

Open:

```text
http://localhost:8501
```

The Streamlit app has:

- Home dashboard: `Real-Time Product Analytics`
- Product list page: `Danh Sach San Pham`
- Comparison page: `So Sanh`

### Home dashboard workflow

1. Open `http://localhost:8501`.
2. In the sidebar, choose data source:
   - `File Simulation`: reads from `data/label/absa_grouped_vietnamese.xlsx`.
   - `Live Predictions`: reads generated files from `data/predictions/*.json`.
3. If using `Live Predictions`, select a prediction file from the sidebar.
4. Click `Start Polling` to refresh charts as files change.
5. Click `Stop Polling` to pause updates.
6. Click `Reset Data` to reset the current Streamlit session state.

The dashboard displays:

- Live sentiment trends.
- Current metrics.
- Aspect breakdown.
- Price vs quality scatter plot if the required aspects exist.

### Product list page workflow

Open the product list page from the Streamlit sidebar.

Use this page to prepare products for comparison.

1. Configure Lazada cookies in the sidebar.
2. Use one of the cookie methods:
   - Login through browser.
   - Upload a cookie `.txt` file.
   - Reuse an existing file in `cookie/lazada_cookies.txt`.
3. Search Lazada products by keyword.
4. Click `Them` / `Add` for products you want to compare.
5. Or paste a Lazada product URL manually.
6. Add at least 2 products.
7. Click the comparison button to move to the comparison page.

### Comparison page workflow

The comparison page runs the ABSA pipeline.

1. Confirm that at least 2 products are selected.
2. Click reset/run again if you want a fresh pipeline run.
3. The page triggers Airflow DAG `realtime_absa_pipeline`.
4. Airflow crawls reviews and sends them to Kafka.
5. The Kafka consumer processes reviews and writes predictions to `data/predictions/`.
6. Streamlit polls prediction files and displays:
   - Radar chart by product.
   - Live review feed.
   - Aspect-level comparison.
   - Overall score per product.
   - Best product based on average aspect score.

### Airflow web workflow

Open:

```text
http://localhost:8080
```

Login:

```text
username: admin
password: admin
```

To monitor a pipeline:

1. Open DAG `realtime_absa_pipeline`.
2. Click the latest DAG run.
3. Open tasks:
   - `trigger_producer`
   - `wait_for_consumer`
   - `aggregate_results`
4. Click `Log` to inspect each task.

### FastAPI web workflow

Open:

```text
http://localhost:8000/docs
```

Use Swagger UI to test:

- `GET /`
- `GET /api/model-info`
- `GET /api/search`
- `GET /api/predictions`
- `GET /api/predictions/{product_id}`
- `POST /api/trigger-absa`
- `GET /api/trigger-status/{product_id}`
- `DELETE /api/predictions/clear`

## Run Individual Components Locally

You can run parts of the system locally without the full Docker stack. For the complete real-time pipeline, Docker Compose is still recommended.

### Create Python virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Linux/WSL:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Install dependencies for the Streamlit app

```powershell
pip install -r app_requirements.txt
```

### Install training/evaluation dependencies

```powershell
pip install -r requirements_training.txt
```

### Install API dependencies locally

The API Dockerfile installs these packages directly. For local API usage:

```powershell
pip install fastapi uvicorn httpx requests transformers scikit-learn pandas numpy tqdm openpyxl
```

Install PyTorch CPU:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install PyTorch CUDA 12.1 example:

```powershell
pip install "torch>=2.1.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Run Streamlit locally

```powershell
streamlit run app/dashboard.py
```

Open:

```text
http://localhost:8501
```

### Run the standalone Lazada crawler web app

```powershell
cd lazada_crawler
pip install -r requirements.txt
streamlit run app.py
```

The standalone crawler can:

- Open Lazada login.
- Save cookies.
- Search products.
- Crawl reviews.
- Bulk crawl by keyword.
- Export CSV.

### Run FastAPI locally

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

If FastAPI runs locally and must call Airflow in Docker, set:

```powershell
$env:AIRFLOW_URL="http://localhost:8080"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/docs
```

### Run Kafka and Spark with Docker, but app code locally

Start infrastructure only:

```powershell
docker compose up -d zookeeper kafka spark-master spark-worker
```

From the host, Kafka bootstrap server is:

```text
localhost:9092
```

From containers, Kafka bootstrap server is:

```text
kafka:29092
```

Run the simulation producer locally:

```powershell
$env:KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
python kafka\producer.py
```

Run the Docker Kafka consumer:

```powershell
docker compose up -d kafka-consumer
docker compose logs -f kafka-consumer
```

## FastAPI Usage

### Health check

PowerShell:

```powershell
Invoke-RestMethod http://localhost:8000/
```

Expected response includes:

```json
{
  "status": "ok",
  "message": "ABSA Prediction API v2.1 (Shared Predictor)",
  "model_loaded": true
}
```

### Get model info

```powershell
Invoke-RestMethod http://localhost:8000/api/model-info
```

### Search Lazada products

```powershell
Invoke-RestMethod "http://localhost:8000/api/search?keyword=dau%20goi&limit=10"
```

### List prediction files

```powershell
Invoke-RestMethod http://localhost:8000/api/predictions
```

### Trigger ABSA pipeline

Request body:

```json
{
  "product_url": "https://www.lazada.vn/products/example-i123456789.html",
  "max_reviews": 50
}
```

Swagger UI path:

```text
POST /api/trigger-absa
```

The API will try to trigger Airflow. If Airflow is offline, it saves a trigger file in:

```text
data/triggers/<product_id>.json
```

## Airflow Usage

### Main DAG

The main real-time DAG is:

```text
realtime_absa_pipeline
```

It has these tasks:

```text
trigger_producer -> wait_for_consumer -> aggregate_results
```

### Manual DAG trigger from Airflow UI

Open:

```text
http://localhost:8080
```

Then:

1. Open `realtime_absa_pipeline`.
2. Click the trigger button.
3. Paste JSON config.

Example config:

```json
{
  "product_id": "123456789",
  "product_url": "https://www.lazada.vn/products/example-i123456789.html",
  "max_reviews": 50
}
```

### Training DAG

The training DAG is:

```text
phobert_absa_training
```

It expects labeled Excel files under:

```text
labeled/
```

Inside Docker, that path becomes:

```text
/opt/airflow/project/labeled
```

If `labeled/` does not exist, create it or update the DAG to point to the actual dataset folder.

## Kafka and Spark Flow

### Kafka topic

The main topic is:

```text
raw_reviews
```

### Producer

`app/lazada_producer.py` sends crawled reviews to Kafka.

Each message contains fields similar to:

```json
{
  "product_id": "123456789",
  "review_content": "San pham tot, giao hang nhanh",
  "rating": 5,
  "review_id": "123456789_...",
  "timestamp": 1710000000.0
}
```

### Consumer

`app/kafka_absa_consumer.py`:

1. Subscribes to `raw_reviews`.
2. Buffers reviews by product.
3. Processes a batch when:
   - batch size reaches `BATCH_SIZE`, or
   - timeout reaches `BATCH_TIMEOUT`.
4. Runs Spark preprocessing and inference.
5. Writes predictions to:

```text
data/predictions/<product_id>.json
```

## Lazada Cookies

Lazada crawling may require valid cookies.

The system checks several possible cookie paths:

```text
cookie/lazada_cookies.json
cookie/lazada_cookies.txt
app/cookie/lazada_cookies.txt
api/lazada_cookies.json
lazada_crawler/lazada_cookies.json
```

Recommended workflow:

1. Open Streamlit at `http://localhost:8501`.
2. Go to the product list page.
3. Use the login/browser button to log in to Lazada.
4. Save cookies through the UI.
5. Confirm files exist in `cookie/`.

Alternative workflow:

1. Export cookies manually.
2. Save them as `cookie/lazada_cookies.txt` or `cookie/lazada_cookies.json`.
3. Restart crawler/API services if needed.

Do not commit real cookies to Git.

## Models and Training

### Existing model artifacts

The repository currently contains these model artifacts locally:

```text
models/logistic_regression_absa/logistic_regression_model.pkl
models/naive_bayes_absa/naive_bayes_model.pkl
models/bilstm_absa/bilstmforabsa_absa.pt
models/cnn_bilstm_absa/cnnbilstmforabsa_absa.pt
models/phobert_absa/phobertforabsamultipolarity_absa.pt
models/xlm_roberta_absa/xlmrobertaforabsa_absa.pt
```

There is also an alternative set under:

```text
models_agm/
```

These files are large and are ignored by `.gitignore` / `.dockerignore`. Make sure they exist on the machine that runs the system.

### Train all methods

```powershell
python train_all_methods.py --data "Augmented Dataset" --output models
```

### Train selected models

```powershell
python train_all_methods.py --data "Augmented Dataset" --output models --model phobert xlm_roberta
```

### Example training options

```powershell
python train_all_methods.py `
  --data "Augmented Dataset" `
  --output models `
  --model phobert xlm_roberta `
  --epochs 5 `
  --batch_size 16 `
  --lr 3e-5 `
  --max_length 256 `
  --label_smoothing 0.1 `
  --gamma 2.0 `
  --sentiment_weight 5.0
```

Supported parser arguments in `train_all_methods.py` include:

```text
--data
--output
--model
--folds
--epochs
--batch_size
--lr
--max_length
--label_smoothing
--gamma
--sentiment_weight
--patience
--threshold_min
--threshold_max
--threshold_steps
```

`setup.sh` contains an example command with `--device cuda`, but the current parser shown in `train_all_methods.py` does not define `--device`. If the script reports `unknown argument --device`, remove that argument or update the parser.

### Ollama predictor option

`app/ollama_predictor.py` supports local Ollama inference.

Defaults:

```text
model: mistral
Windows/local endpoint: http://localhost:11434
Docker endpoint: http://host.docker.internal:11434
```

You can override the endpoint:

```powershell
$env:OLLAMA_HOST="http://localhost:11434"
```

`app/model_config.json` currently contains:

```json
{"active_model": "ollama"}
```

See [Known Implementation Notes](#known-implementation-notes) for an important path note about this config file.

## Runtime Output Files

| Path | Purpose |
| --- | --- |
| `data/predictions/<product_id>.json` | Review-level predictions. |
| `data/predictions/<product_id>_summary.json` | Aggregated aspect summary. |
| `data/predictions/<product_id>.done` | Completion marker. |
| `data/triggers/<product_id>.json` | Local trigger fallback when Airflow is offline. |
| `data/crawled_reviews_buffer.csv` | Temporary crawled/deduplicated review buffer. |
| `airflow/logs/` | Airflow task logs. |

## Known Implementation Notes

These are important when running the current codebase.

### FastAPI predictor interface

`api/main.py` currently calls:

```python
predictor = PhoBERTPredictor()
success = predictor.load_model()
```

But the current `app/absa_predictor.py` defines `PhoBERTPredictor` as requiring `model_path` in the constructor and does not show a `load_model()` method.

If the API fails on startup with a predictor error, update `api/main.py` to instantiate the predictor with a real model path, for example:

```text
models/phobert_absa/phobertforabsamultipolarity_absa.pt
```

or restore a compatible `load_model()` method.

### Kafka consumer model config path

`app/kafka_absa_consumer.py` checks:

```text
/app/model_config.json
```

The existing config file is:

```text
app/model_config.json
```

In Docker, the repository root is mounted as `/app`, so the existing file becomes:

```text
/app/app/model_config.json
```

If you want the consumer to switch model based on config, either:

- copy/symlink `app/model_config.json` to root as `model_config.json`, or
- update `app/kafka_absa_consumer.py` to read `/app/app/model_config.json`.

### Dashboard simulation data

`app/dashboard.py` simulation mode expects:

```text
data/label/absa_grouped_vietnamese.xlsx
```

If this file is missing, use `Live Predictions` or add the dataset file.

### Airflow training data

`phobert_absa_training` expects:

```text
labeled/
```

If the folder is missing, create it or update the DAG.

## Troubleshooting

### Docker build is slow

This project installs PyTorch, Transformers, Spark dependencies, Airflow dependencies, Chrome, and Java. The first build can take a long time.

Retry:

```powershell
docker compose build --no-cache
```

Or build one service:

```powershell
docker compose build streamlit-app
```

### Streamlit opens but shows missing data error

If you see an error about:

```text
data/label/absa_grouped_vietnamese.xlsx
```

fix by one of these options:

- copy the expected Excel file into `data/label/`;
- switch the dashboard to `Live Predictions`;
- run the pipeline first so `data/predictions/*.json` exists.

### No predictions appear after running comparison

Check:

```powershell
docker compose logs -f airflow-scheduler
docker compose logs -f kafka-consumer
docker compose logs -f kafka
```

Also check:

```text
data/predictions/
```

Common causes:

- Lazada cookie expired.
- Kafka consumer is not running.
- Model artifact is missing.
- Predictor interface error.
- Airflow task failed.
- Container cannot write to `data/predictions`.

### Kafka connection error

From host scripts, use:

```text
localhost:9092
```

From Docker containers, use:

```text
kafka:29092
```

Check Kafka logs:

```powershell
docker compose logs -f kafka
```

### Airflow cannot write logs

Create folders:

```powershell
New-Item -ItemType Directory -Force airflow\logs, airflow\plugins
```

Linux/WSL permission fix:

```bash
chmod -R 777 airflow/logs airflow/plugins data
```

Windows fix:

- verify Docker Desktop file sharing;
- make sure files are not locked by editor/antivirus;
- restart Docker Desktop if mounts behave incorrectly.

### Selenium cannot open Chrome

Make sure Chrome is installed.

For headless/server environments, use:

- the requests-based crawler `app/lazada_crawler_simple.py`, or
- configure Chrome headless/Xvfb.

### Lazada search returns no results

Try:

1. refresh Lazada cookies;
2. log in again through the Streamlit UI;
3. reduce request frequency;
4. test another keyword;
5. check whether Lazada has blocked automated traffic.

### FastAPI `/api/trigger-absa` cannot reach Airflow

If FastAPI runs in Docker, default Airflow URL is:

```text
http://airflow-webserver:8080
```

If FastAPI runs on the host, set:

```powershell
$env:AIRFLOW_URL="http://localhost:8080"
```

Then restart FastAPI.

## Security Notes

- Do not commit real Lazada cookies.
- Do not commit `.env` files with secrets.
- Do not expose Airflow with `admin/admin` in production.
- Restrict FastAPI CORS before deployment. The current API allows broad origins.
- Keep model/data artifacts in controlled storage if they contain private or licensed data.
- Rotate cookies if you suspect they were leaked.

