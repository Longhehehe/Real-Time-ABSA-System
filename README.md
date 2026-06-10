# Real-Time ABSA System

He thong phan tich cam xuc theo khia canh (Aspect-Based Sentiment Analysis - ABSA) cho review san pham Lazada. Repo nay gom dashboard Streamlit, API FastAPI, crawler Lazada, pipeline Airflow, Kafka, Spark va cac model ML/DL/Transformer de crawl review, du doan sentiment theo tung aspect, sau do hien thi ket qua theo thoi gian gan thuc.

## 1. Tong quan he thong

### Thanh phan chinh

| Thanh phan | Thu muc / file | Vai tro |
| --- | --- | --- |
| Streamlit dashboard | `app/dashboard.py`, `app/pages/` | Giao dien web: xem analytics, tim san pham, them san pham, so sanh ABSA. |
| FastAPI backend | `api/main.py` | API tra prediction, search Lazada, trigger pipeline ABSA. |
| Lazada crawler | `app/lazada_crawler_simple.py`, `app/lazada_crawler.py`, `lazada_crawler/` | Crawl thong tin/review Lazada bang requests hoac Selenium. |
| Airflow | `airflow/dags/` | Orchestrate luong crawl -> Kafka -> Spark consumer -> aggregate. |
| Kafka | `app/lazada_producer.py`, `kafka/` | Queue review thoi gian thuc qua topic `raw_reviews`. |
| Spark consumer | `app/kafka_absa_consumer.py` | Doc Kafka theo batch, chay inference, ghi JSON ket qua. |
| Model ABSA | `methods/`, `models/`, `models_agm/` | Logistic Regression, Naive Bayes, BiLSTM, CNN-BiLSTM, PhoBERT, XLM-RoBERTa. |
| Training | `train_all_methods.py`, `absa_dataset.py`, `scripts/` | Train/evaluate cac model ABSA. |

### Luong realtime

1. Nguoi dung mo Streamlit tai `http://localhost:8501`.
2. Them it nhat 2 san pham Lazada de so sanh.
3. Streamlit trigger Airflow DAG `realtime_absa_pipeline`.
4. DAG crawl review, loai trung, gui review vao Kafka topic `raw_reviews`.
5. Kafka consumer doc batch review, chay Spark inference va ghi vao `data/predictions/<product_id>.json`.
6. Airflow aggregate thanh `<product_id>_summary.json` va tao marker `<product_id>.done`.
7. Streamlit doc `data/predictions/` de hien thi radar chart, live feed va diem theo aspect.

## 2. Yeu cau moi truong

### Bat buoc

- Python 3.9+ cho Streamlit/Spark; API Docker dung Python 3.10.
- Docker Desktop va Docker Compose v2.
- Git.
- Chrome/Chromium neu dung Selenium crawler hoac dang nhap Lazada tu UI.
- Ket noi internet de tai package, HuggingFace tokenizer/model base, ChromeDriver va goi Docker image.

### Neu dung GPU

- NVIDIA driver hoat dong (`nvidia-smi` chay duoc).
- NVIDIA Container Toolkit hoac Docker Desktop co ho tro GPU qua WSL2.
- Docker Compose hien co khai bao GPU cho Airflow va `kafka-consumer`. Neu may khong co GPU va Compose bao loi `could not select device driver "nvidia"`, hay bo cac block `deploy.resources.reservations.devices` trong `docker-compose.yaml` hoac chay CPU-only.

## 3. Cau truc du lieu quan trong

```text
.
|-- api/                         FastAPI app
|-- app/                         Streamlit, predictor, crawler, Kafka consumer
|-- airflow/dags/                DAG Airflow
|-- kafka/                       Producer/consumer mo phong
|-- lazada_crawler/              App crawler rieng bang Streamlit
|-- methods/                     Dinh nghia model ML/DL/Transformer
|-- models/                      Model artifact dang dung
|-- models_agm/                  Model artifact phien ban khac
|-- data/                        Runtime data, predictions, triggers (can tao)
|-- cookie/                      Cookie Lazada (khong nen commit)
|-- docker-compose.yaml          Stack Docker day du
```

Thu muc `data/` co the chua ton tai sau khi clone. Tao cac thu muc runtime truoc khi chay:

```powershell
New-Item -ItemType Directory -Force data\predictions, data\triggers, airflow\logs, airflow\plugins, cookie
```

Tren Linux/WSL:

```bash
mkdir -p data/predictions data/triggers airflow/logs airflow/plugins cookie
```

## 4. Quyen truy cap file va thu muc

### Windows + Docker Desktop

1. Mo Docker Desktop.
2. Vao `Settings` -> `Resources` -> `File Sharing`.
3. Dam bao o dia/thu muc `C:\Users\Luc\Real-Time-ABSA-System` duoc Docker phep mount.
4. Neu dung WSL2, bat WSL integration cho distro dang dung.
5. Chay terminal bang user co quyen doc/ghi vao repo.

PowerShell khong can `chmod`. Neu container khong ghi duoc prediction/log, thuong la do Docker chua duoc cap quyen file sharing hoac thu muc runtime chua duoc tao.

### Linux/WSL

Cap quyen execute cho shell script:

```bash
chmod +x setup.sh entrypoint.airflow.sh
```

Tao file `.env` cho Airflow de container ghi log dung owner:

```bash
echo "AIRFLOW_UID=$(id -u)" > .env
echo "_AIRFLOW_WWW_USER_USERNAME=admin" >> .env
echo "_AIRFLOW_WWW_USER_PASSWORD=admin" >> .env
```

Cap quyen ghi cho cac thu muc runtime trong moi truong dev:

```bash
mkdir -p data/predictions data/triggers airflow/logs airflow/plugins cookie
chmod -R u+rwX,g+rwX data airflow/logs airflow/plugins cookie
```

Neu Airflow van loi permission trong container, co the dung cach dev nhanh:

```bash
chmod -R 777 data airflow/logs airflow/plugins cookie
```

Neu gap loi `/bin/bash^M`, chuyen line ending:

```bash
dos2unix setup.sh entrypoint.airflow.sh
```

## 5. Chay bang Docker Compose

Day la cach nen dung khi can chay ca Streamlit, Airflow, Kafka, Spark va API.

### Khoi tao va build

```powershell
docker compose build
docker compose up airflow-init
docker compose up -d
```

Kiem tra container:

```powershell
docker compose ps
docker compose logs -f airflow-webserver airflow-scheduler kafka-consumer
```

### Cac URL sau khi chay

| Dich vu | URL | Ghi chu |
| --- | --- | --- |
| Streamlit dashboard | `http://localhost:8501` | Giao dien chinh. |
| FastAPI | `http://localhost:8000` | Root API. |
| FastAPI docs | `http://localhost:8000/docs` | Swagger UI. |
| Airflow | `http://localhost:8080` | Mac dinh `admin/admin`. |
| Spark Master UI | `http://localhost:8081` | Theo doi Spark. |
| Kafka host port | `localhost:9092` | Dung tu may host. |
| Kafka internal port | `kafka:29092` | Dung giua cac container. |

### Dung stack

```powershell
docker compose down
```

Neu muon xoa ca database volume cua Airflow/Postgres:

```powershell
docker compose down -v
```

### Luu y ve `.dockerignore`

Repo dang ignore cac file lon va runtime nhu `*.pt`, `*.pkl`, `data/`, `cookie/`. Khi build image, cac file nay khong duoc copy vao image. Trong `docker-compose.yaml`, cac service co mount `./:/app` hoac `./:/opt/airflow/project`, nen container se thay file local tren may.

Neu build image production khong mount volume, can sua `.dockerignore` hoac copy model/data/cookie bang cach khac.

## 6. Chay local khong Docker

### Tao moi truong Python

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r app_requirements.txt
pip install -r requirements_training.txt
pip install fastapi uvicorn httpx requests tqdm
```

Neu dung GPU, cai PyTorch dung CUDA phu hop voi may. Vi du CUDA 12.1:

```powershell
pip install "torch>=2.1.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Chay Streamlit dashboard

```powershell
streamlit run app/dashboard.py
```

Mo `http://localhost:8501`.

### Chay app crawler Lazada rieng

Repo co mot Streamlit app crawler rieng trong `lazada_crawler/`:

```powershell
cd lazada_crawler
pip install -r requirements.txt
streamlit run app.py
```

App nay phu hop khi can crawl/download review thu cong, bulk crawl theo keyword, luu cookie va export CSV.

### Chay FastAPI

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Mo `http://localhost:8000/docs`.

Luu y: code hien tai cua `api/main.py` dang goi `PhoBERTPredictor()` va `load_model()`, trong khi `app/absa_predictor.py` hien tai nhan `model_path` ngay trong constructor. Neu API loi khi startup, can dong bo lai interface predictor truoc khi dung API prediction.

### Chay Kafka/Spark rieng

Neu muon chay producer/consumer tu host, truoc het bat Kafka/Spark bang Docker:

```powershell
docker compose up -d zookeeper kafka spark-master spark-worker
```

Sau do chay producer mo phong:

```powershell
$env:KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
python kafka\producer.py
```

Consumer realtime duoc thiet ke chay trong container `kafka-consumer` vi can thay Spark master `spark://spark-master:7077`:

```powershell
docker compose up -d kafka-consumer
docker compose logs -f kafka-consumer
```

## 7. Model va training

### Model artifact hien co

Thu muc `models/` dang co cac artifact:

- `models/logistic_regression_absa/logistic_regression_model.pkl`
- `models/naive_bayes_absa/naive_bayes_model.pkl`
- `models/bilstm_absa/bilstmforabsa_absa.pt`
- `models/cnn_bilstm_absa/cnnbilstmforabsa_absa.pt`
- `models/phobert_absa/phobertforabsamultipolarity_absa.pt`
- `models/xlm_roberta_absa/xlmrobertaforabsa_absa.pt`

`models_agm/` co bo artifact tuong tu cho mot phien ban ket qua khac.

### Cau hinh predictor

`app/ollama_predictor.py` ho tro goi Ollama local, mac dinh model `mistral`.

- Tren Windows/local, Ollama endpoint mac dinh la `http://localhost:11434`.
- Trong Docker, endpoint mac dinh la `http://host.docker.internal:11434`.
- Co the doi endpoint bang bien moi truong `OLLAMA_HOST`.

`app/model_config.json` hien dang co:

```json
{"active_model": "ollama"}
```

Luu y: `app/kafka_absa_consumer.py` hien doc config o `/app/model_config.json`, trong khi file that dang nam tai `app/model_config.json`. Neu muon consumer doi model bang config trong Docker, can copy/symlink config ra root repo thanh `model_config.json` hoac sua consumer de doc `/app/app/model_config.json`.

### Train model tu command line

`train_all_methods.py` ho tro cac tham so:

```powershell
python train_all_methods.py --data "Augmented Dataset" --output models --model phobert xlm_roberta --epochs 5 --batch_size 16
```

Train tat ca model trong registry:

```powershell
python train_all_methods.py --data "Augmented Dataset" --output models
```

Mot lenh GPU duoc goi y trong `setup.sh`:

```bash
python train_all_methods.py \
  --data "Augmented Dataset" \
  --output models \
  --model xlm_roberta phobert \
  --device cuda \
  --gamma 2.0 \
  --sentiment_weight 7.0 \
  --label_smoothing 0.1 \
  --epochs 1 \
  --batch_size 16
```

Luu y: `train_all_methods.py` trong code hien tai co parser cho `--data`, `--output`, `--model`, `--folds`, `--epochs`, `--batch_size`, `--lr`, `--max_length`, `--label_smoothing`, `--gamma`, `--sentiment_weight`, `--patience`, `--threshold_min`, `--threshold_max`, `--threshold_steps`. Neu them `--device` ma script bao unknown argument, hay bo tham so do hoac cap nhat parser.

### Training bang Airflow

DAG `phobert_absa_training` se tim data trong:

```text
/opt/airflow/project/labeled
```

Trong repo hien tai chua thay thu muc `labeled/`. Truoc khi trigger DAG nay, can tao/copy data `.xlsx` vao `labeled/` hoac sua DAG de tro den thu muc data thuc te.

## 8. Thao tac tren web

### Streamlit dashboard

Mo:

```text
http://localhost:8501
```

Trang chinh `Real-Time Product Analytics`:

1. Trong sidebar chon `File Simulation` hoac `Live Predictions`.
2. `File Simulation` can file `data/label/absa_grouped_vietnamese.xlsx`.
3. `Live Predictions` doc cac file trong `data/predictions/*.json`.
4. Bam `Start Polling` de dashboard tu dong refresh.
5. Bam `Reset Data` de reset session hien tai.

Trang `Danh Sach San Pham`:

1. Cau hinh cookie Lazada trong sidebar.
2. Co the bam nut mo browser dang nhap Lazada, hoac upload file cookie `.txt`.
3. Tim san pham bang keyword.
4. Bam `Them` de dua san pham vao danh sach so sanh.
5. Hoac dan truc tiep URL san pham Lazada va bam `Them`.
6. Can it nhat 2 san pham de so sanh.

Trang `So Sanh`:

1. Kiem tra danh sach san pham.
2. Bam reset/chay lai neu muon trigger lai pipeline.
3. Ung dung trigger Airflow DAG, cho file prediction cap nhat.
4. Xem radar chart, bang live review va diem tong hop theo aspect.

### Airflow UI

Mo:

```text
http://localhost:8080
```

Dang nhap mac dinh:

```text
username: admin
password: admin
```

DAG quan trong:

- `realtime_absa_pipeline`: crawl/review -> Kafka -> consumer -> aggregate.
- `phobert_absa_training`: train PhoBERT multipolarity, can thu muc `labeled/`.
- `sentiment_model_training`: DAG training mau, dang tro den `train_pipeline.py`.

Trigger DAG `realtime_absa_pipeline` thu cong voi config:

```json
{
  "product_id": "123456789",
  "product_url": "https://www.lazada.vn/products/example-i123456789.html",
  "max_reviews": 50
}
```

Theo doi log cac task:

1. Mo DAG run.
2. Chon task `trigger_producer`, `wait_for_consumer`, hoac `aggregate_results`.
3. Bam `Log`.

### FastAPI

Mo Swagger:

```text
http://localhost:8000/docs
```

Endpoint chinh:

| Method | Path | Chuc nang |
| --- | --- | --- |
| `GET` | `/` | Health check API. |
| `GET` | `/api/model-info` | Thong tin aspect/sentiment. |
| `POST` | `/api/predict-text` | Du doan 1 cau review. |
| `GET` | `/api/search?keyword=...&limit=20` | Tim san pham Lazada. |
| `GET` | `/api/predictions` | Liet ke file prediction. |
| `GET` | `/api/predictions/{product_id}` | Lay prediction cua 1 san pham. |
| `POST` | `/api/trigger-absa` | Trigger pipeline ABSA qua Airflow. |
| `GET` | `/api/trigger-status/{product_id}` | Kiem tra trang thai prediction. |
| `DELETE` | `/api/predictions/clear` | Xoa prediction runtime. |

## 9. Cookie Lazada

He thong tim cookie tai cac duong dan sau:

- `cookie/lazada_cookies.json`
- `cookie/lazada_cookies.txt`
- `app/cookie/lazada_cookies.txt`
- `api/lazada_cookies.json`
- `lazada_crawler/lazada_cookies.json`

Khuyen nghi:

1. Dung UI Streamlit de dang nhap va luu cookie.
2. Dam bao `cookie/` ton tai va container co quyen doc/ghi.
3. Khong commit cookie that len Git.
4. Neu search/crawl tra ve rong, hay dang nhap lai de refresh cookie.

## 10. File output runtime

| File / thu muc | Y nghia |
| --- | --- |
| `data/predictions/<product_id>.json` | Prediction tung review. |
| `data/predictions/<product_id>_summary.json` | Summary theo aspect/sentiment. |
| `data/predictions/<product_id>.done` | Marker pipeline hoan tat. |
| `data/triggers/<product_id>.json` | Trigger local khi Airflow offline. |
| `data/crawled_reviews_buffer.csv` | Buffer review sau crawl/dedup. |
| `airflow/logs/` | Log Airflow. |

## 11. Troubleshooting

### `data/label/absa_grouped_vietnamese.xlsx` not found

Trang dashboard che do `File Simulation` can file nay. Tao dung duong dan hoac chuyen sang `Live Predictions`.

### Airflow khong ghi duoc log/prediction

Tao thu muc va cap quyen:

```bash
mkdir -p data/predictions data/triggers airflow/logs airflow/plugins
chmod -R 777 data airflow/logs airflow/plugins
```

Tren Windows, kiem tra Docker Desktop File Sharing.

### Kafka consumer khong connect Kafka

Kiem tra service:

```powershell
docker compose ps kafka zookeeper kafka-consumer
docker compose logs -f kafka
```

Trong container dung `kafka:29092`; tu host dung `localhost:9092`.

### Airflow DAG trigger thanh cong nhung khong co prediction

Kiem tra:

1. `docker compose logs -f kafka-consumer`
2. Airflow task `wait_for_consumer`
3. Thu muc `data/predictions/`
4. Model artifact trong `models/`
5. Cookie Lazada con hop le

### API loi predictor khi startup

`api/main.py` hien dang goi interface cu (`PhoBERTPredictor()` roi `load_model()`), trong khi `app/absa_predictor.py` hien tai can `model_path`. Can sua API de truyen path model, vi du `models/phobert_absa/phobertforabsamultipolarity_absa.pt`, hoac khoi phuc method `load_model()`.

### Docker build cham hoac fail khi cai Torch/Transformers

Day la binh thuong vi image can tai nhieu package lon. Thu lai:

```powershell
docker compose build --no-cache
```

Neu mang yeu, build tung service:

```powershell
docker compose build streamlit-app
docker compose build absa-api
docker compose build kafka-consumer
```

### Selenium khong mo duoc Chrome

Kiem tra Chrome da cai tren host, hoac chay crawler trong moi truong co GUI. Tren server headless, can cau hinh Chrome headless/Xvfb hoac dung crawler requests-based `lazada_crawler_simple.py`.

## 12. Ghi chu bao mat

- Khong commit `cookie/`, `.env`, model/data noi bo hoac file prediction co thong tin nguoi dung.
- Airflow `admin/admin` chi phu hop local/dev. Khi deploy, doi username/password va secret key.
- FastAPI dang `allow_origins=["*"]`; khi deploy that, nen gioi han origin frontend.
- Cookie Lazada la thong tin dang nhap, can luu local va xoay vong khi nghi ngo bi lo.
