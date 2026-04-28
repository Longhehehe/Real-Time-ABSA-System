```mermaid
flowchart TB
    %% ===== DATA SOURCE =====
    subgraph DS["🛒 DATA SOURCE"]
        Lazada["Lazada<br/>E-commerce"]
    end

    %% ===== DATA INGESTION =====
    subgraph DI["📥 DATA INGESTION"]
        Crawler["Lazada<br/>Crawler"]
        Producer["Kafka<br/>Producer"]
    end

    %% ===== MESSAGE QUEUE =====
    subgraph MQ["📨 MESSAGE QUEUE"]
        Kafka["Apache Kafka"]
        Zookeeper["Zookeeper"]
    end

    %% ===== TRAINING PIPELINE =====
    subgraph TRAINING["🎓 MODEL TRAINING"]
        RawData["Raw Reviews<br/>Data"]
        Labeling["Manual<br/>Labeling"]
        LabeledData["Labeled<br/>Dataset"]
        Preprocessing["Text<br/>Preprocessing"]
        Trainer["PhoBERT<br/>Trainer"]
        TrainedModel["Trained<br/>Model<br/>(.pt file)"]
    end

    %% ===== PROCESSING =====
    subgraph PROCESSING["⚙️ REAL-TIME PROCESSING"]
        Consumer["Kafka<br/>Consumer"]
        subgraph Spark["Spark Cluster"]
            Master["Master"]
            Worker1["Worker 1"]
            Worker2["Worker 2"]
        end
        Predictor["PhoBERT<br/>Predictor<br/>(ABSA)"]
    end

    %% ===== ORCHESTRATION =====
    subgraph ORCH["🎛️ ORCHESTRATION"]
        subgraph Airflow["Apache Airflow"]
            Webserver["Webserver<br/>:8080"]
            Scheduler["Scheduler"]
            DAGs["DAGs"]
        end
        Postgres["PostgreSQL"]
    end

    %% ===== STORAGE =====
    subgraph STORAGE["💾 STORAGE"]
        Predictions["Predictions<br/>JSON Files"]
        ModelStorage["Model<br/>Storage"]
    end

    %% ===== PRESENTATION =====
    subgraph PRESENTATION["📊 PRESENTATION"]
        FastAPI["FastAPI<br/>Backend<br/>:8000"]
        React["React<br/>Frontend<br/>:3000"]
        Streamlit["Streamlit<br/>Dashboard<br/>:8501"]
    end

    %% ===== TRAINING FLOW (Offline) =====
    RawData --> Labeling
    Labeling --> LabeledData
    LabeledData --> Preprocessing
    Preprocessing --> Trainer
    Trainer --> TrainedModel
    TrainedModel --> ModelStorage
    ModelStorage --> Predictor

    %% ===== REAL-TIME FLOW =====
    Lazada --> Crawler
    Crawler --> Producer
    Producer --> Kafka
    Zookeeper -.-> Kafka
    
    Kafka --> Consumer
    Consumer --> Spark
    Master --> Worker1
    Master --> Worker2
    Spark --> Predictor
    
    Predictor --> Predictions
    
    %% ===== ORCHESTRATION FLOW =====
    Postgres -.-> Airflow
    DAGs --> Crawler
    DAGs --> Consumer
    DAGs -.-> Trainer
    
    %% ===== PRESENTATION FLOW =====
    Predictions --> FastAPI
    FastAPI --> React
    Predictions --> Streamlit

    %% ===== STYLING =====
    classDef source fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef ingestion fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef queue fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    classDef process fill:#fff8e1,stroke:#fbc02d,stroke-width:2px
    classDef training fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef orch fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef storage fill:#e0f2f1,stroke:#00796b,stroke-width:2px
    classDef present fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px

    class Lazada source
    class Crawler,Producer ingestion
    class Kafka,Zookeeper queue
    class Consumer,Master,Worker1,Worker2,Predictor process
    class RawData,Labeling,LabeledData,Preprocessing,Trainer,TrainedModel training
    class Webserver,Scheduler,DAGs,Postgres orch
    class Predictions,ModelStorage storage
    class FastAPI,React,Streamlit present
```

---

## 📋 Giải thích luồng:

### 1. Training Pipeline (Offline - Một lần)
```
Raw Reviews → Manual Labeling → Labeled Dataset → Preprocessing → PhoBERT Trainer → Trained Model (.pt)
```

### 2. Real-time Pipeline (Online - Liên tục)
```
Lazada → Crawler → Kafka Producer → Kafka → Consumer → Spark → PhoBERT Predictor → Predictions JSON
```

### 3. Presentation Layer
```
Predictions JSON → FastAPI → React Frontend
                → Streamlit Dashboard
```

---

## 🔧 Cách dùng với Draw.io:

1. Mở https://mermaid.live/
2. Paste code Mermaid (bỏ ``` mermaid và ```)
3. Export PNG/SVG
4. Import vào Draw.io

Hoặc trong Draw.io: **Arrange → Insert → Advanced → Mermaid**
