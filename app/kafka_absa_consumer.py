"""
Kafka Consumer with Spark Pandas UDF
- Consumes raw reviews from 'raw_reviews' topic
- Batches data and triggers Spark job for distributed prediction
- Uses Pandas UDF for preprocessing and inference
"""
import json
import os
import sys
import time
import re
from typing import Dict, List
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
from collections import defaultdict
import threading

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
INPUT_TOPIC = 'raw_reviews'
GROUP_ID = 'absa_spark_consumer_group'
BATCH_SIZE = 20  # Collect reviews before triggering Spark job
BATCH_TIMEOUT = 30  # Max seconds to wait for batch to fill

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_DIR = os.path.join(PROJECT_DIR, 'data', 'predictions')
SPARK_MASTER = os.environ.get('SPARK_MASTER', 'spark://spark-master:7077')

# Buffer for batching
review_buffer = defaultdict(list)  # product_id -> [reviews]
buffer_lock = threading.Lock()


def clean_text(text: str) -> str:
    """Preprocess text before prediction."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Global Spark Session
spark = None

def get_spark_session():
    """Get or create global Spark session."""
    global spark
    if spark is None:
        from pyspark.sql import SparkSession
        print("⚡ Initializing Spark Session...")
        spark = SparkSession.builder \
            .appName("ABSA_Consumer_Service") \
            .master(SPARK_MASTER) \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .config("spark.executor.cores", "1") \
            .config("spark.task.cpus", "1") \
            .getOrCreate()
        print("⚡ Spark Session Ready")
    return spark

def run_spark_prediction(product_id: str, reviews: List[Dict]):
    """
    Trigger Spark job to predict batch of reviews using Pandas UDF.
    """
    try:
        from pyspark.sql.functions import pandas_udf, col
        from pyspark.sql.types import StringType
        import pandas as pd
        
        print(f"🚀 Starting Spark job for {len(reviews)} reviews (Product: {product_id})")
        
        spark = get_spark_session()
        
        # Prepare data
        data = [(r.get('review_content', ''), r.get('rating', 0), r.get('review_id', '')) 
                for r in reviews]
        df = spark.createDataFrame(data, ["review_text", "rating", "review_id"])
        
        # Define Pandas UDFs locally (closures verify imports on workers)
        @pandas_udf(StringType())
        def preprocess_udf(texts: pd.Series) -> pd.Series:
            import re
            def clean(text):
                if not text:
                    return ""
                text = str(text).lower()
                text = re.sub(r'<[^>]+>', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            return texts.apply(clean)
        
        @pandas_udf(StringType())
        def predict_model_udf(texts: pd.Series) -> pd.Series:
            import json
            import os
            import sys
            
            # Ensure we can import from app
            # Ensure we can import from app
            # Dynamic path: c:\...\app\kafka_absa_consumer.py -> PROJECT_ROOT = c:\...\
            # Actually this file is in /app/app/ inside Docker usually, or ./app/ locally
            # We want the directory containing 'app' and 'data' and 'model' folders.
            
            # If __file__ is /path/to/project/app/kafka_absa_consumer.py
            # Then dirname is /path/to/project/app
            # Then dirname(dirname) is /path/to/project
            PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            if PROJECT_ROOT not in sys.path:
                sys.path.insert(0, PROJECT_ROOT)
                
            try:
                from app.absa_predictor import PhoBERTPredictor, ASPECTS
                from app.ollama_predictor import OllamaPredictor
            except ImportError as e:
                return pd.Series([json.dumps({'error': f'ImportError: {e}'}) for _ in range(len(texts))])

            model_path_phobert = os.path.join(PROJECT_ROOT, "model", "bert_absa_model.pth")
            config_path = os.path.join(PROJECT_ROOT, "model_config.json")
            
            # Helper: Load Config
            def get_active_model_type():
                try:
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            cfg = json.load(f)
                            return cfg.get("active_model", "phobert"), os.path.getmtime(config_path)
                except:
                    pass
                return "phobert", 0

            # Helper: Factory
            def load_model_instance(model_type):
                try:
                    if model_type == "ollama":
                        print("📥 Loading Ollama (Mistral)...")
                        return OllamaPredictor()
                    else:
                        print("📥 Loading Custom PhoBERT ABSA Model...")
                        return PhoBERTPredictor(model_path=model_path_phobert)
                except Exception as e:
                    print(f"❌ Model load failed: {e}")
                    return None

            # Singleton State Initialization
            if not hasattr(predict_model_udf, 'predictor'):
                model_type, mtime = get_active_model_type()
                predict_model_udf.predictor = load_model_instance(model_type)
                predict_model_udf.last_config_mtime = mtime
                predict_model_udf.current_model_type = model_type
                
                # PhoBERT specific reload check (only if using phobert)
                predict_model_udf.last_phobert_mtime = 0
                if os.path.exists(model_path_phobert):
                    predict_model_udf.last_phobert_mtime = os.path.getmtime(model_path_phobert)
            
            # --- HOT RELOAD LOGIC ---
            
            # 1. Check Config Change (Model Switch)
            current_type, current_config_mtime = get_active_model_type()
            
            if current_type != getattr(predict_model_udf, 'current_model_type', 'phobert'):
                print(f"🔄 Switching Model: {predict_model_udf.current_model_type} -> {current_type}")
                predict_model_udf.predictor = load_model_instance(current_type)
                predict_model_udf.current_model_type = current_type
                predict_model_udf.last_config_mtime = current_config_mtime
            
            # 2. Check PhoBERT File Update (Only if active)
            elif current_type == "phobert":
                try:
                     if os.path.exists(model_path_phobert):
                        curr_phobert_mtime = os.path.getmtime(model_path_phobert)
                        if curr_phobert_mtime > getattr(predict_model_udf, 'last_phobert_mtime', 0):
                            print("🔄 PhoBERT file updated! Reloading...")
                            predict_model_udf.predictor = load_model_instance("phobert")
                            predict_model_udf.last_phobert_mtime = curr_phobert_mtime
                except: pass

            results = []

            results = []
            
            for text in texts:
                try:
                    if not text:
                        results.append(json.dumps({}))
                        continue
                        
                    sentiment_dict = {}
                    if predict_model_udf.predictor:
                        # predict_single returns {aspect: -1/0/1/2}
                        # We assume the UI handles -1, 0, 1. 2 is N/A.
                        raw_preds = predict_model_udf.predictor.predict_single(text)
                        
                        # Filter out N/A (2) if desired, or keep them.
                        # The UI likely expects keys to exist.
                        # Let's clean it up for the UI (UI expects 1, -1, 0)
                        # If 2 (N/A), maybe we should treat as NEU (0) or omit?
                        # For now, let's keep valid values.
                        sentiment_dict = {k: v for k, v in raw_preds.items() if v != 2}
                    
                    results.append(json.dumps(sentiment_dict, ensure_ascii=False))
                except Exception as e:
                    results.append(json.dumps({'error': str(e)}))
            
            return pd.Series(results)
        
        # Apply UDFs
        df_result = df \
            .withColumn("cleaned_text", preprocess_udf(col("review_text"))) \
            .withColumn("sentiment_json", predict_model_udf(col("cleaned_text")))
        
        # Collect results
        predictions = []
        rows = df_result.collect()
        print(f"📊 Collected {len(rows)} results from Spark")
        
        for row in rows:
            predictions.append({
                'review_id': row['review_id'],
                'original_text': row['review_text'],
                'cleaned_text': row['cleaned_text'],
                'sentiment': json.loads(row['sentiment_json']),
                'rating': row['rating'],
                'processed_at': time.time()
            })
        
        # Note: Do NOT stop spark session here
        
        # Save predictions
        save_predictions(product_id, predictions)
        print(f"✅ Prediction processing complete for {len(predictions)} reviews")
        
        # Log sample for debugging
        if predictions:
            sample = predictions[0]
            print(f"🔎 Sample Prediction: {json.dumps(sample.get('sentiment', {}), ensure_ascii=False)}")
        
    except Exception as e:
        print(f"❌ Spark job failed: {e}")
        # Fallback: save empty or partial results
        import traceback
        traceback.print_exc()


def save_predictions(product_id: str, new_predictions: List[Dict]):
    """Save predictions to JSON file (appending to existing) using ATOMIC WRITE."""
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    file_path = os.path.join(PREDICTIONS_DIR, f"{product_id}.json")
    temp_path = f"{file_path}.tmp"
    
    existing_data = []
    
    # Retry reading existing file to handle race conditions
    max_read_retries = 3
    for i in range(max_read_retries):
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                break  # Success
            except json.JSONDecodeError:
                if i == max_read_retries - 1:
                    print(f"⚠️ Could not read existing file {file_path} after {max_read_retries} attempts. Starting fresh.")
                    existing_data = []
                else:
                    time.sleep(0.1)
        else:
            break

    # Merge data (avoid duplicates based on review_id)
    existing_ids = {item.get('review_id') for item in existing_data if item.get('review_id')}
    
    added_count = 0
    for pred in new_predictions:
        if not pred.get('review_id') or pred.get('review_id') not in existing_ids:
            existing_data.append(pred)
            added_count += 1
    
    # Atomic Write: Write to temp file first, then rename
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk
        
        # Atomic Rename (with Retry for Windows File Locks)
        max_rename_retries = 5
        for i in range(max_rename_retries):
            try:
                os.replace(temp_path, file_path)
                try:
                    os.chmod(file_path, 0o666) # Allow read/write for all (fix PermissionError)
                except Exception as ex:
                    print(f"⚠️ Could not chmod {file_path}: {ex}")
                    
                print(f"💾 Saved {len(existing_data)} predictions (Added {added_count} new) to {file_path}")
                break
            except OSError as e:
                # Windows specific: [WinError 32] The process cannot access the file because it is being used by another process
                if i == max_rename_retries - 1:
                    raise e
                print(f"⚠️ Rename failed (process lock?), retrying {i+1}/{max_rename_retries}...")
                time.sleep(0.5)
        
    except Exception as e:
        print(f"❌ Failed to save predictions atomically: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def process_batch(product_id: str):
    """Process buffered reviews for a product."""
    global review_buffer
    
    with buffer_lock:
        reviews = review_buffer.pop(product_id, [])
    
    if reviews:
        run_spark_prediction(product_id, reviews)


def create_consumer():
    """Create Kafka consumer with retry."""
    while True:
        try:
            return KafkaConsumer(
                INPUT_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                group_id=GROUP_ID,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='earliest',
                consumer_timeout_ms=BATCH_TIMEOUT * 1000
            )
        except NoBrokersAvailable:
            print("⏳ Kafka not ready, retrying in 5s...")
            time.sleep(5)


def run_service():
    """Main consumer loop with batching."""
    print(f"🚀 Starting Kafka Consumer (Spark Pandas UDF mode)")
    print(f"   Kafka: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"   Spark: {SPARK_MASTER}")
    print(f"   Batch Size: {BATCH_SIZE}")
    
    while True:
        try:
            consumer = create_consumer()
            print(f"✅ Subscribed to topic: {INPUT_TOPIC}")
            
            batch_start_time = {}
            
            for message in consumer:
                data = message.value
                product_id = data.get('product_id', 'unknown')
                
                with buffer_lock:
                    review_buffer[product_id].append(data)
                    
                    if product_id not in batch_start_time:
                        batch_start_time[product_id] = time.time()
                    
                    # Check if batch is ready
                    batch_ready = (
                        len(review_buffer[product_id]) >= BATCH_SIZE or
                        time.time() - batch_start_time.get(product_id, 0) > BATCH_TIMEOUT
                    )
                
                if batch_ready:
                    print(f"📦 Batch ready for {product_id} ({len(review_buffer[product_id])} reviews)")
                    process_batch(product_id)
                    batch_start_time.pop(product_id, None)
            
            # Process remaining batches after timeout
            for product_id in list(review_buffer.keys()):
                if review_buffer[product_id]:
                    print(f"📦 Processing remaining batch for {product_id}")
                    process_batch(product_id)
                    
        except Exception as e:
            print(f"❌ Consumer error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    run_service()
