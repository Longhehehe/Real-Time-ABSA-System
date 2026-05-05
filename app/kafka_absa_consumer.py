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

KAFKA_BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
INPUT_TOPIC = 'raw_reviews'
GROUP_ID = 'absa_spark_consumer_group'
BATCH_SIZE = 10                                         
BATCH_TIMEOUT = 15                                                

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_DIR = os.path.join(PROJECT_DIR, 'data', 'predictions')
SPARK_MASTER = os.environ.get('SPARK_MASTER', 'spark://spark-master:7077')

review_buffer = defaultdict(list)                           
buffer_lock = threading.Lock()

def clean_text(text: str) -> str:
    """Preprocess text before prediction."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)               
    text = re.sub(r'\s+', ' ', text).strip()
    return text

spark = None

def get_spark_session():
    """Get or create global Spark session."""
    global spark
    if spark is None:
        from pyspark.sql import SparkSession
                                        
        spark = SparkSession.builder.getOrCreate()
        if spark.sparkContext.master == "local[*]":                                               
             pass                                                                                   
        
        if spark is None or spark.sparkContext.isStopped:
            print(" Initializing Spark Session...")
            spark = SparkSession.builder                .appName("ABSA_Consumer_Service")                                                        
        print(" Initializing Spark Session...")
        spark = SparkSession.builder            .appName("ABSAConsumer")            .master(SPARK_MASTER)            .config("spark.executor.memory", "2g")            .config("spark.driver.memory", "1g")            .config("spark.sql.execution.arrow.pyspark.enabled", "true")            .getOrCreate()
        print(" Spark Session Ready")
    return spark

def _preprocess_text_logic(texts):
    import pandas as pd
    import re
    def clean(text):
        if not text:
            return ""
        text = str(text).lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return texts.apply(clean)

def _predict_model_logic(texts):
    import pandas as pd
    import json
    import os
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    current_dir = os.getcwd()              
    if current_dir not in sys.path:
        sys.path.append(current_dir)
        
    project_root = "/app"
    if project_root not in sys.path:
        sys.path.append(project_root)

    try:
        from app.absa_predictor import PhoBERTPredictor
        from app.ollama_predictor import OllamaPredictor
    except ImportError:
                                         
        try:
           sys.path.append('/opt/airflow/project') 
           from app.absa_predictor import PhoBERTPredictor
           from app.ollama_predictor import OllamaPredictor
        except Exception as e:
            return pd.Series([json.dumps({'error': f'ImportError on worker: {e}'})] * len(texts))

    if not hasattr(_predict_model_logic, 'predictor'):
        _predict_model_logic.predictor = None
        _predict_model_logic.model_type = None
        _predict_model_logic.last_config_check = 0
        print(f"Worker {os.getpid()}: Initializing predictor state")
    
    config_path = os.path.join(project_root, "model_config.json")
    
    now = time.time()
    if now - _predict_model_logic.last_config_check > 30:
        file_mtime = 0
        target_model = "phobert"
        if os.path.exists(config_path):
             try:
                 with open(config_path, 'r') as f:
                     cfg = json.load(f)
                     target_model = cfg.get("active_model", "phobert")
             except: pass
        
        if target_model != _predict_model_logic.model_type or _predict_model_logic.predictor is None:
             try:
                 if target_model == "ollama":
                                                                     
                     _predict_model_logic.predictor = OllamaPredictor()
                 else:
                                                                    
                     _predict_model_logic.predictor = PhoBERTPredictor()
                 
                 _predict_model_logic.model_type = target_model
                 print(f"Worker {os.getpid()}: Loaded model: {target_model}")
             except Exception as e:
                 print(f"Worker {os.getpid()}: Failed load model: {e}")
                 
        _predict_model_logic.last_config_check = now

    predictor = _predict_model_logic.predictor
    
    results = []
    print(f"Worker {os.getpid()}: Processing batch of {len(texts)} texts")
    
    if predictor:
                                                 
        if hasattr(predictor, 'predict_batch'):
                                    
            text_list = texts.tolist()
                                                                             
            batch_results = predictor.predict_batch(text_list, format='multipolarity')
            
            for res in batch_results:
                results.append(json.dumps(res, ensure_ascii=False))
        else:
                              
            def _predict_single_safe(txt):
                try:
                    res = predictor.predict_single(txt)
                    return res['multipolarity']                                 
                except:
                                                 
                    return {asp: {'mentioned': False, 'sentiments': None} for asp in ['Chất lượng sản phẩm', 'Hiệu năng & Trải nghiệm', 'Đúng mô tả', 'Giá cả & Khuyến mãi', 'Vận chuyển', 'Đóng gói', 'Dịch vụ & Thái độ Shop', 'Bảo hành & Đổi trả', 'Tính xác thực']}

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(_predict_single_safe, t) for t in texts]
                for f in futures:
                    results.append(json.dumps(f.result(), ensure_ascii=False))
        
    else:
        results = [json.dumps({'error': 'No predictor loaded'})] * len(texts)

    return pd.Series(results)

def run_spark_prediction(product_id: str, reviews: List[Dict]):
    """
    Trigger Spark job to predict batch of reviews using Pandas UDF.
    """
    try:
        from pyspark.sql.functions import pandas_udf, col
        from pyspark.sql.types import StringType
        
        print(f" Starting Spark job for {len(reviews)} reviews (Product: {product_id})")
        
        spark = get_spark_session()
        
        data = []
        skipped_empty = 0
        for r in reviews:
            text = r.get('review_content') or r.get('review_text') or r.get('reviewContent') or r.get('content') or ''
            text = str(text).strip() if text else ''
            
            if not text:
                skipped_empty += 1
                print(f" Skipping empty review: id={r.get('review_id', 'N/A')}")
                continue
                
            data.append((text, r.get('rating', 0), r.get('review_id', '')))
        
        if skipped_empty > 0:
            print(f" Skipped {skipped_empty} empty reviews out of {len(reviews)}")
        
        if not data:
            print(f" No valid reviews to process after filtering!")
            return
            
        df = spark.createDataFrame(data, ["review_text", "rating", "review_id"])
        
        preprocess_udf = pandas_udf(StringType())(_preprocess_text_logic)
        predict_model_udf = pandas_udf(StringType())(_predict_model_logic)
        
        df_result = df            .repartition(1)            .withColumn("cleaned_text", preprocess_udf(col("review_text")))            .withColumn("sentiment_json", predict_model_udf(col("cleaned_text")))
        
        predictions = []
        rows = df_result.collect()
        print(f" Collected {len(rows)} results from Spark")
        
        for row in rows:
            predictions.append({
                'review_id': row['review_id'],
                'original_text': row['review_text'],
                'cleaned_text': row['cleaned_text'],
                'sentiment': json.loads(row['sentiment_json']),
                'rating': row['rating'],
                'processed_at': time.time()
            })
        
        save_predictions(product_id, predictions)
        print(f" Prediction processing complete for {len(predictions)} reviews")
        
    except Exception as e:
        print(f" Spark job failed: {e}")
        import traceback
        traceback.print_exc()

def save_predictions(product_id: str, new_predictions: List[Dict]):
    """Save predictions to JSON file (appending to existing) using ATOMIC WRITE."""
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    file_path = os.path.join(PREDICTIONS_DIR, f"{product_id}.json")
    temp_path = f"{file_path}.tmp"
    
    existing_data = []
    
    max_read_retries = 3
    for i in range(max_read_retries):
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                break           
            except json.JSONDecodeError:
                if i == max_read_retries - 1:
                    print(f" Could not read existing file {file_path} after {max_read_retries} attempts. Starting fresh.")
                    existing_data = []
                else:
                    time.sleep(0.1)
        else:
            break

    existing_ids = {item.get('review_id') for item in existing_data if item.get('review_id')}
    
    added_count = 0
    for pred in new_predictions:
        if not pred.get('review_id') or pred.get('review_id') not in existing_ids:
            existing_data.append(pred)
            added_count += 1
    
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())                                  
        
        max_rename_retries = 5
        for i in range(max_rename_retries):
            try:
                os.replace(temp_path, file_path)
                try:
                    os.chmod(file_path, 0o666)                                                 
                except Exception as ex:
                    print(f" Could not chmod {file_path}: {ex}")
                    
                print(f" Saved {len(existing_data)} predictions (Added {added_count} new) to {file_path}")
                break
            except OSError as e:
                                                                                                                                
                if i == max_rename_retries - 1:
                    raise e
                print(f" Rename failed (process lock?), retrying {i+1}/{max_rename_retries}...")
                time.sleep(0.5)
        
    except Exception as e:
        print(f" Failed to save predictions atomically: {e}")
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
            print(" Kafka not ready, retrying in 5s...")
            time.sleep(5)

def run_service():
    """Main consumer loop with batching."""
    print(f" Starting Kafka Consumer (Spark Pandas UDF mode)")
    print(f"   Kafka: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"   Spark: {SPARK_MASTER}")
    print(f"   Batch Size: {BATCH_SIZE}")
    
    batch_start_time = {}
    
    while True:
        try:
            consumer = create_consumer()
            print(f" Subscribed to topic: {INPUT_TOPIC}")
            
            for message in consumer:
                data = message.value
                product_id = data.get('product_id', 'unknown')
                
                with buffer_lock:
                    review_buffer[product_id].append(data)
                    
                    if product_id not in batch_start_time:
                        batch_start_time[product_id] = time.time()
                
                with buffer_lock:
                    batch_ready = (
                        len(review_buffer[product_id]) >= BATCH_SIZE or
                        time.time() - batch_start_time.get(product_id, 0) > BATCH_TIMEOUT
                    )
                
                if batch_ready:
                    print(f" Batch ready for {product_id} ({len(review_buffer[product_id])} reviews)")
                    process_batch(product_id)
                    batch_start_time.pop(product_id, None)
                
                current_time = time.time()
                with buffer_lock:
                    products_to_process = []
                    for pid in list(review_buffer.keys()):
                        if pid != product_id and review_buffer[pid]:                                       
                            if current_time - batch_start_time.get(pid, 0) > BATCH_TIMEOUT:
                                products_to_process.append(pid)
                
                for pid in products_to_process:
                    print(f" Timeout triggered for {pid} ({len(review_buffer.get(pid, []))} reviews)")
                    process_batch(pid)
                    batch_start_time.pop(pid, None)
            
            print(f" Consumer poll timeout. Checking for remaining batches...")
            with buffer_lock:
                remaining_products = [pid for pid in list(review_buffer.keys()) if review_buffer[pid]]
            
            for product_id in remaining_products:
                print(f" Processing remaining batch for {product_id} ({len(review_buffer.get(product_id, []))} reviews)")
                process_batch(product_id)
                batch_start_time.pop(product_id, None)
                    
        except Exception as e:
            print(f" Consumer error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)

if __name__ == "__main__":
    run_service()
