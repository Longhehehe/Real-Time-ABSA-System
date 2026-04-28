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
BATCH_SIZE = 10  # Reduced for faster real-time feedback
BATCH_TIMEOUT = 15  # Reduced timeout for quicker batch processing

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_DIR = os.path.join(PROJECT_DIR, 'data', 'predictions')
SPARK_MASTER = os.environ.get('SPARK_MASTER', 'spark://spark-master:7077')

ASPECTS = [
    'Chất lượng sản phẩm',
    'Hiệu năng & Trải nghiệm',
    'Đúng mô tả',
    'Giá cả & Khuyến mãi',
    'Vận chuyển',
    'Đóng gói',
    'Dịch vụ & Thái độ Shop',
    'Bảo hành & Đổi trả',
    'Tính xác thực',
]
SENTIMENT_NAMES = ['NEG', 'POS', 'NEU']

# Buffer for batching
review_buffer = defaultdict(list)  # (product_id, model) -> [reviews]
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
        # Check if active session exists
        spark = SparkSession.builder.getOrCreate()
        if spark.sparkContext.master == "local[*]": # If we got a default one not configured right
             pass # Maybe re-config? Usually safe to just stick with existing or create new if None.
        
        if spark is None or spark.sparkContext.isStopped:
            print("⚡ Initializing Spark Session...")
            spark = SparkSession.builder \
                .appName("ABSA_Consumer_Service") \
        # Optimize for single-machine docker environment
        print("⚡ Initializing Spark Session...")
        spark = SparkSession.builder \
            .appName("ABSAConsumer") \
            .master(SPARK_MASTER) \
            .config("spark.executor.memory", "2g") \
            .config("spark.driver.memory", "1g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        print("⚡ Spark Session Ready")
    return spark

# --- Define UDFs GLOBALLY to avoid closure/pickling overhead ---
# We need to wrap them to ensure imports happen on worker

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

def _predict_model_logic(texts, models=None):
    import pandas as pd
    import json
    import os
    import sys

    def _ensure_paths():
        current_dir = os.getcwd()
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for path in [current_dir, project_root]:
            if path not in sys.path:
                sys.path.append(path)
        return project_root if os.path.isdir(project_root) else current_dir

    project_root = _ensure_paths()

    def _normalize_model_name(raw):
        if not raw:
            return None
        key = str(raw).strip().lower()
        aliases = {
            'phobert': 'phobert_multipolarity',
            'phobert_multipolarity': 'phobert_multipolarity',
            'phobert_legacy': 'phobert_legacy',
            'logistic': 'logistic_regression',
            'logistic_regression': 'logistic_regression',
            'naive': 'naive_bayes',
            'naive_bayes': 'naive_bayes',
            'bilstm': 'bilstm',
            'cnn_bilstm': 'cnn_bilstm',
            'xlm': 'xlm_roberta',
            'xlm_roberta': 'xlm_roberta',
            'ollama': 'ollama',
        }
        return aliases.get(key, key)

    def _get_target_model():
        if models is not None and len(models) > 0:
            candidate_value = models.iloc[0]
            try:
                if pd.isna(candidate_value):
                    candidate_value = None
            except Exception:
                pass
            candidate = _normalize_model_name(candidate_value)
            if candidate:
                return candidate

        config_path = os.path.join(project_root, "model_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                candidate = _normalize_model_name(cfg.get("active_model"))
                if candidate:
                    return candidate
            except Exception:
                pass

        return "phobert_multipolarity"

    def _format_aspect_predictions(mention_scores, sentiment_scores, threshold: float = 0.5):
        results = {}
        for idx, aspect in enumerate(ASPECTS):
            mentioned = float(mention_scores[idx]) > threshold
            sentiments = []
            if mentioned:
                for s_idx, score in enumerate(sentiment_scores[idx]):
                    if float(score) > threshold:
                        sentiments.append(SENTIMENT_NAMES[s_idx])
                if not sentiments:
                    sentiments = ['NEU']

            results[aspect] = {
                'mentioned': bool(mentioned),
                'sentiments': sentiments,
            }
        return results

    if not hasattr(_predict_model_logic, 'state'):
        _predict_model_logic.state = {
            'ml': {},
            'torch': {},
            'phobert': {},
            'ollama': None,
        }

    state = _predict_model_logic.state
    model_key = _get_target_model()

    try:
        if model_key in ('logistic_regression', 'naive_bayes'):
            import pickle
            from methods.ml_models import LogisticRegressionABSA, NaiveBayesABSA

            ml_paths = {
                'logistic_regression': os.path.join(project_root, 'models', 'logistic_regression_absa', 'logistic_regression_model.pkl'),
                'naive_bayes': os.path.join(project_root, 'models', 'naive_bayes_absa', 'naive_bayes_model.pkl'),
            }
            ml_classes = {
                'logistic_regression': LogisticRegressionABSA,
                'naive_bayes': NaiveBayesABSA,
            }

            if model_key not in state['ml']:
                model_path = ml_paths[model_key]
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Missing ML model: {model_path}")
                with open(model_path, 'rb') as f:
                    payload = pickle.load(f)
                model = ml_classes[model_key]()
                model.tfidf = payload['tfidf']
                model.mention_clfs = payload['mention_clfs']
                model.sentiment_clfs = payload['sentiment_clfs']
                state['ml'][model_key] = model

            model = state['ml'][model_key]
            text_list = texts.tolist()
            X = model.tfidf.transform(text_list)
            pred_m, pred_s, _, _ = model.predict(X)
            results = [_format_aspect_predictions(pred_m[i], pred_s[i]) for i in range(len(text_list))]

        elif model_key in ('bilstm', 'cnn_bilstm', 'xlm_roberta'):
            import torch
            from transformers import AutoTokenizer
            from methods.deep_models import BiLSTMForABSA, CNNBiLSTMForABSA
            from methods.transformer_models import XLMRoBERTaForABSA

            torch_specs = {
                'bilstm': {
                    'path': os.path.join(project_root, 'models', 'bilstm_absa', 'bilstmforabsa_absa.pt'),
                    'class': BiLSTMForABSA,
                    'tokenizer': 'vinai/phobert-base',
                    'needs_vocab': True,
                },
                'cnn_bilstm': {
                    'path': os.path.join(project_root, 'models', 'cnn_bilstm_absa', 'cnnbilstmforabsa_absa.pt'),
                    'class': CNNBiLSTMForABSA,
                    'tokenizer': 'vinai/phobert-base',
                    'needs_vocab': True,
                },
                'xlm_roberta': {
                    'path': os.path.join(project_root, 'models', 'xlm_roberta_absa', 'xlmrobertaforabsa_absa.pt'),
                    'class': XLMRoBERTaForABSA,
                    'tokenizer': 'xlm-roberta-base',
                    'needs_vocab': False,
                },
            }

            if model_key not in state['torch']:
                spec = torch_specs[model_key]
                if not os.path.exists(spec['path']):
                    raise FileNotFoundError(f"Missing torch model: {spec['path']}")

                device = "cuda" if torch.cuda.is_available() else "cpu"
                tokenizer = AutoTokenizer.from_pretrained(spec['tokenizer'])
                if spec['needs_vocab']:
                    model = spec['class'](vocab_size=tokenizer.vocab_size)
                else:
                    model = spec['class'](num_aspects=len(ASPECTS))

                try:
                    checkpoint = torch.load(spec['path'], map_location=device, weights_only=False)
                except TypeError:
                    checkpoint = torch.load(spec['path'], map_location=device)
                state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
                model.load_state_dict(state_dict)
                model = model.to(device)
                model.eval()

                state['torch'][model_key] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'device': device,
                }

            payload = state['torch'][model_key]
            tokenizer = payload['tokenizer']
            model = payload['model']
            device = payload['device']

            text_list = texts.tolist()
            encoding = tokenizer(
                text_list,
                truncation=True,
                max_length=256,
                padding='max_length',
                return_tensors='pt',
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            with torch.no_grad():
                logits_m, logits_s = model(input_ids, attention_mask)
                probs_m = torch.sigmoid(logits_m).cpu().numpy()
                probs_s = torch.sigmoid(logits_s).cpu().numpy()

            results = [_format_aspect_predictions(probs_m[i], probs_s[i]) for i in range(len(text_list))]

        elif model_key == 'ollama':
            if state['ollama'] is None:
                from app.ollama_predictor import OllamaPredictor
                state['ollama'] = OllamaPredictor()

            predictor = state['ollama']
            text_list = texts.tolist()
            results = []
            for text in text_list:
                try:
                    res = predictor.predict_single(text)
                    if isinstance(res, dict) and 'multipolarity' in res:
                        results.append(res['multipolarity'])
                    else:
                        results.append(res)
                except Exception:
                    results.append({asp: {'mentioned': False, 'sentiments': []} for asp in ASPECTS})

        else:
            from app.absa_predictor import PhoBERTPredictor, MODEL_PATH, MODEL_PATH_OLD

            phobert_key = model_key if model_key in ('phobert_legacy', 'phobert_multipolarity') else 'phobert_multipolarity'
            if phobert_key not in state['phobert']:
                model_path = MODEL_PATH_OLD if phobert_key == 'phobert_legacy' else MODEL_PATH
                predictor = PhoBERTPredictor(model_path=model_path, use_multipolarity=phobert_key != 'phobert_legacy')
                state['phobert'][phobert_key] = predictor

            predictor = state['phobert'][phobert_key]
            text_list = texts.tolist()
            if hasattr(predictor, 'predict_batch'):
                batch_results = predictor.predict_batch(text_list, format='multipolarity')
                results = [res if isinstance(res, dict) else {} for res in batch_results]
            else:
                results = []
                for text in text_list:
                    try:
                        res = predictor.predict_single(text)
                        results.append(res.get('multipolarity', res))
                    except Exception:
                        results.append({asp: {'mentioned': False, 'sentiments': []} for asp in ASPECTS})

        return pd.Series([json.dumps(res, ensure_ascii=False) for res in results])

    except Exception as e:
        error_payload = json.dumps({'error': f'Inference failure: {e}'})
        return pd.Series([error_payload] * len(texts))


def run_spark_prediction(product_id: str, reviews: List[Dict], model: str):
    """
    Trigger Spark job to predict batch of reviews using Pandas UDF.
    """
    try:
        from pyspark.sql.functions import pandas_udf, col
        from pyspark.sql.types import StringType
        
        print(f"🚀 Starting Spark job for {len(reviews)} reviews (Product: {product_id})")
        
        spark = get_spark_session()
        
        # Prepare data - Handle multiple possible keys for review text
        # Producer sends 'review_content', so check that FIRST
        data = []
        skipped_empty = 0
        for r in reviews:
            text = r.get('review_content') or r.get('review_text') or r.get('reviewContent') or r.get('content') or ''
            text = str(text).strip() if text else ''
            
            # Log and skip empty reviews
            if not text:
                skipped_empty += 1
                print(f"⚠️ Skipping empty review: id={r.get('review_id', 'N/A')}")
                continue
                
            data.append((text, r.get('rating', 0), r.get('review_id', ''), model))
        
        if skipped_empty > 0:
            print(f"⚠️ Skipped {skipped_empty} empty reviews out of {len(reviews)}")
        
        if not data:
            print(f"⚠️ No valid reviews to process after filtering!")
            return
            
        df = spark.createDataFrame(data, ["review_text", "rating", "review_id", "model"])
        
        # Register UDFs
        # Note: Using decorators works, but assigning function prevents re-decoration overhead issue sometimes
        preprocess_udf = pandas_udf(StringType())(_preprocess_text_logic)
        predict_model_udf = pandas_udf(StringType())(_predict_model_logic)
        
        # Apply UDFs
        # Repartition to 1 to ensure we don't scatter 10 items across 10 tasks if unnecessary
        # But we want parallelism if batch > 1. 10 items -> maybe 2 partitions?
        # Actually, let Spark decide or force 1 for small batch to keep it in one executor (better for caching)
        df_result = df \
            .repartition(1) \
            .withColumn("cleaned_text", preprocess_udf(col("review_text"))) \
            .withColumn("sentiment_json", predict_model_udf(col("cleaned_text"), col("model")))
        
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
        
        # Save predictions
        save_predictions(product_id, predictions)
        print(f"✅ Prediction processing complete for {len(predictions)} reviews")
        
    except Exception as e:
        print(f"❌ Spark job failed: {e}")
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


def process_batch(buffer_key):
    """Process buffered reviews for a product/model pair."""
    global review_buffer
    
    with buffer_lock:
        reviews = review_buffer.pop(buffer_key, [])
    
    if reviews:
        product_id, model = buffer_key
        run_spark_prediction(product_id, reviews, model)


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
    
    # Use global batch_start_time to persist across reconnects
    batch_start_time = {}
    
    while True:
        try:
            consumer = create_consumer()
            print(f"✅ Subscribed to topic: {INPUT_TOPIC}")
            
            for message in consumer:
                data = message.value
                product_id = data.get('product_id', 'unknown')
                model = data.get('model') or 'phobert_multipolarity'
                buffer_key = (product_id, model)
                
                with buffer_lock:
                    review_buffer[buffer_key].append(data)
                    
                    if buffer_key not in batch_start_time:
                        batch_start_time[buffer_key] = time.time()
                
                # Check if THIS product's batch is ready
                with buffer_lock:
                    batch_ready = (
                        len(review_buffer[buffer_key]) >= BATCH_SIZE or
                        time.time() - batch_start_time.get(buffer_key, 0) > BATCH_TIMEOUT
                    )
                
                if batch_ready:
                    print(f"📦 Batch ready for {product_id} | {model} ({len(review_buffer[buffer_key])} reviews)")
                    process_batch(buffer_key)
                    batch_start_time.pop(buffer_key, None)
                
                # CRITICAL FIX: Also check ALL OTHER products for timeout
                # This prevents reviews from being orphaned when messages interleave
                current_time = time.time()
                with buffer_lock:
                    products_to_process = []
                    for key in list(review_buffer.keys()):
                        if key != buffer_key and review_buffer[key]:  # Skip current, already handled above
                            if current_time - batch_start_time.get(key, 0) > BATCH_TIMEOUT:
                                products_to_process.append(key)
                
                # Process timed-out batches for other products (outside lock)
                for key in products_to_process:
                    pid, model_name = key
                    print(f"📦 Timeout triggered for {pid} | {model_name} ({len(review_buffer.get(key, []))} reviews)")
                    process_batch(key)
                    batch_start_time.pop(key, None)
            
            # Process remaining batches after consumer timeout (no new messages for consumer_timeout_ms)
            print(f"⏰ Consumer poll timeout. Checking for remaining batches...")
            with buffer_lock:
                remaining_products = [key for key in list(review_buffer.keys()) if review_buffer[key]]
            
            for key in remaining_products:
                pid, model_name = key
                print(f"📦 Processing remaining batch for {pid} | {model_name} ({len(review_buffer.get(key, []))} reviews)")
                process_batch(key)
                batch_start_time.pop(key, None)
                    
        except Exception as e:
            print(f"❌ Consumer error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    run_service()
