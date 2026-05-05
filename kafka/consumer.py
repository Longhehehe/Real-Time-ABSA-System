"""
Kafka Consumer: Consumes reviews from 'raw_reviews' topic, runs ML prediction,
and publishes results to 'predictions' topic.
"""
import json
import os
import sys
import time
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base_path = os.path.join(BASE_DIR, 'prepro', '23520932_23520903_20520692_src', '23520932_23520903_20520692_src')
sys.path.insert(0, base_path)

import os
                           
default_bootstrap = 'localhost:9092'
if os.path.exists('/.dockerenv'):
    default_bootstrap = 'kafka:29092'
    
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', default_bootstrap)
INPUT_TOPIC = 'raw_reviews'
OUTPUT_TOPIC = 'predictions'
CONSUMER_GROUP = 'ml_prediction_group'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model')

def check_and_train_model():
    """Check if model exists, train if not."""
    model_file = os.path.join(MODEL_PATH, 'model.pkl')
    
    if os.path.exists(MODEL_PATH):
        subdirs = [d for d in os.listdir(MODEL_PATH) if os.path.isdir(os.path.join(MODEL_PATH, d))]
        for subdir in subdirs:
            potential_model = os.path.join(MODEL_PATH, subdir, 'model.pkl')
            if os.path.exists(potential_model):
                print(f" Found existing model at {potential_model}")
                return potential_model
    
    if not os.path.exists(model_file):
        print(" No model found. Running training pipeline...")
                                 
        sys.path.insert(0, BASE_DIR)
        from train_pipeline import run_training
        run_training()
        
        if os.path.exists(MODEL_PATH):
            subdirs = [d for d in os.listdir(MODEL_PATH) if os.path.isdir(os.path.join(MODEL_PATH, d))]
            for subdir in sorted(subdirs, reverse=True):              
                potential_model = os.path.join(MODEL_PATH, subdir, 'model.pkl')
                if os.path.exists(potential_model):
                    return potential_model
    
    return model_file

def create_consumer(max_retries=5, retry_delay=5):
    """Create Kafka consumer with retry logic."""
    for attempt in range(max_retries):
        try:
            consumer = KafkaConsumer(
                INPUT_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                group_id=CONSUMER_GROUP,
                auto_offset_reset='earliest',
                value_deserializer=lambda v: json.loads(v.decode('utf-8'))
            )
            print(f" Consumer connected to Kafka, subscribed to '{INPUT_TOPIC}'")
            return consumer
        except NoBrokersAvailable:
            print(f" Attempt {attempt + 1}/{max_retries}: Kafka not ready, retrying in {retry_delay}s...")
            time.sleep(retry_delay)
    raise Exception(" Could not connect to Kafka")

def create_output_producer():
    """Create producer for output topic."""
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8')
    )

def run_consumer():
    """Main consumer loop."""
                                 
    model_path = check_and_train_model()
    print(f"🧠 Using model: {model_path}")
    
    consumer = create_consumer()
    producer = create_output_producer()
    
    print(" Waiting for messages...")
    
    processed_count = 0
    for message in consumer:
        data = message.value
        
        review_content = data.get('reviewContent', '')
        
        sentiment = simulate_prediction(review_content)
        
        result = {
            'review_id': data.get('review_id'),
            'reviewContent': review_content,
            'Product_ID': data.get('Product_ID'),             
            'Product_Category': data.get('Product_Category'),             
            'predicted_sentiment': sentiment,
            'processed_at': time.time()
        }
        
        producer.send(OUTPUT_TOPIC, value=result)
        
        processed_count += 1
        if processed_count % 10 == 0:
            print(f" Processed {processed_count} reviews...")
    
    consumer.close()
    producer.close()

def simulate_prediction(text):
    """Simulate ML prediction based on keywords (for demo purposes)."""
    positive_keywords = ['tốt', 'đẹp', 'nhanh', 'chất lượng', 'hài lòng', 'good', 'great']
    negative_keywords = ['xấu', 'chậm', 'kém', 'tệ', 'thất vọng', 'bad', 'poor']
    
    text_lower = text.lower()
    
    pos_count = sum(1 for kw in positive_keywords if kw in text_lower)
    neg_count = sum(1 for kw in negative_keywords if kw in text_lower)
    
    if pos_count > neg_count:
        return 'POSITIVE'
    elif neg_count > pos_count:
        return 'NEGATIVE'
    else:
        return 'NEUTRAL'

if __name__ == "__main__":
    run_consumer()
