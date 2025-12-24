"""
Kafka Producer: Simulates real-time review streaming from e-commerce platform.
Reads enriched data and publishes to 'raw_reviews' topic with a delay.
"""
import json
import time
import random
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

import os

# Configuration
# Support Docker networking (kafka:9092) vs Localhost
default_bootstrap = 'localhost:9092'
if os.path.exists('/.dockerenv'): # Simple check, or just rely on env var
    default_bootstrap = 'kafka:29092'

KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', default_bootstrap)
TOPIC_NAME = 'raw_reviews'

# Path handling: Relative to this script's location
# kafka/producer.py -> parent -> data/label/absa_enriched.xlsx
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENRICHED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'label', 'absa_enriched.xlsx')

def create_producer(max_retries=5, retry_delay=5):
    """Create Kafka producer with retry logic."""
    for attempt in range(max_retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            print(f"✅ Connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
            return producer
        except NoBrokersAvailable:
            print(f"⏳ Attempt {attempt + 1}/{max_retries}: Kafka not ready, retrying in {retry_delay}s...")
            time.sleep(retry_delay)
    raise Exception("❌ Could not connect to Kafka after multiple attempts")

def load_data():
    """Load enriched data."""
    df = pd.read_excel(ENRICHED_DATA_PATH)
    print(f"📥 Loaded {len(df)} reviews from enriched dataset")
    return df

def stream_reviews(delay_min=0.5, delay_max=2.0, limit=None):
    """
    Stream reviews to Kafka topic.
    Args:
        delay_min/max: Random delay between messages (simulates real-time)
        limit: Maximum number of reviews to send (None = all)
    """
    producer = create_producer()
    df = load_data()
    
    # Shuffle to simulate random order
    df = df.sample(frac=1).reset_index(drop=True)
    
    if limit:
        df = df.head(limit)
    
    print(f"🚀 Starting to stream {len(df)} reviews to topic '{TOPIC_NAME}'...")
    
    for idx, row in df.iterrows():
        # Prepare message payload
        # Only include necessary fields for the consumer
        message = {
            'review_id': idx,
            'reviewContent': str(row.get('reviewContent', '')),
            'Product_ID': row.get('Product_ID', 'Unknown'),
            'Product_Category': row.get('Product_Category', 'Unknown'),
            'timestamp': time.time()
        }
        
        # Use Product_ID as key for partitioning
        key = message['Product_ID']
        
        producer.send(TOPIC_NAME, key=key, value=message)
        
        if (idx + 1) % 10 == 0:
            print(f"📤 Sent {idx + 1} reviews...")
        
        # Simulate real-time delay
        time.sleep(random.uniform(delay_min, delay_max))
    
    producer.flush()
    producer.close()
    print(f"✅ Completed! Sent {len(df)} reviews to Kafka topic '{TOPIC_NAME}'")

if __name__ == "__main__":
    # Stream a limited number for testing
    stream_reviews(delay_min=0.5, delay_max=1.0, limit=50)
