"""
Lazada Producer Module
Sends crawled reviews to Kafka for asynchronous processing.
"""
import json
import os
import time
from typing import List, Dict
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# Configuration
KAFKA_BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
TOPIC_NAME = 'raw_reviews'

def create_producer():
    """Create Kafka Producer with retry."""
    try:
        return KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8')
        )
    except NoBrokersAvailable:
        print("⚠️ Kafka not ready, cannot create producer.")
        return None
    except Exception as e:
        print(f"❌ Error creating producer: {e}")
        return None

def send_reviews_to_kafka(product_id: str, reviews: List[Dict]):
    """
    Send a batch of reviews to Kafka.
    Returns tuple: (success, sent_count)
    """
    producer = create_producer()
    if not producer:
        return False, 0
    
    print(f"📤 Sending {len(reviews)} reviews for Product {product_id} to Kafka...")
    
    import uuid
    sent_count = 0
    skipped_count = 0
    
    for idx, review in enumerate(reviews):
        # Get review content
        content = review.get('review_text') or review.get('reviewContent') or review.get('content', '')
        content = str(content).strip() if content else ''
        
        # Skip empty content
        if not content:
            skipped_count += 1
            print(f"⚠️ Skipping review {idx}: empty content")
            continue
        
        # Generate robust ID: Use existing ID or Fallback to UUID
        # Using UUID is safer than timestamp for batch processing
        r_id = review.get('review_id')
        if not r_id:
            r_id = f"{product_id}_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
            
        message = {
            'product_id': product_id,
            'review_content': content,
            'rating': review.get('rating', 0),
            'review_id': str(r_id),
            'timestamp': time.time()
        }
        producer.send(TOPIC_NAME, value=message)
        sent_count += 1
    
    producer.flush()
    producer.close()
    
    if skipped_count > 0:
        print(f"⚠️ Skipped {skipped_count} empty reviews")
    print(f"✅ Sent {sent_count}/{len(reviews)} reviews successfully!")
    return True, sent_count
