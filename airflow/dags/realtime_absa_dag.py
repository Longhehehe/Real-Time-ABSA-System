"""
Airflow DAG: Real-time ABSA Pipeline
Orchestrates the Producer -> Kafka -> Consumer flow for sentiment analysis.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from datetime import timedelta
import os
import json
import time
from collections import defaultdict

default_args = {
    'owner': 'absa_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Configuration
PROJECT_DIR = '/opt/airflow/project'
PREDICTIONS_DIR = os.path.join(PROJECT_DIR, 'data', 'predictions')


def trigger_producer(**context):
    """
    Task 1: Send reviews to Kafka.
    Option A: Receive pre-crawled reviews from Streamlit via params.
    Option B: Crawl from URL if provided and no reviews data.
    """
    import sys
    sys.path.insert(0, PROJECT_DIR)
    sys.path.insert(0, os.path.join(PROJECT_DIR, 'app'))
    
    from lazada_producer import send_reviews_to_kafka
    
    # Get params
    product_id = context['params'].get('product_id', 'airflow_product')
    product_url = context['params'].get('product_url', '')
    max_reviews = context['params'].get('max_reviews', 100)
    reviews_data = context['params'].get('reviews', [])  # Pre-crawled reviews from Streamlit
    
    reviews = []
    
    # Option A: Use pre-crawled reviews if provided
    if reviews_data:
        print(f"📦 Received {len(reviews_data)} pre-crawled reviews from Streamlit")
        reviews = reviews_data
    
    # Option B: Crawl from URL if no reviews provided
    elif product_url and product_url != 'https://www.lazada.vn/products/...':
        from lazada_crawler import crawl_reviews
        print(f"🔍 Crawling reviews from: {product_url}")
        
        reviews, error = crawl_reviews(
            product_url=product_url,
            cookies_path=os.path.join(PROJECT_DIR, 'cookie', 'lazada_cookies.txt'),
            max_reviews=60,  # Reduced for faster real-time demo
            delay_min=1.0,
            delay_max=2.0,
            item_id=product_id
        )
        
        if error:
            print(f"⚠️ Crawl warning: {error}")
    
    if not reviews:
        raise Exception("No reviews to process! Provide either 'reviews' data or a valid 'product_url'.")

    # Deduplicate reviews using Pandas (Robust)
    import pandas as pd
    
    if reviews:
        df = pd.DataFrame(reviews)
        initial_count = len(df)
        
        # Ensure reviewContent exists
        if 'reviewContent' in df.columns:
            # 1. Normalize content (strip whitespace)
            df['reviewContent'] = df['reviewContent'].fillna('').astype(str).str.strip()
            
            # 2. Create Fuzzy Signature (Lowercase + Remove non-alphanumeric)
            # This catches "Good product." vs "good product" vs "Good  product!"
            df['dedup_key'] = df['reviewContent'].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.replace(r'\s+', '', regex=True)
            
            # 3. Drop duplicates based on this strict signature
            # Also keep subset=['reviewContent'] just in case, but dedup_key is stronger
            df.drop_duplicates(subset=['dedup_key'], keep='first', inplace=True)
            
            # 4. Remove empty content
            df = df[df['reviewContent'] != '']
            
            # Clean up temp col
            df.drop(columns=['dedup_key'], inplace=True)
        
        final_count = len(df)
        if final_count < initial_count:
            print(f"⚠️ Removed {initial_count - final_count} duplicate reviews via Pandas.")
            
        # Save to CSV Buffer (User Request)
        # Save to CSV Buffer (User Request) - Unique per product to allow concurrency
        buffer_file = os.path.join(PROJECT_DIR, 'data', f'crawled_buffer_{product_id}.csv')
        os.makedirs(os.path.dirname(buffer_file), exist_ok=True)
        df.to_csv(buffer_file, index=False, encoding='utf-8-sig')
        print(f"💾 Saved unique reviews to buffer: {buffer_file}")
        
        reviews = df.to_dict('records')

    print(f"📊 Processing {len(reviews)} unique reviews")
    
    # Clear old predictions
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    pred_file = os.path.join(PREDICTIONS_DIR, f"{product_id}.json")
    if os.path.exists(pred_file):
        try:
            os.remove(pred_file)
        except OSError as e:
            print(f"⚠️ Could not remove old predictions (PermissionError): {e}")
            try:
                # Try to truncate if delete fails
                with open(pred_file, 'w'): pass
                print("✅ Truncated old prediction file instead.")
            except:
                print("❌ Could not truncate file either. Old data may persist.")
    
    # Send to Kafka
    success = send_reviews_to_kafka(product_id, reviews)
    
    if not success:
        raise Exception("Failed to send reviews to Kafka!")
    
    # Store info for downstream tasks
    context['ti'].xcom_push(key='product_id', value=product_id)
    context['ti'].xcom_push(key='review_count', value=len(reviews))
    
    print(f"✅ Sent {len(reviews)} reviews for product {product_id}")


def wait_for_consumer(**context):
    """
    Task 2: Poll and wait for Consumer to finish processing.
    """
    product_id = context['ti'].xcom_pull(key='product_id')
    expected_count = context['ti'].xcom_pull(key='review_count')
    
    pred_file = os.path.join(PREDICTIONS_DIR, f"{product_id}.json")
    
    max_wait = 3600  # 60 minutes (Safe for backlog)
    poll_interval = 2  # Faster polling for quicker UI feedback
    elapsed = 0
    
    # Helper to print results
    def print_results(data_chunk):
        # Print Detailed Predictions (Per Review)
        print("\n" + "="*50)
        print(f"📝 DETAILED PREDICTIONS ({len(data_chunk)} reviews)")
        print("="*50)
        
        ASPECTS = ['Chất lượng sản phẩm', 'Trải nghiệm sử dụng', 'Đúng mô tả sản phẩm', 
                  'Hiệu năng sản phẩm', 'Giá cả', 'Khuyến mãi & voucher',
                  'Vận chuyển & giao hàng', 'Đóng gói & bao bì', 'Uy tín & thái độ shop',
                  'Dịch vụ chăm sóc khách hàng', 'Lỗi & bảo hành & hàng giả', 'Đổi trả & bảo hành']
        
        # Map labels to text
        LABEL_MAP = {1: 'POS', 0: 'NEU', -1: 'NEG'}
        
        for idx, item in enumerate(data_chunk):
            # Safe text handling (handle None)
            raw_text = item.get('original_text')
            text = str(raw_text) if raw_text is not None else ""
            text = text.replace('\n', ' ')
            
            if len(text) > 100:
                text = text[:97] + "..."
            
            print(f"\n[Review {idx+1}] {text}")
            sentiment = item.get('sentiment', {})
            
            if not sentiment:
                print("  (No sentiment detected)")
            else:
                for aspect in ASPECTS:
                    label = sentiment.get(aspect)
                    if label is not None:
                        label_str = LABEL_MAP.get(label, str(label))
                        print(f"  - {aspect}: {label_str}")
        
        print("\n" + "="*50)
        
        # Print Prediction Summary
        print(f"📊 PREDICTION RESULTS SUMMARY")
        print("="*50)
        
        # Initialize counts for POS, NEU, NEG
        aspects = defaultdict(lambda: {'POS': 0, 'NEU': 0, 'NEG': 0})
        for item in data_chunk:
            sentiment = item.get('sentiment', {})
            for aspect, label in sentiment.items():
                if label == 1:
                    aspects[aspect]['POS'] += 1
                elif label == 0:
                    aspects[aspect]['NEU'] += 1
                elif label == -1:
                    aspects[aspect]['NEG'] += 1
                
        print(f"{'ASPECT':<30} | {'POS':<5} | {'NEU':<5} | {'NEG':<5}")
        print("-" * 55)
        for aspect, counts in aspects.items():
            print(f"{aspect:<30} | {counts['POS']:<5} | {counts['NEU']:<5} | {counts['NEG']:<5}")
        print("="*50 + "\n")

    elapsed = 0
    while elapsed < max_wait:
        if os.path.exists(pred_file):
            try:
                with open(pred_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if len(data) >= expected_count:
                    print(f"✅ Consumer finished! Processed {len(data)} reviews.")
                    print_results(data)
                    return True
                    
                print(f"⏳ Processed {len(data)}/{expected_count}...")
            except json.JSONDecodeError:
                pass
        
        # Poll logic
        time.sleep(poll_interval)
        elapsed += poll_interval
    
    # Timeout occurred
    print(f"⚠️ Timeout! Printing partial results ({len(data) if 'data' in locals() else 0}/{expected_count})...")
    if 'data' in locals() and data:
        print_results(data)
        
    raise Exception(f"Timeout waiting for Consumer! Only got {len(data) if 'data' in locals() else 0}/{expected_count}")


def aggregate_results(**context):
    """
    Task 3: Aggregate and summarize results.
    """
    product_id = context['ti'].xcom_pull(key='product_id')
    pred_file = os.path.join(PREDICTIONS_DIR, f"{product_id}.json")
    
    with open(pred_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # Count sentiments per aspect
    summary = {}
    for pred in predictions:
        sentiment = pred.get('sentiment', {})
        for aspect, score in sentiment.items():
            if aspect not in summary:
                summary[aspect] = {'positive': 0, 'negative': 0, 'neutral': 0}
            
            if score == 1:
                summary[aspect]['positive'] += 1
            elif score == -1:
                summary[aspect]['negative'] += 1
            elif score == 0:
                summary[aspect]['neutral'] += 1
    
    # Save summary
    summary_file = os.path.join(PREDICTIONS_DIR, f"{product_id}_summary.json")
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"✅ Aggregation complete! Summary saved to {summary_file}")
    except PermissionError:
        print(f"⚠️ PermissionError: Could not save summary to {summary_file}. Printing to stdout instead.")
    except Exception as e:
        print(f"⚠️ Error saving summary: {e}")
    
    print(json.dumps(summary, indent=2, ensure_ascii=False))


with DAG(
    'realtime_absa_pipeline',
    default_args=default_args,
    description='Orchestrate Producer -> Kafka -> Consumer -> Aggregate',
    schedule_interval=None,  # Manual trigger only
    start_date=days_ago(1),
    catchup=False,
    tags=['absa', 'kafka', 'realtime', 'crawl'],
    params={
        'product_id': 'lazada_product',
        'product_url': 'https://www.lazada.vn/products/...',  # Paste Lazada URL here
        'max_reviews': 50,
        'max_reviews': 50,
    },
    max_active_runs=16, # Allow multiple concurrent runs
    concurrency=16,
) as dag:
    
    # Task 1: Trigger Producer
    trigger_producer_task = PythonOperator(
        task_id='trigger_producer',
        python_callable=trigger_producer,
        provide_context=True,
    )
    
    # Task 2: Wait for Consumer
    wait_consumer_task = PythonOperator(
        task_id='wait_for_consumer',
        python_callable=wait_for_consumer,
        provide_context=True,
    )
    
    # Task 3: Aggregate Results
    aggregate_task = PythonOperator(
        task_id='aggregate_results',
        python_callable=aggregate_results,
        provide_context=True,
    )
    
    # Define task dependencies
    trigger_producer_task >> wait_consumer_task >> aggregate_task
