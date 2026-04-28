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
    
    from app.lazada_producer import send_reviews_to_kafka
    
    # Get params - API trigger uses dag_run.conf, manual uses params
    dag_run = context.get('dag_run')
    conf = dag_run.conf if dag_run and dag_run.conf else {}
    params = context.get('params', {})
    
    # Merge conf (priority) with params (fallback)
    product_id = conf.get('product_id') or params.get('product_id', 'airflow_product')
    product_url = conf.get('product_url') or params.get('product_url', '')
    max_reviews = conf.get('max_reviews') or params.get('max_reviews', 100)
    model = conf.get('model') or params.get('model', 'phobert_multipolarity')
    reviews_data = conf.get('reviews') or params.get('reviews', [])  # Pre-crawled reviews
    
    print(f"📋 Config received: product_id={product_id}, product_url={product_url[:50] if product_url else 'N/A'}...")
    
    reviews = []
    
    # Option A: Use pre-crawled reviews if provided
    if reviews_data:
        print(f"📦 Received {len(reviews_data)} pre-crawled reviews from Streamlit")
        reviews = reviews_data
    
    # Option B: Crawl from URL if no reviews provided
    elif product_url and product_url != 'https://www.lazada.vn/products/...':
        print(f"🔍 Crawling reviews from: {product_url}")
        
        # Find cookies path
        cookies_path = None
        for path in [
            os.path.join(PROJECT_DIR, 'cookie', 'lazada_cookies.json'),
            os.path.join(PROJECT_DIR, 'lazada_crawler', 'lazada_cookies.json'),
            os.path.join(PROJECT_DIR, 'app', 'cookie', 'lazada_cookies.json'),
        ]:
            if os.path.exists(path):
                cookies_path = path
                print(f"✅ Found cookies at {cookies_path}")
                break
        
        # Use simple requests-based crawler with balanced mode
        try:
            import sys
            from app.lazada_crawler_simple import crawl_reviews_simple
            
            print("⚖️ Using requests-based crawler with BALANCED MODE...")
            reviews, error = crawl_reviews_simple(
                product_url=product_url,
                cookies_path=cookies_path,
                max_reviews=max_reviews,
                item_id=product_id,
                balanced_mode=True  # Enable balanced crawling
            )
            
            if error:
                print(f"⚠️ Crawler warning: {error}")
            
            print(f"✅ Crawled {len(reviews)} reviews with balanced mode")
                
        except Exception as e:
            print(f"❌ Crawler failed: {e}")
            import traceback
            traceback.print_exc()
            reviews = []
    
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
        buffer_file = os.path.join(PROJECT_DIR, 'data', 'crawled_reviews_buffer.csv')
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
    
    # Send to Kafka (returns tuple: success, sent_count)
    result = send_reviews_to_kafka(product_id, reviews, model=model)
    
    # Handle both old (bool) and new (tuple) return format
    if isinstance(result, tuple):
        success, sent_count = result
    else:
        success = result
        sent_count = len(reviews)
    
    if not success:
        raise Exception("Failed to send reviews to Kafka!")
    
    # Store info for downstream tasks - use actual sent count, not original count
    context['ti'].xcom_push(key='product_id', value=product_id)
    context['ti'].xcom_push(key='review_count', value=sent_count)
    
    print(f"✅ Sent {sent_count} reviews for product {product_id}")


def wait_for_consumer(**context):
    """
    Task 2: Poll and wait for Consumer to finish processing.
    Includes stale detection - if no progress for STALE_TIMEOUT, assume done.
    """
    product_id = context['ti'].xcom_pull(key='product_id')
    expected_count = context['ti'].xcom_pull(key='review_count')
    
    pred_file = os.path.join(PREDICTIONS_DIR, f"{product_id}.json")
    
    max_wait = 1800  # 30 minutes (Increased because sequential processing is slow)
    poll_interval = 5
    stale_timeout = 60  # If no progress for 60 seconds, consider consumer done
    elapsed = 0
    last_count = 0
    stale_timer = 0
    
    # Helper to print results
    def print_results(data_chunk):
        # Print Detailed Predictions (Per Review)
        print("\n" + "="*50)
        print(f"📝 DETAILED PREDICTIONS ({len(data_chunk)} reviews)")
        print("="*50)
        
        ASPECTS = [
            'Chất lượng sản phẩm', 'Hiệu năng & Trải nghiệm', 'Đúng mô tả',
            'Giá cả & Khuyến mãi', 'Vận chuyển', 'Đóng gói',
            'Dịch vụ & Thái độ Shop', 'Bảo hành & Đổi trả', 'Tính xác thực'
        ]
        
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
                        # Handle Multi-Polarity Dictionary Format
                        if isinstance(label, dict):
                            # e.g., {'mentioned': True, 'sentiments': ['POS', 'NEG']}
                            sentiments = label.get('sentiments', [])
                            label_str = ", ".join(sentiments) if sentiments else "NEU"
                        else:
                            # Handle Legacy Integer Format
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
                
                # Normalize label to list of sentiments
                current_sentiments = []
                if isinstance(label, dict):
                     current_sentiments = label.get('sentiments') or []
                elif isinstance(label, int):
                    # Legacy mapping
                    if label == 1: current_sentiments = ['POS']
                    elif label == -1: current_sentiments = ['NEG']
                    elif label == 0: current_sentiments = ['NEU']
                
                # Count
                if not current_sentiments:
                     # If mentioned but empty, count as NEU or ignore? 
                     # Let's count as NEU if structure implies mention
                     pass 
                
                for s in current_sentiments:
                    if s in ['POS', 'NEG', 'NEU']:
                        aspects[aspect][s] += 1

        print(f"{'ASPECT':<30} | {'POS':<5} | {'NEU':<5} | {'NEG':<5}")
        print("-" * 55)
        for aspect, counts in aspects.items():
            print(f"{aspect:<30} | {counts['POS']:<5} | {counts['NEU']:<5} | {counts['NEG']:<5}")
        print("="*50 + "\n")

    elapsed = 0
    while elapsed < max_wait:
        current_count = 0
        if os.path.exists(pred_file):
            try:
                with open(pred_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                current_count = len(data)
                
                if current_count >= expected_count:
                    print(f"✅ Consumer finished! Processed {current_count} reviews.")
                    print_results(data)
                    return True
                
                # Stale detection: check if progress has stopped
                if current_count > last_count:
                    # Progress made, reset stale timer
                    last_count = current_count
                    stale_timer = 0
                else:
                    # No progress
                    stale_timer += poll_interval
                
                # If stale for too long and we have SOME data, accept partial results
                if stale_timer >= stale_timeout and current_count > 0:
                    print(f"⚠️ No progress for {stale_timeout}s. Consumer appears idle.")
                    print(f"📊 Accepting partial results: {current_count}/{expected_count} reviews")
                    print_results(data)
                    # Store actual count for downstream tasks
                    context['ti'].xcom_push(key='actual_count', value=current_count)
                    return True  # Accept partial instead of failing
                    
                print(f"⏳ Processed {current_count}/{expected_count}... (stale: {stale_timer}s)")
            except json.JSONDecodeError:
                pass
        
        # Poll logic
        time.sleep(poll_interval)
        elapsed += poll_interval
    
    # Timeout occurred
    print(f"⚠️ Timeout! Printing partial results ({current_count if 'current_count' in locals() else 0}/{expected_count})...")
    if 'data' in locals() and data:
        print_results(data)
        
    raise Exception(f"Timeout waiting for Consumer! Only got {current_count if 'current_count' in locals() else 0}/{expected_count}")


def aggregate_results(**context):
    """
    Task 3: Aggregate and summarize results.
    """
    product_id = context['ti'].xcom_pull(key='product_id')
    pred_file = os.path.join(PREDICTIONS_DIR, f"{product_id}.json")
    
    with open(pred_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # Count sentiments per aspect
    # Count sentiments per aspect
    summary = {}
    for pred in predictions:
        sentiment = pred.get('sentiment', {})
        for aspect, label in sentiment.items():
            if aspect not in summary:
                summary[aspect] = {'positive': 0, 'negative': 0, 'neutral': 0}
            
            # Normalize label
            current_sentiments = []
            if isinstance(label, dict):
                 current_sentiments = label.get('sentiments') or []
            elif isinstance(label, int):
                # Legacy mapping
                if label == 1: current_sentiments = ['POS']
                elif label == -1: current_sentiments = ['NEG']
                elif label == 0: current_sentiments = ['NEU']
            
            for s in current_sentiments:
                if s == 'POS':
                    summary[aspect]['positive'] += 1
                elif s == 'NEG':
                    summary[aspect]['negative'] += 1
                elif s == 'NEU':
                    summary[aspect]['neutral'] += 1
    
    # Save summary
    summary_file = os.path.join(PREDICTIONS_DIR, f"{product_id}_summary.json")
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Create .done marker file for API to know process is 100% complete
        done_file = os.path.join(PREDICTIONS_DIR, f"{product_id}.done")
        with open(done_file, 'w') as f:
            f.write("completed")
            
        print(f"✅ Aggregation complete. Summary saved to {summary_file}")
        print(f"✅ Created completion marker: {done_file}")
    except PermissionError:
        print(f"⚠️ PermissionError: Could not save summary to {summary_file} or create done file. Printing to stdout instead.")
    except Exception as e:
        print(f"⚠️ Error saving summary or creating done file: {e}")
    
    print(json.dumps(summary, indent=2, ensure_ascii=False))


with DAG(
    'realtime_absa_pipeline',
    default_args=default_args,
    description='Orchestrate Producer -> Kafka -> Consumer -> Aggregate',
    schedule_interval=None,  # Manual trigger only
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=10,  # Allow up to 10 parallel DAG runs
    concurrency=20,  # Allow up to 20 concurrent tasks across all runs
    tags=['absa', 'kafka', 'realtime', 'crawl'],
    params={
        'product_id': 'lazada_product',
        'product_url': 'https://www.lazada.vn/products/...',  # Paste Lazada URL here
        'max_reviews': 50,
    }
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
