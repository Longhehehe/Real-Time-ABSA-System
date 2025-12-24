"""
Trang So Sánh Sản Phẩm - Crawl reviews và hiển thị kết quả ABSA
"""
import streamlit as st
import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import product_manager as pm
import json
from lazada_crawler import crawl_reviews
# from lazada_producer import send_reviews_to_kafka  # Now using Airflow
try:
    from airflow_client import trigger_dag, get_dag_run_status, get_task_instances
except ImportError:
    from app.airflow_client import trigger_dag, get_dag_run_status, get_task_instances
from absa_predictor import (
    aggregate_scores,
    # get_predictor, # Handled by consumer
    ASPECTS, 
    SENTIMENT_MAP
)
import utils

# Page Config
st.set_page_config(
    page_title="So Sánh Sản Phẩm",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    h1 { color: #00CC96; }
    h3 { color: #FAFAFA; }
    .stProgress > div > div > div > div {
        background-color: #00CC96;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar: Cookie Management (Auto-update Feature)
# ---------------------------------------------------------
st.sidebar.markdown("### Quản lý Cookie")
col_c1, col_c2 = st.sidebar.columns(2)

if col_c1.button("Mở Login"):
    from app.lazada_browser import open_lazada_browser
    success, msg = open_lazada_browser()
    if success:
        st.sidebar.success("Đã mở browser!")
    else:
        st.sidebar.error(msg)

if col_c2.button("Lưu Cookie"):
    from app.lazada_browser import save_current_cookies
    # Use default crawler path
    success, msg = save_current_cookies()
    if success:
        st.sidebar.success("Đã cập nhật!")
        st.sidebar.info(msg)
    else:
        st.sidebar.error(msg)
        
st.sidebar.markdown("---")
# ---------------------------------------------------------
# Sidebar: AI Model Selection (Hot Swap)
# ---------------------------------------------------------
st.sidebar.markdown("### Chọn AI Model")

# Load current config
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_config.json")
current_model = "phobert"
if os.path.exists(config_path):
    try:
        with open(config_path, "r") as f:
            current_model = json.load(f).get("active_model", "phobert")
    except: pass

# Map UI names to internal keys
model_options = {
    "PhoBERT (Nhanh & Nhẹ)": "phobert",
    "Mistral (Thông minh & Chậm)": "ollama"
}
# Reverse map for default index
model_keys = list(model_options.values())
default_idx = 0
if current_model in model_keys:
    default_idx = model_keys.index(current_model)

selected_label = st.sidebar.selectbox(
    "Model phân tích:",
    options=list(model_options.keys()),
    index=default_idx
)

selected_key = model_options[selected_label]

# Save if changed
if selected_key != current_model:
    with open(config_path, "w") as f:
        json.dump({"active_model": selected_key}, f)
    st.sidebar.success(f"Đã chuyển sang {selected_key}!")
    st.sidebar.caption("Hệ thống sẽ tự động cập nhật...")
    time.sleep(1) # Visual feedback
    st.rerun()

if selected_key == "ollama":
    st.sidebar.warning("Lưu ý: Mistral chạy rất chậm (1-2s/review). Hãy kiên nhẫn!")

st.sidebar.markdown("---")
# ---------------------------------------------------------


def run_comparison():
    """Execute the comparison workflow: crawl -> predict -> display."""
    products = pm.get_products()
    cookies_path = pm.get_cookies_path()
    
    # Initialize PhoBERT predictor handled by Consumer
    pass
    
    # Progress tracking
    total_steps = len(products) * 2  # crawl + predict for each
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    step = 0
    
    # Phase 1: Parallel DAG Triggering
    active_tasks = []
    status_text.text("Đang kích hoạt phân tích song song cho tất cả sản phẩm...")
    progress_bar.progress(0.0)
    
    # 1. Trigger all DAGs first
    for item_id, product in products.items():
        p_url = product.url
        
        # Fallback if URL is missing
        if not p_url:
            st.warning(f"Sản phẩm '{product.name}' thiếu URL. Đang tạo URL dự phòng...")
            p_url = f"https://www.lazada.vn/products/-i{item_id}.html"
            
        # Trigger Airflow DAG (Let Airflow handle crawling)
        success, dag_run_id = trigger_dag(
            product_id=item_id,
            product_url=p_url,
            max_reviews=150, # Request slightly more to ensure safety
            reviews=None     # Let Airflow crawl
        )
        
        if success:
            active_tasks.append({
                'product_id': item_id,
                'dag_run_id': dag_run_id,
                'name': product.name,
                'status': 'starting'
            })
            st.toast(f"Started: {product.name}")
        else:
            st.error(f"Không thể trigger: {product.name}")

    if not active_tasks:
        st.error("Không có sản phẩm nào được xử lý.")
        return False
        
    st.info(f"Đã kích hoạt {len(active_tasks)} pipeline. Đang chờ kết quả dự đoán (Real-time)...")
    
    # Phase 2: Live Polling & Visualization
    results_container = st.empty()
    max_retries = 300  # 10 minutes timeout
    completed_task_ids = set()
    
    step = 0
    
    for _ in range(max_retries):
        any_data_update = False
        current_processing_count = 0
        
        # 2a. Check prediction files for updates
        for task in active_tasks:
            pid = task['product_id']
            
            # Read prediction file (if exists)
            pred_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'predictions', f"{pid}.json")
            
            if os.path.exists(pred_file):
                try:
                    # Use 'rb' to avoid encoding issues during partial writes, then decode
                    with open(pred_file, 'r', encoding='utf-8', errors='ignore') as f:
                        # Attempt to load even if partial (robustness)
                        content = f.read()
                        if content.strip():
                            # Fix potentially broken JSON if writing isn't atomic (though we fixed atomic writes)
                            # But standard json.loads is strict. 
                            # Since we have atomic writes, we trust the file is valid JSON or renamed recently.
                            data = json.loads(content)
                            
                            if data:
                                # Store text AND sentiment
                                predictions = [
                                    {
                                        'text': d.get('original_text', ''),
                                        'sentiment': d.get('sentiment', {})
                                    } 
                                    for d in data
                                ]
                                pm.update_product_predictions(pid, predictions)
                                
                                # Aggregate scores (need to extract sentiment dict)
                                sentiment_only = [p['sentiment'] for p in predictions]
                                scores = aggregate_scores(sentiment_only)
                                pm.update_product_scores(pid, scores)
                                any_data_update = True
                                current_processing_count += len(data)
                except (json.JSONDecodeError, OSError):
                    pass
        
        
        # 2b. Update UI
        if any_data_update:
            with results_container.container():
                display_results(key_prefix=f"step_{step}")
        
        # 2c. Check Task Completion (Airflow Status)
        finished_newly = False
        running_tasks = []
        
        for task in active_tasks:
            pid = task['product_id']
            if pid in completed_task_ids:
                continue
            
            # Check Airflow status
            dag_status = get_dag_run_status(task['dag_run_id'])
            state = dag_status.get('state')
            
            if state in ['success', 'failed', 'upstream_failed']:
                completed_task_ids.add(pid)
                finished_newly = True
                if state == 'success':
                    st.toast(f"Hoàn tất crawling & predict: {task['name']}")
                else:
                    st.toast(f"Lỗi Pipeline: {task['name']}")
            else:
                running_tasks.append(task['name'])
        
        # Update progress bar
        total_tasks = len(active_tasks)
        finished_count = len(completed_task_ids)
        progress = finished_count / total_tasks
        
        status_text.text(f"Real-time: Đã xử lý {current_processing_count} reviews tổng hợp... ({finished_count}/{total_tasks} Xong)")
        progress_bar.progress(min(progress + 0.05, 1.0)) # Show activity
        
        step += 1
        
        if finished_count == total_tasks:
            break
            
        time.sleep(2)
    
    # Final render
    progress_bar.progress(1.0)
    status_text.success("Phân tích hoàn tất!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    with results_container.container():
        display_results(key_prefix="final")
        
    return True


def display_results(key_prefix: str = ""):
    """
    Display comparison results.
    Args:
        key_prefix: Unique string to prefix widget keys (fix StreamlitDuplicateElementId)
    """
    products = pm.get_products()
    
    # Prepare data for charts
    product_scores = pm.get_products_for_comparison()
    
    if not product_scores:
        st.warning("Chưa có dữ liệu để hiển thị. Vui lòng chạy so sánh trước.")
        return
    
    # --- Section 1: Radar Chart ---
    st.subheader("Biểu đồ Radar So Sánh")
    
    fig_radar = utils.create_radar_chart_multi(product_scores)
    st.plotly_chart(fig_radar, use_container_width=True, key=f"{key_prefix}_radar")
    
    # --- Section 2: detailed Streaming Table (Live Feed) ---
    st.subheader("Live Feed: Chi Tiết Bình Luận & Dự Đoán")
    
    all_reviews_data = []
    seen_reviews = set() # Track (Product, ReducedContent) to prevent visual dupes
    
    # Collect all reviews from all products
    for product_name, product in products.items():
        # Get predictions (Live or Cached)
        preds = product.predictions if product.predictions else []
        
        for p in preds:
            text = p.get('text') or ""
            # Create a simple signature for visual dedup
            # Lowercase + no whitespace
            sig = (product.name, "".join(text.split()).lower())
            
            if sig in seen_reviews:
                continue
            seen_reviews.add(sig)
            
            sentiment = p.get('sentiment', {})
            
            row = {
                "Sản Phẩm": product.name,
                "Nội dung": text
            }
            
            # Add each aspect
            for aspect in ASPECTS:
                val = sentiment.get(aspect)
                # Map value to readable label if needed, or keep raw
                # Assuming val is 1 (POS), 0 (NEU), -1 (NEG) or "POS"..."
                if val == 1 or val == 'POS':
                    display_val = "POS"
                elif val == -1 or val == 'NEG':
                    display_val = "NEG"
                elif val == 0 or val == 'NEU':
                    display_val = "NEU"
                else:
                    display_val = "-"
                
                row[aspect] = display_val
            
            all_reviews_data.append(row)
            
    if all_reviews_data:
        # Show newest first
        df_feed = pd.DataFrame(all_reviews_data[::-1])
        st.dataframe(
            df_feed, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Nội dung": st.column_config.TextColumn("Nội dung", width="large"),
                "Sản Phẩm": st.column_config.TextColumn("Sản Phẩm", width="medium"),
            }
        )
    else:
        st.info("Đang chờ dữ liệu từ Consumer...")
    
    # --- Section 3: Overall Winner ---
    st.subheader("Kết Quả Tổng Hợp")
    
    col1, col2 = st.columns(2)
    
    # Calculate overall scores
    overall_scores = {}
    for product_name, scores in product_scores.items():
        avg = sum(scores.values()) / len(scores) if scores else 0
        overall_scores[product_name] = avg
    
    with col1:
        for product_name, avg_score in overall_scores.items():
            st.metric(
                product_name,
                f"{avg_score:.1f}/100",
                delta=f"{avg_score - 50:.1f}" if avg_score != 50 else None
            )
    
    with col2:
        # Winner
        if overall_scores:
            winner = max(overall_scores, key=overall_scores.get)
            st.success(f"Sản phẩm tốt nhất: **{winner}**")
    
    # --- Section 4: Aspect-wise Bar Chart ---
    st.subheader("So Sánh Theo Từng Khía Cạnh")
    
    # Prepare data for bar chart
    bar_data = []
    for product_name, scores in product_scores.items():
        for aspect, score in scores.items():
            bar_data.append({
                "Sản phẩm": product_name,
                "Khía cạnh": aspect,
                "Điểm": score
            })
    
    df_bar = pd.DataFrame(bar_data)
    
    if not df_bar.empty:
        fig_bar = px.bar(
            df_bar,
            x="Khía cạnh",
            y="Điểm",
            color="Sản phẩm",
            barmode="group",
            text_auto=".1f"
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#FAFAFA',
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_bar, use_container_width=True, key=f"{key_prefix}_bar")
    
    # --- Section 5: Live Review Feed ---
    st.subheader("Live Reviews Feed (Real-time)")
    
    cols = st.columns(len(products))
    
    for idx, (item_id, product) in enumerate(products.items()):
        with cols[idx]:
            st.markdown(f"#### {product.name[:30]}...")
            
            # Use predictions which contain text+sentiment from live feed
            # Fallback to product.reviews if predictions empty (client-side crawl case)
            feed_data = product.predictions if product.predictions else [{'text': r.get('reviewContent'), 'sentiment': {}} for r in product.reviews]
            
            if feed_data:
                # Show latest 50 reviews, scrollable
                with st.container(height=400):
                    for i, item in enumerate(reversed(feed_data)): # Show newest first
                        content = item.get('text', 'N/A')
                        sentiment = item.get('sentiment', {})
                        
                        # Determine overall sentiment color
                        # Basic logic: majority class
                        pos = sum(1 for v in sentiment.values() if v == 2)
                        neg = sum(1 for v in sentiment.values() if v == 0)
                        if pos > neg: 
                            color = "#e6fffa" # Light green
                            icon = ""
                        elif neg > pos: 
                            color = "#fff5f5" # Light red
                            icon = ""
                        else: 
                            color = "#f7fafc" # Grey
                            icon = ""
                            
                        st.markdown(f"""
                        <div style="
                            background-color: {color};
                            color: #1a202c;
                            padding: 10px;
                            border-radius: 8px;
                            margin-bottom: 8px;
                            border: 1px solid #e2e8f0;
                            font-size: 13px;
                        ">
                            <strong>Review #{len(feed_data)-i}</strong><br>
                            {content}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Chưa có reviews...")


def main():
    st.title("So Sánh Sản Phẩm ABSA")
    
    # Initialize
    pm.init_session_state()
    products = pm.get_products()
    
    # Check if we have products
    if not pm.can_compare():
        st.error("Cần ít nhất 2 sản phẩm để so sánh!")
        st.info("Vui lòng quay lại trang **Danh Sách Sản Phẩm** để thêm sản phẩm.")
        
        if st.button("Đi đến Danh Sách Sản Phẩm"):
            st.switch_page("pages/1_Danh_Sách_Sản_Phẩm.py")
        return
    
    # Show current products
    st.markdown("**Sản phẩm so sánh:**")
    for item_url, product in products.items():
         st.markdown(f"- {product.name}")
         
    # Run comparison
    run_comparison()

if __name__ == "__main__":
    main()
