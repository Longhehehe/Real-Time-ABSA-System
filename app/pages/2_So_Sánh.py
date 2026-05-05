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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import product_manager as pm
import json
from lazada_crawler import crawl_reviews
                                                                        
try:
    from airflow_client import trigger_dag, get_dag_run_status, get_task_instances
except ImportError:
    from app.airflow_client import trigger_dag, get_dag_run_status, get_task_instances
from absa_predictor import (
    aggregate_scores,
                                          
    ASPECTS, 
    SENTIMENT_MAP
)
import utils

st.set_page_config(
    page_title="So Sánh Sản Phẩm",
    layout="wide"
)

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
                              
    success, msg = save_current_cookies()
    if success:
        st.sidebar.success("Đã cập nhật!")
        st.sidebar.info(msg)
    else:
        st.sidebar.error(msg)
        
st.sidebar.markdown("---")
                                                           
def run_comparison():
    """Execute the comparison workflow: crawl -> predict -> display."""
    products = pm.get_products()
    cookies_path = pm.get_cookies_path()
    
    pass
    
    total_steps = len(products) * 2                            
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    step = 0
    
    predictions_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'predictions')
    for item_id in products.keys():
        pred_file = os.path.join(predictions_dir, f"{item_id}.json")
        if os.path.exists(pred_file):
            try:
                os.remove(pred_file)
                print(f"🧹 Deleted old predictions: {pred_file}")
            except Exception as e:
                print(f" Could not delete {pred_file}: {e}")
    
    active_tasks = []
    
    current_product_ids = frozenset(products.keys())
    if 'triggered_products' not in st.session_state:
        st.session_state.triggered_products = set()
    if 'active_dag_runs' not in st.session_state:
        st.session_state.active_dag_runs = []
    
    already_triggered = current_product_ids.issubset(st.session_state.triggered_products)
    
    if already_triggered and st.session_state.active_dag_runs:
                                                          
        active_tasks = st.session_state.active_dag_runs
        st.info("Đang tiếp tục theo dõi các pipeline đã kích hoạt...")
    else:
                       
        status_text.text("Đang kích hoạt phân tích song song cho tất cả sản phẩm...")
        progress_bar.progress(0.0)
    
        for item_id, product in products.items():
            p_url = product.url
            
            if not p_url:
                st.warning(f"Sản phẩm '{product.name}' thiếu URL. Đang tạo URL dự phòng...")
                p_url = f"https://www.lazada.vn/products/-i{item_id}.html"
                
            success, dag_run_id = trigger_dag(
                product_id=item_id,
                product_url=p_url,
                max_reviews=150,                                         
                reviews=None                        
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
    
    st.session_state.active_dag_runs = active_tasks
    st.session_state.triggered_products = set(products.keys())
        
    st.info(f"Đã kích hoạt {len(active_tasks)} pipeline. Đang chờ kết quả dự đoán (Real-time)...")
    
    results_container = st.empty()
    max_retries = 900                                         
    completed_task_ids = set()
    
    step = 0
    
    for _ in range(max_retries):
        any_data_update = False
        current_processing_count = 0
        
        for task in active_tasks:
            pid = task['product_id']
            
            pred_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'predictions', f"{pid}.json")
            
            if os.path.exists(pred_file):
                try:
                                                                                          
                    with open(pred_file, 'r', encoding='utf-8', errors='ignore') as f:
                                                                      
                        content = f.read()
                        if content.strip():
                                                                                                                 
                            data = json.loads(content)
                            
                            if data:
                                                          
                                predictions = [
                                    {
                                        'text': d.get('original_text', ''),
                                        'sentiment': d.get('sentiment', {})
                                    } 
                                    for d in data
                                ]
                                pm.update_product_predictions(pid, predictions)
                                
                                sentiment_only = [p['sentiment'] for p in predictions]
                                scores = aggregate_scores(sentiment_only)
                                pm.update_product_scores(pid, scores)
                                any_data_update = True
                                current_processing_count += len(data)
                except (json.JSONDecodeError, OSError):
                    pass
        
        if any_data_update:
            with results_container.container():
                display_results(key_prefix=f"step_{step}")
        
        finished_newly = False
        running_tasks = []
        
        for task in active_tasks:
            pid = task['product_id']
            if pid in completed_task_ids:
                continue
            
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
        
        total_tasks = len(active_tasks)
        finished_count = len(completed_task_ids)
        progress = finished_count / total_tasks
        
        status_text.text(f"Real-time: Đã xử lý {current_processing_count} reviews tổng hợp... ({finished_count}/{total_tasks} Xong)")
        progress_bar.progress(min(progress + 0.05, 1.0))                
        
        step += 1
        
        if finished_count == total_tasks:
            break
            
        time.sleep(2)
    
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
    
    product_scores = pm.get_products_for_comparison()
    
    if not product_scores:
        st.warning("Chưa có dữ liệu để hiển thị. Vui lòng chạy so sánh trước.")
        return
    
    st.subheader("Biểu đồ Radar So Sánh")
    
    fig_radar = utils.create_radar_chart_multi(product_scores)
    st.plotly_chart(fig_radar, use_container_width=True, key=f"{key_prefix}_radar")
    
    st.subheader("Live Feed: Chi Tiết Bình Luận & Dự Đoán")
    
    all_reviews_data = []
    seen_reviews = set()                                                          
    
    for product_name, product in products.items():
                                          
        preds = product.predictions if product.predictions else []
        
        for p in preds:
            text = p.get('text') or ""
                                                        
            sig = (product.name, "".join(text.split()).lower())
            
            if sig in seen_reviews:
                continue
            seen_reviews.add(sig)
            
            sentiment = p.get('sentiment', {})
            
            row = {
                "Sản Phẩm": product.name,
                "Nội dung": text
            }
            
            for aspect in ASPECTS:
                val = sentiment.get(aspect)
                                                                    
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
    
    st.subheader("Kết Quả Tổng Hợp")
    
    col1, col2 = st.columns(2)
    
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
                
        if overall_scores:
            winner = max(overall_scores, key=overall_scores.get)
            st.success(f"Sản phẩm tốt nhất: **{winner}**")
    
    st.subheader("So Sánh Theo Từng Khía Cạnh")
    
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
    
    st.subheader("Live Reviews Feed (Real-time)")
    
    cols = st.columns(len(products))
    
    for idx, (item_id, product) in enumerate(products.items()):
        with cols[idx]:
            st.markdown(f"#### {product.name[:30]}...")
            
            feed_data = product.predictions if product.predictions else [{'text': r.get('reviewContent'), 'sentiment': {}} for r in product.reviews]
            
            if feed_data:
                                                    
                with st.container(height=400):
                    for i, item in enumerate(reversed(feed_data)):                    
                        content = item.get('text', 'N/A')
                        sentiment = item.get('sentiment', {})
                        
                        pos = sum(1 for v in sentiment.values() if v == 2)
                        neg = sum(1 for v in sentiment.values() if v == 0)
                        if pos > neg: 
                            color = "#e6fffa"              
                            icon = ""
                        elif neg > pos: 
                            color = "#fff5f5"            
                            icon = ""
                        else: 
                            color = "#f7fafc"       
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
    
    pm.init_session_state()
    products = pm.get_products()
    
    if not pm.can_compare():
        st.error("Cần ít nhất 2 sản phẩm để so sánh!")
        st.info("Vui lòng quay lại trang **Danh Sách Sản Phẩm** để thêm sản phẩm.")
        
        if st.button("Đi đến Danh Sách Sản Phẩm"):
            st.switch_page("pages/1_Danh_Sách_Sản_Phẩm.py")
        return
    
    st.markdown("**Sản phẩm so sánh:**")
    for item_url, product in products.items():
         st.markdown(f"- {product.name}")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button(" Reset & Chạy Lại"):
                                                      
            if 'triggered_products' in st.session_state:
                del st.session_state.triggered_products
            if 'active_dag_runs' in st.session_state:
                del st.session_state.active_dag_runs
            st.rerun()
         
    run_comparison()

if __name__ == "__main__":
    main()
