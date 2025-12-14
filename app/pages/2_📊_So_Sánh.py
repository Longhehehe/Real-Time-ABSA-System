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
from lazada_crawler import crawl_reviews
from absa_predictor import (
    PhoBERTPredictor, 
    aggregate_scores,
    get_predictor,
    ASPECTS, 
    SENTIMENT_MAP
)
import utils

# Page Config
st.set_page_config(
    page_title="So Sánh Sản Phẩm",
    page_icon="📊",
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


def run_comparison():
    """Execute the comparison workflow: crawl -> predict -> display."""
    products = pm.get_products()
    cookies_path = pm.get_cookies_path()
    
    # Initialize PhoBERT predictor (will auto-train if needed)
    with st.spinner("🧠 Đang tải PhoBERT model..."):
        predictor = get_predictor()
        model_loaded = predictor.load_model()
    
    if not model_loaded:
        st.error("❌ Không thể load hoặc train PhoBERT model!")
        return False
    
    # Progress tracking
    total_steps = len(products) * 2  # crawl + predict for each
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    step = 0
    
    for item_id, product in products.items():
        # Step 1: Crawl reviews
        status_text.text(f"🔄 Đang crawl reviews cho: {product.name}...")
        
        reviews, error = crawl_reviews(
            product.url,
            cookies_path=cookies_path,
            max_reviews=100,
            delay_min=2.0,
            delay_max=4.0,
            item_id=item_id  # Pass item_id directly
        )
        
        step += 1
        progress_bar.progress(step / total_steps)
        
        if error:
            st.warning(f"⚠️ {product.name}: {error}")
            # Use empty reviews
            reviews = []
        
        pm.update_product_reviews(item_id, reviews)
        st.success(f"✅ {product.name}: Đã crawl {len(reviews)} reviews")
        
        # Delay between products to avoid rate limiting
        status_text.text("⏳ Đợi để tránh bị block...")
        time.sleep(5)
        
        # Step 2: Run PhoBERT ABSA predictions
        status_text.text(f"🧠 Đang phân tích cảm xúc với PhoBERT: {product.name}...")
        
        predictions = []
        for review in reviews:
            text = review.get('reviewContent', '')
            
            if text:
                # Use PhoBERT for prediction
                pred = predictor.predict_single(text)
                predictions.append(pred)
        
        pm.update_product_predictions(item_id, predictions)
        
        # Step 3: Aggregate scores
        scores = aggregate_scores(predictions)
        pm.update_product_scores(item_id, scores)
        
        step += 1
        progress_bar.progress(step / total_steps)
    
    progress_bar.progress(1.0)
    status_text.text("✅ Hoàn tất phân tích!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    return True


def display_results():
    """Display comparison results."""
    products = pm.get_products()
    
    # Prepare data for charts
    product_scores = pm.get_products_for_comparison()
    
    if not product_scores:
        st.warning("⚠️ Chưa có dữ liệu để hiển thị. Vui lòng chạy so sánh trước.")
        return
    
    # --- Section 1: Radar Chart ---
    st.subheader("🔍 Biểu đồ Radar So Sánh")
    
    fig_radar = utils.create_radar_chart_multi(product_scores)
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # --- Section 2: Score Table ---
    st.subheader("📊 Bảng Điểm Chi Tiết")
    
    # Create comparison table
    table_data = {"Khía cạnh": ASPECTS}
    
    for product_name, scores in product_scores.items():
        table_data[product_name] = [
            f"{scores.get(asp, 50):.1f}" for asp in ASPECTS
        ]
    
    df_table = pd.DataFrame(table_data)
    st.dataframe(df_table, use_container_width=True, hide_index=True)
    
    # --- Section 3: Overall Winner ---
    st.subheader("🏆 Kết Quả Tổng Hợp")
    
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
            st.success(f"🥇 Sản phẩm tốt nhất: **{winner}**")
    
    # --- Section 4: Aspect-wise Bar Chart ---
    st.subheader("📈 So Sánh Theo Từng Khía Cạnh")
    
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
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # --- Section 5: Sample Reviews ---
    st.subheader("💬 Reviews Tiêu Biểu")
    
    for item_id, product in products.items():
        with st.expander(f"📝 {product.name} ({len(product.reviews)} reviews)"):
            if product.reviews:
                for i, review in enumerate(product.reviews[:5]):
                    content = review.get('reviewContent', 'N/A')
                    rating = review.get('rating', 0)
                    stars = "⭐" * int(rating)
                    st.markdown(f"**{stars}** ({rating}/5)")
                    st.markdown(f"> {content}")
                    st.markdown("---")
            else:
                st.info("Không có reviews")


def main():
    st.title("📊 So Sánh Sản Phẩm ABSA")
    
    # Initialize
    pm.init_session_state()
    products = pm.get_products()
    
    # Check if we have products
    if not pm.can_compare():
        st.error("❌ Cần ít nhất 2 sản phẩm để so sánh!")
        st.info("👉 Vui lòng quay lại trang **Danh Sách Sản Phẩm** để thêm sản phẩm.")
        
        if st.button("🛒 Đi đến Danh Sách Sản Phẩm"):
            st.switch_page("pages/1_🛒_Danh_Sách_Sản_Phẩm.py")
        return
    
    # Show current products
    st.markdown("**Sản phẩm so sánh:**")
    cols = st.columns(len(products))
    for i, (item_id, product) in enumerate(products.items()):
        with cols[i]:
            st.markdown(f"**{product.name}**")
            st.caption(f"ID: {item_id}")
    
    st.markdown("---")
    
    # Check if already analyzed
    has_scores = all(product.scores for product in products.values())
    
    if has_scores:
        st.success("✅ Đã có kết quả phân tích!")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Phân Tích Lại", use_container_width=True):
                run_comparison()
                st.rerun()
        
        with col2:
            if st.button("📊 Xem Kết Quả", type="primary", use_container_width=True):
                display_results()
    else:
        st.info("ℹ️ Nhấn nút bên dưới để bắt đầu crawl reviews và phân tích.")
        
        if st.button("🚀 Bắt Đầu Phân Tích", type="primary", use_container_width=True):
            success = run_comparison()
            if success:
                st.rerun()
    
    # Always show results if available
    if has_scores:
        st.markdown("---")
        display_results()


if __name__ == "__main__":
    main()
