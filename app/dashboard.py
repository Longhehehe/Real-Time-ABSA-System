import streamlit as st
import pandas as pd
import utils
import os
import time
import plotly.express as px

# Page Config
st.set_page_config(
    page_title="Real-Time Product Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    h1 { color: #00CC96; text-align: center; margin-bottom: 2rem; }
    h3 { color: #FAFAFA; }
    .stMetric {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #363945;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("⚡ Real-Time Product Analytics")

    # --- 1. Data Loading (Cached) ---
    @st.cache_data
    def load_cached_data():
        default_path = r'c:/Users/Long/Documents/Hoc_Tap/SE363 (1)/data/label/absa_grouped_vietnamese.xlsx'
        if not os.path.exists(default_path):
            return None, None, "File not found"
        
        df, aspects = utils.load_data(default_path)
        if df is None:
            return None, None, aspects
            
        # Assign Fake Product IDs for simulation
        df = utils.assign_fake_product_ids(df, ['Product A', 'Product B'])
        return df, aspects, None

    df_pool, aspects, error = load_cached_data()

    if error:
        st.error(f"Error: {error}")
        return

    # --- 2. Session State Initialization ---
    if 'stream_data' not in st.session_state:
        # Initial state: Start with a small sample (e.g., 20 rows)
        st.session_state['stream_data'] = df_pool.sample(20).copy()
    
    if 'is_streaming' not in st.session_state:
        st.session_state['is_streaming'] = False

    # --- 3. Sidebar Controls ---
    st.sidebar.header("📡 Streaming Control")
    
    # Data Source Mode
    data_mode = st.sidebar.radio(
        "Data Source",
        ["File Simulation", "Kafka Stream"],
        help="Choose 'Kafka Stream' to consume from Kafka predictions topic"
    )
    
    # Toggle Streaming
    if st.session_state['is_streaming']:
        if st.sidebar.button("🛑 Stop Streaming", type="primary"):
            st.session_state['is_streaming'] = False
            st.rerun()
    else:
        if st.sidebar.button("▶️ Start Streaming"):
            st.session_state['is_streaming'] = True
            st.rerun()
            
    if st.sidebar.button("🔄 Reset Data"):
        st.session_state['stream_data'] = df_pool.sample(20).copy()
        st.session_state['is_streaming'] = False
        st.rerun()

    # Simulation Speed
    speed_factor = st.sidebar.slider("Updates per second", 0.1, 2.0, 0.5)
    
    # Kafka Status Indicator
    if data_mode == "Kafka Stream":
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Kafka Status**")
        try:
            from kafka import KafkaConsumer
            consumer_test = KafkaConsumer(
                bootstrap_servers='localhost:9092',
                consumer_timeout_ms=1000
            )
            consumer_test.close()
            st.sidebar.success("✅ Kafka Connected")
        except Exception as e:
            st.sidebar.error(f"❌ Kafka Offline - {str(e)[:30]}...")
    
    # --- 4. Main Layout ---
    
    # Current Dataset Statistics
    current_df = st.session_state['stream_data']
    total_reviews = len(current_df)
    
    col_metrics_1, col_metrics_2, col_metrics_3 = st.columns(3)
    col_metrics_1.metric("Total Reviews Processed", total_reviews)
    col_metrics_2.metric("Products Tracked", len(current_df['Product_ID'].unique()))
    col_metrics_3.metric("Status", "Live 🔴" if st.session_state['is_streaming'] else "Paused ⏸️")

    st.markdown("---")

    col_chart, col_stats = st.columns([2, 1])

    # Calculate Scores for current data
    products = current_df['Product_ID'].unique()
    product_scores = {}
    
    for prod in sorted(products):
        prod_df = current_df[current_df['Product_ID'] == prod]
        scores = utils.calculate_scores(prod_df, aspects)
        product_scores[prod] = scores

    with col_chart:
        st.subheader("🔍 Real-Time Sentiment Radar")
        fig = utils.create_radar_chart_multi(product_scores)
        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        st.subheader("📊 Live Scores")
        # Display score tables
        summary_data = []
        for prod, scores in product_scores.items():
            avg_score = sum(scores.values()) / len(scores) if scores else 0
            summary_data.append({
                "Product": prod,
                "Avg Score": f"{avg_score:.1f}",
                "Reviews": len(current_df[current_df['Product_ID'] == prod])
            })
        st.dataframe(pd.DataFrame(summary_data), hide_index=True)
        
        with st.expander("Recent Reviews Log"):
            st.dataframe(current_df[['Product_ID', 'reviewContent']].tail(10), hide_index=True)

    # --- New Sentiment Charts Section ---
    st.markdown("---")
    st.subheader("📈 Thống kê cảm xúc theo khía cạnh")
    
    # Mapper for sentiment
    sentiment_map = {1: "POS", 0: "NEU", -1: "NEG"}
    SENTIMENTS = ["POS", "NEU", "NEG"]
    
    aspect_counts = []
    
    # Helper to calculate stats
    # We can filter by product if we want specific product stats, 
    # but for now let's show aggregate or select a product.
    # Let's show aggregate of ALL streamed data for now to keep it simple as per snippet,
    # OR we can add a selector. Let's stick to the snippet logic (general distribution).
    
    # We need to process the data to map 1/0/-1 to strings first for easier counting
    for asp in aspects:
        if asp not in current_df.columns:
            continue
            
        # Get counts for this aspect, mapping numeric to label
        # We need to be careful with NaN or 2 (N/A)
        series = current_df[asp].map(sentiment_map)
        counts = series.value_counts().reindex(SENTIMENTS, fill_value=0)
        
        for sent, cnt in counts.items():
            aspect_counts.append({"Aspect": asp, "Sentiment": sent, "Count": cnt})
            
    df_stats = pd.DataFrame(aspect_counts)

    if not df_stats.empty:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### 🔹 Biểu đồ tổng hợp cảm xúc theo khía cạnh")
            fig_bar = px.bar(
                df_stats,
                x="Aspect", y="Count", color="Sentiment",
                color_discrete_map={"POS": "#33cc33", "NEU": "#cccc00", "NEG": "#ff5050"},
                barmode="group", text_auto=True
            )
            # Custom styling for dark theme
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#FAFAFA'
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart")

        with col2:
            st.markdown("#### 🔹 Tỉ lệ cảm xúc tổng quát")
            df_total = df_stats.groupby("Sentiment")["Count"].sum().reset_index()
            fig_pie = px.pie(
                df_total, names="Sentiment", values="Count",
                color="Sentiment",
                color_discrete_map={"POS": "#33cc33", "NEU": "#cccc00", "NEG": "#ff5050"},
                hole=0.3
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#FAFAFA'
            )
            st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart")

    # --- 5. Streaming Mechanism ---
    if st.session_state['is_streaming']:
        # Simulate new data arrival
        new_batch = df_pool.sample(random_state=None, n=3) # Randomly pick 3 new reviews
        
        # Append to session state
        st.session_state['stream_data'] = pd.concat([st.session_state['stream_data'], new_batch], ignore_index=True)
        
        # Limit memory usage (keep last 1000 reviews max for demo)
        if len(st.session_state['stream_data']) > 1000:
             st.session_state['stream_data'] = st.session_state['stream_data'].tail(1000)
             
        time.sleep(1/speed_factor)
        st.rerun()

if __name__ == "__main__":
    main()
