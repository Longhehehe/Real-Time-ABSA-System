import streamlit as st
import pandas as pd
import utils
import os
import time
import plotly.express as px

# Page Config
st.set_page_config(
    page_title="Real-Time Product Analytics",
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
    st.title("Real-Time Product Analytics")

    # --- 1. Data Loading (Cached) ---
    @st.cache_data
    def load_cached_data():
        # Dynamic path: project/app/dashboard.py -> project/data/label/...
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_path = os.path.join(base_dir, 'data', 'label', 'absa_grouped_vietnamese.xlsx')
        
        if not os.path.exists(default_path):
            # Fallback for Docker mounting structure if needed
            return None, None, f"File not found at {default_path}"
        
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
    st.sidebar.header("Streaming Control")
    
    # Data Source Mode
    data_mode = st.sidebar.radio(
        "Data Source",
        ["File Simulation", "Kafka Stream"],
        help="Choose 'Kafka Stream' to consume from Kafka predictions topic"
    )
    
    # Toggle Streaming
    if st.session_state['is_streaming']:
        if st.sidebar.button("Stop Streaming", type="primary"):
            st.session_state['is_streaming'] = False
            st.rerun()
    else:
        if st.sidebar.button("Start Streaming"):
            st.session_state['is_streaming'] = True
            st.rerun()
            
    if st.sidebar.button("Reset Data"):
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
            st.sidebar.success("Kafka Connected")
        except Exception as e:
            st.sidebar.error(f"Kafka Offline - {str(e)[:30]}...")
    
    # --- 4. Main Layout ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Sentiment Trends")
        utils.plot_sentiment_trend(st.session_state['stream_data'])

    with col2:
        st.subheader("Current Metrics")
        utils.display_metrics(st.session_state['stream_data'])

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Aspect Breakdown")
        utils.plot_aspect_breakdown(st.session_state['stream_data'], aspects)
        
    with col4:
        st.subheader("Price vs Quality")
        if 'Price' in aspects and 'Quality' in aspects:
             fig = px.scatter(
                 st.session_state['stream_data'], 
                 x='Price', 
                 y='Quality', 
                 color='Sentiment',
                 title="Price vs Quality Perceived"
             )
             st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Price/Quality aspects not detected for scatter plot")

    # --- 5. Data Update Logic ---
    if st.session_state['is_streaming']:
        with st.empty():
            if data_mode == "File Simulation":
                 # Simulate new data
                 new_row = df_pool.sample(1)
                 st.session_state['stream_data'] = pd.concat([st.session_state['stream_data'], new_row], ignore_index=True).tail(50)
                 time.sleep(1/speed_factor)
                 st.rerun()
            elif data_mode == "Kafka Stream":
                # Placeholder for Kafka consumption logic
                # Ideally, this would run in a separate thread or use st.runtime.scriptrunner.add_script_run_ctx
                st.info("Kafka streaming not fully implemented in UI loop yet.")
                time.sleep(1)

if __name__ == "__main__":
    main()
