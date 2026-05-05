import streamlit as st
import pandas as pd
import utils
import os
import time
import plotly.express as px
import json

st.set_page_config(
    page_title="Real-Time Product Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

    @st.cache_data
    def load_cached_data():
                                                                          
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_path = os.path.join(base_dir, 'data', 'label', 'absa_grouped_vietnamese.xlsx')
        
        if not os.path.exists(default_path):
                                                              
            return None, None, f"File not found at {default_path}"
        
        df, aspects = utils.load_data(default_path)
        if df is None:
            return None, None, aspects
            
        df = utils.assign_fake_product_ids(df, ['Product A', 'Product B'])
        return df, aspects, None

    df_pool, aspects, error = load_cached_data()

    if error:
        st.error(f"Error: {error}")
        return

    if 'stream_data' not in st.session_state:
                                                                  
        st.session_state['stream_data'] = df_pool.sample(20).copy()
    
    if 'is_streaming' not in st.session_state:
        st.session_state['is_streaming'] = False

    st.sidebar.header("Streaming Control")
    
    data_mode = st.sidebar.radio(
        "Data Source",
        ["File Simulation", "Live Predictions"],
        help="Choose 'Live Predictions' to view real-time analysis results from the Consumer."
    )
    
    selected_live_file = None
    if data_mode == "Live Predictions":
                                   
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pred_dir = os.path.join(base_dir, 'data', 'predictions')
        
        if os.path.exists(pred_dir):
            files = [f for f in os.listdir(pred_dir) if f.endswith('.json')]
            if files:
                selected_live_file = st.sidebar.selectbox("Select Product Results", files)
            else:
                st.sidebar.warning("No prediction files found yet.")
        else:
             st.sidebar.error("Predictions directory not found.")
    
    if st.session_state['is_streaming']:
        if st.sidebar.button("Stop Polling", type="primary"):
            st.session_state['is_streaming'] = False
            st.rerun()
    else:
        if st.sidebar.button("Start Polling"):
            st.session_state['is_streaming'] = True
            st.rerun()
            
    if st.sidebar.button("Reset Data"):
        st.session_state['stream_data'] = df_pool.sample(20).copy()
        st.session_state['is_streaming'] = False
        st.rerun()

    with st.sidebar.expander(" Debug Raw Data"):
        st.write(st.session_state['stream_data'].tail(5))

    speed_factor = st.sidebar.slider("Updates per second", 0.1, 2.0, 0.5)
    
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

    if st.session_state['is_streaming']:
        with st.empty():
            if data_mode == "File Simulation":
                                    
                 new_row = df_pool.sample(1)
                 st.session_state['stream_data'] = pd.concat([st.session_state['stream_data'], new_row], ignore_index=True).tail(50)
                 time.sleep(1/speed_factor)
                 st.rerun()
            elif data_mode == "Live Predictions" and selected_live_file:
                                     
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                file_path = os.path.join(base_dir, 'data', 'predictions', selected_live_file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    if data:
                                            
                        mod_time = os.path.getmtime(file_path)
                        mod_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
                        st.sidebar.info(f"🕒 File updated: {mod_time_str}")

                        rows = []
                        for item in data:
                            row = {}
                                                    
                            sentiments = item.get('sentiment', {})
                            if isinstance(sentiments, str):                                        
                                try: sentiments = json.loads(sentiments)
                                except: sentiments = {}
                                
                            for k, v in sentiments.items():
                                row[k] = v
                                
                            row['rating'] = item.get('rating')
                            row['reviewContent'] = item.get('original_text')
                            rows.append(row)
                            
                        new_df = pd.DataFrame(rows)
                        
                        st.session_state['stream_data'] = new_df.tail(100)                
                        
                    time.sleep(1/speed_factor)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    time.sleep(2)

if __name__ == "__main__":
    main()
