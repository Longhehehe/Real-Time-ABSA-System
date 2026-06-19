import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import random
import time
import streamlit as st

def assign_fake_product_ids(df, product_names=['Product A', 'Product B', 'Product C']):
    """
    Randomly assigns a Product ID to each row in the DataFrame.
    """
    df['Product_ID'] = [random.choice(product_names) for _ in range(len(df))]
    return df

def load_data(file_path):
    """
    Loads data from Excel.
    Handles the specific structure where the header might be on the second row.
    """
    try:
        # Read with header=0 to check structure, but based on inspection, 
        # the meaningful headers are likely on row 1 (second row) if row 0 is grouping.
        # Let's try reading with header=1 (0-indexed, so second row).
        # We'll need to robustly find the header row containing 'reviewContent' or 'Chất lượng sản phẩm'.
        
        # Read first few rows without header
        df_preview = pd.read_excel(file_path, header=None, nrows=5)
        
        # Find the row index that contains 'Chất lượng sản phẩm' or 'reviewContent'
        header_row_idx = 0
        for idx, row in df_preview.iterrows():
            # Convert values to string and strip whitespace for flexible matching
            row_values = [str(v).strip() for v in row.values]
            # STRICT CHECK: Only look for 'Chất lượng sản phẩm' which is unique to the true header row.
            # Do NOT check for 'reviewContent' because it exists in the top grouping row too.
            if 'Chất lượng sản phẩm' in row_values:
                header_row_idx = idx
                break
                
        # Reload with correct header
        df = pd.read_excel(file_path, header=header_row_idx)
        
        # Clean column names (strip whitespace)
        df.columns = [str(c).strip() for c in df.columns]
        
        # List of expected aspects - OPTIMIZED for E-commerce (9 aspects)
        expected_aspects = [
            'Chất lượng sản phẩm',       # Quality, durability, materials
            'Hiệu năng & Trải nghiệm',   # Performance, user experience  
            'Đúng mô tả',                # Accuracy of description
            'Giá cả & Khuyến mãi',       # Price, discounts, value
            'Vận chuyển',                # Shipping speed, delivery
            'Đóng gói',                  # Packaging quality
            'Dịch vụ & Thái độ Shop',    # Customer service, seller attitude
            'Bảo hành & Đổi trả',        # Warranty, returns
            'Tính xác thực',             # Authenticity (fake/genuine)
        ]
        
        # Keep only existing columns
        existing_aspects = [col for col in expected_aspects if col in df.columns]
        
        if not existing_aspects:
            # Fallback: Print what columns we actually found to help debug in the UI
            found_cols = ", ".join(list(df.columns)[:5]) + "..."
            return None, f"Could not find expected aspect columns. Found: {found_cols}"

        return df, existing_aspects

    except Exception as e:
        return None, str(e)

def calculate_scores(df, aspects):
    """
    Calculates average score (0-100) for each aspect.
    Logic: 1 (Pos) -> 100, 0 (Neu) -> 50, -1 (Neg) -> 0.
    2 (N/A) -> Ignored.
    """
    scores = {}
    
    # Mapping
    value_map = {
        1: 100,
        0: 50,
        -1: 0,
        2: np.nan,
        # String support
        'POS': 100, 'Pos': 100, 'pos': 100,
        'NEU': 50, 'Neu': 50, 'neu': 50,
        'NEG': 0, 'Neg': 0, 'neg': 0,
        'None': np.nan, None: np.nan
    }
    
    for aspect in aspects:
        if aspect in df.columns:
            # Robust mapping: Convert to string first
            # Safe convert: handle NaNs/None
            series_str = df[aspect].astype(str).str.replace(r'\.0$', '', regex=True) # "1.0" -> "1"
            
            # Map using string keys
            mapped_series = series_str.map(value_map)
            
            # Calculate mean ignoring NaNs
            avg_score = mapped_series.mean()
            
            # If NaN (no reviews), default to None or 50?
            scores[aspect] = avg_score if not np.isnan(avg_score) else 0
            
    return scores

def generate_mock_product_b(real_scores):
    """
    Generates a simulated Product B by adding random offsets to Product A's scores.
    """
    mock_scores = {}
    for aspect, score in real_scores.items():
        # Random offset between -20 and +20, kept within 0-100
        offset = random.randint(-20, 20)
        new_score = max(0, min(100, score + offset))
        mock_scores[aspect] = new_score
    return mock_scores

def create_radar_chart_multi(product_scores_dict):
    """
    Creates a Radar Chart for multiple products.
    Args:
        product_scores_dict: { 'Product Name': { 'Aspect': score, ... }, ... }
    """
    fig = go.Figure()
    
    # Define a color palette
    colors = ['#00CC96', '#EF553B', '#636EFA', '#AB63FA', '#FFA15A']
    
    for idx, (product_name, scores) in enumerate(product_scores_dict.items()):
        categories = list(scores.keys())
        values = list(scores.values())
        
        # Close the loop for radar chart
        # Plotly Scatterpolar handles closing automatically if fill='toself' usually, 
        # but sometimes explicit closure is safer. 
        # Here we just pass the list, Scatterpolar is usually smart enough.
        
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=product_name,
            line=dict(color=color)
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            )
        ),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=20, b=20),
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig



def display_metrics(df):
    """
    Displays key metrics for the dashboard.
    """
    total_reviews = len(df)
    
    # Calculate global average sentiment across all aspect columns
    # Filter for numeric aspect columns only
    aspect_cols = [c for c in df.columns if c not in ['reviewContent', 'Product_ID', 'Time', 'Sentiment']]
    
    if not aspect_cols:
        st.warning("No aspect columns found for metrics.")
        return

    # Flatten all aspect scores to get a global average (ignoring 2s which are N/A)
    all_scores = df[aspect_cols].values.flatten()
    # Filter out 2 (N/A) and NaNs
    valid_scores = [s for s in all_scores if s != 2 and not np.isnan(s)]
    
    if valid_scores:
        avg_sentiment = np.mean(valid_scores)
        # Map -1..1 to 0..100 roughly for display
        # -1 -> 0, 0 -> 50, 1 -> 100
        display_score = (avg_sentiment + 1) * 50
    else:
        display_score = 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", f"{total_reviews}")
    col2.metric("Avg Sentiment Score", f"{display_score:.1f}/100")
    
    # Mock 'Positive Rate'
    positive_count = len([s for s in valid_scores if s == 1])
    total_valid = len(valid_scores) if valid_scores else 1
    pos_rate = (positive_count / total_valid) * 100
    col3.metric("Positive Rate", f"{pos_rate:.1f}%")

def plot_sentiment_trend(df):
    """
    Plots a line chart of sentiment over time (using index as proxy for time).
    """
    if df.empty:
        st.info("No data to plot.")
        return
        
    # Create a 'Average_Sentiment' for each row
    # Identify aspect columns
    aspect_cols = [c for c in df.columns if c not in ['reviewContent', 'Product_ID', 'timestamp']]
    
    # Helper to avg row
    def get_row_sentiment(row):
        scores = []
        for col in aspect_cols:
            val = row[col]
            if isinstance(val, (int, float)) and val != 2: # Ignore N/A
                scores.append(val)
        return np.mean(scores) if scores else 0

    df_plot = df.copy()
    df_plot['Average_Sentiment'] = df_plot.apply(get_row_sentiment, axis=1)
    
    # Add a pseudo-time axis based on index (assuming stream)
    df_plot['Review_Index'] = range(len(df_plot))
    
    fig = px.line(
        df_plot, 
        x='Review_Index', 
        y='Average_Sentiment', 
        title='Average Sentiment Trend',
        labels={'Review_Index': 'Review Sequence', 'Average_Sentiment': 'Sentiment (-1 to 1)'}
    )
    
    # Update layout for dark theme
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        yaxis=dict(range=[-1.1, 1.1])
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_aspect_breakdown(df, aspects):
    """
    Plots a horizontal bar chart showing the average score for each aspect.
    """
    if df.empty:
        st.info("No data for breakdown.")
        return
        
    scores = calculate_scores(df, aspects)
    
    # Create DF for plotting
    plot_data = pd.DataFrame({
        'Aspect': list(scores.keys()),
        'Score': list(scores.values())
    })
    
    fig = px.bar(
        plot_data,
        x='Score',
        y='Aspect',
        orientation='h',
        title='Average Sentiment Score by Aspect (0-100)',
        text='Score',
        color='Score',
        color_continuous_scale='RdYlGn', # Red to Green
        range_color=[0, 100]
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        yaxis={'categoryorder':'total ascending'}
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    
    st.plotly_chart(fig, use_container_width=True)
