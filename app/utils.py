import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
import time

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
        
        # List of expected aspects based on user request/file inspection
        expected_aspects = [
            'Chất lượng sản phẩm', 'Trải nghiệm sử dụng', 'Đúng mô tả sản phẩm', 'Hiệu năng sản phẩm',
            'Giá cả', 'Khuyến mãi & voucher',
            'Vận chuyển & giao hàng', 'Đóng gói & bao bì',
            'Uy tín & thái độ shop', 'Dịch vụ chăm sóc khách hàng', 'Lỗi & bảo hành & hàng giả', 'Đổi trả & bảo hành'
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
        2: np.nan  # Convert 2 to NaN so it's ignored in mean calculation
    }
    
    for aspect in aspects:
        if aspect in df.columns:
            # Convert values
            series = df[aspect].map(value_map)
            # Calculate mean ignoring NaNs
            avg_score = series.mean()
            
            # If all are NaN (no reviews for this aspect), set to 0 or 50? 
            # Let's set to 0 or keeps as NaN? 0 might be misleading. 
            # Let's default to 0 if NaN.
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

def get_basic_metrics(df):
    """
    Returns basic count of reviews.
    """
    total_reviews = len(df)
    return total_reviews
