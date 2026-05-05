"""
Data Enricher: Adds Product ID and Category columns to the dataset.
This simulates a real e-commerce scenario where reviews belong to different products.
"""
import pandas as pd
import random
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'label', 'absa_grouped_vietnamese.xlsx')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'label', 'absa_enriched.xlsx')

PRODUCT_IDS = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
PRODUCT_CATEGORIES = ['Electronics', 'Fashion', 'Home & Living', 'Beauty', 'Sports']

def enrich_data():
    print(" Starting Data Enrichment...")
    
    try:
        df = pd.read_excel(INPUT_FILE, header=None)
        
        header_idx = 0
        for idx, row in df.iterrows():
            if 'Chất lượng sản phẩm' in [str(v).strip() for v in row.values]:
                header_idx = idx
                break
        
        df = pd.read_excel(INPUT_FILE, header=header_idx)
        print(f" Loaded {len(df)} rows from {INPUT_FILE}")
    except Exception as e:
        print(f" Error loading file: {e}")
        return None
    
    df['Product_ID'] = [random.choice(PRODUCT_IDS) for _ in range(len(df))]
    
    df['Product_Category'] = [random.choice(PRODUCT_CATEGORIES) for _ in range(len(df))]
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f" Saved enriched data to {OUTPUT_FILE}")
    print(f"   - Added columns: Product_ID, Product_Category")
    print(f"   - Sample Product distribution: {df['Product_ID'].value_counts().to_dict()}")
    
    return df

if __name__ == "__main__":
    enrich_data()
