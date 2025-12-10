import pandas as pd

try:
    file_path = r'c:/Users/Long/Documents/Hoc_Tap/SE363 (1)/data/label/absa_grouped_vietnamese.xlsx'
    df = pd.read_excel(file_path)
    print("ALL COLUMNS:", list(df.columns))
    
    # Check for product identifying columns
    potential_id_cols = [c for c in df.columns if 'id' in c.lower() or 'product' in c.lower() or 'type' in c.lower() or 'name' in c.lower()]
    print("ID Columns Candidates:", potential_id_cols)

    # Check unique values for a few aspects
    aspects = ["Chất lượng sản phẩm", "Giá cả"]
    for aspect in aspects:
        if aspect in df.columns:
            print(f"\nUnique values for '{aspect}':")
            print(df[aspect].unique()[:10]) # Show first 10 unique
            print(df[aspect].value_counts().head())
        else:
            print(f"Column '{aspect}' not found")

except Exception as e:
    print(e)
