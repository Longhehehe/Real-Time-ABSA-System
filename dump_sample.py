import pandas as pd
import sys

try:
    file_path = r'c:/Users/Long/Documents/Hoc_Tap/SE363 (1)/data/label/absa_grouped_vietnamese.xlsx'
    df = pd.read_excel(file_path)
    
    with open('data_sample.txt', 'w', encoding='utf-8') as f:
        f.write("COLUMNS:\n")
        for col in df.columns:
            f.write(f"'{col}'\n")
        
        f.write("\n\nHEAD(5):\n")
        f.write(df.head().to_string())
        
        # Check first column distinct values to guess if it is product ID
        f.write(f"\n\nFirst Column: {df.columns[0]}\n")
        f.write(str(df.iloc[:, 0].unique()[:20]))

except Exception as e:
    with open('data_sample.txt', 'w', encoding='utf-8') as f:
        f.write(str(e))
