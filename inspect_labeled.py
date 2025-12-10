import pandas as pd

try:
    file_path = r'c:/Users/Long/Documents/Hoc_Tap/SE363 (1)/model/test_flow_reviews_1_labeled_full.xlsx'
    df = pd.read_excel(file_path, nrows=5)
    print("COLUMNS:", list(df.columns))
    print("First row values:", df.iloc[0].tolist())
except Exception as e:
    print(e)
