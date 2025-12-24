import pandas as pd

try:
    file_path = r'c:/Users/Long/Documents/Hoc_Tap/SE363 (1)/data/label/absa_grouped_vietnamese.xlsx'
    df = pd.read_excel(file_path, nrows=5)
    print("Columns:", df.columns.tolist())
    print("Data Types:\n", df.dtypes)
    print("First row:\n", df.iloc[0])
except Exception as e:
    print(e)
