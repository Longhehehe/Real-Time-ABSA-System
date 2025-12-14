#!/usr/bin/env python
# Simple test script
import sys
import os

PROJECT_DIR = '/opt/airflow/project'
src_path = os.path.join(PROJECT_DIR, 'prepro', '23520932_23520903_20520692_src', '23520932_23520903_20520692_src')
sys.path.insert(0, src_path)

print(f"PROJECT_DIR: {PROJECT_DIR}")
print(f"src_path: {src_path}")
print(f"src_path exists: {os.path.exists(src_path)}")

# Check data file
data_path = os.path.join(PROJECT_DIR, 'data', 'label', 'absa_grouped_vietnamese.xlsx')
print(f"data_path: {data_path}")
print(f"data_path exists: {os.path.exists(data_path)}")

# Check VnCoreNLP
vn_path = os.getenv('VNCORENLP_PATH', '/opt/vncorenlp/VnCoreNLP-1.1.1.jar')
print(f"VNCORENLP_PATH: {vn_path}")
print(f"VnCoreNLP exists: {os.path.exists(vn_path)}")

# Try imports
try:
    from Src.train_multinb_rf import MultinomialNBModel
    print("Import MultinomialNBModel: OK")
except Exception as e:
    print(f"Import MultinomialNBModel: FAILED - {e}")

try:
    from Src.preprocessing import preprocess_text
    print("Import preprocess_text: OK")
except Exception as e:
    print(f"Import preprocess_text: FAILED - {e}")
