#!/usr/bin/env python
"""Debug script to check model loading in Streamlit container."""
import os
import sys
import pickle

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'best_model')

print(f"BASE_DIR: {BASE_DIR}")
print(f"MODEL_DIR: {MODEL_DIR}")
print(f"MODEL_DIR exists: {os.path.exists(MODEL_DIR)}")

# Find model folders
if os.path.exists(MODEL_DIR):
    print(f"\nContents of MODEL_DIR:")
    for item in os.listdir(MODEL_DIR):
        item_path = os.path.join(MODEL_DIR, item)
        print(f"  - {item} (dir: {os.path.isdir(item_path)})")
        
        if os.path.isdir(item_path):
            print(f"    Contents:")
            for subitem in os.listdir(item_path):
                print(f"      - {subitem}")
            
            # Try to load model
            model_file = os.path.join(item_path, 'model.pkl')
            vectorizer_file = os.path.join(item_path, 'vectorizer.pkl')
            
            if os.path.exists(model_file):
                print(f"\n    Trying to load {model_file}...")
                try:
                    with open(model_file, 'rb') as f:
                        model = pickle.load(f)
                    print(f"    ✅ Model loaded successfully! Type: {type(model)}")
                except Exception as e:
                    print(f"    ❌ Error loading model: {e}")
            
            if os.path.exists(vectorizer_file):
                print(f"\n    Trying to load {vectorizer_file}...")
                try:
                    with open(vectorizer_file, 'rb') as f:
                        vectorizer = pickle.load(f)
                    print(f"    ✅ Vectorizer loaded successfully! Type: {type(vectorizer)}")
                except Exception as e:
                    print(f"    ❌ Error loading vectorizer: {e}")
