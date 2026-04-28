"""
ABSA Shared Utilities — Constants, Data Loading, Prediction.

This module contains shared resources used by all models:
  - ASPECTS, LABEL_TO_INDEX (constants)
  - ABSADatasetMultiPolarity (PyTorch Dataset)
  - load_data_multipolarity (Data loader)
  - predict_multipolarity (Inference)

Model definitions are in methods/ package.
Training is handled by train_all_methods.py.
"""
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
from transformers import AutoTokenizer

ASPECTS = [
    'Chất lượng sản phẩm',                                       
    'Hiệu năng & Trải nghiệm',                                 
    'Đúng mô tả',                                         
    'Giá cả & Khuyến mãi',                                
    'Vận chuyển',                                          
    'Đóng gói',                                     
    'Dịch vụ & Thái độ Shop',                                       
    'Bảo hành & Đổi trả',                           
    'Tính xác thực',                                          
]

LABEL_TO_INDEX = {
    -1: 0,                   
    1: 1,                    
    0: 2,                    
}

class ABSADatasetMultiPolarity(Dataset):
    """PyTorch Dataset for Multi-Polarity ABSA.
    labels_s is [batch, num_aspects, 3] multi-hot instead of [batch, num_aspects].
    """

    def __init__(self, texts: List[str], labels_m: np.ndarray, labels_s: np.ndarray,
                 tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels_m = labels_m
        self.labels_s = labels_s
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]), truncation=True,
            max_length=self.max_length, padding='max_length', return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels_m': torch.tensor(self.labels_m[idx], dtype=torch.float),
            'labels_s': torch.tensor(self.labels_s[idx], dtype=torch.float),
        }

def load_data_multipolarity(data_path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Load and preprocess data from Excel/CSV for MULTI-POLARITY format.

    Returns:
        texts, labels_m [N, 9], labels_s [N, 9, 3]
    """
    import glob

    if os.path.isdir(data_path):
        print(f"   Loading from folder: {data_path}")
        files = sorted(glob.glob(os.path.join(data_path, '*.xlsx')))
        print(f"   Found {len(files)} files")
        dfs = []
        for f in files:
            df = pd.read_excel(f)
            dfs.append(df)
            print(f"      - {os.path.basename(f)}: {len(df)} rows")
        df = pd.concat(dfs, ignore_index=True)
        print(f"   Total: {len(df)} rows")
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)

    texts = df['reviewContent'].tolist()
    labels_m, labels_s = [], []

    for _, row in df.iterrows():
        row_m, row_s = [], []
        for aspect in ASPECTS:
            if aspect in df.columns:
                val = row[aspect]
                val_str = str(val).strip() if pd.notna(val) else '2'
                val_str = val_str.replace('[', '').replace(']', '').strip()

                if val_str in ('2', 'nan', ''):
                    row_m.append(0)
                    row_s.append([0, 0, 0])
                else:
                    row_m.append(1)
                    sv = [0, 0, 0]
                    if ',' in val_str:
                        try:
                            for lbl in [int(x.strip()) for x in val_str.split(',')]:
                                if lbl in LABEL_TO_INDEX:
                                    sv[LABEL_TO_INDEX[lbl]] = 1
                        except ValueError:
                            sv[2] = 1
                    else:
                        try:
                            lbl = int(float(val_str))
                            sv[LABEL_TO_INDEX.get(lbl, 2)] = 1
                        except ValueError:
                            sv[2] = 1
                    row_s.append(sv)
            else:
                row_m.append(0)
                row_s.append([0, 0, 0])
        labels_m.append(row_m)
        labels_s.append(row_s)

    return texts, np.array(labels_m, dtype=np.float32), np.array(labels_s, dtype=np.float32)
