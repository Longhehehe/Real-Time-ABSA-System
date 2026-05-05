import pandas as pd
import numpy as np
import os
import ast

ASPECTS = [
    'Chất lượng sản phẩm', 'Hiệu năng & Trải nghiệm', 'Đúng mô tả', 
    'Giá cả & Khuyến mãi', 'Vận chuyển', 'Đóng gói', 
    'Dịch vụ & Thái độ Shop', 'Bảo hành & Đổi trả', 'Tính xác thực',
]

def parse_label(val):
    if pd.isna(val) or str(val).strip() in ('2', 'nan', ''):
        return None
    val_str = str(val).replace('[', '').replace(']', '').strip()
    try:
        if ',' in val_str:
            return [int(x.strip()) for x in val_str.split(',')]
        return int(float(val_str))
    except:
        return None

def analyze():
    file_path = r'c:\Users\Luc\Real-Time-ABSA-System\Augmented Dataset\augmented_result.xlsx'
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_excel(file_path)
    print(f"Total rows: {len(df)}")
    
    pos_total, neg_total, neu_total = 0, 0, 0
    
    for aspect in ASPECTS:
        if aspect not in df.columns: continue
        labels = df[aspect].apply(parse_label)
        for lbl in labels:
            if lbl is None: continue
            if isinstance(lbl, list):
                if 1 in lbl: pos_total += 1
                if -1 in lbl: neg_total += 1
                if 0 in lbl: neu_total += 1
            else:
                if lbl == 1: pos_total += 1
                elif lbl == -1: neg_total += 1
                elif lbl == 0: neu_total += 1
                
    print("\nPer-aspect sentiment distribution:")
    for aspect in ASPECTS:
        if aspect not in df.columns: continue
        labels = df[aspect].apply(parse_label)
        p, n, neu = 0, 0, 0
        for lbl in labels:
            if lbl is None: continue
            if isinstance(lbl, list):
                if 1 in lbl: p += 1
                if -1 in lbl: n += 1
                if 0 in lbl: neu += 1
            else:
                if lbl == 1: p += 1
                elif lbl == -1: n += 1
                elif lbl == 0: neu += 1
        print(f"  Aspect {ASPECTS.index(aspect):<2}: POS={p:<5} NEG={n:<5} NEU={neu:<5}")

if __name__ == "__main__":
    analyze()
