import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from absa_dataset import ASPECTS

ASPECT_MAP_EN = {
    'Chất lượng sản phẩm': 'Product Quality',
    'Hiệu năng & Trải nghiệm': 'Performance & Exp',
    'Đúng mô tả': 'Correct Desc',
    'Giá cả & Khuyến mãi': 'Price & Promo',
    'Vận chuyển': 'Shipping',
    'Đóng gói': 'Packaging',
    'Dịch vụ & Thái độ Shop': 'Service & Attitude',
    'Bảo hành & Đổi trả': 'Warranty & Returns',
    'Tính xác thực': 'Authenticity'
}

# The user wants these classes specifically
ALLOWED_LABELS = ['-1', '0', '1', '2', '-1, 1']
LABEL_NAMES = ['Negative', 'Neutral', 'Positive', 'N/A', 'Mixed (-1, 1)']

def normalize_label(val):
    """Normalize labels, keeping -1, 1 separate and filtering out forbidden mixed classes."""
    if pd.isna(val):
        return "2"
    val_str = str(val).strip()
    if val_str in ('2', '2.0', 'nan', ''):
        return "2"
    
    # Handle mixed labels
    if "," in val_str:
        # Extract unique integer parts
        try:
            parts = sorted(list(set([str(int(float(x.strip()))) for x in val_str.split(",")])))
            # The only allowed mixed label is -1 and 1
            if parts == ['-1', '1']:
                return "-1, 1"
            # If it's something else like [-1, 0] or [1, 0], we'll treat it as "Invalid" for filtering
            return "INVALID_MIXED"
        except:
            return "2"
            
    try:
        label = str(int(float(val_str)))
        if label in ALLOWED_LABELS:
            return label
        return "2"
    except:
        return "2"

def main():
    true_file = os.path.join(ROOT, 'True_Test_Data', 'dev_cleaned_cleaned_cleaned.xlsx')
    pred_file = os.path.join(ROOT, 'True_Test_Data', 'dev_predicted.xlsx')
    viz_dir = os.path.join(ROOT, 'visualizations', 'confusion_matrices')
    os.makedirs(viz_dir, exist_ok=True)
    
    if not os.path.exists(true_file) or not os.path.exists(pred_file):
        print("Missing files for evaluation.")
        return

    df_true = pd.read_excel(true_file)
    df_pred = pd.read_excel(pred_file)

    print(f"\n{'='*80}")
    print(f"  ABSA Evaluation Results (Cleaned Labels + Multi-Polarity)")
    print(f"{'='*80}\n")

    all_true = []
    all_pred = []

    for aspect in ASPECTS:
        if aspect not in df_true.columns or aspect not in df_pred.columns:
            continue
            
        y_true_raw = df_true[aspect].apply(normalize_label).tolist()
        y_pred_raw = df_pred[aspect].apply(normalize_label).tolist()
        
        # Filter out "INVALID_MIXED" cases as requested (xóa khỏi metric)
        y_true = []
        y_pred = []
        for t, p in zip(y_true_raw, y_pred_raw):
            if t != "INVALID_MIXED" and p != "INVALID_MIXED":
                y_true.append(t)
                y_pred.append(p)
        
        if not y_true:
            continue

        all_true.extend(y_true)
        all_pred.extend(y_pred)
        
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=ALLOWED_LABELS)
        
        aspect_en = ASPECT_MAP_EN.get(aspect, aspect)
        print(f"--- Aspect: {aspect_en} ---")
        print(f"Accuracy: {acc:.4f}\n")
        
        # Plotting
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
        plt.title(f'Confusion Matrix: {aspect_en}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        save_path = os.path.join(viz_dir, f"{aspect_en.replace(' ', '_').lower()}_cm.png")
        plt.savefig(save_path)
        plt.close()
        print(f"  Saved confusion matrix to: {save_path}")

    # Global Summary
    print(f"\n{'='*80}")
    print("  OVERALL CLASSIFICATION REPORT (Target Classes Only)")
    print(f"{'='*80}")
    print(classification_report(all_true, all_pred, target_names=LABEL_NAMES, labels=ALLOWED_LABELS, zero_division=0))

if __name__ == "__main__":
    main()
