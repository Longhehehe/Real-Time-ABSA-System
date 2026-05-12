import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import argparse
from tqdm import tqdm
from datetime import datetime

# Add root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from absa_dataset import ASPECTS
from methods.evaluation import compute_all_metrics
from app.absa_predictor import GeneralABSAPredictor

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

ALLOWED_LABELS = ['-1', '0', '1', '2', '1, -1']
LABEL_NAMES = ['Negative', 'Neutral', 'Positive', 'N/A', 'Mixed (1, -1)']

def normalize_label(val):
    if pd.isna(val): return "2"
    val_str = str(val).strip()
    if val_str in ('2', '2.0', 'nan', ''): return "2"
    if "," in val_str:
        try:
            parts = sorted(list(set([str(int(float(x.strip()))) for x in val_str.split(",")])), reverse=True)
            if parts == ['1', '-1']: return "1, -1"
            return "INVALID_MIXED"
        except: return "2"
    try:
        label = str(int(float(val_str)))
        if label in ALLOWED_LABELS: return label
        return "2"
    except: return "2"

def main():
    parser = argparse.ArgumentParser(description='Evaluate ABSA model on True Test Data')
    parser.add_argument('--model', '-m', type=str, default='xlm', 
                        help='Model keyword (phobert|xlm|bilstm|cnn_bilstm|lr|nb) or explicit path')
    parser.add_argument('--data', '-d', type=str, default='dev_augmented_500.xlsx',
                        help='True test data filename in True_Test_Data folder')
    args = parser.parse_args()

    # 1. Resolve model path
    model_map = {
        'phobert': os.path.join(ROOT, 'models', 'phobert_absa', 'phobertforabsamultipolarity_absa.pt'),
        'xlm': os.path.join(ROOT, 'models', 'xlm_roberta_absa', 'xlmrobertaforabsa_absa.pt'),
        'bilstm': os.path.join(ROOT, 'models', 'bilstm_absa', 'bilstmforabsa_absa.pt'),
        'cnn_bilstm': os.path.join(ROOT, 'models', 'cnn_bilstm_absa', 'cnnbilstmforabsa_absa.pt'),
        'lr': os.path.join(ROOT, 'models', 'logistic_regression_absa', 'logistic_regression_model.pkl'),
        'nb': os.path.join(ROOT, 'models', 'naive_bayes_absa', 'naive_bayes_model.pkl')
    }
    
    model_path = model_map.get(args.model.lower(), args.model)
    if not os.path.exists(model_path):
        # Try local path
        model_path = os.path.join(ROOT, args.model)
        if not os.path.exists(model_path):
            print(f"Model path not found: {args.model}")
            print(f"Available keywords: {list(model_map.keys())}")
            return

    true_file = os.path.join(ROOT, 'True_Test_Data', args.data)
    if not os.path.exists(true_file):
        print(f"Missing true data file: {true_file}")
        return

    # 2. Initialize Predictor
    predictor = GeneralABSAPredictor(model_path)
    
    df_true = pd.read_excel(true_file)
    num_samples = len(df_true)
    
    print(f"\n{'='*80}")
    print(f"  ABSA LIVE EVALUATION: {predictor.model_class_name}")
    print(f"  Testing on {num_samples} samples from: {os.path.basename(true_file)}")
    print(f"{'='*80}\n")

    # Prepare data structures
    num_aspects = len(ASPECTS)
    true_m = np.zeros((num_samples, num_aspects))
    pred_m = np.zeros((num_samples, num_aspects))
    true_s = np.zeros((num_samples, num_aspects, 3)) 
    pred_s = np.zeros((num_samples, num_aspects, 3))
    prob_m = np.zeros((num_samples, num_aspects))
    prob_s = np.zeros((num_samples, num_aspects, 3))
    
    sentiment_idx_map = {"-1": 0, "1": 1, "0": 2}
    all_y_true, all_y_pred = [], []

    # 3. Run Inference
    print("Running inference...")
    for i, row in tqdm(df_true.iterrows(), total=num_samples):
        text = str(row['reviewContent'])
        res = predictor.predict_single(text)
        
        prob_m[i] = res['probs']['mention']
        prob_s[i] = res['probs']['sentiment']
        
        for a_idx, aspect in enumerate(ASPECTS):
            t_val = normalize_label(row[aspect])
            if t_val == "INVALID_MIXED": continue
            
            if t_val != "2":
                true_m[i, a_idx] = 1
                for part in t_val.split(","):
                    p = part.strip()
                    if p in sentiment_idx_map: true_s[i, a_idx, sentiment_idx_map[p]] = 1
            
            multi_info = res['multipolarity'][aspect]
            if multi_info['mentioned']:
                pred_m[i, a_idx] = 1
                for s in multi_info['sentiments']:
                    s_idx = 0 if s == 'NEG' else (1 if s == 'POS' else 2)
                    pred_s[i, a_idx, s_idx] = 1
            
            p_val = res['legacy'][aspect]
            all_y_true.append(t_val)
            all_y_pred.append(str(p_val))

    # 4. Compute Metrics
    print("\nComputing comprehensive metrics...")
    metrics = compute_all_metrics(true_m, pred_m, true_s, pred_s, prob_m, prob_s)
    
    # Save to JSON
    out_dir = os.path.join(ROOT, 'test_results')
    os.makedirs(out_dir, exist_ok=True)
    safe_model_name = args.model.replace('/', '_').replace('\\', '_').replace(':', '_').replace('.pt', '').replace('.pkl', '')
    out_file = os.path.join(out_dir, f"{safe_model_name}_test_results.json")
    
    result_data = {
        "model": predictor.model_class_name,
        "test_data": args.data,
        "num_samples": num_samples,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "timestamp": datetime.now().isoformat()
    }
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"  [+] Saved evaluation metrics to: {out_file}")

    # 5. Display
    print(f"\n{'='*80}")
    print(f"  FINAL RESULTS FOR TEST SET")
    print(f"{'='*80}")
    
    groups = {
        "Mention Detection": ["mention_precision_macro", "mention_recall_macro", "mention_f1_macro", "mention_auc_roc", "mention_auc_pr"],
        "Sentiment Classification": ["sentiment_precision_macro", "sentiment_recall_macro", "sentiment_f1_macro", "sentiment_f1_samples", "sentiment_auc_roc", "sentiment_auc_pr"],
        "Combined (7 Symmetric Metrics)": ["combined_precision_macro", "combined_recall_macro", "combined_f1_macro", "combined_f1_micro", "combined_f1_weighted", "combined_auc_roc", "combined_auc_pr"],
        "Overall Performance": ["combined_score"]
    }
    
    for g_name, keys in groups.items():
        print(f"\n  --- {g_name} ---")
        for k in keys:
            val = metrics.get(k)
            if val is not None: print(f"  {k:.<35} {val:.4f}")
    
    print(f"\n{'='*80}")
    print("  FLAT CLASSIFICATION REPORT")
    print(f"{'='*80}")
    # Convert legacy back to names for report
    report_pred = [LABEL_NAMES[ALLOWED_LABELS.index(p)] if p in ALLOWED_LABELS else "N/A" for p in all_y_pred]
    report_true = [LABEL_NAMES[ALLOWED_LABELS.index(t)] if t in ALLOWED_LABELS else "N/A" for t in all_y_true]
    print(classification_report(report_true, report_pred, zero_division=0))

if __name__ == "__main__":
    main()
