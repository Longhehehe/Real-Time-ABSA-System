"""
Shared evaluation engine for Multi-Polarity ABSA.
Computes comprehensive metrics per fold: Precision, Recall, F1 (macro/micro/weighted),
AUC-ROC, AUC-PR for both Mention Detection and Sentiment Classification.
"""
import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)


def compute_all_metrics(
    true_m: np.ndarray,
    pred_m: np.ndarray,
    true_s: np.ndarray,
    pred_s: np.ndarray,
    prob_m: np.ndarray = None,
    prob_s: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for one fold/split.

    Args:
        true_m: Ground truth mention labels [N, num_aspects]
        pred_m: Predicted mention labels [N, num_aspects]
        true_s: Ground truth sentiment [N, num_aspects, 3] multi-hot
        pred_s: Predicted sentiment [N, num_aspects, 3] multi-hot
        prob_m: Mention probabilities [N, num_aspects] (for AUC)
        prob_s: Sentiment probabilities [N, num_aspects, 3] (for AUC)

    Returns:
        Dict with all metrics
    """
    metrics = {}

    # --- Mention Detection Metrics ---
    m_true_flat = true_m.flatten()
    m_pred_flat = pred_m.flatten()

    metrics['mention_precision_macro'] = precision_score(m_true_flat, m_pred_flat, average='macro', zero_division=0)
    metrics['mention_recall_macro'] = recall_score(m_true_flat, m_pred_flat, average='macro', zero_division=0)
    metrics['mention_f1_macro'] = f1_score(m_true_flat, m_pred_flat, average='macro', zero_division=0)
    metrics['mention_f1_micro'] = f1_score(m_true_flat, m_pred_flat, average='micro', zero_division=0)
    metrics['mention_f1_weighted'] = f1_score(m_true_flat, m_pred_flat, average='weighted', zero_division=0)

    # Mention AUC-ROC & AUC-PR
    if prob_m is not None:
        try:
            metrics['mention_auc_roc'] = roc_auc_score(m_true_flat, prob_m.flatten())
        except ValueError:
            metrics['mention_auc_roc'] = 0.0
        try:
            metrics['mention_auc_pr'] = average_precision_score(m_true_flat, prob_m.flatten())
        except ValueError:
            metrics['mention_auc_pr'] = 0.0
    else:
        metrics['mention_auc_roc'] = 0.0
        metrics['mention_auc_pr'] = 0.0

    # --- Sentiment Metrics (only on mentioned aspects) ---
    mentioned_mask = true_m.flatten() == 1
    true_s_flat = true_s.reshape(-1, 3)
    pred_s_flat = pred_s.reshape(-1, 3)

    true_s_mentioned = true_s_flat[mentioned_mask]
    pred_s_mentioned = pred_s_flat[mentioned_mask]

    if len(true_s_mentioned) > 0:
        metrics['sentiment_precision_macro'] = precision_score(true_s_mentioned, pred_s_mentioned, average='macro', zero_division=0)
        metrics['sentiment_recall_macro'] = recall_score(true_s_mentioned, pred_s_mentioned, average='macro', zero_division=0)
        metrics['sentiment_f1_macro'] = f1_score(true_s_mentioned, pred_s_mentioned, average='macro', zero_division=0)
        metrics['sentiment_f1_micro'] = f1_score(true_s_mentioned, pred_s_mentioned, average='micro', zero_division=0)
        metrics['sentiment_f1_weighted'] = f1_score(true_s_mentioned, pred_s_mentioned, average='weighted', zero_division=0)
        metrics['sentiment_f1_samples'] = f1_score(true_s_mentioned, pred_s_mentioned, average='samples', zero_division=0)

        if prob_s is not None:
            prob_s_flat = prob_s.reshape(-1, 3)
            prob_s_mentioned = prob_s_flat[mentioned_mask]
            try:
                metrics['sentiment_auc_roc'] = roc_auc_score(true_s_mentioned, prob_s_mentioned, average='macro')
            except ValueError:
                metrics['sentiment_auc_roc'] = 0.0
            try:
                metrics['sentiment_auc_pr'] = average_precision_score(true_s_mentioned, prob_s_mentioned, average='macro')
            except ValueError:
                metrics['sentiment_auc_pr'] = 0.0
        else:
            metrics['sentiment_auc_roc'] = 0.0
            metrics['sentiment_auc_pr'] = 0.0
    else:
        for k in ['sentiment_precision_macro', 'sentiment_recall_macro',
                   'sentiment_f1_macro', 'sentiment_f1_micro', 'sentiment_f1_weighted',
                   'sentiment_f1_samples', 'sentiment_auc_roc', 'sentiment_auc_pr']:
            metrics[k] = 0.0

    # Combined F1
    metrics['combined_f1'] = (metrics['mention_f1_macro'] + metrics['sentiment_f1_samples']) / 2

    return metrics


def print_fold_summary(all_fold_metrics: List[Dict], model_name: str):
    """Print per-fold table similar to Bảng 5.1 & 5.3."""
    n = len(all_fold_metrics)
    print(f"\n{'='*90}")
    print(f"  {model_name} — {n}-Fold Cross-Validation Results")
    print(f"{'='*90}")

    key_groups = {
        'Mention Detection': [
            ('Precision (macro)', 'mention_precision_macro'),
            ('Recall (macro)', 'mention_recall_macro'),
            ('F1-macro', 'mention_f1_macro'),
            ('F1-micro', 'mention_f1_micro'),
            ('F1-weighted', 'mention_f1_weighted'),
            ('AUC-ROC', 'mention_auc_roc'),
            ('AUC-PR', 'mention_auc_pr'),
        ],
        'Sentiment (Multi-Polarity)': [
            ('Precision (macro)', 'sentiment_precision_macro'),
            ('Recall (macro)', 'sentiment_recall_macro'),
            ('F1-macro', 'sentiment_f1_macro'),
            ('F1-micro', 'sentiment_f1_micro'),
            ('F1-weighted', 'sentiment_f1_weighted'),
            ('F1-samples', 'sentiment_f1_samples'),
            ('AUC-ROC', 'sentiment_auc_roc'),
            ('AUC-PR', 'sentiment_auc_pr'),
        ],
        'Combined': [
            ('Combined F1', 'combined_f1'),
        ],
    }

    for group_name, keys in key_groups.items():
        print(f"\n  --- {group_name} ---")
        header = f"  {'Metric':<22}" + "".join(f"{'Fold '+str(i+1):>10}" for i in range(n)) + f"{'Average':>10}{'Std Dev':>10}"
        print(header)
        print("  " + "-" * (22 + 10 * n + 20))

        for display_name, key in keys:
            vals = [m[key] for m in all_fold_metrics]
            avg = np.mean(vals)
            std = np.std(vals)
            row = f"  {display_name:<22}" + "".join(f"{v:>10.4f}" for v in vals) + f"{avg:>10.4f}{std:>10.5f}"
            print(row)


def build_comparison_table(all_results: Dict[str, Dict]):
    """Print final comparison table across all models."""
    print(f"\n{'='*110}")
    print(f"  FINAL MODEL COMPARISON — Multi-Polarity ABSA")
    print(f"{'='*110}")

    header_keys = [
        ('Prec(M)', 'mention_precision_macro'),
        ('Rec(M)', 'mention_recall_macro'),
        ('F1mac(M)', 'mention_f1_macro'),
        ('ROC(M)', 'mention_auc_roc'),
        ('Prec(S)', 'sentiment_precision_macro'),
        ('Rec(S)', 'sentiment_recall_macro'),
        ('F1mac(S)', 'sentiment_f1_macro'),
        ('ROC(S)', 'sentiment_auc_roc'),
        ('Combined', 'combined_f1'),
    ]

    header = f"  {'Model':<24}" + "".join(f"{h:>10}" for h, _ in header_keys)
    print(header)
    print("  " + "-" * (24 + 10 * len(header_keys)))

    for model_name, result in all_results.items():
        avg = result.get('avg_metrics', {})
        row = f"  {model_name:<24}" + "".join(f"{avg.get(k, 0.0):>10.4f}" for _, k in header_keys)
        print(row)

    print(f"{'='*110}\n")
