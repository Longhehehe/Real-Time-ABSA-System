"""
Multi-Method ABSA Training Orchestrator
Selectively train and compare models for Multi-Polarity ABSA.

Usage:
    python train_all_methods.py --model phobert          # Train 1 model
    python train_all_methods.py --model bilstm cnn_bilstm # Train multiple
    python train_all_methods.py                           # Train ALL 6 models

Models available:
    ML-Based:    logistic_regression, naive_bayes
    Deep-Based:  bilstm, cnn_bilstm
    Transformer: phobert, xlm_roberta
"""
import os
import sys
import json
import argparse
import warnings
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

from training_plots import (
    save_loss_plot,
    save_metric_bars,
    save_models_metrics_comparison,
    save_fold_scores_plot,
)

from absa_dataset import (
    load_data_multipolarity,
    ABSADatasetMultiPolarity,
    ASPECTS,
)

from methods import (
    compute_all_metrics, print_fold_summary, build_comparison_table,
    LogisticRegressionABSA, NaiveBayesABSA,
    BiLSTMForABSA, CNNBiLSTMForABSA,
    PhoBERTForABSAMultiPolarity, XLMRoBERTaForABSA,
)

NUM_ASPECTS = len(ASPECTS)

class BCEFocalLoss(nn.Module):
    """
    Focal Loss for Multi-label classification with Alpha balancing and Label Smoothing.
    Helps the model focus on hard examples and minority classes.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha # Can be a float or a tensor of shape (num_classes,)
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, mask=None):
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_term = (1 - pt) ** self.gamma

        if self.alpha is not None:
            # If alpha is a tensor, we need to apply it per-class
            if torch.is_tensor(self.alpha):
                # Ensure alpha is on the same device as logits
                alpha = self.alpha.to(logits.device)
                # alpha_term: if target > 0.5 use alpha, else use 1-alpha
                alpha_term = torch.where(targets > 0.5, alpha, 1 - alpha)
            else:
                alpha_term = torch.where(targets > 0.5, self.alpha, 1 - self.alpha)
            focal_loss = alpha_term * focal_term * bce_loss
        else:
            focal_loss = focal_term * bce_loss

        if mask is not None:
            # If mask is (batch, num_aspects), expand to (batch, num_aspects, 3) for sentiment
            if mask.dim() < targets.dim():
                mask = mask.unsqueeze(-1)
            focal_loss = focal_loss * mask

        if self.reduction == 'mean':
            if mask is not None:
                # Average only over masked (mentioned) aspects
                return focal_loss.sum() / (mask.sum() * targets.size(-1) + 1e-9)
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ── Dynamic Threshold Tuning Helpers ────────────────────────────────────────

def _threshold_grid(min_t: float = 0.1, max_t: float = 0.9, steps: int = 17) -> np.ndarray:
    return np.linspace(min_t, max_t, steps)


def _collect_probs(model, loader, device: str):
    """Run model on loader, return true labels and predicted probabilities."""
    model.eval()
    tm, ts, pm, ps = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            logits_m, logits_s = model(ids, mask)
            pm.append(torch.sigmoid(logits_m).cpu().numpy())
            ps.append(torch.sigmoid(logits_s).cpu().numpy())
            tm.append(batch['labels_m'].numpy())
            ts.append(batch['labels_s'].numpy())
    return (
        np.vstack(tm),
        np.concatenate(ts, axis=0),
        np.vstack(pm),
        np.concatenate(ps, axis=0),
    )


def _tune_thresholds_m(prob_m: np.ndarray, true_m: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Per-aspect threshold search on mention head."""
    best = np.full(true_m.shape[1], 0.5, dtype=np.float32)
    for a in range(true_m.shape[1]):
        y_true = true_m[:, a]
        if y_true.sum() == 0:
            continue
        best_f1 = -1.0
        for t in grid:
            y_pred = (prob_m[:, a] >= t).astype(int)
            score  = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1, best[a] = score, t
    return best


def _tune_thresholds_s(prob_s: np.ndarray, true_s: np.ndarray,
                       true_m: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Per-aspect per-class threshold search on sentiment head."""
    best = np.full((true_s.shape[1], true_s.shape[2]), 0.5, dtype=np.float32)
    for a in range(true_s.shape[1]):
        mask = true_m[:, a] == 1
        if mask.sum() == 0:
            continue
        for c in range(true_s.shape[2]):
            y_true   = true_s[mask, a, c]
            best_f1  = -1.0
            for t in grid:
                y_pred = (prob_s[mask, a, c] >= t).astype(int)
                score  = f1_score(y_true, y_pred, zero_division=0)
                if score > best_f1:
                    best_f1, best[a, c] = score, t
    return best


def _apply_thresholds(prob_m, prob_s, th_m, th_s, enforce_neu: bool = True):
    pred_m = (prob_m >= th_m.reshape(1, -1)).astype(np.float32)
    pred_s = (prob_s >= th_s.reshape(1, th_s.shape[0], th_s.shape[1])).astype(np.float32)
    if enforce_neu:
        no_sent = (pred_m == 1) & (pred_s.sum(axis=2) == 0)
        pred_s[no_sent, 2] = 1.0
    return pred_m, pred_s


MODEL_REGISTRY = {
    'logistic_regression': ('ML-Based', LogisticRegressionABSA),
    'naive_bayes':         ('ML-Based', NaiveBayesABSA),
    'bilstm':              ('Deep-Based', BiLSTMForABSA),
    'cnn_bilstm':          ('Deep-Based', CNNBiLSTMForABSA),
    'phobert':             ('Transformer', PhoBERTForABSAMultiPolarity),
    'xlm_roberta':         ('Transformer', XLMRoBERTaForABSA),
}

def train_ml_kfold(
    model_class,
    texts: List[str],
    labels_m: np.ndarray,
    labels_s: np.ndarray,
    output_dir: str,
    n_folds: int = 5,
    threshold_min: float = 0.1,
    threshold_max: float = 0.9,
    threshold_steps: int = 17,
) -> Dict:
    """Train ML model with K-Fold CV + Dynamic Threshold Tuning.
    Note: Early Stopping does not apply to ML models (no epochs).
    """
    name = model_class.__name__
    print(f"\n{'='*70}")
    print(f"  Training {name} with {n_folds}-Fold CV")
    print(f"{'='*70}")

    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X = tfidf.fit_transform(texts)

    grid  = _threshold_grid(threshold_min, threshold_max, threshold_steps)
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_fold_metrics = []
    best_fold, best_f1 = 0, -1.0
    best_th_m, best_th_s = None, None

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n  FOLD {fold+1}/{n_folds} — Train: {len(train_idx)}, Val: {len(val_idx)}")

        model = model_class()
        model.tfidf = tfidf
        model.fit(X[train_idx], labels_m[train_idx], labels_s[train_idx])

        # Get raw probabilities from ML model
        _, _, prob_m, prob_s = model.predict(X[val_idx])
        true_m = labels_m[val_idx]
        true_s = labels_s[val_idx]

        # Dynamic threshold tuning on val set
        th_m = _tune_thresholds_m(prob_m, true_m, grid)
        th_s = _tune_thresholds_s(prob_s, true_s, true_m, grid)
        pred_m, pred_s = _apply_thresholds(prob_m, prob_s, th_m, th_s)

        fm = compute_all_metrics(true_m, pred_m, true_s, pred_s, prob_m, prob_s)
        all_fold_metrics.append(fm)

        score = fm['combined_score']
        print(f"    M-F1={fm['mention_f1_macro']:.4f} | S-F1s={fm['sentiment_f1_samples']:.4f} "
              f"| M-AUC={fm['mention_auc_roc']:.4f} | S-AUC={fm['sentiment_auc_roc']:.4f} "
              f"| Combined={score:.4f}")

        if score > best_f1:
            best_f1   = score
            best_fold = fold + 1
            best_th_m = th_m
            best_th_s = th_s
            model.save(output_dir)
            print(f"      ✓ New best (fold {fold+1}, score={score:.4f})")

    # Persist best thresholds alongside the saved ML model
    os.makedirs(output_dir, exist_ok=True)
    if best_th_m is not None:
        import json as _json
        th_path = os.path.join(output_dir, 'thresholds.json')
        _json.dump(
            {'thresholds_m': best_th_m.tolist(), 'thresholds_s': best_th_s.tolist()},
            open(th_path, 'w')
        )
        print(f"  Thresholds saved → {th_path}")

    # Save per-fold scores chart (substitute for training loss, since ML has no epochs)
    save_fold_scores_plot(
        all_fold_metrics,
        output_dir,
        filename='fold_scores.png',
        title=f"{name} — Per-Fold Validation Scores ({n_folds}-Fold CV)",
    )

    print_fold_summary(all_fold_metrics, name)
    return _save_results(name, all_fold_metrics, best_fold, best_f1, n_folds, output_dir)

def train_deep_kfold(
    model_class,
    tokenizer,
    texts: np.ndarray,
    labels_m: np.ndarray,
    labels_s: np.ndarray,
    output_dir: str,
    n_folds: int = 5,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    device: str = "cpu",
    is_transformer: bool = False,
    mention_weight: float = 2.0,
    sentiment_weight: float = 5.0,
    label_smoothing: float = 0.1,
    gamma: float = 2.0,
    patience: int = 3,
    threshold_min: float = 0.1,
    threshold_max: float = 0.9,
    threshold_steps: int = 17,
) -> Dict:
    """Train Deep/Transformer model with K-Fold CV."""
    name = model_class.__name__
    print(f"\n{'='*70}")
    print(f"  Training {name} with {n_folds}-Fold CV on {device}")
    print(f"{'='*70}")

    vocab_size = tokenizer.vocab_size
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_fold_metrics = []
    best_fold, best_f1 = 0, -1.0
    best_state = None
    best_th_m  = None
    best_th_s  = None
    fold_epoch_losses = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(texts)):
        print(f"\n  FOLD {fold+1}/{n_folds} — Train: {len(train_idx)}, Val: {len(val_idx)}")

        train_ds = ABSADatasetMultiPolarity(texts[train_idx].tolist(), labels_m[train_idx], labels_s[train_idx], tokenizer, max_length)
        val_ds = ABSADatasetMultiPolarity(texts[val_idx].tolist(), labels_m[val_idx], labels_s[val_idx], tokenizer, max_length)
        
        # Calculate weights for WeightedRandomSampler based on minority aspects
        train_labels_m = labels_m[train_idx]
        aspect_counts = train_labels_m.sum(axis=0) + 1e-6 # Tránh chia cho 0
        aspect_weights = len(train_labels_m) / aspect_counts
        
        # Gán trọng số cho từng sample dựa trên aspect hiếm nhất mà nó chứa
        sample_weights = np.zeros(len(train_labels_m))
        for i in range(len(train_labels_m)):
            present = np.where(train_labels_m[i] == 1)[0]
            if len(present) > 0:
                sample_weights[i] = aspect_weights[present].max()
            else:
                sample_weights[i] = 1.0 # Trọng số cơ bản cho mẫu "Không nhắc đến"

        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = model_class(num_aspects=NUM_ASPECTS) if is_transformer else model_class(vocab_size=vocab_size)
        model = model.to(device)

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)
        
        # Calculate alpha weights dynamically for this fold to handle class imbalance.
        # For mention detection:
        m_pos_counts = torch.tensor(labels_m[train_idx].sum(axis=0))
        m_total = len(train_idx)
        # alpha = (total - pos) / total -> gives more weight to positive class if it's rare
        # But standard alpha is weight for positive class. Let's use: alpha_pos = 1 - (pos/total)
        alpha_m = 1.0 - (m_pos_counts / m_total).clamp(min=0.05, max=0.95)

        # For sentiment:
        s_labels = labels_s[train_idx] # (N, 9, 3)
        s_pos_counts = torch.tensor(s_labels.sum(axis=(0))) # (9, 3)
        # Only count mentions for the denominator
        s_mentions = torch.tensor(labels_m[train_idx].sum(axis=0)).unsqueeze(-1).expand(-1, 3)
        alpha_s = 1.0 - (s_pos_counts / (s_mentions + 1e-6)).clamp(min=0.05, max=0.95)

        # Initialize Focal Loss with dynamic alpha and label smoothing
        crit_m = BCEFocalLoss(gamma=gamma, alpha=alpha_m, label_smoothing=label_smoothing)
        crit_s = BCEFocalLoss(gamma=gamma, alpha=alpha_s, label_smoothing=label_smoothing)

        grid = _threshold_grid(threshold_min, threshold_max, threshold_steps)

        # ── Per-epoch training + validation + early stopping ─────────────────
        epoch_losses    = []
        epochs_no_improve = 0
        fold_best_f1    = -1.0
        fold_best_state = None
        fold_best_th_m  = None
        fold_best_th_s  = None

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Fold {fold+1} Ep {epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                ids  = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                lm   = batch['labels_m'].to(device)
                ls   = batch['labels_s'].to(device)

                logits_m, logits_s = model(ids, mask)
                loss_m = crit_m(logits_m, lm)
                loss_s = crit_s(logits_s, ls, mask=lm)
                loss   = mention_weight * loss_m + sentiment_weight * loss_s

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                train_loss += loss.item()

            avg_loss = train_loss / max(len(train_loader), 1)
            epoch_losses.append(avg_loss)

            # ── Validation with Dynamic Threshold Tuning ─────────────────────
            true_m, true_s, prob_m, prob_s = _collect_probs(model, val_loader, device)
            th_m  = _tune_thresholds_m(prob_m, true_m, grid)
            th_s  = _tune_thresholds_s(prob_s, true_s, true_m, grid)
            pred_m, pred_s = _apply_thresholds(prob_m, prob_s, th_m, th_s)

            fm = compute_all_metrics(true_m, pred_m, true_s, pred_s, prob_m, prob_s)
            score = fm['combined_score']

            print(f"    Fold {fold+1} Ep {epoch+1}: loss={avg_loss:.4f} | "
                  f"M-F1={fm['mention_f1_macro']:.4f} | "
                  f"S-F1s={fm['sentiment_f1_samples']:.4f} | "
                  f"S-AUC={fm['sentiment_auc_roc']:.4f} | "
                  f"Combined={score:.4f}")

            if score > fold_best_f1:
                fold_best_f1    = score
                fold_best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                fold_best_th_m  = th_m.copy()
                fold_best_th_s  = th_s.copy()
                epochs_no_improve = 0
                print(f"      ✓ New fold best (score={score:.4f})")
            else:
                epochs_no_improve += 1
                print(f"      No improve {epochs_no_improve}/{patience}")
                if epochs_no_improve >= patience:
                    print(f"      Early stopping at epoch {epoch+1}.")
                    break



        # Re-evaluate with best state to get final fold metrics
        if fold_best_state is not None:
            model.load_state_dict(fold_best_state)
        true_m, true_s, prob_m, prob_s = _collect_probs(model, val_loader, device)
        th_m = fold_best_th_m if fold_best_th_m is not None else _tune_thresholds_m(prob_m, true_m, grid)
        th_s = fold_best_th_s if fold_best_th_s is not None else _tune_thresholds_s(prob_s, true_s, true_m, grid)
        pred_m, pred_s = _apply_thresholds(prob_m, prob_s, th_m, th_s)
        fm = compute_all_metrics(true_m, pred_m, true_s, pred_s, prob_m, prob_s)

        all_fold_metrics.append(fm)
        if epoch_losses:
            fold_epoch_losses.append(epoch_losses)

        print(f"    ── Fold {fold+1} Final: Combined={fm['combined_score']:.4f} | "
              f"M-F1={fm['mention_f1_macro']:.4f} | S-F1s={fm['sentiment_f1_samples']:.4f}")

        if fm['combined_score'] > best_f1:
            best_f1    = fm['combined_score']
            best_fold  = fold + 1
            best_state = fold_best_state
            best_th_m  = th_m
            best_th_s  = th_s
            print(f"     ★ Global best updated! (Fold {fold+1}, score={best_f1:.4f})")

        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    print_fold_summary(all_fold_metrics, name)

    os.makedirs(output_dir, exist_ok=True)
    if best_state:
        torch.save({
            'model_state_dict': best_state,
            'model_class': name,
            'aspects': ASPECTS,
            'best_combined_score': best_f1,
            'best_fold': best_fold,
            'multi_polarity': True,
            'thresholds_m': best_th_m.tolist() if best_th_m is not None else None,
            'thresholds_s': best_th_s.tolist() if best_th_s is not None else None,
        }, os.path.join(output_dir, f'{name.lower()}_absa.pt'))

    if fold_epoch_losses:
        avg_epoch_losses = np.mean(fold_epoch_losses, axis=0).tolist()
        save_loss_plot(
            avg_epoch_losses,
            output_dir,
            filename='training_loss_by_epoch.png',
            title=f"{name} Training Loss by Epoch (Avg {n_folds}-Fold)",
        )

    return _save_results(name, all_fold_metrics, best_fold, best_f1, n_folds, output_dir)

def _save_results(name, all_fold_metrics, best_fold, best_f1, n_folds, output_dir):
    """Compute averages and save results.json."""
    avg = {}
    for key in all_fold_metrics[0]:
        vals = [m[key] for m in all_fold_metrics]
        avg[key] = float(np.mean(vals))
        avg[f'{key}_std'] = float(np.std(vals))

    results = {
        'model': name,
        'n_folds': n_folds,
        'best_fold': best_fold,
        'best_combined_f1': best_f1,
        'avg_metrics': avg,
        'fold_results': all_fold_metrics,
        'timestamp': datetime.now().isoformat(),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results

def train_selected_models(
    data_path: str,
    model_names: List[str],
    base_output_dir: str = "./models",
    n_folds: int = 5,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    max_length: int = 256,
    device: str = None,
    label_smoothing: float = 0.1,
    gamma: float = 2.0,
    sentiment_weight: float = 5.0,
    patience: int = 3,
    threshold_min: float = 0.1,
    threshold_max: float = 0.9,
    threshold_steps: int = 17,
):
    """Train selected models and produce comparison if >1."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'#'*70}")
    print(f"  MULTI-METHOD ABSA TRAINING")
    print(f"  Models: {', '.join(model_names)}")
    print(f"  Device: {device} | Folds: {n_folds} | Epochs: {epochs}")
    print(f"{'#'*70}")

    print("\n Loading data...")
    texts, labels_m, labels_s = load_data_multipolarity(data_path)
    clean_texts = []
    clean_labels_m = []
    clean_labels_s = []

    for idx, text in enumerate(texts):
        if text is None or (isinstance(text, float) and np.isnan(text)):
            continue
        text_str = str(text).strip()
        if not text_str:
            continue
        clean_texts.append(text_str)
        clean_labels_m.append(labels_m[idx])
        clean_labels_s.append(labels_s[idx])

    if len(clean_texts) != len(texts):
        print(f"  Filtered {len(texts) - len(clean_texts)} empty/NaN texts")

    texts = clean_texts
    labels_m = np.array(clean_labels_m)
    labels_s = np.array(clean_labels_s)
    texts_np = np.array(texts)
    print(f"  Samples: {len(texts)} | Mention: {labels_m.shape} | Sentiment: {labels_s.shape}")

    all_results = {}
    tokenizers_cache = {}

    for model_key in model_names:
        if model_key not in MODEL_REGISTRY:
            print(f"\n  Unknown model: {model_key}. Skipping.")
            continue

        category, model_class = MODEL_REGISTRY[model_key]
        out_dir = os.path.join(base_output_dir, f'{model_key}_absa')

        if category == 'ML-Based':
            result = train_ml_kfold(
                model_class, texts, labels_m, labels_s, out_dir, n_folds,
                threshold_min=threshold_min, threshold_max=threshold_max,
                threshold_steps=threshold_steps,
            )

        elif category == 'Deep-Based':
            if 'phobert' not in tokenizers_cache:
                print("\n Loading PhoBERT tokenizer...")
                tokenizers_cache['phobert'] = AutoTokenizer.from_pretrained("vinai/phobert-base")
            tok = tokenizers_cache['phobert']
            result = train_deep_kfold(
                model_class, tok, texts_np, labels_m, labels_s, out_dir,
                n_folds, epochs, batch_size, learning_rate=1e-3,
                max_length=max_length, device=device, is_transformer=False,
                label_smoothing=0.0, gamma=gamma, sentiment_weight=sentiment_weight,
                patience=patience, threshold_min=threshold_min,
                threshold_max=threshold_max, threshold_steps=threshold_steps,
            )

        elif category == 'Transformer':
            if model_key == 'phobert':
                tok_name = 'phobert'
                tok_hf = "vinai/phobert-base"
            else:
                tok_name = 'xlm_roberta'
                tok_hf = "xlm-roberta-base"

            if tok_name not in tokenizers_cache:
                print(f"\n Loading {tok_hf} tokenizer...")
                tokenizers_cache[tok_name] = AutoTokenizer.from_pretrained(tok_hf)
            tok = tokenizers_cache[tok_name]

            result = train_deep_kfold(
                model_class, tok, texts_np, labels_m, labels_s, out_dir,
                n_folds, epochs, batch_size, learning_rate=learning_rate,
                max_length=max_length, device=device, is_transformer=True,
                mention_weight=2.0, sentiment_weight=sentiment_weight,
                label_smoothing=label_smoothing, gamma=gamma,
                patience=patience, threshold_min=threshold_min,
                threshold_max=threshold_max, threshold_steps=threshold_steps,
            )

        display_name = model_key.replace('_', ' ').title()
        all_results[display_name] = result

        avg = result.get('avg_metrics', {})
        metric_order = [
            ('Mention F1',           'mention_f1_macro'),
            ('Sentiment F1',         'sentiment_f1_macro'),
            ('Sentiment F1 (samps)', 'sentiment_f1_samples'),
            ('Mention AUC-ROC',      'mention_auc_roc'),
            ('Sentiment AUC-ROC',    'sentiment_auc_roc'),
            ('Combined Score',       'combined_score'),
        ]
        metrics = {label: float(avg.get(key, 0.0)) for label, key in metric_order}
        save_metric_bars(
            metrics,
            out_dir,
            filename='metrics_comparison.png',
            title=f"{display_name} Metrics Comparison",
        )

    if len(all_results) > 1:
        build_comparison_table(all_results)

        metric_order = [
            ('Mention F1', 'mention_f1_macro'),
            ('Sentiment F1', 'sentiment_f1_macro'),
            ('Sentiment F1 (samples)', 'sentiment_f1_samples'),
            ('Combined F1', 'combined_f1'),
        ]
        models_metrics = {
            name: result.get('avg_metrics', {})
            for name, result in all_results.items()
        }
        save_models_metrics_comparison(
            models_metrics,
            base_output_dir,
            filename='all_models_metrics_comparison.png',
            title='Model Metrics Comparison',
            metric_order=metric_order,
        )

    combined_path = os.path.join(base_output_dir, 'all_models_comparison.json')
    serializable = {name: {'model': r['model'], 'best_fold': r['best_fold'],
                           'best_combined_f1': r['best_combined_f1'], 'avg_metrics': r['avg_metrics']}
                    for name, r in all_results.items()}
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"\n Results saved to: {combined_path}")

    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Method ABSA Training")
    parser.add_argument('--data', type=str, default=None, help='Path to labeled data')
    parser.add_argument('--output', type=str, default='./models', help='Output directory')
    parser.add_argument('--model', type=str, nargs='+', default=None,
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model(s) to train (default: all)')
    parser.add_argument('--folds', type=int, default=5, help='K-Fold splits')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs (deep/transformer)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing for transformers')
    parser.add_argument('--gamma', type=float, default=2.0, help='Gamma for Focal Loss')
    parser.add_argument('--sentiment_weight', type=float, default=5.0, help='Weight for sentiment loss')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--threshold_min', type=float, default=0.1, help='Min threshold for grid search')
    parser.add_argument('--threshold_max', type=float, default=0.9, help='Max threshold for grid search')
    parser.add_argument('--threshold_steps', type=int, default=17, help='Number of steps in threshold grid')

    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = args.data or os.path.join(BASE_DIR, 'New database')

    if not os.path.exists(data_path):
        print(f" Data not found: {data_path}")
        sys.exit(1)

    models = args.model if args.model else list(MODEL_REGISTRY.keys())

    train_selected_models(
        data_path=data_path,
        model_names=models,
        base_output_dir=args.output,
        n_folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        label_smoothing=args.label_smoothing,
        gamma=args.gamma,
        sentiment_weight=args.sentiment_weight,
        patience=args.patience,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_steps=args.threshold_steps,
    )
