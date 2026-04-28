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
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

warnings.filterwarnings('ignore')

from training_plots import (
    save_loss_plot,
    save_metric_bars,
    save_models_metrics_comparison,
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
) -> Dict:
    """Train ML model with K-Fold CV."""
    name = model_class.__name__
    print(f"\n{'='*70}")
    print(f"  Training {name} with {n_folds}-Fold CV")
    print(f"{'='*70}")

    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X = tfidf.fit_transform(texts)

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_fold_metrics = []
    best_fold, best_f1 = 0, 0

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n  FOLD {fold+1}/{n_folds} — Train: {len(train_idx)}, Val: {len(val_idx)}")

        model = model_class()
        model.tfidf = tfidf
        model.fit(X[train_idx], labels_m[train_idx], labels_s[train_idx])

        pred_m, pred_s, prob_m, prob_s = model.predict(X[val_idx])
        fm = compute_all_metrics(labels_m[val_idx], pred_m, labels_s[val_idx], pred_s, prob_m, prob_s)
        all_fold_metrics.append(fm)

        print(f"    F1-macro(M): {fm['mention_f1_macro']:.4f} | F1-samples(S): {fm['sentiment_f1_samples']:.4f} | Combined: {fm['combined_f1']:.4f}")

        if fm['combined_f1'] > best_f1:
            best_f1 = fm['combined_f1']
            best_fold = fold + 1
            model.save(output_dir)

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
) -> Dict:
    """Train Deep/Transformer model with K-Fold CV."""
    name = model_class.__name__
    print(f"\n{'='*70}")
    print(f"  Training {name} with {n_folds}-Fold CV on {device}")
    print(f"{'='*70}")

    vocab_size = tokenizer.vocab_size
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_fold_metrics = []
    best_fold, best_f1 = 0, 0
    best_state = None
    fold_epoch_losses = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(texts)):
        print(f"\n  FOLD {fold+1}/{n_folds} — Train: {len(train_idx)}, Val: {len(val_idx)}")

        train_ds = ABSADatasetMultiPolarity(texts[train_idx].tolist(), labels_m[train_idx], labels_s[train_idx], tokenizer, max_length)
        val_ds = ABSADatasetMultiPolarity(texts[val_idx].tolist(), labels_m[val_idx], labels_s[val_idx], tokenizer, max_length)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        model = model_class(num_aspects=NUM_ASPECTS) if is_transformer else model_class(vocab_size=vocab_size)
        model = model.to(device)

        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)
        crit_m = nn.BCEWithLogitsLoss()
        crit_s = nn.BCEWithLogitsLoss()

        epoch_losses = []
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Fold {fold+1} Ep {epoch+1}", leave=False)
            for batch in pbar:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                lm = batch['labels_m'].to(device)
                ls = batch['labels_s'].to(device)

                logits_m, logits_s = model(ids, mask)
                loss = crit_m(logits_m, lm) + crit_s(logits_s, ls)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                train_loss += loss.item()

            avg_loss = train_loss / max(len(train_loader), 1)
            epoch_losses.append(avg_loss)

        if epoch_losses:
            fold_epoch_losses.append(epoch_losses)

        model.eval()
        pm, tm, ps, ts, prm, prs = [], [], [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                logits_m, logits_s = model(ids, mask)

                p_m = torch.sigmoid(logits_m).cpu().numpy()
                p_s = torch.sigmoid(logits_s).cpu().numpy()

                pm.append((p_m > 0.5).astype(float))
                tm.append(batch['labels_m'].numpy())
                ps.append((p_s > 0.5).astype(float))
                ts.append(batch['labels_s'].numpy())
                prm.append(p_m)
                prs.append(p_s)

        fm = compute_all_metrics(
            np.vstack(tm), np.vstack(pm),
            np.concatenate(ts), np.concatenate(ps),
            np.vstack(prm), np.concatenate(prs),
        )
        all_fold_metrics.append(fm)

        print(f"    F1-macro(M): {fm['mention_f1_macro']:.4f} | F1-samples(S): {fm['sentiment_f1_samples']:.4f} | Combined: {fm['combined_f1']:.4f}")

        if fm['combined_f1'] > best_f1:
            best_f1 = fm['combined_f1']
            best_fold = fold + 1
            best_state = model.state_dict().copy()
            print(f"     New best! (Fold {fold+1})")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print_fold_summary(all_fold_metrics, name)

    os.makedirs(output_dir, exist_ok=True)
    if best_state:
        torch.save({
            'model_state_dict': best_state,
            'model_class': name,
            'aspects': ASPECTS,
            'best_f1': best_f1,
            'best_fold': best_fold,
            'multi_polarity': True,
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
            result = train_ml_kfold(model_class, texts, labels_m, labels_s, out_dir, n_folds)

        elif category == 'Deep-Based':
            if 'phobert' not in tokenizers_cache:
                print("\n Loading PhoBERT tokenizer...")
                tokenizers_cache['phobert'] = AutoTokenizer.from_pretrained("vinai/phobert-base")
            tok = tokenizers_cache['phobert']
            result = train_deep_kfold(
                model_class, tok, texts_np, labels_m, labels_s, out_dir,
                n_folds, epochs, batch_size, learning_rate=1e-3,
                max_length=max_length, device=device, is_transformer=False,
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
            )

        display_name = model_key.replace('_', ' ').title()
        all_results[display_name] = result

        avg = result.get('avg_metrics', {})
        metric_order = [
            ('Mention F1', 'mention_f1_macro'),
            ('Sentiment F1', 'sentiment_f1_macro'),
            ('Sentiment F1 (samples)', 'sentiment_f1_samples'),
            ('Combined F1', 'combined_f1'),
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

    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = args.data or os.path.join(BASE_DIR, 'data', 'labeled')

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
    )
