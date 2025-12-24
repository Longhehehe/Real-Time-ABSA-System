"""
PhoBERT Multi-Polarity ABSA Trainer
Train PhoBERT model for Aspect-Based Sentiment Analysis with MULTI-POLARITY support.

Key difference from original:
- Each aspect can have MULTIPLE sentiment labels (e.g., both POS and NEG)
- Uses BCEWithLogitsLoss instead of CrossEntropyLoss for sentiment
- Data format: '0,1' means both NEG and POS for same aspect
"""
import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import json

# Aspect names - OPTIMIZED for E-commerce (9 aspects)
ASPECTS = [
    'Chất lượng sản phẩm',       # Quality, durability, materials
    'Hiệu năng & Trải nghiệm',   # Performance, user experience  
    'Đúng mô tả',                # Accuracy of description
    'Giá cả & Khuyến mãi',       # Price, discounts, value
    'Vận chuyển',                # Shipping speed, delivery
    'Đóng gói',                  # Packaging quality
    'Dịch vụ & Thái độ Shop',    # Customer service, seller attitude
    'Bảo hành & Đổi trả',        # Warranty, returns
    'Tính xác thực',             # Authenticity (fake/genuine)
]


# Label mapping for MULTI-POLARITY format (USER DEFINED):
# Raw label values:
#   -1 = Tiêu cực (NEG)
#    1 = Tích cực (POS)
#    0 = Trung lập (NEU)
#    2 = Không đề cập (Not mentioned)
#
# Multi-polarity: '-1,1' = Both NEG and POS
#
# Internal representation (multi-hot vector [NEG, POS, NEU]):
#   - [1, 0, 0] = NEG only (-1)
#   - [0, 1, 0] = POS only (1)
#   - [0, 0, 1] = NEU only (0)
#   - [1, 1, 0] = Both NEG and POS (-1,1)
#   - [0, 0, 0] = Not mentioned (2)

# Mapping from raw label to index in sentiment vector
LABEL_TO_INDEX = {
    -1: 0,   # NEG -> index 0
    1: 1,    # POS -> index 1
    0: 2,    # NEU -> index 2
}


class ABSADatasetMultiPolarity(Dataset):
    """Dataset for Multi-Polarity ABSA.
    
    Key difference: labels_s is now [batch, num_aspects, 3] instead of [batch, num_aspects]
    """
    
    def __init__(
        self, 
        texts: List[str], 
        labels_m: np.ndarray,  # Mention labels: [batch, num_aspects]
        labels_s: np.ndarray,  # Sentiment labels: [batch, num_aspects, 3] - multi-hot!
        tokenizer,
        max_length: int = 256
    ):
        self.texts = texts
        self.labels_m = labels_m
        self.labels_s = labels_s
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels_m': torch.tensor(self.labels_m[idx], dtype=torch.float),
            'labels_s': torch.tensor(self.labels_s[idx], dtype=torch.float)  # Float for BCE!
        }


class PhoBERTForABSAMultiPolarity(nn.Module):
    """PhoBERT model with multi-task learning for Multi-Polarity ABSA.
    
    Uses hard parameter sharing with two task heads:
    - Mention detection: Binary classification per aspect
    - Sentiment classification: MULTI-LABEL classification per aspect (can have multiple sentiments!)
    """
    
    def __init__(self, num_aspects: int = 9, dropout: float = 0.3):
        super().__init__()
        
        # Load PhoBERT backbone
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        hidden_size = self.phobert.config.hidden_size  # 768
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        
        # Task heads
        self.head_m = nn.Linear(hidden_size, num_aspects)  # Mention detection
        self.head_s = nn.Linear(hidden_size, num_aspects * 3)  # Sentiment (3 classes per aspect)
        
        self.num_aspects = num_aspects
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Returns:
            logits_m: Mention logits (batch_size, num_aspects)
            logits_s: Sentiment logits (batch_size, num_aspects, 3) - use sigmoid, NOT softmax!
        """
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        h_cls = self.dropout(cls_output)
        
        # Task-specific predictions
        logits_m = self.head_m(h_cls)
        logits_s = self.head_s(h_cls).view(-1, self.num_aspects, 3)
        
        return logits_m, logits_s


def load_data_multipolarity(data_path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Load and preprocess data from Excel or CSV file for MULTI-POLARITY format.
    
    Label format (USER DEFINED):
        -1 = NEG (Tiêu cực)
         1 = POS (Tích cực)
         0 = NEU (Trung lập)
         2 = Not mentioned (Không đề cập)
        '-1,1' = Both NEG and POS (multi-polarity)
    
    Returns:
        texts: List of review texts
        labels_m: Mention labels (batch_size, num_aspects) - binary
        labels_s: Sentiment labels (batch_size, num_aspects, 3) - multi-hot vectors!
    """
    import glob
    
    # Check if data_path is a directory (load all xlsx files)
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
    
    # Extract labels for each aspect
    labels_m = []  # Mention labels
    labels_s = []  # Sentiment labels - NOW MULTI-HOT!
    
    for idx, row in df.iterrows():
        row_labels_m = []
        row_labels_s = []
        
        for aspect in ASPECTS:
            if aspect in df.columns:
                val = row[aspect]
                
                # Handle multi-polarity format:
                # 2 or NaN: Not mentioned
                # -1: NEG, 1: POS, 0: NEU
                # '-1,1' or '[-1,1]': Both NEG and POS
                
                # Convert to string for uniform handling
                val_str = str(val).strip() if pd.notna(val) else '2'
                
                # Remove brackets if present: '[-1,1]' -> '-1,1'
                val_str = val_str.replace('[', '').replace(']', '').strip()
                
                if val_str == '2' or val_str == 'nan' or val_str == '':
                    # Not mentioned
                    row_labels_m.append(0)
                    row_labels_s.append([0, 0, 0])  # Padding
                else:
                    # Mentioned
                    row_labels_m.append(1)
                    
                    # Parse sentiment(s)
                    sentiment_vector = [0, 0, 0]  # [NEG, POS, NEU]
                    
                    if ',' in val_str:
                        # Multi-label: "-1,1" or "-1,0" etc.
                        try:
                            labels = [int(x.strip()) for x in val_str.split(',')]
                            for label in labels:
                                if label in LABEL_TO_INDEX:
                                    sentiment_vector[LABEL_TO_INDEX[label]] = 1
                        except ValueError:
                            # Fallback: treat as neutral
                            sentiment_vector[2] = 1
                    else:
                        # Single label: -1, 1, or 0
                        try:
                            label = int(float(val_str))
                            if label in LABEL_TO_INDEX:
                                sentiment_vector[LABEL_TO_INDEX[label]] = 1
                            else:
                                sentiment_vector[2] = 1  # Default to neutral
                        except ValueError:
                            sentiment_vector[2] = 1
                    
                    row_labels_s.append(sentiment_vector)
            else:
                # Aspect not in data
                row_labels_m.append(0)
                row_labels_s.append([0, 0, 0])
        
        labels_m.append(row_labels_m)
        labels_s.append(row_labels_s)

    
    # labels_s shape: [num_samples, num_aspects, 3]
    return texts, np.array(labels_m, dtype=np.float32), np.array(labels_s, dtype=np.float32)



def train_model_multipolarity(
    data_path: str,
    output_dir: str = "./models/phobert_absa_multipolarity",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    device: str = None
):
    """Train PhoBERT Multi-Polarity ABSA model."""
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🚀 PhoBERT Multi-Polarity ABSA Training")
    print(f"📱 Device: {device}")
    
    # Load tokenizer
    print("📥 Loading PhoBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    # Load data
    print(f"📊 Loading data from {data_path}...")
    texts, labels_m, labels_s = load_data_multipolarity(data_path)
    print(f"   Total samples: {len(texts)}")
    print(f"   Labels shape - Mention: {labels_m.shape}, Sentiment: {labels_s.shape}")
    
    # Check for multi-polarity samples
    multi_polarity_count = np.sum(np.sum(labels_s, axis=-1) > 1)
    print(f"   Multi-polarity samples (>1 sentiment per aspect): {multi_polarity_count}")
    
    # Split data
    train_texts, val_texts, train_labels_m, val_labels_m, train_labels_s, val_labels_s = train_test_split(
        texts, labels_m, labels_s, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(train_texts)}, Val: {len(val_texts)}")
    
    # Create datasets
    train_dataset = ABSADatasetMultiPolarity(train_texts, train_labels_m, train_labels_s, tokenizer, max_length)
    val_dataset = ABSADatasetMultiPolarity(val_texts, val_labels_m, val_labels_s, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    print("🧠 Creating PhoBERT Multi-Polarity ABSA model...")
    model = PhoBERTForABSAMultiPolarity(num_aspects=len(ASPECTS))
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # Loss functions - BOTH use BCE for multi-label!
    criterion_m = nn.BCEWithLogitsLoss()  # Binary for mention detection
    criterion_s = nn.BCEWithLogitsLoss()  # Multi-label for sentiment (CHANGED!)
    
    # Training loop with early stopping
    best_val_f1 = 0
    patience = 3  # Early stopping patience
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\n📈 Epoch {epoch + 1}/{epochs}")
        
        # === TRAINING ===
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(train_pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_m = batch['labels_m'].to(device)
            labels_s = batch['labels_s'].to(device)  # Now [batch, aspects, 3]
            
            # Forward pass
            logits_m, logits_s = model(input_ids, attention_mask)
            
            # Compute multi-task loss
            loss_m = criterion_m(logits_m, labels_m)
            
            # Multi-label sentiment loss (CHANGED!)
            # logits_s: [batch, aspects, 3]
            # labels_s: [batch, aspects, 3]
            loss_s = criterion_s(logits_s, labels_s)
            
            # Combined loss
            loss = loss_m + loss_s
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"   Train Loss: {avg_train_loss:.4f}")
        
        # === VALIDATION ===
        model.eval()
        val_preds_m = []
        val_true_m = []
        val_preds_s = []
        val_true_s = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_m = batch['labels_m'].to(device)
                labels_s = batch['labels_s'].to(device)
                
                logits_m, logits_s = model(input_ids, attention_mask)
                
                # Predictions - BOTH use sigmoid threshold for multi-label!
                preds_m = (torch.sigmoid(logits_m) > 0.5).float()
                preds_s = (torch.sigmoid(logits_s) > 0.5).float()  # Multi-label!
                
                val_preds_m.append(preds_m.cpu().numpy())
                val_true_m.append(labels_m.cpu().numpy())
                val_preds_s.append(preds_s.cpu().numpy())
                val_true_s.append(labels_s.cpu().numpy())
        
        val_preds_m = np.vstack(val_preds_m)
        val_true_m = np.vstack(val_true_m)
        val_preds_s = np.concatenate(val_preds_s, axis=0)
        val_true_s = np.concatenate(val_true_s, axis=0)
        
        # Calculate metrics
        f1_m = f1_score(val_true_m.flatten(), val_preds_m.flatten(), average='macro', zero_division=0)
        
        # Multi-label F1 for sentiment - ONLY on mentioned aspects!
        # Filter out non-mentioned samples (where all labels are 0)
        mentioned_mask = val_true_m.flatten() == 1  # [batch * aspects]
        
        # Reshape sentiment arrays
        true_s_flat = val_true_s.reshape(-1, 3)  # [batch * aspects, 3]
        pred_s_flat = val_preds_s.reshape(-1, 3)
        
        # Only keep mentioned rows for evaluation
        true_s_mentioned = true_s_flat[mentioned_mask]
        pred_s_mentioned = pred_s_flat[mentioned_mask]
        
        if len(true_s_mentioned) > 0:
            # Use samples average for multi-label
            f1_s = f1_score(
                true_s_mentioned,
                pred_s_mentioned,
                average='samples',
                zero_division=0
            )
            
            # Also calculate micro F1 as alternative metric
            f1_s_micro = f1_score(
                true_s_mentioned.flatten(),
                pred_s_mentioned.flatten(),
                average='binary',
                zero_division=0
            )
        else:
            f1_s = 0.0
            f1_s_micro = 0.0
        
        combined_f1 = (f1_m + f1_s) / 2
        
        print(f"   Val Mention F1: {f1_m:.4f}")
        print(f"   Val Sentiment F1 (multi-label, samples): {f1_s:.4f}")
        print(f"   Val Sentiment F1 (micro): {f1_s_micro:.4f}")
        print(f"   Val Combined F1: {combined_f1:.4f}")
        
        # Save best model
        if combined_f1 > best_val_f1:
            best_val_f1 = combined_f1
            patience_counter = 0
            print(f"   ✅ New best model! Saving...")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'tokenizer_name': 'vinai/phobert-base',
                'aspects': ASPECTS,
                'best_f1': best_val_f1,
                'mention_f1': f1_m,
                'sentiment_f1': f1_s,
                'multi_polarity': True  # Flag for inference
            }, os.path.join(output_dir, 'phobert_absa_multipolarity.pt'))
            
            # Save config
            config = {
                'aspects': ASPECTS,
                'max_length': max_length,
                'best_f1': float(best_val_f1),
                'mention_f1': float(f1_m),
                'sentiment_f1': float(f1_s),
                'multi_polarity': True,
                'sentiment_classes': ['NEG', 'POS', 'NEU']
            }
            with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement. Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"   🛑 Early stopping triggered!")
                break
    
    print(f"\n🎉 Training complete! Best F1: {best_val_f1:.4f}")
    print(f"📁 Model saved to: {output_dir}")
    
    return output_dir


def train_model_kfold(
    data_path: str,
    output_dir: str = "./models/phobert_absa_multipolarity",
    n_folds: int = 5,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    device: str = None
):
    """
    Train PhoBERT Multi-Polarity ABSA model using K-Fold Cross-Validation.
    
    Args:
        data_path: Path to training data
        output_dir: Directory to save model
        n_folds: Number of folds for cross-validation
        epochs: Number of training epochs per fold
        batch_size: Batch size
        learning_rate: Learning rate
        max_length: Max sequence length
        device: Device to use (cuda/cpu)
    
    Returns:
        Dict with fold results and best model info
    """
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🚀 PhoBERT Multi-Polarity ABSA Training with {n_folds}-Fold CV")
    print(f"📱 Device: {device}")
    
    # Load tokenizer
    print("📥 Loading PhoBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    # Load data
    print(f"📊 Loading data from {data_path}...")
    texts, labels_m, labels_s = load_data_multipolarity(data_path)
    texts = np.array(texts)  # Convert to numpy for indexing
    
    print(f"   Total samples: {len(texts)}")
    print(f"   Labels shape - Mention: {labels_m.shape}, Sentiment: {labels_s.shape}")
    
    # Check for multi-polarity samples
    multi_polarity_count = np.sum(np.sum(labels_s, axis=-1) > 1)
    print(f"   Multi-polarity samples (>1 sentiment per aspect): {multi_polarity_count}")
    
    # K-Fold setup
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Track results across folds
    fold_results = []
    best_overall_f1 = 0
    best_fold = 0
    best_model_state = None
    
    print(f"\n{'='*60}")
    print(f"Starting {n_folds}-Fold Cross-Validation")
    print(f"{'='*60}")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(texts)):
        print(f"\n📂 FOLD {fold + 1}/{n_folds}")
        print(f"   Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        
        # Split data for this fold
        train_texts = texts[train_idx]
        val_texts = texts[val_idx]
        train_labels_m = labels_m[train_idx]
        val_labels_m = labels_m[val_idx]
        train_labels_s = labels_s[train_idx]
        val_labels_s = labels_s[val_idx]
        
        # Create datasets
        train_dataset = ABSADatasetMultiPolarity(
            train_texts.tolist(), train_labels_m, train_labels_s, tokenizer, max_length
        )
        val_dataset = ABSADatasetMultiPolarity(
            val_texts.tolist(), val_labels_m, val_labels_s, tokenizer, max_length
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Create fresh model for each fold
        model = PhoBERTForABSAMultiPolarity(num_aspects=len(ASPECTS))
        model = model.to(device)
        
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )
        
        # Loss functions
        criterion_m = nn.BCEWithLogitsLoss()
        criterion_s = nn.BCEWithLogitsLoss()
        
        # Training loop for this fold
        best_fold_f1 = 0
        
        for epoch in range(epochs):
            # === TRAINING ===
            model.train()
            train_loss = 0
            train_pbar = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}")
            
            for batch in train_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_labels_m = batch['labels_m'].to(device)
                batch_labels_s = batch['labels_s'].to(device)
                
                # Forward pass
                logits_m, logits_s = model(input_ids, attention_mask)
                
                # Compute loss
                loss_m = criterion_m(logits_m, batch_labels_m)
                loss_s = criterion_s(logits_s, batch_labels_s)
                loss = loss_m + loss_s
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # === VALIDATION after all epochs ===
        model.eval()
        val_preds_m = []
        val_true_m = []
        val_preds_s = []
        val_true_s = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_labels_m = batch['labels_m'].to(device)
                batch_labels_s = batch['labels_s'].to(device)
                
                logits_m, logits_s = model(input_ids, attention_mask)
                
                preds_m = (torch.sigmoid(logits_m) > 0.5).float()
                preds_s = (torch.sigmoid(logits_s) > 0.5).float()
                
                val_preds_m.append(preds_m.cpu().numpy())
                val_true_m.append(batch_labels_m.cpu().numpy())
                val_preds_s.append(preds_s.cpu().numpy())
                val_true_s.append(batch_labels_s.cpu().numpy())
        
        val_preds_m = np.vstack(val_preds_m)
        val_true_m = np.vstack(val_true_m)
        val_preds_s = np.concatenate(val_preds_s, axis=0)
        val_true_s = np.concatenate(val_true_s, axis=0)
        
        # Calculate metrics
        f1_m = f1_score(val_true_m.flatten(), val_preds_m.flatten(), average='macro', zero_division=0)
        
        # Multi-label F1 - only on mentioned aspects
        mentioned_mask = val_true_m.flatten() == 1
        true_s_flat = val_true_s.reshape(-1, 3)
        pred_s_flat = val_preds_s.reshape(-1, 3)
        
        if mentioned_mask.sum() > 0:
            true_s_mentioned = true_s_flat[mentioned_mask]
            pred_s_mentioned = pred_s_flat[mentioned_mask]
            f1_s = f1_score(true_s_mentioned, pred_s_mentioned, average='samples', zero_division=0)
        else:
            f1_s = 0.0
        
        combined_f1 = (f1_m + f1_s) / 2
        
        print(f"   Fold {fold+1} Results:")
        print(f"      Mention F1: {f1_m:.4f}")
        print(f"      Sentiment F1: {f1_s:.4f}")
        print(f"      Combined F1: {combined_f1:.4f}")
        
        # Save fold result
        fold_results.append({
            'fold': fold + 1,
            'mention_f1': f1_m,
            'sentiment_f1': f1_s,
            'combined_f1': combined_f1
        })
        
        # Track best model across all folds
        if combined_f1 > best_overall_f1:
            best_overall_f1 = combined_f1
            best_fold = fold + 1
            best_model_state = model.state_dict().copy()
            print(f"   ✅ New best model! (Fold {fold+1})")
        
        # Clean up
        del model
        torch.cuda.empty_cache() if device == "cuda" else None
    
    # Summary
    print(f"\n{'='*60}")
    print(f"K-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    
    avg_mention_f1 = np.mean([r['mention_f1'] for r in fold_results])
    avg_sentiment_f1 = np.mean([r['sentiment_f1'] for r in fold_results])
    avg_combined_f1 = np.mean([r['combined_f1'] for r in fold_results])
    std_combined_f1 = np.std([r['combined_f1'] for r in fold_results])
    
    print(f"Average Mention F1:   {avg_mention_f1:.4f}")
    print(f"Average Sentiment F1: {avg_sentiment_f1:.4f}")
    print(f"Average Combined F1:  {avg_combined_f1:.4f} (+/- {std_combined_f1:.4f})")
    print(f"Best Fold:            {best_fold} (F1: {best_overall_f1:.4f})")
    
    # Save best model
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model and load best state
    final_model = PhoBERTForABSAMultiPolarity(num_aspects=len(ASPECTS))
    final_model.load_state_dict(best_model_state)
    
    torch.save({
        'model_state_dict': best_model_state,
        'tokenizer_name': 'vinai/phobert-base',
        'aspects': ASPECTS,
        'best_f1': best_overall_f1,
        'best_fold': best_fold,
        'avg_f1': avg_combined_f1,
        'std_f1': std_combined_f1,
        'n_folds': n_folds,
        'multi_polarity': True
    }, os.path.join(output_dir, 'phobert_absa_multipolarity.pt'))
    
    # Save config
    config = {
        'aspects': ASPECTS,
        'max_length': max_length,
        'best_f1': float(best_overall_f1),
        'avg_f1': float(avg_combined_f1),
        'std_f1': float(std_combined_f1),
        'best_fold': best_fold,
        'n_folds': n_folds,
        'multi_polarity': True,
        'sentiment_classes': ['NEG', 'POS', 'NEU'],
        'fold_results': fold_results
    }
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 K-Fold Training complete!")
    print(f"📁 Best model (Fold {best_fold}) saved to: {output_dir}")
    
    return {
        'output_dir': output_dir,
        'fold_results': fold_results,
        'best_fold': best_fold,
        'best_f1': best_overall_f1,
        'avg_f1': avg_combined_f1,
        'std_f1': std_combined_f1
    }

def predict_multipolarity(
    texts: List[str],
    model_path: str = "./models/phobert_absa_multipolarity",
    device: str = None,
    threshold: float = 0.5
) -> List[Dict]:
    """
    Predict multi-polarity sentiments for given texts.
    
    Returns:
        List of predictions, each containing:
        - aspect: {
            'mentioned': bool,
            'sentiments': List[str]  # Can have multiple: ['NEG', 'POS']
          }
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    checkpoint = torch.load(
        os.path.join(model_path, 'phobert_absa_multipolarity.pt'),
        map_location=device
    )
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint['tokenizer_name'])
    model = PhoBERTForABSAMultiPolarity(num_aspects=len(ASPECTS))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    sentiment_names = ['NEG', 'POS', 'NEU']
    results = []
    
    with torch.no_grad():
        for text in texts:
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=256,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            logits_m, logits_s = model(input_ids, attention_mask)
            
            # Get predictions
            preds_m = (torch.sigmoid(logits_m) > threshold).squeeze().cpu().numpy()
            preds_s = (torch.sigmoid(logits_s) > threshold).squeeze().cpu().numpy()
            
            prediction = {'text': text, 'aspects': {}}
            
            for i, aspect in enumerate(ASPECTS):
                mentioned = bool(preds_m[i])
                
                if mentioned:
                    # Get all sentiments above threshold
                    sentiments = [
                        sentiment_names[j]
                        for j in range(3)
                        if preds_s[i, j]
                    ]
                    
                    # Default to NEU if no sentiment detected
                    if not sentiments:
                        sentiments = ['NEU']
                else:
                    sentiments = None
                
                prediction['aspects'][aspect] = {
                    'mentioned': mentioned,
                    'sentiments': sentiments
                }
            
            results.append(prediction)
    
    return results


if __name__ == "__main__":
    # Default paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Use ALL labeled data from data/labeled folder (11 files, ~10,500 reviews)
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'labeled')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'models', 'phobert_absa_multipolarity')
    
    # Check if folder exists
    if not os.path.exists(DATA_PATH):
        print(f"⚠️ Labeled data folder not found at {DATA_PATH}")
        print(f"Please run auto_label_absa.py first!")
        sys.exit(1)

    # Choose training mode
    USE_KFOLD = True  # Set to False for standard train/test split
    N_FOLDS = 5
    
    if USE_KFOLD:
        print(f"\n📊 Using {N_FOLDS}-Fold Cross-Validation")
        train_model_kfold(
            data_path=DATA_PATH,
            output_dir=OUTPUT_DIR,
            n_folds=N_FOLDS,
            epochs=5,           # 5 epochs per fold
            batch_size=2,       # Small batch for small dataset
            learning_rate=3e-5
        )
    else:
        print("\n📊 Using Standard Train/Test Split (80/20)")
        train_model_multipolarity(
            data_path=DATA_PATH,
            output_dir=OUTPUT_DIR,
            epochs=5,
            batch_size=2,
            learning_rate=3e-5
        )
    
    # Test prediction
    print("\n🔍 Testing prediction...")
    test_texts = [
        "Áo đẹp nhưng vải hơi mỏng. Giao hàng nhanh!",
        "Sản phẩm tốt, giá hợp lý."
    ]
    
    try:
        predictions = predict_multipolarity(test_texts, OUTPUT_DIR)
        for pred in predictions:
            print(f"\nText: {pred['text']}")
            for aspect, info in pred['aspects'].items():
                if info['mentioned']:
                    print(f"  - {aspect}: {info['sentiments']}")
    except Exception as e:
        print(f"Prediction test skipped: {e}")

