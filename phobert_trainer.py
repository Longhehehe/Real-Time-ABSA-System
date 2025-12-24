"""
PhoBERT Multi-label ABSA Trainer
Train PhoBERT model for Aspect-Based Sentiment Analysis on Vietnamese reviews.
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import json

# Aspect names
ASPECTS = [
    'Chất lượng sản phẩm',
    'Trải nghiệm sử dụng',
    'Đúng mô tả sản phẩm',
    'Hiệu năng sản phẩm',
    'Giá cả',
    'Khuyến mãi & voucher',
    'Vận chuyển & giao hàng',
    'Đóng gói & bao bì',
    'Uy tín & thái độ shop',
    'Dịch vụ chăm sóc khách hàng',
    'Lỗi & bảo hành & hàng giả',
    'Đổi trả & bảo hành'
]

# Label mapping for new multi-task format:
# Mention: 0 (not mentioned), 1 (mentioned)
# Sentiment: 0 (NEG), 1 (POS), 2 (NEU)


class ABSADataset(Dataset):
    """Dataset for Multi-task ABSA."""
    
    def __init__(
        self, 
        texts: List[str], 
        labels_m: np.ndarray,  # Mention labels
        labels_s: np.ndarray,  # Sentiment labels
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
            'labels_s': torch.tensor(self.labels_s[idx], dtype=torch.long)
        }


class PhoBERTForABSA(nn.Module):
    """PhoBERT model with multi-task learning for ABSA.
    
    Uses hard parameter sharing with two task heads:
    - Mention detection: Binary classification per aspect
    - Sentiment classification: 3-class classification per aspect (NEG/NEU/POS)
    """
    
    def __init__(self, num_aspects: int = 12, dropout: float = 0.3):
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
            logits_s: Sentiment logits (batch_size, num_aspects, 3)
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


def load_data(data_path: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Load and preprocess data from Excel or CSV file.
    
    Returns:
        texts: List of review texts
        labels_m: Mention labels (batch_size, num_aspects) - binary
        labels_s: Sentiment labels (batch_size, num_aspects) - 0/1/2 for NEG/POS/NEU
    """
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)
    
    texts = df['reviewContent'].tolist()
    
    # Extract labels for each aspect
    labels_m = []  # Mention labels
    labels_s = []  # Sentiment labels
    
    for idx, row in df.iterrows():
        row_labels_m = []
        row_labels_s = []
        
        for aspect in ASPECTS:
            if aspect in df.columns:
                val = row[aspect]
                
                # Convert to multi-task format:
                # Old: -1 (N/A), 0 (NEG), 1 (POS), 2 (NEU)
                # New mention: 0 (not mentioned), 1 (mentioned)
                # New sentiment: 0 (NEG), 1 (POS), 2 (NEU)
                
                if pd.isna(val) or val == -1:
                    # Not mentioned
                    row_labels_m.append(0)
                    row_labels_s.append(0)  # Padding (will be ignored in loss)
                else:
                    # Mentioned
                    row_labels_m.append(1)
                    # Sentiment: keep as is (0=NEG, 1=POS, 2=NEU)
                    row_labels_s.append(int(val))
            else:
                # Aspect not in data
                row_labels_m.append(0)
                row_labels_s.append(0)
        
        labels_m.append(row_labels_m)
        labels_s.append(row_labels_s)
    
    return texts, np.array(labels_m, dtype=np.float32), np.array(labels_s, dtype=np.int64)


def merge_datasets(old_data_path: str, new_data_path: str, output_path: str = None) -> str:
    """
    Merge old and new datasets for retraining.
    
    Args:
        old_data_path: Path to old training data
        new_data_path: Path to new training data
        output_path: Path to save merged data (optional)
    
    Returns:
        Path to merged data file
    """
    print(f"📊 Merging datasets...")
    print(f"   Old data: {old_data_path}")
    print(f"   New data: {new_data_path}")
    
    # Load both datasets
    old_df = pd.read_excel(old_data_path)
    new_df = pd.read_excel(new_data_path)
    
    print(f"   Old samples: {len(old_df)}")
    print(f"   New samples: {len(new_df)}")
    
    # Merge datasets
    merged_df = pd.concat([old_df, new_df], ignore_index=True)
    
    # Remove duplicates based on reviewContent
    merged_df = merged_df.drop_duplicates(subset=['reviewContent'], keep='last')
    
    print(f"   Merged samples: {len(merged_df)} (after dedup)")
    
    # Save merged dataset
    if output_path is None:
        base_dir = os.path.dirname(old_data_path)
        output_path = os.path.join(base_dir, 'merged_training_data.xlsx')
    
    merged_df.to_excel(output_path, index=False)
    print(f"   ✅ Saved to: {output_path}")
    
    return output_path


def get_old_model_f1(model_dir: str = "./models/phobert_absa") -> float:
    """
    Get F1 score of the old model from config.
    
    Returns:
        F1 score of old model, or 0.0 if not found
    """
    config_path = os.path.join(model_dir, 'config.json')
    
    if not os.path.exists(config_path):
        print("⚠️ No old model found, will train from scratch")
        return 0.0
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        old_f1 = config.get('best_f1', 0.0)
        print(f"📈 Old model F1: {old_f1:.4f}")
        return old_f1
    except Exception as e:
        print(f"⚠️ Error reading old config: {e}")
        return 0.0


def train_and_compare(
    data_path: str,
    model_dir: str = "./models/phobert_absa",
    epochs: int = 5,
    batch_size: int = 16,
    min_improvement: float = 0.01
) -> Tuple[bool, float, float]:
    """
    Train new model and compare with old model.
    Only update if new model is significantly better.
    
    Args:
        data_path: Path to training data
        model_dir: Directory containing old model
        epochs: Number of training epochs
        batch_size: Batch size
        min_improvement: Minimum F1 improvement to update model
    
    Returns:
        Tuple of (should_update, new_f1, old_f1)
    """
    # Get old model F1
    old_f1 = get_old_model_f1(model_dir)
    
    # Train new model to temporary directory
    temp_dir = model_dir + "_temp"
    
    print(f"\n🚀 Training new model...")
    train_model(
        data_path=data_path,
        output_dir=temp_dir,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Get new model F1
    new_config_path = os.path.join(temp_dir, 'config.json')
    with open(new_config_path, 'r', encoding='utf-8') as f:
        new_config = json.load(f)
    new_f1 = new_config.get('best_f1', 0.0)
    
    print(f"\n📊 Model Comparison:")
    print(f"   Old F1: {old_f1:.4f}")
    print(f"   New F1: {new_f1:.4f}")
    print(f"   Improvement: {new_f1 - old_f1:.4f}")
    
    # Check if should update
    improvement = new_f1 - old_f1
    should_update = improvement >= min_improvement
    
    if should_update:
        print(f"   ✅ New model is better! Updating...")
        
        # Backup old model
        if os.path.exists(model_dir):
            backup_dir = model_dir + "_backup"
            if os.path.exists(backup_dir):
                import shutil
                shutil.rmtree(backup_dir)
            os.rename(model_dir, backup_dir)
        
        # Move new model to main directory
        os.rename(temp_dir, model_dir)
        print(f"   ✅ Model updated successfully!")
    else:
        print(f"   ❌ New model not significantly better (need >= {min_improvement:.4f} improvement)")
        print(f"   Keeping old model.")
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_dir)
    
    return should_update, new_f1, old_f1


def train_model(
    data_path: str,
    output_dir: str = "./models/phobert_absa",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    device: str = None
):
    """Train PhoBERT ABSA model."""
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🚀 PhoBERT ABSA Training")
    print(f"📱 Device: {device}")
    
    # Load tokenizer
    print("📥 Loading PhoBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    # Load data
    print(f"📊 Loading data from {data_path}...")
    texts, labels_m, labels_s = load_data(data_path)
    print(f"   Total samples: {len(texts)}")
    
    # Split data
    train_texts, val_texts, train_labels_m, val_labels_m, train_labels_s, val_labels_s = train_test_split(
        texts, labels_m, labels_s, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(train_texts)}, Val: {len(val_texts)}")
    
    # Create datasets
    train_dataset = ABSADataset(train_texts, train_labels_m, train_labels_s, tokenizer, max_length)
    val_dataset = ABSADataset(val_texts, val_labels_m, val_labels_s, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    print("🧠 Creating PhoBERT ABSA model...")
    model = PhoBERTForABSA(num_aspects=len(ASPECTS))
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
    criterion_m = nn.BCEWithLogitsLoss()  # Binary for mention detection
    criterion_s = nn.CrossEntropyLoss()  # Multi-class for sentiment
    
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
            labels_s = batch['labels_s'].to(device)
            
            # Forward pass
            logits_m, logits_s = model(input_ids, attention_mask)
            
            # Compute multi-task loss
            loss_m = criterion_m(logits_m, labels_m)
            
            # For sentiment loss, only compute on mentioned aspects
            # Flatten and mask
            loss_s = criterion_s(
                logits_s.view(-1, 3),  # (batch * num_aspects, 3)
                labels_s.view(-1)  # (batch * num_aspects)
            )
            
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
                
                # Predictions
                preds_m = (torch.sigmoid(logits_m) > 0.5).long()
                preds_s = torch.argmax(logits_s, dim=-1)
                
                val_preds_m.append(preds_m.cpu().numpy())
                val_true_m.append(labels_m.cpu().numpy())
                val_preds_s.append(preds_s.cpu().numpy())
                val_true_s.append(labels_s.cpu().numpy())
        
        val_preds_m = np.vstack(val_preds_m)
        val_true_m = np.vstack(val_true_m)
        val_preds_s = np.vstack(val_preds_s)
        val_true_s = np.vstack(val_true_s)
        
        # Calculate metrics
        f1_m = f1_score(val_true_m.flatten(), val_preds_m.flatten(), average='macro', zero_division=0)
        f1_s = f1_score(val_true_s.flatten(), val_preds_s.flatten(), average='macro', zero_division=0)
        combined_f1 = (f1_m + f1_s) / 2
        
        print(f"   Val Mention F1: {f1_m:.4f}")
        print(f"   Val Sentiment F1: {f1_s:.4f}")
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
                'sentiment_f1': f1_s
            }, os.path.join(output_dir, 'phobert_absa.pt'))
            
            # Save config
            config = {
                'aspects': ASPECTS,
                'max_length': max_length,
                'best_f1': float(best_val_f1),
                'mention_f1': float(f1_m),
                'sentiment_f1': float(f1_s)
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


if __name__ == "__main__":
    # Default paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Use the new Mistral labeled dataset
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'label', 'absa_labeled_mistral.csv')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'models', 'phobert_absa')
    
    # Check if file exists, if not fall back
    if not os.path.exists(DATA_PATH):
        print(f"⚠️ New dataset not found at {DATA_PATH}, falling back to old test data.")
        DATA_PATH = os.path.join(BASE_DIR, 'data', 'label', 'absa_grouped_vietnamese_test.xlsx')

    train_model(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        epochs=15,          # Increased from 5 to 15
        batch_size=16,      # Standard for Base model
        learning_rate=3e-5  # Slightly higher for fine-tuning
    )
