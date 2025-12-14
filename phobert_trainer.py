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

# Label mapping: -1=Negative, 0=Neutral, 1=Positive, 2=N/A (or missing)
NUM_LABELS = 4  # -1, 0, 1, 2 mapped to 0, 1, 2, 3


class ABSADataset(Dataset):
    """Dataset for Multi-label ABSA."""
    
    def __init__(
        self, 
        texts: List[str], 
        labels: np.ndarray,
        tokenizer,
        max_length: int = 256
    ):
        self.texts = texts
        self.labels = labels
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
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class PhoBERTForABSA(nn.Module):
    """PhoBERT model with multiple classification heads for ABSA."""
    
    def __init__(self, num_aspects: int = 12, num_labels: int = 4, dropout: float = 0.3):
        super().__init__()
        
        # Load PhoBERT
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        hidden_size = self.phobert.config.hidden_size  # 768
        
        # Shared layers
        self.dropout = nn.Dropout(dropout)
        self.shared_fc = nn.Linear(hidden_size, 256)
        self.relu = nn.ReLU()
        
        # Individual classification heads for each aspect
        self.aspect_classifiers = nn.ModuleList([
            nn.Linear(256, num_labels) for _ in range(num_aspects)
        ])
        
        self.num_aspects = num_aspects
    
    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Shared layers
        x = self.dropout(cls_output)
        x = self.shared_fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Individual aspect predictions
        aspect_logits = []
        for classifier in self.aspect_classifiers:
            logits = classifier(x)
            aspect_logits.append(logits)
        
        # Stack: (batch_size, num_aspects, num_labels)
        return torch.stack(aspect_logits, dim=1)


def load_data(data_path: str) -> Tuple[List[str], np.ndarray]:
    """Load and preprocess data from Excel file."""
    df = pd.read_excel(data_path)
    
    texts = df['reviewContent'].tolist()
    
    # Extract labels for each aspect
    labels = []
    for idx, row in df.iterrows():
        row_labels = []
        for aspect in ASPECTS:
            if aspect in df.columns:
                val = row[aspect]
                # Map: -1->0, 0->1, 1->2, NaN/other->3
                if pd.isna(val):
                    mapped = 3
                elif val == -1:
                    mapped = 0
                elif val == 0:
                    mapped = 1
                elif val == 1:
                    mapped = 2
                else:
                    mapped = 3
                row_labels.append(mapped)
            else:
                row_labels.append(3)  # N/A
        labels.append(row_labels)
    
    return texts, np.array(labels)


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
    texts, labels = load_data(data_path)
    print(f"   Total samples: {len(texts)}")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(train_texts)}, Val: {len(val_texts)}")
    
    # Create datasets
    train_dataset = ABSADataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = ABSADataset(val_texts, val_labels, tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    print("🧠 Creating PhoBERT ABSA model...")
    model = PhoBERTForABSA(num_aspects=len(ASPECTS), num_labels=NUM_LABELS)
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_f1 = 0
    
    for epoch in range(epochs):
        print(f"\n📈 Epoch {epoch + 1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc="Training")
        
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids, attention_mask)
            
            # Compute loss for each aspect
            loss = 0
            for i in range(len(ASPECTS)):
                loss += criterion(logits[:, i, :], labels[:, i])
            loss /= len(ASPECTS)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"   Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1)
                
                val_preds.append(preds.cpu().numpy())
                val_true.append(labels.cpu().numpy())
        
        val_preds = np.vstack(val_preds)
        val_true = np.vstack(val_true)
        
        # Calculate metrics
        f1_scores = []
        for i, aspect in enumerate(ASPECTS):
            f1 = f1_score(val_true[:, i], val_preds[:, i], average='macro', zero_division=0)
            f1_scores.append(f1)
        
        avg_f1 = np.mean(f1_scores)
        print(f"   Val Macro F1: {avg_f1:.4f}")
        
        # Save best model
        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            print(f"   ✅ New best model! Saving...")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'tokenizer_name': 'vinai/phobert-base',
                'aspects': ASPECTS,
                'num_labels': NUM_LABELS,
                'best_f1': best_val_f1
            }, os.path.join(output_dir, 'phobert_absa.pt'))
            
            # Save config
            config = {
                'aspects': ASPECTS,
                'num_labels': NUM_LABELS,
                'max_length': max_length,
                'best_f1': float(best_val_f1)
            }
            with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 Training complete! Best F1: {best_val_f1:.4f}")
    print(f"📁 Model saved to: {output_dir}")
    
    return output_dir


if __name__ == "__main__":
    # Default paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'label', 'absa_grouped_vietnamese_test.xlsx')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'models', 'phobert_absa')
    
    train_model(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        epochs=5,
        batch_size=16
    )
