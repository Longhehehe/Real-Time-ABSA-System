#!/usr/bin/env python3
# train_absa_optimized.py
# ======================================
# Training script tối ưu với model nhẹ hơn và early stopping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import json
import os
import gc
import warnings
from datetime import datetime
from tqdm import tqdm
import logging
import sys

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ABSA_Training")

# ============================================
# CONFIGURATION
# ============================================
# Model selection (từ nặng đến nhẹ)
# "xlm-roberta-base" (270M params) - chất lượng cao nhất nhưng chậm
# "distilroberta-base" (82M params) - cân bằng tốt
# "distilbert-base-uncased" (66M params) - nhanh, chất lượng khá
# "phobert-base" (135M params) - tốt cho tiếng Việt

MODEL_NAME = "distilroberta-base"  # ✅ Model cân bằng tốt
MAX_LEN = 64
BATCH_SIZE = 8  # Tăng batch size để train nhanh hơn
ACCUMULATION_STEPS = 4  # Gradient accumulation
EPOCHS = 1
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 2  # Stop sớm nếu không cải thiện

ASPECTS = ["Price", "Shipping", "Outlook", "Quality", "Size", "Shop_Service", "General", "Others"]
SENTIMENTS = ["NEG", "POS", "NEU"]

# Paths
# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check if mistral labeled data exists, otherwise use what's available
# Based on file listing, we have 'data/label/absa_labeled_mistral.csv'
DATA_PATH = os.path.join(BASE_DIR, 'data', 'label', 'absa_labeled_mistral.csv')
if not os.path.exists(DATA_PATH):
    # Fallback to other files if mistral one is missing? 
    # Or keep it, user might have it.
    pass

MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

CURRENT_MODEL_PATH = os.path.join(MODEL_DIR, "best_absa_hardshare.pt")
NEW_MODEL_PATH = os.path.join(MODEL_DIR, "absa_model_candidate.pt")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
BACKUP_METRICS_PATH = os.path.join(MODEL_DIR, f"model_metrics_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🔥 Training on: {DEVICE}")

if DEVICE.type == "cpu":
    torch.set_num_threads(4)
    logger.info("⚙️  CPU mode: 4 threads")
else:
    logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")

# ============================================
# MODEL DEFINITION
# ============================================
class ABSAModel(nn.Module):
    """Lightweight ABSA model"""
    def __init__(self, model_name=MODEL_NAME, num_aspects=len(ASPECTS)):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        H = self.backbone.config.hidden_size
        
        # Regularization
        self.dropout = nn.Dropout(0.3)
        
        # Task heads
        self.head_m = nn.Linear(H, num_aspects)
        self.head_s = nn.Linear(H, num_aspects * 3)
    
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = self.dropout(out.last_hidden_state[:, 0, :])
        
        logits_m = self.head_m(h_cls)
        logits_s = self.head_s(h_cls).view(-1, len(ASPECTS), 3)
        
        return logits_m, logits_s

# ============================================
# DATASET
# ============================================
class ABSADataset(Dataset):
    """ABSA Dataset"""
    def __init__(self, texts, labels_m, labels_s, tokenizer, max_len):
        self.texts = texts
        self.labels_m = labels_m
        self.labels_s = labels_s
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels_m": torch.tensor(self.labels_m[idx], dtype=torch.float),
            "labels_s": torch.tensor(self.labels_s[idx], dtype=torch.long),
        }

# ============================================
# DATA LOADING
# ============================================
def load_data():
    """Load and prepare training data"""
    logger.info("Loading data...")
    
    # Try CSV first
    if os.path.exists(DATA_PATH):
        logger.info(f"Loading from CSV: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        # Fallback to PostgreSQL
        logger.info("CSV not found. Loading from PostgreSQL...")
        try:
            from sqlalchemy import create_engine
            engine = create_engine("postgresql://airflow:airflow@postgres:5432/airflow")
            df = pd.read_sql("SELECT * FROM absa_results ORDER BY RANDOM() LIMIT 5000", engine)
            logger.info(f"Loaded {len(df)} rows from PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    # Extract texts
    if "ReviewText" in df.columns:
        texts = df["ReviewText"].astype(str).values
    elif "Review" in df.columns:
        texts = df["Review"].astype(str).values
    else:
        texts = df[df.columns[0]].astype(str).values
    
    # Clean texts
    texts = [t.strip() for t in texts if t.strip()]
    
    # === SỬA CHÍNH TẠI ĐÂY ===
    aspect_cols = [asp for asp in ASPECTS if asp in df.columns]
    if not aspect_cols:
        raise ValueError(f"No aspect columns found. Expected one of: {ASPECTS}")

    # Lấy dữ liệu thô từ các cột aspect
    raw_labels = df[aspect_cols].fillna(-1).values  # Đảm bảo NaN → -1

    # === MENTION LABELS: -1 → 0, còn lại → 1 ===
    labels_m = (raw_labels != -1).astype(int)  # 1 dòng!

    # === SENTIMENT LABELS: -1 → 0, 0→0(NEG), 1→1(POS), 2→2(NEU) ===
    labels_s = np.where(raw_labels == -1, 0, raw_labels)  # -1 → 0
    # Không cần map thêm vì: 0→0, 1→1, 2→2 → ĐÚNG!

    logger.info(f"Loaded {len(texts)} samples")
    logger.info(f"Aspect columns: {aspect_cols}")
    logger.info(f"Mention stats: {np.sum(labels_m)} mentions out of {labels_m.size}")
    
    return texts, labels_m, labels_s.astype(int)

# ============================================
# TRAINING
# ============================================
def train_epoch(model, loader, optimizer, scheduler, device, accumulation_steps=1):
    """Train one epoch"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(loader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_m = batch["labels_m"].to(device)
        labels_s = batch["labels_s"].to(device)
        
        # Forward
        logits_m, logits_s = model(input_ids, attention_mask)
        
        # Loss
        loss_m = F.binary_cross_entropy_with_logits(logits_m, labels_m)
        loss_s = F.cross_entropy(logits_s.view(-1, 3), labels_s.view(-1))
        loss = loss_m + loss_s
        
        # Backward (with gradient accumulation)
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        progress_bar.set_postfix({"loss": loss.item() * accumulation_steps})
        
        # Cleanup
        del input_ids, attention_mask, labels_m, labels_s, loss
        
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return total_loss / len(loader)

def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()
    
    all_preds_m = []
    all_labels_m = []
    all_preds_s = []
    all_labels_s = []
    
    progress_bar = tqdm(loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_m = batch["labels_m"].to(device)
            labels_s = batch["labels_s"].to(device)
            
            # Forward
            logits_m, logits_s = model(input_ids, attention_mask)
            
            # Predictions
            preds_m = (torch.sigmoid(logits_m) > 0.5).long()
            preds_s = torch.argmax(logits_s, dim=-1)
            
            all_preds_m.extend(preds_m.cpu().numpy())
            all_labels_m.extend(labels_m.cpu().numpy())
            all_preds_s.extend(preds_s.cpu().numpy().flatten())
            all_labels_s.extend(labels_s.cpu().numpy().flatten())
            
            # Cleanup
            del input_ids, attention_mask, labels_m, labels_s
    
    # Calculate metrics
    f1_m = f1_score(all_labels_m, all_preds_m, average="macro", zero_division=0)
    f1_s = f1_score(all_labels_s, all_preds_s, average="macro", zero_division=0)
    acc_s = accuracy_score(all_labels_s, all_preds_s)
    
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    return {
        "mention_f1": f1_m,
        "sentiment_f1": f1_s,
        "sentiment_acc": acc_s,
        "combined_f1": (f1_m + f1_s) / 2
    }

# ============================================
# MAIN TRAINING LOOP
# ============================================
def train_model():
    """Main training function"""
    logger.info("="*60)
    logger.info("🚀 ABSA Model Training")
    logger.info("="*60)
    
    # Load data
    texts, labels_m, labels_s = load_data()
    
    # Split data
    X_train, X_val, y_m_train, y_m_val, y_s_train, y_s_val = train_test_split(
        texts, labels_m, labels_s,
        test_size=0.2,
        random_state=42
    )
    
    logger.info(f"📊 Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Tokenizer
    logger.info(f"🔤 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Datasets
    train_dataset = ABSADataset(X_train, y_m_train, y_s_train, tokenizer, MAX_LEN)
    val_dataset = ABSADataset(X_val, y_m_val, y_s_val, tokenizer, MAX_LEN)
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda")
    )
    
    # Model
    logger.info(f"🤖 Initializing model: {MODEL_NAME}")
    model = ABSAModel().to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"📊 Total params: {total_params:,}")
    logger.info(f"📊 Trainable params: {trainable_params:,}")
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    total_steps = len(train_loader) * EPOCHS // ACCUMULATION_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.info(f"⚙️  Total steps: {total_steps}")
    logger.info(f"⚙️  Warmup steps: {warmup_steps}")
    
    # Training loop
    best_f1 = 0
    patience_counter = 0
    history = []
    
    logger.info("="*60)
    logger.info("🏋️  Starting training...")
    logger.info("="*60)
    
    for epoch in range(EPOCHS):
        logger.info(f"\n📈 Epoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, DEVICE, ACCUMULATION_STEPS
        )
        
        # Evaluate
        metrics = evaluate(model, val_loader, DEVICE)
        
        # Log
        logger.info(f"Loss: {train_loss:.4f}")
        logger.info(f"Mention F1: {metrics['mention_f1']:.4f}")
        logger.info(f"Sentiment F1: {metrics['sentiment_f1']:.4f}")
        logger.info(f"Sentiment Acc: {metrics['sentiment_acc']:.4f}")
        logger.info(f"Combined F1: {metrics['combined_f1']:.4f}")
        
        # Save history
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **metrics
        })
        
        # Save best model
        if metrics["combined_f1"] > best_f1:
            best_f1 = metrics["combined_f1"]
            torch.save(model.state_dict(), NEW_MODEL_PATH)
            logger.info(f"✅ New best model saved! F1: {best_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"⏳ No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logger.info(f"🛑 Early stopping triggered!")
            break
    
    # Save metrics
    final_metrics = {
        "timestamp": datetime.now().isoformat(),
        "model_name": MODEL_NAME,
        "mention_f1": metrics["mention_f1"],
        "sentiment_f1": metrics["sentiment_f1"],
        "sentiment_acc": metrics["sentiment_acc"],
        "combined_f1": best_f1,
        "epochs_trained": epoch + 1,
        "total_epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "model_path": NEW_MODEL_PATH,
        "training_history": history
    }
    
    # Backup old metrics if exists
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            old_metrics = json.load(f)
        with open(BACKUP_METRICS_PATH, 'w') as f:
            json.dump(old_metrics, f, indent=2)
        logger.info(f"💾 Backed up old metrics to {BACKUP_METRICS_PATH}")
    
    # Save new metrics
    with open(METRICS_PATH, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    logger.info("="*60)
    logger.info("✅ Training completed!")
    logger.info(f"🎯 Best Combined F1: {best_f1:.4f}")
    logger.info(f"💾 Model saved to: {NEW_MODEL_PATH}")
    logger.info(f"📊 Metrics saved to: {METRICS_PATH}")
    logger.info("="*60)
    
    return final_metrics

# ============================================
# ENTRY POINT
# ============================================
if __name__ == "__main__":
    try:
        metrics = train_model()
        logger.info("\n🎉 Training finished successfully!")
        logger.info(json.dumps(metrics, indent=2, ensure_ascii=False))
        
    except KeyboardInterrupt:
        logger.info("\n⛔ Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
