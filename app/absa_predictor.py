"""
PhoBERT ABSA Predictor Module
Load trained PhoBERT model and predict aspect-based sentiment.
If model not found, automatically train it.
"""
import os
import sys
import json
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'phobert_absa')
MODEL_PATH = os.path.join(MODEL_DIR, 'phobert_absa.pt')
CONFIG_PATH = os.path.join(MODEL_DIR, 'config.json')

# Aspect categories - OPTIMIZED for E-commerce (9 aspects)
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


# Label mapping for new multi-task format:
# Mention: 0 (not mentioned), 1 (mentioned)
# Sentiment: 0 (NEG), 1 (POS), 2 (NEU)
SENTIMENT_LABELS = {0: 'NEGATIVE', 1: 'POSITIVE', 2: 'NEUTRAL'}

# Map model sentiment to old format for compatibility
SENTIMENT_TO_OLD = {
    0: -1,  # NEG
    1: 1,   # POS
    2: 0    # NEU
}

SENTIMENT_MAP = {
    1: 'POSITIVE',
    0: 'NEUTRAL',
    -1: 'NEGATIVE',
    2: 'N/A'
}


class PhoBERTForABSA(nn.Module):
    """PhoBERT model with multi-task learning for ABSA.
    
    Uses hard parameter sharing with two task heads:
    - Mention detection: Binary classification per aspect
    - Sentiment classification: 3-class classification per aspect (NEG/POS/NEU)
    """
    
    def __init__(self, num_aspects: int = 12, dropout: float = 0.3):
        super().__init__()
        
        from transformers import AutoModel
        
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


class PhoBERTPredictor:
    """PhoBERT ABSA Predictor - loads model and makes predictions."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.max_length = 256
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load trained PhoBERT model."""
        if model_path is None:
            model_path = MODEL_PATH
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            print("🚀 Starting automatic training...")
            return self._auto_train()
        
        try:
            from transformers import AutoTokenizer
            
            print(f"📥 Loading PhoBERT model from {model_path}...")
            
            # Load checkpoint (weights_only=False for compatibility)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Load tokenizer - try local first, then HuggingFace
            tokenizer_local_path = os.path.join(MODEL_DIR, 'tokenizer')
            if os.path.exists(tokenizer_local_path):
                print(f"📥 Loading tokenizer from local: {tokenizer_local_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_local_path)
            else:
                try:
                    print("📥 Downloading tokenizer from HuggingFace (first time only)...")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        checkpoint.get('tokenizer_name', 'vinai/phobert-base') if isinstance(checkpoint, dict) and 'tokenizer_name' in checkpoint else 'vinai/phobert-base'
                    )
                except:
                     self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
            
            # Load model with new multi-task architecture
            self.model = PhoBERTForABSA(
                num_aspects=len(ASPECTS)
            )
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load config if exists
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.max_length = config.get('max_length', 256)
            
            self.model_loaded = True
            print(f"✅ Model loaded successfully! (F1: {checkpoint.get('best_f1', 'N/A'):.4f})")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def _auto_train(self) -> bool:
        """Automatically train model if not found."""
        try:
            from phobert_trainer import train_model
            
            data_path = os.path.join(BASE_DIR, 'data', 'label', 'absa_grouped_vietnamese_test.xlsx')
            
            if not os.path.exists(data_path):
                print(f"❌ Training data not found: {data_path}")
                return False
            
            print("🎓 Auto-training PhoBERT model...")
            output_dir = train_model(
                data_path=data_path,
                output_dir=MODEL_DIR,
                epochs=5,
                batch_size=16
            )
            
            # Load the trained model
            return self.load_model(os.path.join(output_dir, 'phobert_absa.pt'))
            
        except Exception as e:
            print(f"❌ Auto-training failed: {e}")
            return False
    
    def predict_single(self, text: str) -> Dict[str, int]:
        """
        Predict sentiment for a single review.
        Returns dict mapping aspect to sentiment (-1=NEG, 0=NEU, 1=POS, 2=N/A).
        """
        if not self.model_loaded:
            # Try to load/train model
            if not self.load_model():
                print("❌ Cannot make prediction - model not available")
                return {asp: 2 for asp in ASPECTS}  # All N/A
        
        try:
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Predict with multi-task model
            with torch.no_grad():
                logits_m, logits_s = self.model(input_ids, attention_mask)
                
                # Mention predictions (binary)
                preds_m = (torch.sigmoid(logits_m) > 0.5).squeeze(0).cpu().numpy()
                
                # Sentiment predictions (0=NEG, 1=POS, 2=NEU)
                preds_s = torch.argmax(logits_s, dim=-1).squeeze(0).cpu().numpy()
            
            # Map to old format for compatibility
            results = {}
            for i, aspect in enumerate(ASPECTS):
                if preds_m[i]:  # Aspect is mentioned
                    # Convert sentiment: 0=NEG->-1, 1=POS->1, 2=NEU->0
                    results[aspect] = SENTIMENT_TO_OLD.get(int(preds_s[i]), 0)
                else:  # Not mentioned
                    results[aspect] = 2  # N/A
            
            return results
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return {asp: 2 for asp in ASPECTS}
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, int]]:
        """Predict sentiment for a batch of reviews."""
        return [self.predict_single(text) for text in texts]


def aggregate_scores(predictions: List[Dict[str, int]], aspects: List[str] = None) -> Dict[str, float]:
    """
    Aggregate predictions across multiple reviews into scores (0-100).
    
    Args:
        predictions: List of prediction dicts (aspect -> sentiment)
        aspects: List of aspects to include (default: all)
    
    Returns:
        Dict mapping aspect to average score (0-100)
    """
    if aspects is None:
        aspects = ASPECTS
    
    if not predictions:
        return {asp: 50.0 for asp in aspects}
    
    # Value mapping: 1 -> 100, 0 -> 50, -1 -> 0, 2 -> NaN
    value_map = {1: 100, 0: 50, -1: 0, 2: np.nan}
    
    scores = {}
    for aspect in aspects:
        values = []
        for pred in predictions:
            val = pred.get(aspect, 2)
            mapped = value_map.get(val, np.nan)
            if not np.isnan(mapped):
                values.append(mapped)
        
        if values:
            scores[aspect] = np.mean(values)
        else:
            scores[aspect] = 50.0  # Default to neutral
    
    return scores


# Rating-based prediction (backup for when model struggles)
def rating_based_prediction(text: str, rating: int) -> Dict[str, int]:
    """
    Use review rating to determine base sentiment, refined by keywords.
    Rating: 1-2 = Negative, 3 = Neutral, 4-5 = Positive
    """
    text_lower = text.lower()
    
    # Base sentiment from rating
    if rating <= 2:
        base_sentiment = -1  # Negative
    elif rating == 3:
        base_sentiment = 0   # Neutral
    else:
        base_sentiment = 1   # Positive
    
    # Aspect-specific keywords for refinement
    aspect_keywords = {
        'Chất lượng sản phẩm': {
            'pos': ['chất lượng tốt', 'chất lượng', 'đẹp', 'bền', 'chắc chắn', 'tuyệt vời'],
            'neg': ['kém', 'xấu', 'tệ', 'lỗi', 'hỏng', 'rách', 'dở']
        },
        'Giá cả': {
            'pos': ['giá tốt', 'rẻ', 'hợp lý', 'đáng tiền', 'phải chăng', 'giá rẻ'],
            'neg': ['đắt', 'mắc', 'không đáng', 'giá cao', 'chặt chém']
        },
        'Vận chuyển & giao hàng': {
            'pos': ['giao nhanh', 'nhanh', 'đúng hạn', 'giao hàng tốt', 'ship nhanh'],
            'neg': ['giao chậm', 'chậm', 'trễ', 'delay', 'lâu', 'đợi lâu']
        },
        'Đóng gói & bao bì': {
            'pos': ['đóng gói cẩn thận', 'đóng gói đẹp', 'gói kỹ', 'đóng gói tốt'],
            'neg': ['đóng gói kém', 'móp', 'bẹp', 'hư hộp', 'đóng gói sơ sài']
        },
        'Uy tín & thái độ shop': {
            'pos': ['shop uy tín', 'nhiệt tình', 'tư vấn tốt', 'thân thiện', 'shop tốt'],
            'neg': ['shop tệ', 'thái độ kém', 'không nhiệt tình', 'lừa đảo']
        },
    }
    
    results = {}
    
    for aspect in ASPECTS:
        if aspect in aspect_keywords:
            keywords = aspect_keywords[aspect]
            pos_count = sum(1 for kw in keywords['pos'] if kw in text_lower)
            neg_count = sum(1 for kw in keywords['neg'] if kw in text_lower)
            
            if pos_count > neg_count:
                results[aspect] = 1
            elif neg_count > pos_count:
                results[aspect] = -1
            else:
                results[aspect] = base_sentiment
        else:
            # No specific keywords, use base sentiment
            results[aspect] = base_sentiment
    
    return results


# Global predictor instance
_predictor = None

def get_predictor() -> PhoBERTPredictor:
    """Get or create global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = PhoBERTPredictor()
    return _predictor


# Test function
if __name__ == "__main__":
    print("=== PhoBERT ABSA Predictor Test ===")
    
    predictor = PhoBERTPredictor()
    
    if predictor.load_model():
        test_reviews = [
            "Sản phẩm chất lượng tốt, giao hàng nhanh, đóng gói cẩn thận",
            "Sản phẩm kém chất lượng, giao hàng chậm, shop thái độ tệ"
        ]
        
        for review in test_reviews:
            print(f"\nReview: {review[:50]}...")
            pred = predictor.predict_single(review)
            for asp, val in pred.items():
                sentiment = SENTIMENT_MAP.get(val, 'N/A')
                print(f"  {asp}: {sentiment}")
