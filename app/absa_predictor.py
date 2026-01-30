"""
PhoBERT ABSA Predictor Module (Multi-Polarity)
Load trained PhoBERT model and predict aspect-based sentiment.
Supports multi-label sentiment (e.g., both POSITIVE and NEGATIVE for same aspect).
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

# Model paths - Using NEW multipolarity model
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'phobert_absa_multipolarity')
MODEL_PATH = os.path.join(MODEL_DIR, 'phobert_absa_multipolarity.pt')
CONFIG_PATH = os.path.join(MODEL_DIR, 'config.json')

# Fallback to old model if multipolarity not available
MODEL_DIR_OLD = os.path.join(BASE_DIR, 'models', 'phobert_absa')
MODEL_PATH_OLD = os.path.join(MODEL_DIR_OLD, 'phobert_absa.pt')

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

# Sentiment label indices for multi-label format
SENTIMENT_NAMES = ['NEG', 'POS', 'NEU']  # Index 0, 1, 2

# Map multi-label to old format for compatibility
SENTIMENT_MAP = {
    'POS': 1,
    'NEU': 0,
    'NEG': -1,
    'N/A': 2
}

SENTIMENT_MAP_REVERSE = {
    1: 'POSITIVE',
    0: 'NEUTRAL',
    -1: 'NEGATIVE',
    2: 'N/A'
}


class PhoBERTForABSAMultiPolarity(nn.Module):
    """PhoBERT model with multi-task learning for Multi-Polarity ABSA.
    
    Uses hard parameter sharing with two task heads:
    - Mention detection: Binary classification per aspect
    - Sentiment classification: MULTI-LABEL classification per aspect (can have multiple sentiments!)
    """
    
    def __init__(self, num_aspects: int = 9, dropout: float = 0.3):
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


# Legacy model class for backward compatibility
class PhoBERTForABSA(nn.Module):
    """PhoBERT model (old single-label version)."""
    
    def __init__(self, num_aspects: int = 9, dropout: float = 0.3):
        super().__init__()
        
        from transformers import AutoModel
        
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        hidden_size = self.phobert.config.hidden_size
        
        self.dropout = nn.Dropout(dropout)
        self.head_m = nn.Linear(hidden_size, num_aspects)
        self.head_s = nn.Linear(hidden_size, num_aspects * 3)
        self.num_aspects = num_aspects
    
    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        h_cls = self.dropout(cls_output)
        logits_m = self.head_m(h_cls)
        logits_s = self.head_s(h_cls).view(-1, self.num_aspects, 3)
        return logits_m, logits_s


class PhoBERTPredictor:
    """PhoBERT ABSA Predictor - loads model and makes predictions.
    Supports multi-polarity (multi-label) sentiment prediction.
    """
    
    def __init__(self, model_path: Optional[str] = None, use_multipolarity: bool = True):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        self.max_length = 256
        self.is_multipolarity = use_multipolarity
        self.threshold = 0.5  # Threshold for sigmoid predictions
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load trained PhoBERT model (multipolarity or legacy)."""
        
        # Try multipolarity model first
        if model_path is None:
            if os.path.exists(MODEL_PATH):
                model_path = MODEL_PATH
                self.is_multipolarity = True
                model_dir = MODEL_DIR
            elif os.path.exists(MODEL_PATH_OLD):
                model_path = MODEL_PATH_OLD
                self.is_multipolarity = False
                model_dir = MODEL_DIR_OLD
            else:
                print(f"❌ No model found!")
                print(f"   Checked: {MODEL_PATH}")
                print(f"   Checked: {MODEL_PATH_OLD}")
                return False
        else:
            model_dir = os.path.dirname(model_path)
            self.is_multipolarity = 'multipolarity' in model_path
        
        try:
            from transformers import AutoTokenizer
            
            mode_str = "Multi-Polarity" if self.is_multipolarity else "Legacy"
            print(f"📥 Loading PhoBERT {mode_str} model from {model_path}...")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Load tokenizer
            tokenizer_local_path = os.path.join(model_dir, 'tokenizer')
            if os.path.exists(tokenizer_local_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_local_path)
            else:
                tokenizer_name = 'vinai/phobert-base'
                if isinstance(checkpoint, dict) and 'tokenizer_name' in checkpoint:
                    tokenizer_name = checkpoint['tokenizer_name']
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # Load model with appropriate architecture
            if self.is_multipolarity:
                self.model = PhoBERTForABSAMultiPolarity(num_aspects=len(ASPECTS))
            else:
                self.model = PhoBERTForABSA(num_aspects=len(ASPECTS))
            
            # Load weights
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load config if exists
            config_path = os.path.join(model_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.max_length = config.get('max_length', 256)
            
            self.model_loaded = True
            f1_score = checkpoint.get('best_f1', 'N/A')
            if isinstance(f1_score, (int, float)):
                print(f"✅ Model loaded successfully! (F1: {f1_score:.4f})")
            else:
                print(f"✅ Model loaded successfully!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_single(self, text: str) -> Dict[str, any]:
        """
        Predict sentiment for a single review.
        
        Returns:
            Dict with two formats:
            - 'legacy': {aspect: int} for backward compatibility (-1=NEG, 0=NEU, 1=POS, 2=N/A)
            - 'multipolarity': {aspect: {'mentioned': bool, 'sentiments': List[str]}}
        """
        if not self.model_loaded:
            if not self.load_model():
                print("❌ Cannot make prediction - model not available")
                return {
                    'legacy': {asp: 2 for asp in ASPECTS},
                    'multipolarity': {asp: {'mentioned': False, 'sentiments': None} for asp in ASPECTS}
                }
        
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
            
            # Predict
            with torch.no_grad():
                logits_m, logits_s = self.model(input_ids, attention_mask)
                
                # Mention predictions (binary)
                preds_m = (torch.sigmoid(logits_m) > self.threshold).squeeze(0).cpu().numpy()
                
                if self.is_multipolarity:
                    # Multi-label: apply sigmoid and threshold
                    preds_s = (torch.sigmoid(logits_s) > self.threshold).squeeze(0).cpu().numpy()
                else:
                    # Single-label: argmax
                    preds_s = torch.argmax(logits_s, dim=-1).squeeze(0).cpu().numpy()
            
            # Build results
            legacy_result = {}
            multi_result = {}
            
            for i, aspect in enumerate(ASPECTS):
                mentioned = bool(preds_m[i])
                
                if mentioned:
                    if self.is_multipolarity:
                        # Multi-label: get all sentiments above threshold
                        sentiments = [SENTIMENT_NAMES[j] for j in range(3) if preds_s[i, j]]
                        
                        # Default to NEU if no sentiment detected
                        if not sentiments:
                            sentiments = ['NEU']
                        
                        # For legacy format, pick primary sentiment (POS > NEG > NEU priority)
                        if 'POS' in sentiments and 'NEG' in sentiments:
                            # Mixed sentiment - could show as Neutral or pick one
                            legacy_result[aspect] = 0  # Neutral
                        elif 'POS' in sentiments:
                            legacy_result[aspect] = 1
                        elif 'NEG' in sentiments:
                            legacy_result[aspect] = -1
                        else:
                            legacy_result[aspect] = 0
                    else:
                        # Single-label
                        sentiment_idx = int(preds_s[i])
                        # Map: 0=NEG->-1, 1=POS->1, 2=NEU->0
                        sentiment_to_legacy = {0: -1, 1: 1, 2: 0}
                        legacy_result[aspect] = sentiment_to_legacy.get(sentiment_idx, 0)
                        sentiments = [SENTIMENT_NAMES[sentiment_idx]]
                    
                    multi_result[aspect] = {
                        'mentioned': True,
                        'sentiments': sentiments
                    }
                else:
                    legacy_result[aspect] = 2  # N/A
                    multi_result[aspect] = {
                        'mentioned': False,
                        'sentiments': None
                    }
            
            return {
                'legacy': legacy_result,
                'multipolarity': multi_result
            }
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'legacy': {asp: 2 for asp in ASPECTS},
                'multipolarity': {asp: {'mentioned': False, 'sentiments': None} for asp in ASPECTS}
            }
    
    def predict_single_legacy(self, text: str) -> Dict[str, int]:
        """Predict and return only legacy format (for backward compatibility)."""
        result = self.predict_single(text)
        return result['legacy']
    
    def predict_single_multipolarity(self, text: str) -> Dict[str, Dict]:
        """Predict and return only multipolarity format."""
        result = self.predict_single(text)
        return result['multipolarity']
    
    def predict_batch(self, texts: List[str], format: str = 'legacy') -> List[Dict]:
        """
        Predict sentiment for a batch of reviews.
        
        Args:
            texts: List of review texts
            format: 'legacy', 'multipolarity', or 'both'
        """
        results = []
        for text in texts:
            pred = self.predict_single(text)
            if format == 'legacy':
                results.append(pred['legacy'])
            elif format == 'multipolarity':
                results.append(pred['multipolarity'])
            else:
                results.append(pred)
        return results


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


def aggregate_multipolarity_scores(predictions: List[Dict], aspects: List[str] = None) -> Dict[str, Dict]:
    """
    Aggregate multipolarity predictions into detailed scores.
    
    Returns:
        Dict mapping aspect to:
        - score: float (0-100)
        - sentiment_distribution: {POS: %, NEG: %, NEU: %}
        - review_count: int
        - has_mixed: bool (has reviews with multiple sentiments)
    """
    if aspects is None:
        aspects = ASPECTS
    
    if not predictions:
        return {asp: {'score': 50.0, 'sentiment_distribution': {}, 'review_count': 0, 'has_mixed': False} for asp in aspects}
    
    result = {}
    
    for aspect in aspects:
        sentiment_counts = {'POS': 0, 'NEG': 0, 'NEU': 0}
        total_mentioned = 0
        mixed_count = 0
        
        for pred in predictions:
            if aspect in pred and pred[aspect].get('mentioned'):
                total_mentioned += 1
                sentiments = pred[aspect].get('sentiments', [])
                
                if len(sentiments) > 1:
                    mixed_count += 1
                
                for s in sentiments:
                    if s in sentiment_counts:
                        sentiment_counts[s] += 1
        
        # Calculate score based on sentiment distribution
        if total_mentioned > 0:
            # Weighted score: POS=100, NEU=50, NEG=0
            total_votes = sum(sentiment_counts.values())
            if total_votes > 0:
                score = (sentiment_counts['POS'] * 100 + sentiment_counts['NEU'] * 50) / total_votes
            else:
                score = 50.0
            
            # Calculate percentages
            distribution = {k: round(v / total_votes * 100, 1) if total_votes > 0 else 0 for k, v in sentiment_counts.items()}
        else:
            score = 50.0
            distribution = {}
        
        result[aspect] = {
            'score': round(score, 1),
            'sentiment_distribution': distribution,
            'review_count': total_mentioned,
            'has_mixed': mixed_count > 0
        }
    
    return result


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
    print("=== PhoBERT ABSA Multi-Polarity Predictor Test ===")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    predictor = PhoBERTPredictor()
    
    if predictor.load_model():
        test_reviews = [
            "Sản phẩm chất lượng tốt, giao hàng nhanh, đóng gói cẩn thận",
            "Sản phẩm kém chất lượng, giao hàng chậm, shop thái độ tệ",
            "Áo đẹp nhưng vải hơi mỏng. Giao hàng nhanh!",  # Mixed sentiment
            "Giá rẻ nhưng chất lượng không tương xứng",  # Conflicting
        ]
        
        print(f"\n🔍 Testing {len(test_reviews)} reviews...\n")
        
        for review in test_reviews:
            print(f"Review: {review[:60]}...")
            result = predictor.predict_single(review)
            
            print("  [Multipolarity Format]:")
            for asp, info in result['multipolarity'].items():
                if info['mentioned']:
                    print(f"    {asp}: {info['sentiments']}")
            
            print("  [Legacy Format]:")
            for asp, val in result['legacy'].items():
                if val != 2:  # Skip N/A
                    sentiment = SENTIMENT_MAP_REVERSE.get(val, 'N/A')
                    print(f"    {asp}: {sentiment}")
            print()
