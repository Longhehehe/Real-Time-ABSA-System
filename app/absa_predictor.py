from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import os
import sys
import pickle
import numpy as np

# Add root to path to import methods
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from absa_dataset import ASPECTS
from methods import (
    LogisticRegressionABSA, NaiveBayesABSA,
    BiLSTMForABSA, CNNBiLSTMForABSA,
    PhoBERTForABSAMultiPolarity, XLMRoBERTaForABSA
)

class GeneralABSAPredictor:
    """
    A robust predictor that can handle any model type from the registry:
    - ML-Based (LR, NB)
    - Deep-Based (BiLSTM, CNN-BiLSTM)
    - Transformer (PhoBERT, XLM-R)
    """
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.tfidf = None
        self.model_type = None # 'ml', 'deep', 'transformer'
        self.model_class_name = None
        
        # Thresholds (per aspect/class)
        self.mention_threshold = 0.5
        self.sentiment_threshold = 0.5
        
        self._load_anything()

    def _load_anything(self):
        print(f"Loading model from: {self.model_path}")
        
        # Check if it's a pickle (ML) or a torch file (DL/Transformer)
        if self.model_path.endswith('.pkl'):
            self._load_ml()
        elif self.model_path.endswith('.pt'):
            self._load_torch()
        else:
            # Try to determine from path if extension is missing
            if 'logistic' in self.model_path.lower() or 'naive' in self.model_path.lower():
                self._load_ml()
            else:
                self._load_torch()

    def _load_ml(self):
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Determine class from filename or data
        filename = os.path.basename(self.model_path).lower()
        if 'logistic' in filename:
            self.model = LogisticRegressionABSA()
        else:
            self.model = NaiveBayesABSA()
            
        self.model.tfidf = data['tfidf']
        self.model.mention_clfs = data['mention_clfs']
        self.model.sentiment_clfs = data['sentiment_clfs']
        self.tfidf = data['tfidf']
        self.model_type = 'ml'
        self.model_class_name = self.model.__class__.__name__
        print(f"Successfully loaded ML model: {self.model_class_name}")

    def _load_torch(self):
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # 1. Identify model class
        if isinstance(checkpoint, dict):
            self.model_class_name = checkpoint.get('model_class')
            state_dict = checkpoint.get('model_state_dict')
            
            # Load thresholds if they exist (new format)
            if 'thresholds_m' in checkpoint and checkpoint['thresholds_m'] is not None:
                self.mention_threshold = np.array(checkpoint['thresholds_m'])
            if 'thresholds_s' in checkpoint and checkpoint['thresholds_s'] is not None:
                self.sentiment_threshold = np.array(checkpoint['thresholds_s'])
        else:
            state_dict = checkpoint
            # Fallback identification based on keys
            if 'conv1.weight' in state_dict: self.model_class_name = 'CNNBiLSTMForABSA'
            elif 'lstm.weight_ih_l0' in state_dict: self.model_class_name = 'BiLSTMForABSA'
            elif 'roberta.embeddings.word_embeddings.weight' in state_dict:
                # Check model path for XLM vs PhoBERT
                if 'xlm' in self.model_path.lower():
                    self.model_class_name = 'XLMRoBERTaForABSA'
                else:
                    self.model_class_name = 'PhoBERTForABSAMultiPolarity'
        
        # 2. Instantiate and Load
        num_aspects = len(ASPECTS)
        
        if self.model_class_name == 'PhoBERTForABSAMultiPolarity':
            self.model = PhoBERTForABSAMultiPolarity(num_aspects=num_aspects)
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            self.model_type = 'transformer'
        elif self.model_class_name == 'XLMRoBERTaForABSA':
            self.model = XLMRoBERTaForABSA(num_aspects=num_aspects)
            self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
            self.model_type = 'transformer'
        elif self.model_class_name in ['BiLSTMForABSA', 'CNNBiLSTMForABSA']:
            # For Deep models, we use PhoBERT tokenizer for pre-processing
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            # Determine vocab_size directly from the state_dict to avoid mismatches
            vocab_size = state_dict['embedding.weight'].shape[0]
            if self.model_class_name == 'BiLSTMForABSA':
                self.model = BiLSTMForABSA(vocab_size=vocab_size)
            else:
                self.model = CNNBiLSTMForABSA(vocab_size=vocab_size)
            self.model_type = 'deep'
        else:
            raise ValueError(f"Unknown model class: {self.model_class_name}")

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"Successfully loaded Torch model: {self.model_class_name} ({self.model_type})")

    def predict_single(self, text: str) -> dict:
        if self.model_type == 'ml':
            return self._predict_ml(text)
        else:
            return self._predict_torch(text)

    def _predict_ml(self, text: str) -> dict:
        X_tfidf = self.tfidf.transform([text])
        # ml_models predict returns: pred_m, pred_s, prob_m, prob_s
        pm, ps, prob_m_raw, prob_s_raw = self.model.predict(X_tfidf)
        
        # ML models already output "hard" predictions, but we want the probs for metric calculation
        prob_m = prob_m_raw[0]
        prob_s = prob_s_raw[0]
        
        return self._format_output(prob_m, prob_s)

    def _predict_torch(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding='max_length')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            logits_m, logits_s = self.model(input_ids, attention_mask)
            prob_m = torch.sigmoid(logits_m).squeeze(0).cpu().numpy()
            prob_s = torch.sigmoid(logits_s).squeeze(0).cpu().numpy()
            
        return self._format_output(prob_m, prob_s)

    def _format_output(self, prob_m, prob_s) -> dict:
        # th_m can be float (0.5) or array (9,)
        if isinstance(self.mention_threshold, np.ndarray): th_m = self.mention_threshold
        else: th_m = np.full(len(ASPECTS), self.mention_threshold)
            
        if isinstance(self.sentiment_threshold, np.ndarray): th_s = self.sentiment_threshold
        else: th_s = np.full((len(ASPECTS), 3), self.sentiment_threshold)
        
        preds_m = (prob_m >= th_m).astype(int)
        preds_s = (prob_s >= th_s).astype(int)
        
        # Enforce Neutral if no sentiment but mentioned
        for i in range(len(ASPECTS)):
            if preds_m[i] == 1 and preds_s[i].sum() == 0:
                preds_s[i, 2] = 1 # NEU
        
        multi_result = {}
        legacy_result = {}
        SENTIMENT_NAMES = ['NEG', 'POS', 'NEU']
        
        for i, aspect in enumerate(ASPECTS):
            is_m = bool(preds_m[i])
            active_s = [SENTIMENT_NAMES[j] for j in range(3) if preds_s[i, j] == 1]
            
            multi_result[aspect] = {
                'mentioned': is_m,
                'sentiments': active_s if is_m else []
            }
            
            # Legacy mapping for evaluation script
            if not is_m:
                legacy_result[aspect] = "2"
            else:
                if 'POS' in active_s and 'NEG' in active_s:
                    legacy_result[aspect] = "1, -1"
                elif 'POS' in active_s:
                    legacy_result[aspect] = "1"
                elif 'NEG' in active_s:
                    legacy_result[aspect] = "-1"
                else:
                    legacy_result[aspect] = "0"
                
        return {
            'multipolarity': multi_result,
            'legacy': legacy_result,
            'probs': {
                'mention': prob_m,
                'sentiment': prob_s
            }
        }

# Maintain backward compatibility
class PhoBERTPredictor(GeneralABSAPredictor):
    def __init__(self, model_path: str, device: str = None):
        super().__init__(model_path, device)

class XLMPredictor(GeneralABSAPredictor):
    def __init__(self, model_path: str, device: str = None):
        super().__init__(model_path, device)
