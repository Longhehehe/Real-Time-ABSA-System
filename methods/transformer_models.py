"""
Transformer-Based ABSA Models: PhoBERT, XLM-RoBERTa.
Uses pretrained transformer backbone + dual task heads for multi-polarity ABSA.
"""
import torch.nn as nn
from transformers import AutoModel

NUM_ASPECTS = 9

class PhoBERTForABSAMultiPolarity(nn.Module):
    """PhoBERT model with multi-task learning for Multi-Polarity ABSA.

    Uses hard parameter sharing with two task heads:
    - Mention detection: Binary classification per aspect
    - Sentiment classification: MULTI-LABEL classification per aspect (can have multiple sentiments!)
    """

    def __init__(self, num_aspects: int = NUM_ASPECTS, dropout: float = 0.3):
        super().__init__()

        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        hidden_size = self.phobert.config.hidden_size       

        self.dropout = nn.Dropout(dropout)
        self.head_m = nn.Linear(hidden_size, num_aspects)
        self.head_s = nn.Linear(hidden_size, num_aspects * 3)
        self.num_aspects = num_aspects

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        h = self.dropout(cls_output)
        logits_m = self.head_m(h)
        logits_s = self.head_s(h).view(-1, self.num_aspects, 3)
        return logits_m, logits_s

class XLMRoBERTaForABSA(nn.Module):
    """XLM-RoBERTa model for multi-task multi-polarity ABSA."""

    def __init__(self, num_aspects: int = NUM_ASPECTS, dropout: float = 0.3, model_name: str = "xlm-roberta-base"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size       
        self.dropout = nn.Dropout(dropout)
        self.head_m = nn.Linear(hidden_size, num_aspects)
        self.head_s = nn.Linear(hidden_size, num_aspects * 3)
        self.num_aspects = num_aspects

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        h = self.dropout(cls_output)
        logits_m = self.head_m(h)
        logits_s = self.head_s(h).view(-1, self.num_aspects, 3)
        return logits_m, logits_s
