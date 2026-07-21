"""Task-aware model definitions used by the official experiment runner."""

from __future__ import annotations

from typing import Dict, Tuple, Type

import torch
import torch.nn as nn
from transformers import AutoModel

from methods.deep_models import BiLSTMForABSA, CNNBiLSTMForABSA
from methods.transformer_models import PhoBERTForABSAMultiPolarity, XLMRoBERTaForABSA


MODEL_NAMES = (
    "logistic_regression",
    "naive_bayes",
    "bilstm",
    "cnn_bilstm",
    "phobert",
    "xlm_roberta",
)

NEURAL_MODEL_NAMES = ("bilstm", "cnn_bilstm", "phobert", "xlm_roberta")
TRANSFORMER_MODEL_NAMES = ("phobert", "xlm_roberta")
TOKENIZER_NAMES = {
    "bilstm": "vinai/phobert-base",
    "cnn_bilstm": "vinai/phobert-base",
    "phobert": "vinai/phobert-base",
    "xlm_roberta": "xlm-roberta-base",
}


class BiLSTMForSentiment(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 3,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
        else:
            _, (hidden, _) = self.lstm(embedded)
        representation = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(self.dropout(representation))


class CNNBiLSTMForSentiment(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 3,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.lstm = nn.LSTM(
            128,
            hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        values = self.embedding(input_ids).permute(0, 2, 1)
        values = self.pool(torch.relu(self.bn1(self.conv1(values))))
        values = self.pool(torch.relu(self.bn2(self.conv2(values))))
        values = values.permute(0, 2, 1)
        _, (hidden, _) = self.lstm(values)
        representation = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.classifier(self.dropout(representation))


class TransformerForSentiment(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        self.model_name = model_name
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(outputs.last_hidden_state[:, 0, :]))


def create_neural_model(
    model_name: str,
    task: str,
    vocab_size: int,
    num_aspects: int = 0,
) -> nn.Module:
    if task == "acsa":
        if model_name == "bilstm":
            return BiLSTMForABSA(vocab_size=vocab_size, num_aspects=num_aspects)
        if model_name == "cnn_bilstm":
            return CNNBiLSTMForABSA(vocab_size=vocab_size, num_aspects=num_aspects)
        if model_name == "phobert":
            return PhoBERTForABSAMultiPolarity(num_aspects=num_aspects, dropout=0.3)
        if model_name == "xlm_roberta":
            return XLMRoBERTaForABSA(num_aspects=num_aspects, dropout=0.3)
    elif task == "global_sentiment":
        if model_name == "bilstm":
            return BiLSTMForSentiment(vocab_size=vocab_size)
        if model_name == "cnn_bilstm":
            return CNNBiLSTMForSentiment(vocab_size=vocab_size)
        if model_name == "phobert":
            return TransformerForSentiment("vinai/phobert-base")
        if model_name == "xlm_roberta":
            return TransformerForSentiment("xlm-roberta-base")
    raise ValueError(f"Unsupported model/task combination: {model_name}/{task}")
