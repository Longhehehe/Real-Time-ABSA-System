"""
Deep-Based ABSA Models: BiLSTM, CNN-BiLSTM.
PyTorch models with Embedding layer + dual task heads for multi-polarity ABSA.
"""
import torch
import torch.nn as nn

NUM_ASPECTS = 9


class BiLSTMForABSA(nn.Module):
    """BiLSTM model for multi-task multi-polarity ABSA."""

    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=2,
            bidirectional=True, batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.head_m = nn.Linear(hidden_dim * 2, NUM_ASPECTS)
        self.head_s = nn.Linear(hidden_dim * 2, NUM_ASPECTS * 3)

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths, batch_first=True, enforce_sorted=False
            )
            output, (h_n, _) = self.lstm(packed)
        else:
            output, (h_n, _) = self.lstm(emb)

        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        h = self.dropout(h)

        logits_m = self.head_m(h)
        logits_s = self.head_s(h).view(-1, NUM_ASPECTS, 3)
        return logits_m, logits_s


class CNNBiLSTMForABSA(nn.Module):
    """CNN + BiLSTM model for multi-task multi-polarity ABSA."""

    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(
            128, hidden_dim, num_layers=2,
            bidirectional=True, batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.head_m = nn.Linear(hidden_dim * 2, NUM_ASPECTS)
        self.head_s = nn.Linear(hidden_dim * 2, NUM_ASPECTS * 3)

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids)
        x = emb.permute(0, 2, 1)

        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))

        x = x.permute(0, 2, 1)
        output, (h_n, _) = self.lstm(x)

        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        h = self.dropout(h)

        logits_m = self.head_m(h)
        logits_s = self.head_s(h).view(-1, NUM_ASPECTS, 3)
        return logits_m, logits_s
