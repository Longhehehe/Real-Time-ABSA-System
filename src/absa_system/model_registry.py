"""Canonical six-model registry for comparable ABSA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch.nn as nn

from .model import (
    AspectEvidenceModel,
    BiLSTMABSA,
    CNNBiLSTMABSA,
    TransformerCLSABSA,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    tokenizer_name: str | None
    checkpoint_filename: str
    iterative: bool
    supports_evidence: bool


MODEL_SPECS: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        "logistic_regression", "classical", None, "model.pkl", False, False
    ),
    "naive_bayes": ModelSpec(
        "naive_bayes", "classical", None, "model.pkl", False, False
    ),
    "bilstm": ModelSpec(
        "bilstm", "deep", "vinai/phobert-base", "model.pt", True, False
    ),
    "cnn_bilstm": ModelSpec(
        "cnn_bilstm", "deep", "vinai/phobert-base", "model.pt", True, False
    ),
    "phobert": ModelSpec(
        "phobert", "transformer", "vinai/phobert-base", "model.pt", True, True
    ),
    "xlm_roberta": ModelSpec(
        "xlm_roberta", "transformer", "xlm-roberta-base", "model.pt", True, False
    ),
}

MODEL_NAMES = tuple(MODEL_SPECS)


def get_model_spec(model_name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[str(model_name)]
    except KeyError as exc:
        raise ValueError(
            f"unknown model {model_name!r}; expected one of {list(MODEL_NAMES)}"
        ) from exc


def effective_neural_config(
    model_name: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve model-specific settings without mutating the frozen input."""

    spec = get_model_spec(model_name)
    if not spec.iterative:
        raise ValueError(f"{model_name} is not a neural model")
    resolved = dict(config)
    resolved["model_name"] = model_name
    resolved["backbone_name"] = spec.tokenizer_name
    if not spec.supports_evidence:
        resolved["evidence_weight"] = 0.0
    if spec.family == "deep":
        resolved["gradient_checkpointing"] = False
        resolved["head_lr"] = float(config.get("deep_learning_rate", 0.001))
        resolved["amp"] = bool(config.get("deep_amp", False))
    return resolved


def build_neural_model(
    model_name: str,
    *,
    tokenizer: Any,
    config: Mapping[str, Any],
) -> nn.Module:
    spec = get_model_spec(model_name)
    if not spec.iterative:
        raise ValueError(f"{model_name} is not a neural model")
    dropout = float(config.get("dropout", 0.2))
    if model_name == "phobert":
        return AspectEvidenceModel(
            backbone_name=str(spec.tokenizer_name),
            dropout=dropout,
            local_files_only=bool(config.get("local_files_only", False)),
        )
    if model_name == "xlm_roberta":
        return TransformerCLSABSA(
            backbone_name=str(spec.tokenizer_name),
            dropout=float(config.get("baseline_dropout", 0.3)),
            local_files_only=bool(config.get("local_files_only", False)),
        )
    vocab_size = int(
        getattr(tokenizer, "model_vocab_size", None) or len(tokenizer)
    )
    padding_idx = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    kwargs = {
        "vocab_size": vocab_size,
        "padding_idx": padding_idx,
        "embedding_dim": int(config.get("embedding_dim", 128)),
        "hidden_dim": int(config.get("recurrent_hidden_dim", 256)),
        "dropout": float(config.get("baseline_dropout", 0.3)),
    }
    if model_name == "bilstm":
        return BiLSTMABSA(**kwargs)
    if model_name == "cnn_bilstm":
        return CNNBiLSTMABSA(
            **kwargs,
            convolution_dim=int(config.get("convolution_dim", 128)),
        )
    raise ValueError(f"unsupported neural model: {model_name}")
