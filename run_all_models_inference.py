import argparse
import json
import os
import pickle
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer

from absa_dataset import ASPECTS
from methods import (
    LogisticRegressionABSA,
    NaiveBayesABSA,
    BiLSTMForABSA,
    CNNBiLSTMForABSA,
    PhoBERTForABSAMultiPolarity,
    XLMRoBERTaForABSA,
)

SENTIMENT_NAMES = ["NEG", "POS", "NEU"]
DEFAULT_TEXT = (
    "Hang chinh hang. Check thong tin ro rang. "
    "Hang mall ma thay danh gia nhieu nguoi noi hang gia cung hoang mang. "
    "Giao hang nhanh."
)

MODEL_DIRS = {
    "logistic_regression": "./models/logistic_regression_absa",
    "naive_bayes": "./models/naive_bayes_absa",
    "bilstm": "./models/bilstm_absa",
    "cnn_bilstm": "./models/cnn_bilstm_absa",
    "phobert": "./models/phobert_absa",
    "xlm_roberta": "./models/xlm_roberta_absa",
}


def _find_checkpoint(model_dir: str, hints: Optional[List[str]] = None) -> str:
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    candidates = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoint found in {model_dir}")

    if hints:
        for hint in hints:
            for fname in candidates:
                if hint in fname.lower():
                    return os.path.join(model_dir, fname)

    return os.path.join(model_dir, sorted(candidates)[0])


def _format_predictions(mention_scores, sentiment_scores, threshold: float) -> Dict[str, Dict]:
    predictions = {}
    for idx, aspect in enumerate(ASPECTS):
        mentioned = mention_scores[idx] > threshold
        sentiments = []
        if mentioned:
            for s_idx, score in enumerate(sentiment_scores[idx]):
                if score > threshold:
                    sentiments.append(SENTIMENT_NAMES[s_idx])
            if not sentiments:
                sentiments = ["NEU"]

        predictions[aspect] = {
            "mentioned": bool(mentioned),
            "sentiments": sentiments,
        }
    return predictions


def _predict_torch_model(
    model,
    tokenizer,
    text: str,
    max_length: int,
    threshold: float,
    device: str,
) -> Dict[str, Dict]:
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits_m, logits_s = model(input_ids, attention_mask)
        probs_m = torch.sigmoid(logits_m)[0].cpu().numpy()
        probs_s = torch.sigmoid(logits_s)[0].cpu().numpy()

    return _format_predictions(probs_m, probs_s, threshold)


def _predict_ml_model(model, text: str, threshold: float) -> Dict[str, Dict]:
    X = model.tfidf.transform([text])
    pred_m, pred_s, _, _ = model.predict(X)
    mention_scores = pred_m[0]
    sentiment_scores = pred_s[0]
    return _format_predictions(mention_scores, sentiment_scores, threshold)


def run_logistic_regression(text: str, model_dir: str, threshold: float) -> Dict:
    path = os.path.join(model_dir, "logistic_regression_model.pkl")
    with open(path, "rb") as f:
        payload = pickle.load(f)

    model = LogisticRegressionABSA()
    model.tfidf = payload["tfidf"]
    model.mention_clfs = payload["mention_clfs"]
    model.sentiment_clfs = payload["sentiment_clfs"]
    predictions = _predict_ml_model(model, text, threshold)
    return {"model_dir": model_dir, "predictions": predictions}


def run_naive_bayes(text: str, model_dir: str, threshold: float) -> Dict:
    path = os.path.join(model_dir, "naive_bayes_model.pkl")
    with open(path, "rb") as f:
        payload = pickle.load(f)

    model = NaiveBayesABSA()
    model.tfidf = payload["tfidf"]
    model.mention_clfs = payload["mention_clfs"]
    model.sentiment_clfs = payload["sentiment_clfs"]
    predictions = _predict_ml_model(model, text, threshold)
    return {"model_dir": model_dir, "predictions": predictions}


def run_bilstm(text: str, model_dir: str, max_length: int, threshold: float, device: str) -> Dict:
    checkpoint_path = _find_checkpoint(model_dir, ["bilstm", "bilstmforabsa"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = BiLSTMForABSA(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    predictions = _predict_torch_model(model, tokenizer, text, max_length, threshold, device)
    return {"model_dir": model_dir, "checkpoint": checkpoint_path, "predictions": predictions}


def run_cnn_bilstm(text: str, model_dir: str, max_length: int, threshold: float, device: str) -> Dict:
    checkpoint_path = _find_checkpoint(model_dir, ["cnn", "cnnbilstm", "cnnbi"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = CNNBiLSTMForABSA(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    predictions = _predict_torch_model(model, tokenizer, text, max_length, threshold, device)
    return {"model_dir": model_dir, "checkpoint": checkpoint_path, "predictions": predictions}


def run_phobert(text: str, model_dir: str, max_length: int, threshold: float, device: str) -> Dict:
    checkpoint_path = _find_checkpoint(model_dir, ["phobert", "absamultipolarity"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = PhoBERTForABSAMultiPolarity(num_aspects=len(ASPECTS))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    predictions = _predict_torch_model(model, tokenizer, text, max_length, threshold, device)
    return {"model_dir": model_dir, "checkpoint": checkpoint_path, "predictions": predictions}


def run_xlm_roberta(text: str, model_dir: str, max_length: int, threshold: float, device: str) -> Dict:
    checkpoint_path = _find_checkpoint(model_dir, ["xlm", "roberta"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = XLMRoBERTaForABSA(num_aspects=len(ASPECTS))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    predictions = _predict_torch_model(model, tokenizer, text, max_length, threshold, device)
    return {"model_dir": model_dir, "checkpoint": checkpoint_path, "predictions": predictions}


def _run_with_timing(name: str, func, *args, **kwargs) -> Dict:
    start = time.perf_counter()
    try:
        payload = func(*args, **kwargs)
        error = None
    except Exception as exc:
        payload = {}
        error = str(exc)
    end = time.perf_counter()

    payload["timing"] = {"seconds": round(end - start, 4)}
    if error:
        payload["error"] = error
    payload["name"] = name
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ABSA inference for all models")
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT)
    parser.add_argument("--output", type=str, default="reports/all_models_prediction.json")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    start_wall = datetime.now(timezone.utc)
    start_perf = time.perf_counter()

    results = {}
    results["logistic_regression"] = _run_with_timing(
        "logistic_regression",
        run_logistic_regression,
        args.text,
        MODEL_DIRS["logistic_regression"],
        args.threshold,
    )
    results["naive_bayes"] = _run_with_timing(
        "naive_bayes",
        run_naive_bayes,
        args.text,
        MODEL_DIRS["naive_bayes"],
        args.threshold,
    )
    results["bilstm"] = _run_with_timing(
        "bilstm",
        run_bilstm,
        args.text,
        MODEL_DIRS["bilstm"],
        args.max_length,
        args.threshold,
        device,
    )
    results["cnn_bilstm"] = _run_with_timing(
        "cnn_bilstm",
        run_cnn_bilstm,
        args.text,
        MODEL_DIRS["cnn_bilstm"],
        args.max_length,
        args.threshold,
        device,
    )
    results["phobert"] = _run_with_timing(
        "phobert",
        run_phobert,
        args.text,
        MODEL_DIRS["phobert"],
        args.max_length,
        args.threshold,
        device,
    )
    results["xlm_roberta"] = _run_with_timing(
        "xlm_roberta",
        run_xlm_roberta,
        args.text,
        MODEL_DIRS["xlm_roberta"],
        args.max_length,
        args.threshold,
        device,
    )

    end_perf = time.perf_counter()
    end_wall = datetime.now(timezone.utc)

    output = {
        "text": args.text,
        "models": results,
        "timing": {
            "seconds": round(end_perf - start_perf, 4),
            "start_time": start_wall.isoformat(),
            "end_time": end_wall.isoformat(),
            "device": device,
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved prediction to: {args.output}")


if __name__ == "__main__":
    main()
