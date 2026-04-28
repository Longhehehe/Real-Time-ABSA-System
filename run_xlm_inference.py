import argparse
import json
import os
import time
from datetime import datetime, timezone

import torch
from transformers import AutoTokenizer

from absa_dataset import ASPECTS
from methods import XLMRoBERTaForABSA

SENTIMENT_NAMES = ["NEG", "POS", "NEU"]
DEFAULT_TEXT = (
    "Hang chinh hang. Check thong tin ro rang. "
    "Hang mall ma thay danh gia nhieu nguoi noi hang gia cung hoang mang. "
    "Giao hang nhanh."
)


def resolve_checkpoint(model_dir: str) -> str:
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    candidates = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoint found in {model_dir}")

    preferred = [f for f in candidates if "xlmroberta" in f.lower()]
    chosen = preferred[0] if preferred else sorted(candidates)[0]
    return os.path.join(model_dir, chosen)


def predict(text: str, model_dir: str, device: str, max_length: int, threshold: float) -> dict:
    checkpoint_path = resolve_checkpoint(model_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = XLMRoBERTaForABSA(num_aspects=len(ASPECTS))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

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

    predictions = {}
    for idx, aspect in enumerate(ASPECTS):
        mentioned = probs_m[idx] > threshold
        sentiments = []
        if mentioned:
            for s_idx, score in enumerate(probs_s[idx]):
                if score > threshold:
                    sentiments.append(SENTIMENT_NAMES[s_idx])
            if not sentiments:
                sentiments = ["NEU"]

        predictions[aspect] = {
            "mentioned": bool(mentioned),
            "sentiments": sentiments,
        }

    return {
        "model_dir": model_dir,
        "checkpoint": checkpoint_path,
        "text": text,
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XLM-RoBERTa ABSA inference")
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT, help="Input review text")
    parser.add_argument("--model_dir", type=str, default="./models/xlm_roberta_absa")
    parser.add_argument("--output", type=str, default="reports/xlm_prediction.json")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    start_wall = datetime.now(timezone.utc)
    start_perf = time.perf_counter()

    result = predict(
        text=args.text,
        model_dir=args.model_dir,
        device=device,
        max_length=args.max_length,
        threshold=args.threshold,
    )

    end_perf = time.perf_counter()
    end_wall = datetime.now(timezone.utc)

    result["timing"] = {
        "seconds": round(end_perf - start_perf, 4),
        "start_time": start_wall.isoformat(),
        "end_time": end_wall.isoformat(),
        "device": device,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved prediction to: {args.output}")


if __name__ == "__main__":
    main()
