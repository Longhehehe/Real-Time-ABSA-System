"""Simple CLI to run a single example through `app.absa_predictor.PhoBERTPredictor`.

Usage examples:
  python predict_example.py --model phobert --text "Sản phẩm đẹp, giao hàng nhanh"
  python predict_example.py --model xlm --text "Sản phẩm kém, giao hàng chậm"
  python predict_example.py --model models/phobert_absa/my_phobert.pt --text "..."

The script will try common model paths under the `models/` folder when given the keywords
`phobert` or `xlm`.
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from app.absa_predictor import PhoBERTPredictor, ASPECTS
import torch

# XLM model class lives in methods.transformer_models
from methods.transformer_models import XLMRoBERTaForABSA
from transformers import AutoTokenizer


def find_model_path(choice: str) -> Optional[str]:
    # If user provides explicit path, use it
    if os.path.exists(choice):
        return choice

    # Known candidate paths for 'phobert' and 'xlm' keywords
    candidates = []
    lowered = choice.lower()
    if 'phobert' in lowered or lowered in ('phobert', 'phobert.pt'):
        candidates = [
            os.path.join('models', 'phobert_absa', 'phobertforabsamultipolarity_absa.pt'),
            os.path.join('models', 'phobert_absa', 'phobert_absa.pt'),
            os.path.join('models', 'phobert_absa_multipolarity', 'phobert_absa_multipolarity.pt'),
            os.path.join('models', 'phobert_absa', 'phobertforabsamultipolarity_absa.pt')
        ]
    elif 'xlm' in lowered or lowered in ('xlm', 'xlm.pt'):
        candidates = [
            os.path.join('models', 'xlm_roberta_absa', 'xlmrobertaforabsa_absa.pt'),
            os.path.join('models', 'xlm_roberta_absa', 'xlm_roberta_absa.pt'),
            os.path.join('models', 'xlm_roberta_absa', 'xlm.pt')
        ]
    else:
        # Not a recognized keyword and not an existing path
        return None

    for rel in candidates:
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            return p

    # Try relative candidate names (without joining ROOT) in case tests expect that
    for rel in candidates:
        if os.path.exists(rel):
            return rel

    return None


def pretty_print_prediction(result: dict) -> None:
    print('\nPrediction (Multipolarity format):')
    multi = result.get('multipolarity', {})
    for asp in ASPECTS:
        info = multi.get(asp, {'mentioned': False, 'sentiments': None})
        if info.get('mentioned'):
            print(f"- {asp}: mentioned, sentiments={info.get('sentiments')}")
        else:
            print(f"- {asp}: not mentioned")

    print('\nLegacy format (compact):')
    legacy = result.get('legacy', {})
    for asp in ASPECTS:
        val = legacy.get(asp, 2)
        if val != 2:
            # Map: -1 NEG, 0 NEU, 1 POS
            lbl = 'POSITIVE' if val == 1 else ('NEUTRAL' if val == 0 else 'NEGATIVE')
            print(f"- {asp}: {lbl}")


def main():
    parser = argparse.ArgumentParser(description='Predict a single example using ABSA predictor')
    parser.add_argument('--model', '-m', default='phobert', help='Model keyword (phobert|xlm) or explicit .pt path')
    parser.add_argument('--text', '-t', required=False, help='Text to analyze (if omitted, script will prompt)')
    args = parser.parse_args()

    text = args.text
    if not text:
        try:
            text = input('Enter text to analyze: ').strip()
        except KeyboardInterrupt:
            print('\nCancelled')
            return

    model_path = find_model_path(args.model)
    if model_path is None:
        print('Model not found for choice:', args.model)
        print('Tried known candidate paths under the models/ folder.')
        return

    print('Using model:', model_path)

    predictor = PhoBERTPredictor()

    model_dir = os.path.dirname(model_path)
    # If the selected model folder or path mentions 'xlm', load using the XLM model class
    if 'xlm' in os.path.basename(model_dir).lower() or 'xlm' in model_path.lower():
        try:
            print('Detected XLM model. Loading XLMRoBERTaForABSA...')
            checkpoint = torch.load(model_path, map_location=predictor.device)

            tokenizer_local_path = os.path.join(model_dir, 'tokenizer')
            if os.path.exists(tokenizer_local_path):
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_local_path)
            else:
                tokenizer_name = 'xlm-roberta-base'
                if isinstance(checkpoint, dict) and 'tokenizer_name' in checkpoint:
                    tokenizer_name = checkpoint['tokenizer_name']
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            model = XLMRoBERTaForABSA(num_aspects=len(ASPECTS))
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model = model.to(predictor.device)
            model.eval()

            predictor.model = model
            predictor.tokenizer = tokenizer
            predictor.model_loaded = True
            print('XLM model loaded successfully.')
        except Exception as e:
            print('Failed to load XLM model:', e)
            return
    else:
        ok = predictor.load_model(model_path)
        if not ok:
            print('Failed to load model. Aborting.')
            return

    print('\nRunning prediction...')
    result = predictor.predict_single(text)
    pretty_print_prediction(result)


if __name__ == '__main__':
    main()
