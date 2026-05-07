import os
import sys
import pandas as pd
import torch
from tqdm import tqdm

# Add root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.absa_predictor import XLMPredictor, ASPECTS

def main():
    input_file = os.path.join(ROOT, 'True_Test_Data', 'dev_augmented_500.xlsx')
    output_file = os.path.join(ROOT, 'True_Test_Data', 'dev_predicted_xlm.xlsx')
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    # Initialize predictor
    predictor = XLMPredictor()
    
    # Try to find the correct model path
    model_path = os.path.join(ROOT, 'models', 'xlm_roberta_absa', 'xlm_roberta_large_absa.pt')
    if not os.path.exists(model_path):
        # Try alternate names if not found
        model_path = os.path.join(ROOT, 'models', 'xlm_roberta_absa', 'xlm_roberta_absa.pt')
        
    print(f"Loading model from: {model_path}")
    ok = predictor.load_model(model_path)
    if not ok:
        print("Failed to load model. Aborting.")
        return

    # Load data
    print(f"Reading data from: {input_file}")
    df = pd.read_excel(input_file)
    
    if 'reviewContent' not in df.columns:
        print("Error: 'reviewContent' column not found in the input file.")
        return

    # Predict
    print(f"Predicting for {len(df)} reviews using XLM-RoBERTa...")
    results = []
    
    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row['reviewContent'])
        pred = predictor.predict_single(text)
        
        # We'll use the legacy format for simple columns (-1, 0, 1, 2)
        # or we could use the multipolarity format if the user prefers.
        # Given the aspects list, we'll create one column per aspect.
        res_row = {'reviewContent': text}
        for aspect in ASPECTS:
            multi_info = pred['multipolarity'][aspect]
            if multi_info['mentioned']:
                sents = multi_info['sentiments']
                if len(sents) > 1:
                    # e.g., ["POS", "NEG"] -> "1, -1"
                    val_list = []
                    if "POS" in sents: val_list.append("1")
                    if "NEG" in sents: val_list.append("-1")
                    if "NEU" in sents: val_list.append("0")
                    res_row[aspect] = ", ".join(val_list)
                else:
                    sent = sents[0]
                    res_row[aspect] = 1 if sent == "POS" else (-1 if sent == "NEG" else 0)
            else:
                res_row[aspect] = 2 # N/A
        
        results.append(res_row)

    # Save
    out_df = pd.DataFrame(results)
    out_df.to_excel(output_file, index=False)
    print(f"Saved XLM predictions to: {output_file}")

if __name__ == '__main__':
    main()
