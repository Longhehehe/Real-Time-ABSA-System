import sys
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add the correct path to find Src modules
# The structure is: prepro/23520932_23520903_20520692_src/23520932_23520903_20520692_src/Src
base_path = r'c:/Users/Long/Documents/Hoc_Tap/SE363 (1)/prepro/23520932_23520903_20520692_src/23520932_23520903_20520692_src'
sys.path.insert(0, base_path)

from Src.train_multinb_rf import MultinomialNBModel
from Src.preprocessing import preprocess_text

def run_training():
    print("🚀 Starting Training Pipeline...")
    
    # 1. Load Data
    data_path = r'c:/Users/Long/Documents/Hoc_Tap/SE363 (1)/data/label/absa_grouped_vietnamese.xlsx'
    print(f"📥 Loading data from: {data_path}")
    
    # Handle the specific header structure of the file
    try:
        df = pd.read_excel(data_path, header=None)
        # Find header index logic (simplified from utils.py)
        header_idx = 0
        for idx, row in df.iterrows():
            if 'Chất lượng sản phẩm' in [str(v).strip() for v in row.values]:
                header_idx = idx
                break
        
        df = pd.read_excel(data_path, header=header_idx)
        print(f"✅ Data loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Prepare Data for Model
    # The model expects a 'content' and 'label' column?
    # Wait, the existing code in train_multinb_rf.py expects 'label' column.
    # The current excel file `absa_grouped_vietnamese` seems to be ASPECT-BASED labels (wide format), not simple sentiment per text.
    # We need to flatten this or aggregate it to train a general sentiment model, OR train aspect-specific models.
    # Given the request is simple "train model", let's create a GENERAL sentiment label for the text by averaging aspects.
    # OR simpler: Use the file `test_flow_reviews_1_labeled_full.xlsx` if it has 'content' and 'label'.
    # User said NOT to use `test_flow_reviews_1_labels`. 
    # So we must use `absa_grouped_vietnamese.xlsx`.
    
    # Strategy: Convert Aspect Scores to a Single Sentiment Label for the General Model
    # If Avg Score > 0.5 -> Positive (1), < 0.5 -> Negative (0)? 
    # The model classes are likely mapped.
    # Let's check Src/train_multinb_rf.py again. It uses LabelEncoder.
    
    # Let's assume we train on the aspect 'Chất lượng sản phẩm' as the primary label for now, or mix all aspects.
    # For demonstration, let's take 'Chất lượng sản phẩm' as the label.
    # Values: 1 (Pos), 0 (Neu), -1 (Neg), 2 (N/A).
    
    print("⚙️ Preprocessing data...")
    if 'reviewContent' not in df.columns:
        print("❌ 'reviewContent' column not found.")
        return

    # Flatten logic: Create a dataset of (text, label) pairs
    # We will exclude N/A (2)
    training_data = []
    
    target_aspect = 'Chất lượng sản phẩm'
    if target_aspect not in df.columns:
         print(f"❌ Aspect '{target_aspect}' not found for training target.")
         return

    filtered = df[df[target_aspect] != 2].dropna(subset=['reviewContent', target_aspect])
    
    # The existing model `train_model_with_preprocessed` expects a DataFrame with 'label' column.
    train_df = pd.DataFrame({
        'content': filtered['reviewContent'],
        'label': filtered[target_aspect] # -1, 0, 1
    })
    
    print(f"📊 Training samples: {len(train_df)}")

    # 3. Preprocess Text
    print("Cleaning text...")
    
    # Priority: Environment Variable (Docker) > Local Path (Dev)
    env_vncorenlp_path = os.getenv('VNCORENLP_PATH')
    local_vncorenlp_path = r'c:/Users/Long/Documents/Hoc_Tap/SE363 (1)/VnCoreNLP/VnCoreNLP-1.1.1.jar'
    
    if env_vncorenlp_path and os.path.exists(env_vncorenlp_path):
        vncorenlp_path = env_vncorenlp_path
        print(f"🌍 Using VnCoreNLP from ENV: {vncorenlp_path}")
    else:
        vncorenlp_path = local_vncorenlp_path # Fallback
        print(f"🏠 Using VnCoreNLP from LOCAL: {vncorenlp_path}")
    # Note: If user doesn't have VnCoreNLP jar at this exact path, it might fail. 
    # Let's try to run relevant preprocessing without full VnCoreNLP if possible, or assume it exists.
    # Inspecting parameters: `vncorenlp_path` is passed.
    
    # To avoid Java dependency issues in this "Demo" verify phase if the user environment is restricted,
    # I will wrap this in try-except.
    
    try:
        # 4. Train Model
        print("🧠 Training MultinomialNB Model...")
        model = MultinomialNBModel(vncorenlp_path=vncorenlp_path, use_parallel=False)
        
        # Preprocess
        texts = [str(x) for x in train_df['content'].tolist()]
        # We might mock preprocessing if VnCoreNLP fails, but let's try real call first.
        # Actually `preprocess_text` in their Src likely calls VnCoreNLP.
        
        # NOTE: Since I can't guarantee User has Java/VnCoreNLP running correct, 
        # I might need to simplify or assume it works. 
        # For the script to be valid, I write standard code.
        
        preprocessed_texts = preprocess_text(texts, use_parallel=False) 
        
        results, _, _, _ = model.train_model_with_preprocessed(
            train_df, 
            preprocessed_texts, 
            use_count=True, 
            use_feature_selection=True
        )
        
        print(f"✅ Training completed. Accuracy: {results.get('accuracy', 0.0):.4f}")
        
        # 5. Save Model
        output_dir = r'c:/Users/Long/Documents/Hoc_Tap/SE363 (1)/models/best_model'
        os.makedirs(output_dir, exist_ok=True)
        # The class `MultiNomialNBModel` creates a timestamped folder. We might want a fixed folder for the dashboard to load.
        # I'll modify the save logic or just rename after.
        saved_path = model.save_model(output_dir)
        print(f"💾 Model saved to: {saved_path}")
        
        # Save a 'latest' pointer or copy files to a fixed 'latest' dir for the dashboard?
        # For simplicity, let dashboard pick the latest or valid one.
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_training()
