import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import joblib
import time
from multiprocessing import cpu_count
from Src.preprocessing import preprocess_text
from Src.feature_enginerring import ParallelVietnameseSentimentFeatureExtractor, VietnameseSentimentFeatureExtractor
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MultinomialNBModel:
    def __init__(self, vncorenlp_path: str = "./VnCoreNLP/VnCoreNLP-1.1.1.jar", 
                 use_parallel: bool = True, n_workers: int = None):
        self.use_parallel = use_parallel
        self.n_workers = n_workers or min(cpu_count(), 4)
        if use_parallel:
            self.feature_extractor = ParallelVietnameseSentimentFeatureExtractor(vncorenlp_path, n_workers)
        else:
            self.feature_extractor = VietnameseSentimentFeatureExtractor(vncorenlp_path)
            
        self.label_encoder = LabelEncoder()
        self.model = None 
        self.is_fitted = False

    def train_model_with_preprocessed(self, df, preprocessed_texts, test_size=0.2, random_state=42, use_count=True, use_feature_selection=True, use_svd=True):
        print(f"Using {len(preprocessed_texts)} preprocessed samples")
        
        # Encode labels
        labels = self.label_encoder.fit_transform(df['label'])
        
        # Extract features with custom settings
        with tqdm(desc="Feature extraction") as pbar:
            if self.use_parallel and isinstance(self.feature_extractor, ParallelVietnameseSentimentFeatureExtractor):
                features = self.feature_extractor.main_extractor.extract_batch_features(
                    preprocessed_texts, y=labels, 
                    use_count=use_count, 
                    use_feature_selection=use_feature_selection, use_svd=use_svd
                )
            else:
                features = self.feature_extractor.extract_batch_features(
                    preprocessed_texts, y=labels,
                    use_count=use_count, 
                    use_feature_selection=use_feature_selection, use_svd=use_svd
                )
            pbar.update(1)
        
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state, stratify=labels)
        \
        with tqdm(desc="Training MultinomialNB") as pbar:
            start_time = time.time()
            
            # For MultinomialNB, ensure non-negative features
            X_train_processed = np.maximum(X_train, 0)
            X_test_processed = np.maximum(X_test, 0)
            
            self.model = MultinomialNB(alpha=1.0)
            self.model.fit(X_train_processed, y_train)
            y_pred = self.model.predict(X_test_processed)
            
            training_time = time.time() - start_time
            pbar.update(1)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'training_time': training_time
        }
        
        self.is_fitted = True
        return results, X_test, y_test, y_pred
    
    def predict(self, text):
        processed_text = preprocess_text([text], use_parallel=False)
        if isinstance(self.feature_extractor, ParallelVietnameseSentimentFeatureExtractor):
            features = self.feature_extractor.extract_batch_features_parallel(processed_text, use_parallel=False, use_feature_selection=False)
        else:
            features = self.feature_extractor.extract_batch_features(processed_text, use_feature_selection=False)
        features = np.maximum(features, 0)
        
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0] if hasattr(self.model, 'predict_proba') else None
        label = self.label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(probabilities) if probabilities is not None else None
        
        return label, confidence, probabilities
    
    def save_model(self, model_path="./models"):
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        
        model_dir = f"{model_path}/MultinomialNB_{int(time.time())}"
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.model, f"{model_dir}/model.pkl")
        joblib.dump(self.label_encoder, f"{model_dir}/label_encoder.pkl")
        
        return model_dir
    
    def load_model(self, model_path):
        self.model = joblib.load(f"{model_path}/model.pkl")
        self.label_encoder = joblib.load(f"{model_path}/label_encoder.pkl")
        self.is_fitted = True
    
    def close(self):
        self.feature_extractor.close()
