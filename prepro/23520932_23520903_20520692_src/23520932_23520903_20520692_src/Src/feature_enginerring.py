import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from multiprocessing import cpu_count
import threading
from scipy import sparse

class VietnameseSentimentFeatureExtractor:
    def __init__(self, vncorenlp_path: str = None):
        self.vncorenlp_path = vncorenlp_path 
        self.count_vectorizer = CountVectorizer(
            max_features=1500,  
            ngram_range=(1, 2),
            min_df=5,  
            max_df=0.8,
            lowercase=True
        )
        self.feature_selector = SelectKBest(chi2, k=500) 
        self.svd = TruncatedSVD(n_components=30, random_state=42)  
        self.is_fitted = False
    
    def extract_count_features(self, texts: List[str], fit: bool = True):
        if fit:
            features = self.count_vectorizer.fit_transform(texts)
        else:
            features = self.count_vectorizer.transform(texts)
        return features 
    
    def apply_feature_selection(self, X, y: np.ndarray = None, fit: bool = True):
        if fit and y is not None:
            if sparse.issparse(X):
                X.data = np.maximum(X.data, 0)
            else:
                X = np.maximum(X, 0)
            features = self.feature_selector.fit_transform(X, y)
        else:
            if sparse.issparse(X):
                X.data = np.maximum(X.data, 0)
            else:
                X = np.maximum(X, 0)
            features = self.feature_selector.transform(X)
        return features
    
    def apply_svd(self, X, fit: bool = True):
        if fit:
            features = self.svd.fit_transform(X)
        else:
            features = self.svd.transform(X)
        return features
    
    def extract_batch_features(self, texts: List[str], y: np.ndarray = None, use_count: bool = True, use_feature_selection: bool = True, use_svd: bool = True):
        if not texts:
            return np.array([])
        
        features_list = []
        # Extract Count features
        if use_count:
            count_features = self.extract_count_features(texts, fit=not self.is_fitted)
            features_list.append(count_features)
        
        # Combine features (horizontally stack sparse matrices)
        if features_list:
            combined_features = sparse.hstack(features_list, format='csr')
        else:
            return np.array([])
        
        # Apply feature selection (works with sparse matrices)
        if use_feature_selection and y is not None:
            combined_features = self.apply_feature_selection(combined_features, y, fit=not self.is_fitted)
        
        # Apply dimensionality reduction (prefer SVD for sparse matrices)
        if use_svd:
            combined_features = self.apply_svd(combined_features, fit=not self.is_fitted)
        self.is_fitted = True
        
        # Convert to dense only at the very end and only if needed
        if sparse.issparse(combined_features):
            combined_features = combined_features.toarray()
        
        return combined_features


class ParallelVietnameseSentimentFeatureExtractor:
    def __init__(self, vncorenlp_path: str = None, n_workers: int = None):
        self.vncorenlp_path = vncorenlp_path  # Keep for compatibility
        self.n_workers = n_workers or min(cpu_count(), 4)
        self.main_extractor = VietnameseSentimentFeatureExtractor(vncorenlp_path)
        self._local = threading.local()
    
    def _get_extractor(self):
        if not hasattr(self._local, 'extractor'):
            self._local.extractor = VietnameseSentimentFeatureExtractor(self.vncorenlp_path)
        return self._local.extractor
    
    def extract_batch_features_parallel(self, texts: List[str], y: np.ndarray = None, **kwargs) -> np.ndarray:
        if not texts:
            return np.array([])
        return self.main_extractor.extract_batch_features(texts, y, **kwargs)
