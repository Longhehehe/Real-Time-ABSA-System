"""
ML-Based ABSA Models: Logistic Regression, Naive Bayes.
Uses TF-IDF features + Binary Relevance (one classifier per native aspect).
"""
import os
import pickle
import numpy as np
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.dummy import DummyClassifier

DEFAULT_NUM_ASPECTS = 9


class MLBaseABSA:
    """Base class for ML-based multi-task ABSA models using TF-IDF."""

    def __init__(self, name: str, max_features: int = 10000,
                 num_aspects: int = DEFAULT_NUM_ASPECTS):
        self.name = name
        self.max_features = max_features
        self.num_aspects = int(num_aspects)
        if self.num_aspects <= 0:
            raise ValueError("num_aspects must be positive")
        self.tfidf = None
        self.mention_clfs = []
        self.sentiment_clfs = []

    def _create_mention_clf(self):
        raise NotImplementedError

    def _create_sentiment_clf(self):
        raise NotImplementedError

    def _build_classifiers(self):
        self.mention_clfs = [self._create_mention_clf() for _ in range(self.num_aspects)]
        self.sentiment_clfs = [self._create_sentiment_clf() for _ in range(self.num_aspects)]

    def fit(self, X_tfidf, labels_m: np.ndarray, labels_s: np.ndarray):
        """Train all sub-classifiers."""
        self._build_classifiers()
        if labels_m.shape[1] != self.num_aspects or labels_s.shape[1] != self.num_aspects:
            raise ValueError("Label aspect dimension does not match num_aspects")
        for i in range(self.num_aspects):
            target_m = labels_m[:, i]
            if np.unique(target_m).size < 2:
                self.mention_clfs[i] = DummyClassifier(
                    strategy='constant', constant=int(target_m[0])
                )
            self.mention_clfs[i].fit(X_tfidf, target_m)
            mask = labels_m[:, i] == 1
            if mask.sum() > 0:
                self.sentiment_clfs[i].fit(X_tfidf[mask], labels_s[mask, i, :])

    @staticmethod
    def _positive_probability(clf, X):
        probabilities = clf.predict_proba(X)
        classes = np.asarray(getattr(clf, 'classes_', []))
        if probabilities.ndim == 1:
            return probabilities
        if probabilities.shape[1] == 1:
            return np.ones(X.shape[0]) if classes.size and classes[0] == 1 else np.zeros(X.shape[0])
        matches = np.where(classes == 1)[0]
        return probabilities[:, int(matches[0])] if matches.size else probabilities[:, -1]

    def predict_proba(self, X_tfidf):
        """Return raw mention/sentiment probabilities for every sample/aspect."""
        n = X_tfidf.shape[0]
        prob_m = np.zeros((n, self.num_aspects), dtype=np.float32)
        prob_s = np.zeros((n, self.num_aspects, 3), dtype=np.float32)

        for i in range(self.num_aspects):
            prob_m[:, i] = self._positive_probability(self.mention_clfs[i], X_tfidf)
            sentiment_clf = self.sentiment_clfs[i]
            if hasattr(sentiment_clf, 'estimators_'):
                values = np.asarray(sentiment_clf.predict_proba(X_tfidf))
                if values.ndim == 1:
                    values = values[:, None]
                width = min(values.shape[1], 3)
                prob_s[:, i, :width] = values[:, :width]
        return prob_m, prob_s

    def predict(self, X_tfidf):
        """Predict mention + sentiment. Returns pred_m, pred_s, prob_m, prob_s."""
        n = X_tfidf.shape[0]
        prob_m, prob_s = self.predict_proba(X_tfidf)
        pred_m = (prob_m >= 0.5).astype(np.float32)
        pred_s = (prob_s >= 0.5).astype(np.float32)
        pred_s *= pred_m[:, :, None]
        no_sentiment = (pred_m == 1) & (pred_s.sum(axis=2) == 0)
        pred_s[:, :, 2][no_sentiment] = 1

        return pred_m, pred_s, prob_m, prob_s

    def save(self, output_dir: str, metadata: Dict = None):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{self.name}_model.pkl'), 'wb') as f:
            pickle.dump({
                'tfidf': self.tfidf,
                'mention_clfs': self.mention_clfs,
                'sentiment_clfs': self.sentiment_clfs,
                'num_aspects': self.num_aspects,
                'metadata': metadata or {},
            }, f)


class LogisticRegressionABSA(MLBaseABSA):
    """Logistic Regression for multi-task multi-polarity ABSA."""

    def __init__(self, num_aspects: int = DEFAULT_NUM_ASPECTS):
        super().__init__('logistic_regression', num_aspects=num_aspects)

    def _create_mention_clf(self):
        return LogisticRegression(max_iter=1000, C=1.0, random_state=42)

    def _create_sentiment_clf(self):
        return OneVsRestClassifier(LogisticRegression(max_iter=1000, C=1.0, random_state=42))


class NaiveBayesABSA(MLBaseABSA):
    """Naive Bayes for multi-task multi-polarity ABSA."""

    def __init__(self, num_aspects: int = DEFAULT_NUM_ASPECTS):
        super().__init__('naive_bayes', num_aspects=num_aspects)

    def _create_mention_clf(self):
        return MultinomialNB(alpha=1.0)

    def _create_sentiment_clf(self):
        return OneVsRestClassifier(MultinomialNB(alpha=1.0))
