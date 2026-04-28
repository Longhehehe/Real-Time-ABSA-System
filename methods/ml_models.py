"""
ML-Based ABSA Models: Logistic Regression, Naive Bayes.
Uses TF-IDF features + Binary Relevance (9 classifiers per task head).
"""
import os
import pickle
import numpy as np
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier

NUM_ASPECTS = 9


class MLBaseABSA:
    """Base class for ML-based multi-task ABSA models using TF-IDF."""

    def __init__(self, name: str, max_features: int = 10000):
        self.name = name
        self.max_features = max_features
        self.tfidf = None
        self.mention_clfs = []
        self.sentiment_clfs = []

    def _create_mention_clf(self):
        raise NotImplementedError

    def _create_sentiment_clf(self):
        raise NotImplementedError

    def _build_classifiers(self):
        self.mention_clfs = [self._create_mention_clf() for _ in range(NUM_ASPECTS)]
        self.sentiment_clfs = [self._create_sentiment_clf() for _ in range(NUM_ASPECTS)]

    def fit(self, X_tfidf, labels_m: np.ndarray, labels_s: np.ndarray):
        """Train all sub-classifiers."""
        self._build_classifiers()
        for i in range(NUM_ASPECTS):
            self.mention_clfs[i].fit(X_tfidf, labels_m[:, i])
            mask = labels_m[:, i] == 1
            if mask.sum() > 0:
                self.sentiment_clfs[i].fit(X_tfidf[mask], labels_s[mask, i, :])

    def predict(self, X_tfidf):
        """Predict mention + sentiment. Returns pred_m, pred_s, prob_m, prob_s."""
        n = X_tfidf.shape[0]
        pred_m = np.zeros((n, NUM_ASPECTS))
        pred_s = np.zeros((n, NUM_ASPECTS, 3))
        prob_m = np.zeros((n, NUM_ASPECTS))
        prob_s = np.zeros((n, NUM_ASPECTS, 3))

        for i in range(NUM_ASPECTS):
            pred_m[:, i] = self.mention_clfs[i].predict(X_tfidf)
            try:
                prob_m[:, i] = self.mention_clfs[i].predict_proba(X_tfidf)[:, 1]
            except Exception:
                prob_m[:, i] = pred_m[:, i]

            mask = pred_m[:, i] == 1
            if mask.sum() > 0:
                try:
                    pred_s[mask, i, :] = self.sentiment_clfs[i].predict(X_tfidf[mask])
                    prob_s[mask, i, :] = self.sentiment_clfs[i].predict_proba(X_tfidf[mask])
                except Exception:
                    pred_s[mask, i, :] = 0

        return pred_m, pred_s, prob_m, prob_s

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{self.name}_model.pkl'), 'wb') as f:
            pickle.dump({
                'tfidf': self.tfidf,
                'mention_clfs': self.mention_clfs,
                'sentiment_clfs': self.sentiment_clfs,
            }, f)


class LogisticRegressionABSA(MLBaseABSA):
    """Logistic Regression for multi-task multi-polarity ABSA."""

    def __init__(self):
        super().__init__('logistic_regression')

    def _create_mention_clf(self):
        return LogisticRegression(max_iter=1000, C=1.0, random_state=42)

    def _create_sentiment_clf(self):
        return OneVsRestClassifier(LogisticRegression(max_iter=1000, C=1.0, random_state=42))


class NaiveBayesABSA(MLBaseABSA):
    """Naive Bayes for multi-task multi-polarity ABSA."""

    def __init__(self):
        super().__init__('naive_bayes')

    def _create_mention_clf(self):
        return MultinomialNB(alpha=1.0)

    def _create_sentiment_clf(self):
        return OneVsRestClassifier(MultinomialNB(alpha=1.0))
