import numpy as np
import nltk
from collections import defaultdict, Counter
import math
import re
from typing import List, Tuple, Dict, Optional, Set
import logging
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import warnings
import time
import os
warnings.filterwarnings('ignore')

class OptimizedHMMPOSTagger:
    def __init__(self, alpha=0.0003, min_word_freq=2, use_word_features=True,
                 smoothing_method='kneser_ney', beam_width=5, 
                 use_parallel=True, cache_size=50000, stability_threshold=1e-15,
                 context_window=3, dynamic_feature_weights=True):
        self.word_to_idx = {}
        self.tag_to_idx = {}
        self.idx_to_tag = {}
        self.vocab_size = 0
        self.num_tags = 0

        self.alpha = alpha
        self.min_word_freq = min_word_freq
        self.use_word_features = use_word_features
        self.smoothing_method = smoothing_method
        self.beam_width = beam_width
        self.use_parallel = use_parallel
        self.cache_size = cache_size
        self.stability_threshold = stability_threshold
        self.context_window = context_window
        self.dynamic_feature_weights = dynamic_feature_weights

        self.transition_probs = None
        self.emission_probs = None
        self.initial_probs = None
        self.log_transition_probs = None
        self.log_emission_probs = None
        self.log_initial_probs = None

        self.feature_probs = {
            'suffix': defaultdict(lambda: defaultdict(float)),
            'prefix': defaultdict(lambda: defaultdict(float)),
            'shape': defaultdict(lambda: defaultdict(float)),
            'case': defaultdict(lambda: defaultdict(float)),
            'length': defaultdict(lambda: defaultdict(float)),
            'bigram': defaultdict(lambda: defaultdict(float)),
            'trigram': defaultdict(lambda: defaultdict(float)),
            'cluster': defaultdict(lambda: defaultdict(float)),
            'morphology': defaultdict(lambda: defaultdict(float))
        }
        self.feature_weights = {}

        self.unknown_word_idx = None
        self.tag_counts = None
        self.word_tag_counts = defaultdict(Counter)
        self.cache_unknown_emissions = {}
        self.feature_cache = {}

        self.special_tokens = [
            '<UNK>', '<RARE>', '<NUM>', '<PUNCT>', '<HYPHEN>', 
            '<APOSTROPHE>', '<EMAIL>', '<URL>', '<CURRENCY>', '<DATE>',
            '<TIME>', '<PERCENT>', '<ORDINAL>', '<CARDINAL>', '<ABBREVIATION>',
            '<PROPER>', '<FOREIGN>', '<SYMBOL>', '<ALLCAPS>', '<CAP>', '<SUFFIX_ING>'
        ]

        self.training_stats = {}
        self.stability_metrics = {}

        self.kneser_ney_discount = 0.75
        self.interpolation_weights = {'unigram': 0.1, 'bigram': 0.3, 'trigram': 0.6}

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def extract_enhanced_features(self, word: str, position: int = -1, sentence: List[str] = None) -> Dict:
        cache_key = f"{word}_{position}_{len(sentence) if sentence else 0}"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        features = self._extract_morphological_features(word)
        if sentence and position >= 0:
            features.update(self._extract_contextual_features(word, position, sentence))
            features.update(self._extract_positional_features(word, position, sentence))
        if len(self.feature_cache) < self.cache_size:
            self.feature_cache[cache_key] = features
        return features

    def _extract_morphological_features(self, word: str) -> Dict:
        features = {}
        word_lower = word.lower()
        features['length'] = min(len(word), 20)
        features['is_alpha'] = word.isalpha()
        features['is_title'] = word.istitle()
        features['is_upper'] = word.isupper()
        features['is_lower'] = word.islower()
        features['is_capitalized'] = word[0].isupper() if word else False
        features['shape'] = self.get_word_shape(word)
        features['simplified_shape'] = self.get_simplified_shape(word)
        features['case_pattern'] = self.get_case_pattern(word)
        features['has_digit'] = any(c.isdigit() for c in word)
        features['has_hyphen'] = '-' in word
        features['has_apostrophe'] = "'" in word
        features['has_punct'] = any(c in '.,!?;:()[]{}' for c in word)
        features['digit_ratio'] = sum(1 for c in word if c.isdigit()) / len(word) if word else 0
        features['vowel_ratio'] = sum(1 for c in word_lower if c in 'aeiou') / len(word) if word else 0
        for i in range(1, min(7, len(word_lower) + 1)):
            if i <= len(word_lower):
                features[f'prefix_{i}'] = word_lower[:i]
                features[f'suffix_{i}'] = word_lower[-i:]
        morphological_patterns = {
            'verb_patterns': ['ing', 'ed', 'er', 'est', 's', 'es', 'en'],
            'noun_patterns': ['tion', 'sion', 'ness', 'ment', 'ity', 'ty', 'ism', 'ist', 'ance', 'ence'],
            'adj_patterns': ['able', 'ible', 'ful', 'less', 'ous', 'ive', 'al', 'ic', 'ical', 'ary', 'ish'],
            'adv_patterns': ['ly', 'ward', 'wise', 'ways']
        }
        for pattern_type, patterns in morphological_patterns.items():
            for pattern in patterns:
                if word_lower.endswith(pattern):
                    features[f'morphology_{pattern_type}_{pattern}'] = len(pattern) / len(word_lower)
        features['is_email'] = bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', word))
        features['is_url'] = bool(re.match(r'^https?://|^www\.', word))
        features['is_currency'] = any(c in word for c in '$€£¥¢₹₽')
        features['is_date'] = bool(re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', word))
        features['is_time'] = bool(re.match(r'\d{1,2}:\d{2}', word))
        features['is_percent'] = word.endswith('%')
        features['is_ordinal'] = bool(re.match(r'\d+(st|nd|rd|th)$', word_lower))
        features['is_number'] = word.replace(',', '').replace('.', '').isdigit()
        features['complexity'] = self.calculate_complexity(word)
        return features

    def _extract_contextual_features(self, word: str, position: int, sentence: List[str]) -> Dict:
        features = {}
        sentence_len = len(sentence)
        features['position_ratio'] = position / max(sentence_len - 1, 1)
        features['is_sentence_start'] = position == 0
        features['is_sentence_end'] = position == sentence_len - 1
        features['sentence_length_bin'] = min(sentence_len // 5, 10)
        for offset in range(-self.context_window, self.context_window + 1):
            if offset == 0:
                continue
            ctx_pos = position + offset
            if 0 <= ctx_pos < sentence_len:
                ctx_word = sentence[ctx_pos].lower()
                features[f'context_{offset}'] = ctx_word
                features[f'context_{offset}_cap'] = sentence[ctx_pos][0].isupper()
                features[f'context_{offset}_len'] = len(ctx_word)
        if position > 0:
            features['bigram_context'] = f"{sentence[position-1].lower()}_{word.lower()}"
        if position > 1:
            features['trigram_context'] = f"{sentence[position-2].lower()}_{sentence[position-1].lower()}_{word.lower()}"
        return features

    def _extract_positional_features(self, word: str, position: int, sentence: List[str]) -> Dict:
        features = {}
        features['after_punct'] = position > 0 and any(c in '.,!?;:' for c in sentence[position-1])
        features['before_punct'] = position < len(sentence)-1 and any(c in '.,!?;:' for c in sentence[position+1])
        if position > 0:
            features['prev_cap'] = sentence[position-1][0].isupper()
        if position < len(sentence) - 1:
            features['next_cap'] = sentence[position+1][0].isupper()
        return features

    def get_word_shape(self, word: str) -> str:
        if not word:
            return ""
        shape = re.sub(r'[A-Z]', 'X', word)
        shape = re.sub(r'[a-z]', 'x', shape)
        shape = re.sub(r'[0-9]', 'd', shape)
        shape = re.sub(r'[.,!?;:]', '.', shape)
        return shape[:15]

    def get_simplified_shape(self, word: str) -> str:
        shape = self.get_word_shape(word)
        if not shape:
            return ""
        simplified = shape[0]
        for i in range(1, len(shape)):
            if shape[i] != shape[i-1]:
                simplified += shape[i]
        return simplified

    def get_case_pattern(self, word: str) -> str:
        if not word:
            return 'EMPTY'
        elif word.isupper():
            return 'ALL_UPPER'
        elif word.islower():
            return 'ALL_LOWER'
        elif word.istitle():
            return 'TITLE_CASE'
        elif word[0].isupper():
            return 'FIRST_UPPER'
        else:
            return 'MIXED_CASE'

    def calculate_complexity(self, word: str) -> float:
        if not word:
            return 0.0
        complexity = 0.0
        complexity += len(word) * 0.1
        complexity += sum(1 for c in word if not c.isalpha()) * 0.3
        complexity += len(set(word.lower())) / len(word) * 2
        return complexity

    def build_enhanced_vocabulary(self, traintext: List[List[str]], trainlabel: List[List[str]]):
        word_counts = Counter()
        tag_counts = Counter()
        feature_counters = {key: defaultdict(Counter) for key in self.feature_probs}
        total_words = 0
        for sent_idx, (sentence, tags) in enumerate(zip(traintext, trainlabel)):
            for i, (word, tag) in enumerate(zip(sentence, tags)):
                word_lower = word.lower()
                word_counts[word_lower] += 1
                tag_counts[tag] += 1
                self.word_tag_counts[word_lower][tag] += 1
                total_words += 1
                if self.use_word_features:
                    features = self.extract_enhanced_features(word, i, sentence)
                    self._collect_feature_statistics(tag, features, feature_counters)
        self.word_to_idx = {token: i for i, token in enumerate(self.special_tokens)}
        vocab_idx = len(self.special_tokens)
        frequent_words = {word: count for word, count in word_counts.items() 
                         if count >= self.min_word_freq}
        for word in sorted(frequent_words.keys()):
            self.word_to_idx[word] = vocab_idx
            vocab_idx += 1
        self.unknown_word_idx = 0
        self.vocab_size = len(self.word_to_idx)
        sorted_tags = sorted(tag_counts.keys())
        self.tag_to_idx = {tag: i for i, tag in enumerate(sorted_tags)}
        self.idx_to_tag = {i: tag for tag, i in self.tag_to_idx.items()}
        self.num_tags = len(self.tag_to_idx)
        self.tag_counts = tag_counts
        if self.use_word_features:
            self._build_feature_probabilities(feature_counters)
            if self.dynamic_feature_weights:
                self._calculate_dynamic_weights()
        self.training_stats = {
            'total_words': total_words,
            'vocab_size': self.vocab_size,
            'num_tags': self.num_tags,
            'oov_rate': 1 - len(frequent_words) / len(word_counts)
        }

    def _collect_feature_statistics(self, tag: str, features: Dict, feature_counters: Dict):
        for key, value in features.items():
            if key.startswith('suffix_'):
                feature_counters['suffix'][value][tag] += 1
            elif key.startswith('prefix_'):
                feature_counters['prefix'][value][tag] += 1
            elif key.startswith('morphology_'):
                feature_counters['morphology'][value][tag] += 1
        feature_counters['shape'][features.get('shape', 'UNK')][tag] += 1
        feature_counters['case'][features.get('case_pattern', 'UNK')][tag] += 1
        feature_counters['length'][features.get('length', 0)][tag] += 1
        if 'bigram_context' in features:
            feature_counters['bigram'][features['bigram_context']][tag] += 1
        if 'trigram_context' in features:
            feature_counters['trigram'][features['trigram_context']][tag] += 1
        cluster = self._assign_word_cluster(features)
        feature_counters['cluster'][cluster][tag] += 1

    def _assign_word_cluster(self, features: Dict) -> str:
        if features.get('is_email', False):
            return 'EMAIL_CLUSTER'
        elif features.get('is_url', False):
            return 'URL_CLUSTER'
        elif features.get('is_currency', False):
            return 'CURRENCY_CLUSTER'
        elif features.get('is_date', False):
            return 'DATE_CLUSTER'
        elif features.get('is_time', False):
            return 'TIME_CLUSTER'
        elif features.get('is_number', False):
            return 'NUMBER_CLUSTER'
        elif features.get('has_digit', False):
            return 'ALPHANUMERIC_CLUSTER'
        elif features.get('has_hyphen', False):
            return 'HYPHENATED_CLUSTER'
        elif any(k.startswith('morphology_verb') for k in features):
            return 'VERB_MORPH_CLUSTER'
        elif any(k.startswith('morphology_noun') for k in features):
            return 'NOUN_MORPH_CLUSTER'
        elif any(k.startswith('morphology_adj') for k in features):
            return 'ADJ_MORPH_CLUSTER'
        elif any(k.startswith('morphology_adv') for k in features):
            return 'ADV_MORPH_CLUSTER'
        else:
            return f"SHAPE_{features.get('simplified_shape', 'UNK')}_CLUSTER"

    def _build_feature_probabilities(self, feature_counters: Dict):
        def build_prob_dict(counts_dict, prob_dict, min_count=2, smoothing=0.01):
            for feature, tag_counter in counts_dict.items():
                total = sum(tag_counter.values())
                if total >= min_count:
                    smoothed_total = total + smoothing * self.num_tags
                    for tag in self.tag_to_idx.keys():
                        count = tag_counter.get(tag, 0)
                        prob_dict[feature][tag] = (count + smoothing) / smoothed_total
        for feature_type, counts in feature_counters.items():
            build_prob_dict(counts, self.feature_probs[feature_type])

    def _calculate_dynamic_weights(self):
        base_weights = {
            'suffix': 4.0, 'prefix': 2.0, 'shape': 3.0, 'case': 2.5,
            'length': 1.0, 'bigram': 3.5, 'trigram': 5.0, 'cluster': 3.0,
            'morphology': 4.5
        }
        for feature_type, probs_dict in self.feature_probs.items():
            if not probs_dict:
                self.feature_weights[feature_type] = base_weights.get(feature_type, 1.0)
                continue
            entropies = []
            for feature_probs in probs_dict.values():
                if feature_probs:
                    entropy = -sum(p * math.log2(p) for p in feature_probs.values() if p > 0)
                    entropies.append(entropy)
            if entropies:
                avg_entropy = np.mean(entropies)
                max_entropy = math.log2(self.num_tags)
                informativeness = 1 - (avg_entropy / max_entropy)
                coverage = len(probs_dict) / (self.vocab_size + 1)
                weight = base_weights.get(feature_type, 1.0) * (1 + informativeness) * (1 + coverage)
                self.feature_weights[feature_type] = weight
            else:
                self.feature_weights[feature_type] = base_weights.get(feature_type, 1.0)

    def enhanced_word_to_index(self, word: str) -> int:
        word_lower = word.lower()
        if word_lower in self.word_to_idx:
            return self.word_to_idx[word_lower]
        # Pseudoword mapping cải tiến
        if word.isdigit():
            return self.word_to_idx.get('<NUM>', self.unknown_word_idx)
        if word.isupper():
            return self.word_to_idx.get('<ALLCAPS>', self.unknown_word_idx)
        if word.istitle():
            return self.word_to_idx.get('<CAP>', self.unknown_word_idx)
        if re.match(r'.*ing$', word_lower):
            return self.word_to_idx.get('<SUFFIX_ING>', self.unknown_word_idx)
        if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', word):
            return self.word_to_idx.get('<EMAIL>', self.unknown_word_idx)
        if word.startswith(('http://', 'https://', 'www.')):
            return self.word_to_idx.get('<URL>', self.unknown_word_idx)
        if any(c in word for c in '$€£¥¢₹₽'):
            return self.word_to_idx.get('<CURRENCY>', self.unknown_word_idx)
        if re.match(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', word):
            return self.word_to_idx.get('<DATE>', self.unknown_word_idx)
        if re.match(r'\d{1,2}:\d{2}', word):
            return self.word_to_idx.get('<TIME>', self.unknown_word_idx)
        if word.endswith('%'):
            return self.word_to_idx.get('<PERCENT>', self.unknown_word_idx)
        if re.match(r'\d+(st|nd|rd|th)$', word_lower):
            return self.word_to_idx.get('<ORDINAL>', self.unknown_word_idx)
        if word.replace(',', '').replace('.', '').isdigit():
            return self.word_to_idx.get('<CARDINAL>', self.unknown_word_idx)
        if word[0].isupper() and word.isalpha():
            return self.word_to_idx.get('<PROPER>', self.unknown_word_idx)
        if any(c in '.,!?;:()[]{}' for c in word):
            return self.word_to_idx.get('<PUNCT>', self.unknown_word_idx)
        if '-' in word:
            return self.word_to_idx.get('<HYPHEN>', self.unknown_word_idx)
        if "'" in word:
            return self.word_to_idx.get('<APOSTROPHE>', self.unknown_word_idx)
        return self.unknown_word_idx

    def get_enhanced_unknown_word_emission_probs(self, word: str, position: int = -1, 
                                               sentence: List[str] = None) -> np.ndarray:
        cache_key = f"{word}_{position}_{len(sentence) if sentence else 0}"
        if cache_key in self.cache_unknown_emissions:
            return self.cache_unknown_emissions[cache_key]
        probs = np.array([self.tag_counts[self.idx_to_tag[i]] for i in range(self.num_tags)], dtype=float)
        probs = probs / np.sum(probs)
        if not self.use_word_features:
            self.cache_unknown_emissions[cache_key] = probs
            return probs
        features = self.extract_enhanced_features(word, position, sentence)
        for feature_type, feature_probs in self.feature_probs.items():
            weight = self.feature_weights.get(feature_type, 1.0)
            if feature_type == 'suffix':
                for i in range(1, min(7, len(word) + 1)):
                    suffix = word.lower()[-i:]
                    if suffix in feature_probs:
                        for tag, prob in feature_probs[suffix].items():
                            tag_idx = self.tag_to_idx[tag]
                            probs[tag_idx] *= (1 + prob * weight * (i / 6.0))
            elif feature_type == 'prefix':
                for i in range(1, min(5, len(word) + 1)):
                    prefix = word.lower()[:i]
                    if prefix in feature_probs:
                        for tag, prob in feature_probs[prefix].items():
                            tag_idx = self.tag_to_idx[tag]
                            probs[tag_idx] *= (1 + prob * weight * (i / 4.0))
            else:
                feature_keys = []
                if feature_type == 'shape':
                    feature_keys = [features.get('shape')]
                elif feature_type == 'case':
                    feature_keys = [features.get('case_pattern')]
                elif feature_type == 'length':
                    feature_keys = [features.get('length')]
                elif feature_type == 'bigram':
                    feature_keys = [features.get('bigram_context')]
                elif feature_type == 'trigram':
                    feature_keys = [features.get('trigram_context')]
                elif feature_type == 'cluster':
                    feature_keys = [self._assign_word_cluster(features)]
                elif feature_type == 'morphology':
                    feature_keys = [v for k, v in features.items() if k.startswith('morphology_')]
                for feature_key in feature_keys:
                    if feature_key and feature_key in feature_probs:
                        for tag, prob in feature_probs[feature_key].items():
                            tag_idx = self.tag_to_idx[tag]
                            probs[tag_idx] *= (1 + prob * weight)
        # Ổn định hóa xác suất
        probs = np.maximum(probs, self.stability_threshold)
        probs = probs / np.sum(probs)
        if len(self.cache_unknown_emissions) < self.cache_size:
            self.cache_unknown_emissions[cache_key] = probs
        return probs

    def estimate_enhanced_parameters(self, traintext: List[List[str]], trainlabel: List[List[str]]):
        initial_counts = np.zeros(self.num_tags, dtype=np.float64)
        transition_counts = np.zeros((self.num_tags, self.num_tags), dtype=np.float64)
        emission_counts = np.zeros((self.num_tags, self.vocab_size), dtype=np.float64)
        tag_total_counts = np.zeros(self.num_tags, dtype=np.float64)
        for sentence, tags in zip(traintext, trainlabel):
            if not sentence:
                continue
            word_indices = [self.enhanced_word_to_index(word) for word in sentence]
            tag_indices = [self.tag_to_idx[tag] for tag in tags]
            initial_counts[tag_indices[0]] += 1
            for word_idx, tag_idx in zip(word_indices, tag_indices):
                emission_counts[tag_idx, word_idx] += 1
                tag_total_counts[tag_idx] += 1
            for i in range(len(tag_indices) - 1):
                transition_counts[tag_indices[i], tag_indices[i+1]] += 1
        total_initial = np.sum(initial_counts)
        self.initial_probs = (initial_counts + self.alpha) / (total_initial + self.alpha * self.num_tags)
        self.transition_probs = np.zeros((self.num_tags, self.num_tags), dtype=np.float64)
        for i in range(self.num_tags):
            row_sum = np.sum(transition_counts[i])
            if row_sum > 0:
                if self.smoothing_method == 'kneser_ney':
                    self.transition_probs[i] = self._apply_kneser_ney_smoothing(
                        transition_counts[i], row_sum
                    )
                else:
                    mle_probs = transition_counts[i] / row_sum
                    uniform_probs = np.ones(self.num_tags) / self.num_tags
                    lambda_val = min(0.95, 0.85 + 0.1 * (row_sum / (row_sum + 50)))
                    self.transition_probs[i] = lambda_val * mle_probs + (1 - lambda_val) * uniform_probs
            else:
                self.transition_probs[i] = np.ones(self.num_tags) / self.num_tags
        self.emission_probs = np.zeros((self.num_tags, self.vocab_size), dtype=np.float64)
        for i in range(self.num_tags):
            if tag_total_counts[i] > 0:
                self.emission_probs[i] = (emission_counts[i] + self.alpha) / \
                                       (tag_total_counts[i] + self.alpha * self.vocab_size)
            else:
                self.emission_probs[i] = np.ones(self.vocab_size) / self.vocab_size
        self.transition_probs = np.maximum(self.transition_probs, self.stability_threshold)
        self.emission_probs = np.maximum(self.emission_probs, self.stability_threshold)
        self.initial_probs = np.maximum(self.initial_probs, self.stability_threshold)
        self.transition_probs = self.transition_probs / np.sum(self.transition_probs, axis=1, keepdims=True)
        self.emission_probs = self.emission_probs / np.sum(self.emission_probs, axis=1, keepdims=True)
        self.initial_probs = self.initial_probs / np.sum(self.initial_probs)
        # Lấy log với epsilon để tránh log(0)
        self.log_transition_probs = np.log(self.transition_probs + 1e-15)
        self.log_emission_probs = np.log(self.emission_probs + 1e-15)
        self.log_initial_probs = np.log(self.initial_probs + 1e-15)

    def _apply_kneser_ney_smoothing(self, counts: np.ndarray, total: float) -> np.ndarray:
        discount = self.kneser_ney_discount
        num_nonzero = np.sum(counts > 0)
        if num_nonzero == 0:
            return np.ones(len(counts)) / len(counts)
        discounted_counts = np.maximum(counts - discount, 0)
        alpha = discount * num_nonzero / total
        backoff_prob = 1.0 / self.num_tags
        probs = (discounted_counts / total) + alpha * backoff_prob
        return probs

    def viterbi_decode(self, sentence: List[str]) -> List[str]:
        if not sentence:
            return []
        T = len(sentence)
        word_indices = [self.enhanced_word_to_index(word) for word in sentence]
        log_V = np.full((T, self.num_tags), -np.inf, dtype=np.float64)
        backpointer = np.zeros((T, self.num_tags), dtype=np.int32)
        for s in range(self.num_tags):
            word_idx = word_indices[0]
            if word_idx == self.unknown_word_idx:
                unk_emissions = self.get_enhanced_unknown_word_emission_probs(
                    sentence[0], 0, sentence
                )
                emission_prob = unk_emissions[s]
            else:
                emission_prob = self.emission_probs[s, word_idx]
            log_emission = np.log(emission_prob + 1e-15)
            log_V[0, s] = self.log_initial_probs[s] + log_emission
        for t in range(1, T):
            for s in range(self.num_tags):
                word_idx = word_indices[t]
                if word_idx == self.unknown_word_idx:
                    unk_emissions = self.get_enhanced_unknown_word_emission_probs(
                        sentence[t], t, sentence
                    )
                    emission_prob = unk_emissions[s]
                else:
                    emission_prob = self.emission_probs[s, word_idx]
                log_emission = np.log(emission_prob + 1e-15)
                trans_scores = log_V[t-1] + self.log_transition_probs[:, s]
                # Ổn định hóa: nếu quá nhỏ thì chuẩn hóa lại
                if np.any(trans_scores < -1e10):
                    trans_scores -= np.max(trans_scores)
                best_prev = np.argmax(trans_scores)
                log_V[t, s] = trans_scores[best_prev] + log_emission
                backpointer[t, s] = best_prev
        best_path = [np.argmax(log_V[T-1])]
        for t in range(T-1, 0, -1):
            best_path.insert(0, backpointer[t, best_path[0]])
        return [self.idx_to_tag[tag_idx] for tag_idx in best_path]

    def train(self, traintext: List[List[str]], trainlabel: List[List[str]]):
        start_time = time.time()
        self.build_enhanced_vocabulary(traintext, trainlabel)
        self.estimate_enhanced_parameters(traintext, trainlabel)
        training_time = time.time() - start_time
        self.training_stats['training_time'] = training_time

    def predict(self, testtext: List[List[str]], use_beam_search: bool = False) -> List[List[str]]:
        if not testtext:
            return []
        if self.use_parallel and len(testtext) > 100:
            return self._predict_parallel(testtext)
        else:
            return self._predict_sequential(testtext)

    def _predict_sequential(self, testtext: List[List[str]]) -> List[List[str]]:
        predictions = []
        for sentence in testtext:
            try:
                tags = self.viterbi_decode(sentence)
                predictions.append(tags)
            except Exception as e:
                self.logger.warning(f"Error processing sentence: {e}")
                most_frequent_tag = max(self.tag_counts, key=self.tag_counts.get)
                predictions.append([most_frequent_tag] * len(sentence))
        return predictions

    def _predict_parallel(self, testtext: List[List[str]]) -> List[List[str]]:
        def safe_decode(sentence):
            try:
                return self.viterbi_decode(sentence)
            except:
                most_frequent_tag = max(self.tag_counts, key=self.tag_counts.get)
                return [most_frequent_tag] * len(sentence)
        num_workers = min(mp.cpu_count() - 1, len(testtext), 8)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            predictions = list(executor.map(safe_decode, testtext))
        return predictions

def run_optimized_enhanced(traintext, trainlabel, testtext, **kwargs):
    tagger = OptimizedHMMPOSTagger(**kwargs)
    tagger.train(traintext, trainlabel)
    predictions = tagger.predict(testtext)
    return predictions, tagger

def run(traintext, trainlabel, testtext):
    predictions, model = run_optimized_enhanced(
        traintext, trainlabel, testtext,
        alpha=0.0003,
        min_word_freq=2,
        use_word_features=True,
        smoothing_method='kneser_ney',
        beam_width=1,
        use_parallel=True,
        dynamic_feature_weights=True
    )
    return predictions
