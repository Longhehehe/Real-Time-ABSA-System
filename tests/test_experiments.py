from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.adapters import (
    load_uit_jsonl,
    load_vlsp2016_tsv,
    load_vlsp2018_txt,
)
from experiments.metrics import (
    apply_acsa_thresholds,
    compute_acsa_metrics,
    compute_global_sentiment_metrics,
    tune_acsa_thresholds,
)
from experiments.modeling import MODEL_NAMES
from experiments.profiles import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_CONFIG_PATH,
    load_prepared_profile,
    load_registry,
    resolve_profile_ids,
)
from experiments.runner import ExperimentRunner, RunOptions, _resume_compatible, _run_ml_seed
from experiments.schema import PreparedProfile, SplitData
from methods.deep_models import BiLSTMForABSA, CNNBiLSTMForABSA


class RegistryTests(unittest.TestCase):
    def test_profile_taxonomies_are_native_and_complete(self):
        registry = load_registry(DEFAULT_CONFIG_PATH)
        profiles = registry["profiles"]
        self.assertEqual(len(profiles["uit_visd4sa"]["aspects"]), 10)
        self.assertEqual(len(profiles["vlsp2018_hotel"]["aspects"]), 34)
        self.assertEqual(len(profiles["vlsp2018_restaurant"]["aspects"]), 12)
        self.assertEqual(profiles["vlsp2016"]["task"], "global_sentiment")
        self.assertEqual(profiles["vlsp2016"]["aspects"], [])

    def test_default_matrix_has_72_runs(self):
        registry = load_registry(DEFAULT_CONFIG_PATH)
        profiles = resolve_profile_ids(["all"], registry)
        with tempfile.TemporaryDirectory() as tmp:
            runner = ExperimentRunner(RunOptions(artifact_root=Path(tmp)))
            matrix = runner.run_matrix(profiles, MODEL_NAMES, [42, 52, 62], dry_run=True)
        self.assertEqual(matrix["n_runs"], 72)

    def test_resume_rejects_smoke_result_for_full_run(self):
        expected = {
            "profile_id": "uit_visd4sa",
            "task": "acsa",
            "model": "phobert",
            "seed": 42,
            "protocol": "official_train_dev_test",
            "max_epochs": 10,
            "patience": 3,
            "batch_size_requested": 16,
            "max_length": 256,
            "dataset_stats": {"train": 7782, "dev": 1112, "test": 2225},
        }
        self.assertTrue(_resume_compatible({"config": expected}, expected))
        smoke = json.loads(json.dumps(expected))
        smoke["max_epochs"] = 1
        smoke["dataset_stats"]["train"] = 64
        self.assertFalse(_resume_compatible({"config": smoke}, expected))


class AdapterTests(unittest.TestCase):
    def test_uit_collapses_multiple_spans_to_multi_polarity(self):
        config = {
            "aspects": ["BATTERY", "CAMERA"],
            "polarity_map": {
                "NEGATIVE": "NEG",
                "POSITIVE": "POS",
                "NEUTRAL": "NEU",
            },
        }
        row = {
            "text": "pin tốt nhưng đôi lúc tụt nhanh",
            "labels": [
                [0, 7, "BATTERY#POSITIVE"],
                [18, 999, "BATTERY#NEGATIVE"],
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            split, audit = load_uit_jsonl("uit", "train", path, config)
        self.assertEqual(split.labels_m.tolist(), [[1.0, 0.0]])
        self.assertEqual(split.labels_s[0, 0].tolist(), [1.0, 1.0, 0.0])
        self.assertEqual(audit["invalid_span_count"], 1)

    def test_vlsp2018_keeps_native_aspect(self):
        config = {
            "aspects": ["FOOD#QUALITY", "SERVICE#GENERAL"],
            "polarity_map": {"negative": "NEG", "positive": "POS", "neutral": "NEU"},
        }
        content = (
            "#1\nĐồ ăn ngon nhưng phục vụ chậm\n"
            "{FOOD#QUALITY, positive}, {SERVICE#GENERAL, negative}\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.txt"
            path.write_text(content, encoding="utf-8")
            split, _ = load_vlsp2018_txt("vlsp", "train", path, config)
        self.assertEqual(split.labels_m.tolist(), [[1.0, 1.0]])
        self.assertEqual(split.labels_s[0, 0].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(split.labels_s[0, 1].tolist(), [1.0, 0.0, 0.0])

    def test_vlsp2016_skips_empty_text_and_keeps_three_class_index(self):
        config = {
            "polarity_map": {"NEG": "NEG", "POS": "POS", "NEU": "NEU"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.tsv"
            path.write_text("hay\tPOS\n\ufeff\tNEU\ntệ\tNEG\n", encoding="utf-8")
            split, audit = load_vlsp2016_tsv("vlsp2016", "train", path, config)
        self.assertEqual(split.labels.tolist(), [1, 0])
        self.assertEqual(audit["skipped_empty_text_count"], 1)


class MetricsAndModelTests(unittest.TestCase):
    def test_tfidf_is_fit_on_train_only(self):
        def split(name, texts, labels):
            return SplitData(
                profile_id="synthetic",
                task="global_sentiment",
                split=name,
                texts=texts,
                sample_ids=[f"{name}-{idx}" for idx in range(len(texts))],
                labels=np.asarray(labels, dtype=np.int64),
            )

        profile = PreparedProfile(
            profile_id="synthetic",
            task="global_sentiment",
            aspects=[],
            train=split(
                "train",
                ["xấu sản phẩm", "tốt sản phẩm", "bình thường", "rất xấu", "rất tốt", "tạm ổn"],
                [0, 1, 2, 0, 1, 2],
            ),
            dev=split("dev", ["devonlytoken tốt", "devonlytoken xấu", "devonlytoken thường"], [1, 0, 2]),
            test=split("test", ["testonlytoken tốt", "testonlytoken xấu", "testonlytoken thường"], [1, 0, 2]),
            audit={"clean_train_rows": 6},
            metadata={},
        )
        profile.validate()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _run_ml_seed(
                profile,
                "logistic_regression",
                42,
                run_dir,
                RunOptions(artifact_root=run_dir),
            )
            with (run_dir / "model.pkl").open("rb") as handle:
                payload = pickle.load(handle)
        vocabulary = payload["tfidf"].vocabulary_
        self.assertNotIn("devonlytoken", vocabulary)
        self.assertNotIn("testonlytoken", vocabulary)

    def test_acsa_thresholds_and_end_to_end_metric(self):
        true_m = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
        true_s = np.zeros((2, 2, 3), dtype=np.float32)
        true_s[0, 0, 1] = 1
        true_s[1, 1, 0] = 1
        prob_m = np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32)
        prob_s = np.zeros((2, 2, 3), dtype=np.float32)
        prob_s[0, 0, 1] = 0.9
        prob_s[1, 1, 0] = 0.9
        thresholds = tune_acsa_thresholds(true_m, true_s, prob_m, prob_s)
        pred_m, pred_s = apply_acsa_thresholds(prob_m, prob_s, *thresholds)
        metrics = compute_acsa_metrics(true_m, pred_m, true_s, pred_s, prob_m, prob_s)
        self.assertEqual(metrics["end_to_end_f1_micro"], 1.0)
        self.assertEqual(metrics["exact_match_accuracy"], 1.0)

    def test_global_sentiment_metric_contract(self):
        metrics = compute_global_sentiment_metrics(
            np.asarray([0, 1, 2]), np.asarray([0, 1, 1])
        )
        self.assertIn("f1_macro", metrics)
        self.assertEqual(len(metrics["confusion_matrix"]), 3)
        self.assertEqual(set(metrics["per_class"]), {"NEG", "POS", "NEU"})

    def test_dynamic_deep_output_shapes(self):
        inputs = torch.randint(0, 50, (2, 12))
        mask = torch.ones_like(inputs)
        for model_class, num_aspects in ((BiLSTMForABSA, 10), (CNNBiLSTMForABSA, 12)):
            model = model_class(vocab_size=50, hidden_dim=8, embedding_dim=8, num_aspects=num_aspects)
            mention, sentiment = model(inputs, mask)
            self.assertEqual(tuple(mention.shape), (2, num_aspects))
            self.assertEqual(tuple(sentiment.shape), (2, num_aspects, 3))


class RealDataIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(DEFAULT_CACHE_ROOT.exists(), "prepared benchmark cache is not available")
    def test_prepared_profiles_have_no_cross_split_leakage(self):
        expected = {
            "uit_visd4sa": 10,
            "vlsp2018_hotel": 34,
            "vlsp2018_restaurant": 12,
            "vlsp2016": 0,
        }
        for profile_id, num_aspects in expected.items():
            profile = load_prepared_profile(profile_id)
            profile.validate()
            self.assertEqual(len(profile.aspects), num_aspects)
            self.assertFalse(set(profile.train.texts).intersection(profile.dev.texts))
            self.assertFalse(set(profile.train.texts).intersection(profile.test.texts))


if __name__ == "__main__":
    unittest.main()
