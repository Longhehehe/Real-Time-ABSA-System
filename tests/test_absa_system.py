from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

from absa_system.data import (
    build_model_ready_release,
    sha256_file,
    stratified_group_split,
    validate_model_ready_release,
)
from absa_system.losses import compute_absa_loss
from absa_system.metrics import Thresholds, apply_thresholds, compute_metrics
from absa_system.model import AspectEvidenceModel
from absa_system.schema import (
    ABSAExample,
    ASPECTS,
    AnnotationSchemaError,
    EvidenceSpan,
    POLARITIES,
    parse_ai_annotation_record,
    sha256_text,
)
from absa_system.tokenization import build_evidence_masks
from absa_system.training import seal_training_run, validate_training_run


def annotation_row(index: int, *, mixed: bool = False) -> dict:
    text = f"Sản phẩm số {index} tốt nhưng giao hàng chậm."
    quality_start = text.index("tốt")
    shipping_start = text.index("giao hàng chậm")
    aspects = []
    for aspect in ASPECTS:
        if aspect == "Chất lượng sản phẩm":
            label = "1, -1" if mixed else 1
            evidence = [
                {
                    "start": quality_start,
                    "end": quality_start + len("tốt"),
                    "text": "tốt",
                    "polarity": "positive",
                }
            ]
            if mixed:
                evidence.append(
                    {
                        "start": shipping_start,
                        "end": shipping_start + len("giao hàng chậm"),
                        "text": "giao hàng chậm",
                        "polarity": "negative",
                    }
                )
        elif aspect == "Vận chuyển":
            label = -1
            evidence = [
                {
                    "start": shipping_start,
                    "end": shipping_start + len("giao hàng chậm"),
                    "text": "giao hàng chậm",
                    "polarity": "negative",
                }
            ]
        else:
            label = 2
            evidence = []
        aspects.append(
            {
                "aspect": aspect,
                "label": label,
                "evidence": evidence,
                "uncertainty_codes": [],
            }
        )
    return {
        "sample_id": f"sample-{index:04d}",
        "reviewContent": text,
        "review_text_sha256": sha256_text(text),
        "artifact_status": "AI_PSEUDO_LABEL_PENDING_HUMAN_VERIFICATION",
        "annotation": {
            "annotation_status": "LABELED",
            "aspects": aspects,
        },
        "source": {
            "product_id": f"product-{index // 2}",
            "category": "test",
        },
    }


class SchemaTests(unittest.TestCase):
    def test_mixed_label_and_evidence_are_preserved(self):
        row = annotation_row(1, mixed=True)
        example = parse_ai_annotation_record(
            row,
            source_package="fixture",
            source_domain="test",
        )
        self.assertEqual(example.sentiment_labels[0], (1, 1, 0))
        self.assertEqual(example.mention_labels[0], 1)
        self.assertEqual(len(example.evidence), 3)

    def test_invalid_evidence_offset_fails_closed(self):
        row = annotation_row(1)
        row["annotation"]["aspects"][0]["evidence"][0]["start"] += 1
        with self.assertRaises(AnnotationSchemaError):
            parse_ai_annotation_record(
                row,
                source_package="fixture",
                source_domain="test",
            )


class EvidenceAlignmentTests(unittest.TestCase):
    def test_character_span_maps_to_overlapping_tokens(self):
        result = build_evidence_masks(
            offsets=[(0, 0), (0, 3), (4, 8), (9, 12), (0, 0)],
            special_tokens_mask=[1, 0, 0, 0, 1],
            evidence=[
                {
                    "aspect_index": 0,
                    "polarity_index": 1,
                    "start": 4,
                    "end": 10,
                }
            ],
        )
        self.assertEqual(result["mention_evidence_mask"][0].tolist(), [0, 0, 1, 1, 0])
        self.assertEqual(
            result["polarity_evidence_mask"][0, 1].tolist(),
            [0, 0, 1, 1, 0],
        )


class MetricTests(unittest.TestCase):
    def test_threshold_application_keeps_mixed_and_enforces_neutral_exclusivity(self):
        mention_prob = np.full((1, len(ASPECTS)), 0.1)
        sentiment_prob = np.full((1, len(ASPECTS), len(POLARITIES)), 0.1)
        mention_prob[0, 0] = 0.9
        sentiment_prob[0, 0] = [0.8, 0.9, 0.7]
        thresholds = Thresholds(
            mention=np.full(len(ASPECTS), 0.5),
            sentiment=np.full((len(ASPECTS), len(POLARITIES)), 0.5),
        )
        mention, sentiment = apply_thresholds(
            mention_prob,
            sentiment_prob,
            thresholds,
        )
        self.assertEqual(mention[0, 0], 1)
        self.assertEqual(sentiment[0, 0].tolist(), [1, 1, 0])

        metrics = compute_metrics(mention, sentiment, mention, sentiment)
        self.assertEqual(metrics["exact_set_match"], 1.0)
        self.assertEqual(metrics["mixed"]["f1"], 1.0)


class GroupSplitTests(unittest.TestCase):
    def test_group_split_is_deterministic_and_disjoint(self):
        examples = []
        group_map = {}
        for index in range(30):
            mentions = (1,) + (0,) * (len(ASPECTS) - 1)
            sentiments = ((0, 1, 0),) + ((0, 0, 0),) * (len(ASPECTS) - 1)
            example = ABSAExample(
                sample_id=f"s-{index}",
                text=f"text {index}",
                review_text_sha256=sha256_text(f"text {index}"),
                annotation_status="LABELED",
                mention_labels=mentions,
                sentiment_labels=sentiments,
                evidence=(),
                source_package="fixture",
                source_domain="test",
                source_metadata={},
                label_provenance="fixture",
            )
            examples.append(example)
            group_map[example.sample_id] = f"group-{index // 2}"
        left = stratified_group_split(
            examples,
            group_map,
            ratios={"train": 0.6, "dev": 0.2, "test": 0.2},
            seed=7,
        )
        right = stratified_group_split(
            examples,
            group_map,
            ratios={"train": 0.6, "dev": 0.2, "test": 0.2},
            seed=7,
        )
        self.assertEqual(left, right)
        self.assertEqual(set(left.values()), {"train", "dev", "test"})


class DummyEncoder(nn.Module):
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(100, hidden_size)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class ModelTests(unittest.TestCase):
    def test_model_shapes_and_loss_backward(self):
        model = AspectEvidenceModel(
            backbone_name="dummy",
            encoder=DummyEncoder(),
            dropout=0.0,
        )
        batch_size, sequence_length = 2, 7
        input_ids = torch.randint(0, 100, (batch_size, sequence_length))
        attention_mask = torch.ones_like(input_ids)
        content_mask = attention_mask.clone()
        content_mask[:, 0] = 0
        output = model(input_ids, attention_mask, content_mask)
        self.assertEqual(
            output.mention_logits.shape,
            (batch_size, len(ASPECTS)),
        )
        self.assertEqual(
            output.sentiment_logits.shape,
            (batch_size, len(ASPECTS), len(POLARITIES)),
        )
        batch = {
            "mention_labels": torch.zeros(batch_size, len(ASPECTS)),
            "sentiment_labels": torch.zeros(
                batch_size, len(ASPECTS), len(POLARITIES)
            ),
            "mention_evidence_mask": torch.zeros(
                batch_size, len(ASPECTS), sequence_length
            ),
            "polarity_evidence_mask": torch.zeros(
                batch_size,
                len(ASPECTS),
                len(POLARITIES),
                sequence_length,
            ),
            "mention_evidence_available": torch.zeros(
                batch_size, len(ASPECTS)
            ),
            "polarity_evidence_available": torch.zeros(
                batch_size, len(ASPECTS), len(POLARITIES)
            ),
        }
        batch["mention_labels"][:, 0] = 1
        batch["sentiment_labels"][:, 0, 1] = 1
        loss = compute_absa_loss(output, batch)
        loss.total.backward()
        self.assertTrue(torch.isfinite(loss.total))
        self.assertIsNotNone(model.aspect_queries.grad)


class ReleaseBuilderTests(unittest.TestCase):
    def test_tiny_release_builds_and_validates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "labels.jsonl"
            with source.open("w", encoding="utf-8", newline="\n") as handle:
                for index in range(30):
                    handle.write(
                        json.dumps(
                            annotation_row(index, mixed=index % 7 == 0),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            config = {
                "schema_version": "absa-model-data-config/1.0.0",
                "seed": 7,
                "allowed_statuses": ["LABELED"],
                "split": {"train": 0.6, "dev": 0.2, "test": 0.2},
                "sources": [
                    {
                        "name": "fixture",
                        "path": "labels.jsonl",
                        "domain": "test",
                    }
                ],
                "reservation_ledgers": [],
                "curation_sources": [],
            }
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(config, ensure_ascii=False),
                encoding="utf-8",
            )
            output = root / "release"
            manifest = build_model_ready_release(
                project_root=root,
                config_path=config_path,
                output_dir=output,
            )
            result = validate_model_ready_release(output)
            self.assertEqual(result["status"], "VALID")
            self.assertEqual(result["records"], 30)
            self.assertEqual(manifest["leakage_audit"]["group_overlap"]["train_test"], 0)


class TrainingArtifactTests(unittest.TestCase):
    def test_training_run_is_sealed_and_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.pt").write_bytes(b"checkpoint")
            checkpoint_digest = sha256_file(root / "model.pt")
            payloads = {
                "run.json": {
                    "status": "COMPLETED",
                    "data_release_id": "release-test",
                    "data_manifest_sha256": "a" * 64,
                    "sample_limits": {"train": 2, "dev": 2, "test": 2},
                    "checkpoint": {"sha256": checkpoint_digest},
                },
                "thresholds.json": {"selected_on": "dev"},
                "training_config.json": {"seed": 1},
                "test_metrics.json": {"macro_f1": 0.0},
            }
            for name, payload in payloads.items():
                (root / name).write_text(
                    json.dumps(payload), encoding="utf-8"
                )
            (root / "epochs.jsonl").write_text(
                json.dumps({"epoch": 1}) + "\n", encoding="utf-8"
            )

            manifest = seal_training_run(root)
            self.assertEqual(manifest["status"], "SEALED_SMOKE_RUN")
            self.assertEqual(validate_training_run(root)["status"], "VALID")

            (root / "thresholds.json").write_text(
                json.dumps({"selected_on": "test"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                validate_training_run(root)


if __name__ == "__main__":
    unittest.main()
