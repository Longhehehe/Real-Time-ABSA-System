"""Official train/dev/test experiment execution for all profiles and models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import importlib.metadata
import json
import math
import os
import pickle
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from methods.ml_models import LogisticRegressionABSA, NaiveBayesABSA

from .artifacts import aggregate_model_runs, aggregate_profile, write_json
from .metrics import (
    apply_acsa_thresholds,
    compute_acsa_metrics,
    compute_global_sentiment_metrics,
    tune_acsa_thresholds,
)
from .modeling import (
    MODEL_NAMES,
    NEURAL_MODEL_NAMES,
    TOKENIZER_NAMES,
    TRANSFORMER_MODEL_NAMES,
    create_neural_model,
)
from .profiles import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    PreparedProfile,
    SplitData,
    load_prepared_profile,
)


DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "experiments"


@dataclass
class RunOptions:
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT
    cache_root: Path = DEFAULT_CACHE_ROOT
    config_path: Path = DEFAULT_CONFIG_PATH
    max_epochs: int = 10
    patience: int = 3
    batch_size: int = 16
    max_length: int = 256
    transformer_lr: float = 3e-5
    deep_lr: float = 1e-3
    threshold_min: float = 0.1
    threshold_max: float = 0.9
    threshold_steps: int = 17
    max_train_samples: Optional[int] = None
    max_dev_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    device: Optional[str] = None


class EncodedTextDataset(Dataset):
    def __init__(self, split: SplitData, tokenizer, max_length: int):
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.split.texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.split.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item: Dict[str, torch.Tensor] = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.split.task == "acsa":
            item["labels_m"] = torch.tensor(self.split.labels_m[index], dtype=torch.float32)
            item["labels_s"] = torch.tensor(self.split.labels_s[index], dtype=torch.float32)
        else:
            item["labels"] = torch.tensor(int(self.split.labels[index]), dtype=torch.long)
        return item


class BCEFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.label_smoothing:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        loss = ((1.0 - pt) ** self.gamma) * bce
        if mask is not None:
            while mask.ndim < loss.ndim:
                mask = mask.unsqueeze(-1)
            mask = mask.expand_as(loss)
            denominator = mask.sum().clamp(min=1.0)
            return (loss * mask).sum() / denominator
        return loss.mean()


def set_global_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        pass


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def environment_versions() -> Dict[str, Any]:
    versions: Dict[str, Any] = {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    for package in ("transformers", "numpy", "pandas", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _subset_split(split: SplitData, maximum: Optional[int], seed: int) -> SplitData:
    if maximum is None or maximum >= len(split.texts):
        return split
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(split.texts), size=maximum, replace=False)).tolist()
    return split.subset(indices)


def subset_profile(profile: PreparedProfile, options: RunOptions) -> PreparedProfile:
    if not any((options.max_train_samples, options.max_dev_samples, options.max_test_samples)):
        return profile
    subset = PreparedProfile(
        profile_id=profile.profile_id,
        task=profile.task,
        aspects=list(profile.aspects),
        train=_subset_split(profile.train, options.max_train_samples, 1101),
        dev=_subset_split(profile.dev, options.max_dev_samples, 1102),
        test=_subset_split(profile.test, options.max_test_samples, 1103),
        audit=profile.audit,
        metadata=profile.metadata,
    )
    subset.validate()
    return subset


def _dataset_stats(profile: PreparedProfile) -> Dict[str, Any]:
    return {
        "train": len(profile.train.texts),
        "dev": len(profile.dev.texts),
        "test": len(profile.test.texts),
        "num_aspects": len(profile.aspects),
        "raw_train": profile.audit.get("raw_train_rows"),
        "clean_train": profile.audit.get("clean_train_rows"),
        "removed_train_leakage": profile.audit.get(
            "removed_train_rows_due_to_dev_test_overlap"
        ),
        "source_manifest": profile.metadata.get("source_manifest"),
    }


def _base_run_config(
    profile: PreparedProfile,
    model_name: str,
    seed: int,
    options: RunOptions,
) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "profile_id": profile.profile_id,
        "task": profile.task,
        "model": model_name,
        "seed": seed,
        "protocol": "official_train_dev_test",
        "aspects": profile.aspects,
        "sentiment_order": ["NEG", "POS", "NEU"],
        "max_epochs": options.max_epochs,
        "patience": options.patience,
        "batch_size_requested": options.batch_size,
        "max_length": options.max_length,
        "tfidf": {
            "max_features": 10000,
            "ngram_range": [1, 2],
            "sublinear_tf": True,
        },
        "threshold_grid": {
            "min": options.threshold_min,
            "max": options.threshold_max,
            "steps": options.threshold_steps,
        },
        "dataset_stats": _dataset_stats(profile),
    }


def _resume_compatible(existing: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    """Reject stale smoke/parameter runs instead of silently resuming them."""

    config = existing.get("config") or {}
    scalar_keys = (
        "profile_id",
        "task",
        "model",
        "seed",
        "protocol",
        "max_epochs",
        "patience",
        "batch_size_requested",
        "max_length",
    )
    if any(config.get(key) != expected.get(key) for key in scalar_keys):
        return False
    existing_stats = config.get("dataset_stats", {})
    expected_stats = expected.get("dataset_stats", {})
    return all(existing_stats.get(key) == expected_stats.get(key) for key in ("train", "dev", "test"))


def _run_ml_seed(
    profile: PreparedProfile,
    model_name: str,
    seed: int,
    run_dir: Path,
    options: RunOptions,
) -> Dict[str, Any]:
    set_global_seed(seed)
    started = time.monotonic()
    config = _base_run_config(profile, model_name, seed, options)
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    train_features = vectorizer.fit_transform(profile.train.texts)
    dev_features = vectorizer.transform(profile.dev.texts)
    test_features = vectorizer.transform(profile.test.texts)

    if profile.task == "acsa":
        model = (
            LogisticRegressionABSA(num_aspects=len(profile.aspects))
            if model_name == "logistic_regression"
            else NaiveBayesABSA(num_aspects=len(profile.aspects))
        )
        model.tfidf = vectorizer
        model.fit(train_features, profile.train.labels_m, profile.train.labels_s)
        dev_prob_m, dev_prob_s = model.predict_proba(dev_features)
        thresholds_m, thresholds_s = tune_acsa_thresholds(
            profile.dev.labels_m,
            profile.dev.labels_s,
            dev_prob_m,
            dev_prob_s,
            options.threshold_min,
            options.threshold_max,
            options.threshold_steps,
        )
        dev_pred_m, dev_pred_s = apply_acsa_thresholds(
            dev_prob_m, dev_prob_s, thresholds_m, thresholds_s
        )
        dev_metrics = compute_acsa_metrics(
            profile.dev.labels_m,
            dev_pred_m,
            profile.dev.labels_s,
            dev_pred_s,
            dev_prob_m,
            dev_prob_s,
        )
        test_prob_m, test_prob_s = model.predict_proba(test_features)
        test_pred_m, test_pred_s = apply_acsa_thresholds(
            test_prob_m, test_prob_s, thresholds_m, thresholds_s
        )
        test_metrics = compute_acsa_metrics(
            profile.test.labels_m,
            test_pred_m,
            profile.test.labels_s,
            test_pred_s,
            test_prob_m,
            test_prob_s,
        )
        thresholds = {
            "thresholds_m": thresholds_m.tolist(),
            "thresholds_s": thresholds_s.tolist(),
            "selected_on": "dev",
        }
        write_json(run_dir / "thresholds.json", thresholds)
        payload = {
            "schema_version": 2,
            "task": profile.task,
            "profile_id": profile.profile_id,
            "aspects": profile.aspects,
            "sentiment_order": ["NEG", "POS", "NEU"],
            "num_aspects": len(profile.aspects),
            "model_class": model.__class__.__name__,
            "tfidf": vectorizer,
            "mention_clfs": model.mention_clfs,
            "sentiment_clfs": model.sentiment_clfs,
            "thresholds_m": thresholds_m,
            "thresholds_s": thresholds_s,
            "metadata": config,
        }
        checkpoint_file = "model.pkl"
    else:
        classifier = (
            LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
            if model_name == "logistic_regression"
            else MultinomialNB(alpha=1.0)
        )
        classifier.fit(train_features, profile.train.labels)
        dev_prediction = classifier.predict(dev_features)
        dev_metrics = compute_global_sentiment_metrics(profile.dev.labels, dev_prediction)
        test_prediction = classifier.predict(test_features)
        test_metrics = compute_global_sentiment_metrics(profile.test.labels, test_prediction)
        payload = {
            "schema_version": 2,
            "task": profile.task,
            "profile_id": profile.profile_id,
            "aspects": [],
            "sentiment_order": ["NEG", "POS", "NEU"],
            "model_class": classifier.__class__.__name__,
            "tfidf": vectorizer,
            "classifier": classifier,
            "metadata": config,
        }
        checkpoint_file = "model.pkl"

    with (run_dir / checkpoint_file).open("wb") as handle:
        pickle.dump(payload, handle)
    return {
        "schema_version": 2,
        "status": "completed",
        "profile_id": profile.profile_id,
        "task": profile.task,
        "model": model_name,
        "seed": seed,
        "best_epoch": None,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "training_loss": [],
        "checkpoint_file": checkpoint_file,
        "config": config,
        "duration_seconds": time.monotonic() - started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


@torch.no_grad()
def _collect_neural_predictions(
    model: nn.Module,
    loader: DataLoader,
    task: str,
    device: str,
) -> Tuple[np.ndarray, ...]:
    model.eval()
    if task == "acsa":
        true_m: List[np.ndarray] = []
        true_s: List[np.ndarray] = []
        prob_m: List[np.ndarray] = []
        prob_s: List[np.ndarray] = []
        for batch in loader:
            logits_m, logits_s = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            true_m.append(batch["labels_m"].numpy())
            true_s.append(batch["labels_s"].numpy())
            prob_m.append(torch.sigmoid(logits_m).cpu().numpy())
            prob_s.append(torch.sigmoid(logits_s).cpu().numpy())
        return (
            np.concatenate(true_m),
            np.concatenate(true_s),
            np.concatenate(prob_m),
            np.concatenate(prob_s),
        )

    labels: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        labels.append(batch["labels"].numpy())
        probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def _evaluate_neural(
    model: nn.Module,
    loader: DataLoader,
    profile: PreparedProfile,
    device: str,
    options: RunOptions,
    thresholds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[Dict[str, Any], Optional[Tuple[np.ndarray, np.ndarray]]]:
    predictions = _collect_neural_predictions(model, loader, profile.task, device)
    if profile.task == "acsa":
        true_m, true_s, prob_m, prob_s = predictions
        if thresholds is None:
            thresholds = tune_acsa_thresholds(
                true_m,
                true_s,
                prob_m,
                prob_s,
                options.threshold_min,
                options.threshold_max,
                options.threshold_steps,
            )
        pred_m, pred_s = apply_acsa_thresholds(prob_m, prob_s, *thresholds)
        return (
            compute_acsa_metrics(true_m, pred_m, true_s, pred_s, prob_m, prob_s),
            thresholds,
        )
    true_labels, probabilities = predictions
    predicted = np.argmax(probabilities, axis=1)
    return compute_global_sentiment_metrics(true_labels, predicted), None


def _train_neural_attempt(
    profile: PreparedProfile,
    model_name: str,
    seed: int,
    run_dir: Path,
    options: RunOptions,
    batch_size: int,
    gradient_accumulation: int,
    tokenizer,
    device: str,
) -> Dict[str, Any]:
    set_global_seed(seed)
    started = time.monotonic()
    train_dataset = EncodedTextDataset(profile.train, tokenizer, options.max_length)
    dev_dataset = EncodedTextDataset(profile.dev, tokenizer, options.max_length)
    test_dataset = EncodedTextDataset(profile.test, tokenizer, options.max_length)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = create_neural_model(
        model_name,
        profile.task,
        vocab_size=len(tokenizer),
        num_aspects=len(profile.aspects),
    ).to(device)
    learning_rate = options.transformer_lr if model_name in TRANSFORMER_MODEL_NAMES else options.deep_lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / gradient_accumulation))
    total_updates = updates_per_epoch * options.max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_updates // 10,
        num_training_steps=total_updates,
    )
    mention_loss = BCEFocalLoss(
        gamma=2.0,
        label_smoothing=0.1 if model_name in TRANSFORMER_MODEL_NAMES else 0.0,
    )
    sentiment_loss = BCEFocalLoss(
        gamma=2.0,
        label_smoothing=0.1 if model_name in TRANSFORMER_MODEL_NAMES else 0.0,
    )
    global_loss = nn.CrossEntropyLoss()
    primary_metric = "end_to_end_f1_micro" if profile.task == "acsa" else "f1_macro"
    best_score = -float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_thresholds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    best_dev_metrics: Optional[Dict[str, Any]] = None
    epochs_without_improvement = 0
    training_loss: List[float] = []

    for epoch in range(1, options.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        batches = 0
        progress = tqdm(
            train_loader,
            desc=f"{profile.profile_id}/{model_name}/seed{seed}/epoch{epoch}",
            leave=False,
        )
        for batch_index, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if profile.task == "acsa":
                labels_m = batch["labels_m"].to(device)
                labels_s = batch["labels_s"].to(device)
                logits_m, logits_s = model(input_ids, attention_mask)
                loss = 2.0 * mention_loss(logits_m, labels_m)
                loss = loss + 5.0 * sentiment_loss(logits_s, labels_s, mask=labels_m)
            else:
                logits = model(input_ids, attention_mask)
                loss = global_loss(logits, batch["labels"].to(device))

            (loss / gradient_accumulation).backward()
            running_loss += float(loss.detach().cpu())
            batches += 1
            should_step = batch_index % gradient_accumulation == 0 or batch_index == len(train_loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

        epoch_loss = running_loss / max(batches, 1)
        training_loss.append(epoch_loss)
        dev_metrics, candidate_thresholds = _evaluate_neural(
            model, dev_loader, profile, device, options, thresholds=None
        )
        score = float(dev_metrics[primary_metric])
        print(
            f"{profile.profile_id}/{model_name}/seed={seed} epoch={epoch} "
            f"loss={epoch_loss:.4f} dev_{primary_metric}={score:.4f}"
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_thresholds = candidate_thresholds
            best_dev_metrics = dev_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= options.patience:
                break

    if best_state is None or best_dev_metrics is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    test_metrics, _ = _evaluate_neural(
        model,
        test_loader,
        profile,
        device,
        options,
        thresholds=best_thresholds,
    )
    config = _base_run_config(profile, model_name, seed, options)
    config.update(
        {
            "batch_size_actual": batch_size,
            "gradient_accumulation": gradient_accumulation,
            "effective_batch_size": batch_size * gradient_accumulation,
            "learning_rate": learning_rate,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "tokenizer_name": TOKENIZER_NAMES[model_name],
            "device": device,
        }
    )
    checkpoint = {
        "schema_version": 2,
        "profile_id": profile.profile_id,
        "task": profile.task,
        "model_class": model.__class__.__name__,
        "model_name": model_name,
        "model_state_dict": best_state,
        "aspects": profile.aspects,
        "num_aspects": len(profile.aspects),
        "sentiment_order": ["NEG", "POS", "NEU"],
        "tokenizer_name": TOKENIZER_NAMES[model_name],
        "max_length": options.max_length,
        "seed": seed,
        "hyperparameters": config,
        "source_manifest": profile.metadata.get("source_manifest"),
        "best_epoch": best_epoch,
        "best_dev_score": best_score,
        "thresholds_m": best_thresholds[0].tolist() if best_thresholds else None,
        "thresholds_s": best_thresholds[1].tolist() if best_thresholds else None,
    }
    checkpoint_file = "model.pt"
    torch.save(checkpoint, run_dir / checkpoint_file)
    if best_thresholds:
        write_json(
            run_dir / "thresholds.json",
            {
                "thresholds_m": best_thresholds[0].tolist(),
                "thresholds_s": best_thresholds[1].tolist(),
                "selected_on": "dev",
                "best_epoch": best_epoch,
            },
        )
    return {
        "schema_version": 2,
        "status": "completed",
        "profile_id": profile.profile_id,
        "task": profile.task,
        "model": model_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "dev_metrics": best_dev_metrics,
        "test_metrics": test_metrics,
        "training_loss": training_loss,
        "checkpoint_file": checkpoint_file,
        "config": config,
        "duration_seconds": time.monotonic() - started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


class ExperimentRunner:
    def __init__(self, options: RunOptions):
        self.options = options
        self.options.artifact_root = self.options.artifact_root.resolve()
        self.options.artifact_root.mkdir(parents=True, exist_ok=True)
        self._tokenizers: Dict[str, Any] = {}
        self.commit = git_commit()
        self.environment = environment_versions()

    def tokenizer(self, model_name: str):
        tokenizer_name = TOKENIZER_NAMES[model_name]
        if tokenizer_name not in self._tokenizers:
            self._tokenizers[tokenizer_name] = AutoTokenizer.from_pretrained(tokenizer_name)
        return self._tokenizers[tokenizer_name]

    def run_seed(
        self,
        profile: PreparedProfile,
        model_name: str,
        seed: int,
        run_dir: Path,
    ) -> Dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        if model_name not in NEURAL_MODEL_NAMES:
            return _run_ml_seed(profile, model_name, seed, run_dir, self.options)

        device = self.options.device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = self.tokenizer(model_name)
        candidates = [(self.options.batch_size, 1)]
        if model_name in TRANSFORMER_MODEL_NAMES and device.startswith("cuda"):
            candidates = [(16, 1), (8, 2), (4, 4)]
            if self.options.batch_size not in (16, 8, 4):
                candidates.insert(0, (self.options.batch_size, max(1, 16 // self.options.batch_size)))

        last_error: Optional[BaseException] = None
        for batch_size, accumulation in candidates:
            try:
                return _train_neural_attempt(
                    profile,
                    model_name,
                    seed,
                    run_dir,
                    self.options,
                    batch_size,
                    accumulation,
                    tokenizer,
                    device,
                )
            except BaseException as error:
                if not _is_cuda_oom(error):
                    raise
                last_error = error
                print(
                    f"CUDA OOM for {profile.profile_id}/{model_name}/seed={seed} "
                    f"with batch={batch_size}; retrying smaller physical batch."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        raise RuntimeError("All CUDA OOM fallback configurations failed") from last_error

    def run_matrix(
        self,
        profiles: Iterable[str],
        models: Iterable[str],
        seeds: Iterable[int],
        resume: bool = False,
        fail_fast: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        profile_ids = list(profiles)
        model_names = list(models)
        seed_values = [int(seed) for seed in seeds]
        unknown_models = sorted(set(model_names) - set(MODEL_NAMES))
        if unknown_models:
            raise ValueError(f"Unknown model(s): {', '.join(unknown_models)}")
        matrix = [
            {"profile": profile_id, "model": model_name, "seed": seed}
            for profile_id in profile_ids
            for model_name in model_names
            for seed in seed_values
        ]
        if dry_run:
            return {"n_runs": len(matrix), "runs": matrix}

        summary: Dict[str, Any] = {"completed": [], "failed": []}
        for profile_id in profile_ids:
            original_profile = load_prepared_profile(
                profile_id,
                cache_root=self.options.cache_root,
                config_path=self.options.config_path,
            )
            profile = subset_profile(original_profile, self.options)
            profile_dir = self.options.artifact_root / profile_id
            profile_dir.mkdir(parents=True, exist_ok=True)
            write_json(profile_dir / "dataset_audit.json", original_profile.audit)
            model_results: Dict[str, Dict[str, Any]] = {}

            for model_name in model_names:
                model_dir = profile_dir / model_name
                for seed in seed_values:
                    run_dir = model_dir / "runs" / f"seed_{seed}"
                    result_path = run_dir / "results.json"
                    if resume and result_path.exists():
                        existing = json.loads(result_path.read_text(encoding="utf-8"))
                        expected_config = _base_run_config(
                            profile, model_name, seed, self.options
                        )
                        if existing.get("status") == "completed" and _resume_compatible(
                            existing, expected_config
                        ):
                            print(f"Skipping completed {profile_id}/{model_name}/seed={seed}")
                            continue
                        if existing.get("status") == "completed":
                            print(
                                f"Re-running incompatible prior result for "
                                f"{profile_id}/{model_name}/seed={seed}"
                            )
                    write_json(
                        run_dir / "status.json",
                        {
                            "status": "running",
                            "profile_id": profile_id,
                            "model": model_name,
                            "seed": seed,
                            "started_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    try:
                        result = self.run_seed(profile, model_name, seed, run_dir)
                        write_json(result_path, result)
                        write_json(run_dir / "config.json", result["config"])
                        write_json(run_dir / "status.json", {"status": "completed"})
                        summary["completed"].append(
                            {"profile": profile_id, "model": model_name, "seed": seed}
                        )
                    except Exception as error:
                        failure = {
                            "status": "failed",
                            "profile_id": profile_id,
                            "model": model_name,
                            "seed": seed,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                        }
                        write_json(result_path, failure)
                        write_json(run_dir / "status.json", failure)
                        summary["failed"].append(failure)
                        if fail_fast:
                            raise

                try:
                    model_results[model_name] = aggregate_model_runs(
                        model_dir=model_dir,
                        profile_id=profile_id,
                        task=profile.task,
                        model_name=model_name,
                        aspects=profile.aspects,
                        requested_seeds=seed_values,
                        dataset_stats=_dataset_stats(profile),
                        git_commit=self.commit,
                        environment=self.environment,
                    )
                except ValueError:
                    if fail_fast:
                        raise
            aggregate_profile(model_results, profile_dir)
        write_json(self.options.artifact_root / "run_summary.json", summary)
        return summary
