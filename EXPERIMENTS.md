# Reproducible experiments on four external-data profiles

This branch adds an official train/dev/test benchmark pipeline for six model families:

- Logistic Regression
- Naive Bayes
- BiLSTM
- CNN-BiLSTM
- PhoBERT
- XLM-RoBERTa

The four profiles are `uit_visd4sa`, `vlsp2018_hotel`, `vlsp2018_restaurant`, and `vlsp2016`. The first three are ACSA profiles with their native aspect taxonomies. VLSP 2016 is a separate three-class global-sentiment task. No dataset aspect is semantically mapped to the original Lazada taxonomy.

## 1. Git and data policy

All implementation work belongs to the `experiments` branch. Raw third-party data, processed caches, checkpoints, predictions, and logs are deliberately excluded from Git. Only code, configuration, documentation, and published JSON/CSV/PNG reports are versioned.

Expected raw layout:

```text
absa data/
  UIT-ViSD4SA-main/UIT-ViSD4SA-main/data/
    train.jsonl
    dev.jsonl
    test.jsonl
  absa-vlsp-2018-main/absa-vlsp-2018-main/datasets/
    vlsp2018_hotel/*.txt
    vlsp2018_restaurant/*.txt
  vlsp2016/
    SA-2016.train
    SA-2016.dev
    SA-2016.test
    SA-2016.dev_test
```

Dataset sources and citations are recorded in `configs/datasets.json`. The preparation command records the exact file sizes and SHA-256 values in `.experiment_cache/processed/source_manifest.json`.

## 2. Server preflight

Activate the dedicated Conda environment after every SSH login:

```bash
cd ~/Real-Time-ABSA-System
source ~/anaconda3/etc/profile.d/conda.sh
conda activate absa
unset PYTHONPATH
hash -r

which python
python --version
python -m pip --version
```

Install and verify dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_training.txt
python -m pip check

nvidia-smi
python - <<'PY'
import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY

df -h .
```

Reserve at least 40 GB. The runner executes sequentially on one GPU and can resume completed seeds.

## 3. Prepare and audit all data

```bash
python scripts/prepare_experiment_data.py --profiles all
```

This command:

- parses each native format;
- converts UIT spans to review-level multi-polarity ACSA labels;
- normalizes only Unicode/whitespace representation;
- retains native aspects;
- reports malformed span boundaries;
- removes from train every exact text present in dev or test;
- leaves dev/test and within-split duplicates untouched;
- writes normalized JSONL under `.experiment_cache/processed/`.

Expected clean split sizes for the current source files:

| Profile | Train | Dev | Test | Aspects |
| --- | ---: | ---: | ---: | ---: |
| `uit_visd4sa` | 7,782 | 1,112 | 2,225 | 10 |
| `vlsp2018_hotel` | 2,999 | 2,000 | 600 | 34 |
| `vlsp2018_restaurant` | 2,923 | 1,290 | 500 | 12 |
| `vlsp2016` | 4,983 | 100 | 1,050 | N/A |

`SA-2016.dev_test` is audited but is not used for model selection or final scoring.

## 4. Validate the matrix and run smoke tests

Confirm that the default command expands to 72 runs:

```bash
python scripts/run_experiments.py --dry-run
```

Run fast ML smoke tests across all profiles:

```bash
python scripts/run_experiments.py \
  --profiles all \
  --models logistic_regression \
  --seeds 42 \
  --smoke \
  --fail-fast
```

Run the required one-epoch PhoBERT smoke test on small subsets:

```bash
python scripts/run_experiments.py \
  --profiles all \
  --models phobert \
  --seeds 42 \
  --smoke \
  --fail-fast
```

Smoke artifacts are not publishable as final results. The runner compares dataset sizes and hyperparameters before resuming, so a full run automatically replaces an incompatible smoke result; a separate `--artifact-root` is still useful to keep the server workspace tidy.

## 5. Run the full official matrix

```bash
python scripts/run_experiments.py \
  --profiles uit_visd4sa vlsp2018_hotel vlsp2018_restaurant vlsp2016 \
  --models logistic_regression naive_bayes bilstm cnn_bilstm phobert xlm_roberta \
  --seeds 42 52 62 \
  --resume
```

The fixed protocol is:

- fit on clean train only;
- early stopping and threshold selection on dev only;
- evaluate test only after the best dev epoch is selected;
- select the representative seed by dev primary metric;
- report mean and standard deviation over seeds 42, 52, and 62.

Transformer OOM fallback is automatic: physical batch 16, then 8 with accumulation 2, then 4 with accumulation 4. Effective batch size remains 16 and the actual values are written to `config.json`.

Use `tmux` on the server:

```bash
tmux new -s absa-experiments
```

Detach with `Ctrl+B`, then `D`; reconnect with `tmux attach -t absa-experiments`.

## 6. Artifact contract

Heavy local/server artifacts:

```text
artifacts/experiments/<profile>/<model>/
  config.json
  results.json
  metrics_comparison.png
  seed_scores.png
  training_loss_by_epoch.png   # neural models
  confusion_matrix.png         # VLSP 2016
  thresholds.json              # ACSA
  best_model.pt|pkl            # ignored by Git
  runs/seed_42|52|62/
```

`results.json` uses schema version 2. It contains `avg_metrics` plus `*_std`, complete `seed_results`, native aspects, dataset statistics, environment versions, source checksums, and the Git commit. `seed_results` intentionally replaces the historically inaccurate `fold_results` field.

Primary metrics:

- ACSA: end-to-end `Aspect#Polarity` micro-F1.
- VLSP 2016: sentiment macro-F1.

Historical mention/sentiment/combined metrics remain available for comparison with `models/` and `models_agm/`, but combined score is not the external benchmark's primary metric.

## 7. Validate and publish lightweight reports

Do not publish until all 72 runs are complete:

```bash
python scripts/validate_experiment_artifacts.py
python scripts/publish_experiment_results.py
```

Published files appear under:

```text
reports/experiments/
  uit_visd4sa/
  vlsp2018_hotel/
  vlsp2018_restaurant/
  vlsp2016/
  summary_acsa.csv
  summary_sentiment.csv
  run_manifest.json
```

The publisher copies only root-level JSON/CSV/PNG files for each model. It never copies checkpoints, per-review predictions, logs, cache files, or raw data.

## 8. Automated checks and Git publication

```bash
python -m unittest discover -s tests -p "test_experiments.py" -v
python -m compileall experiments scripts methods app/absa_predictor.py
git status --short
```

Stage explicit paths only. Never stage `absa data/`, `.experiment_cache/`, `artifacts/experiments/`, `.pt`, or `.pkl` files.

```bash
git add .gitignore EXPERIMENTS.md configs/datasets.json experiments methods \
  app/absa_predictor.py scripts/prepare_experiment_data.py \
  scripts/run_experiments.py scripts/validate_experiment_artifacts.py \
  scripts/publish_experiment_results.py tests/test_experiments.py

git commit -m "Add reproducible multi-profile experiment pipeline"
git push -u origin experiments
```

After full server experiments, explicitly add only `reports/experiments/`, commit, and push again.
