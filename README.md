# Vietnamese Multi-Polarity ABSA Research System

This repository contains a provenance-aware Vietnamese e-commerce review
corpus, its collector, and an evidence-aware multi-polarity ABSA model. The
frozen 2026-07-25 base snapshot is preserved, and a separately versioned
2026-07-28 incremental delta has been processed without rewriting the base
release.
Historical ABSA data is preserved under `legacy/data/`. The former training,
API, dashboard, Kafka, Airflow and experiment source code is available under
`legacy/system/` and in the Git branch `archive/pre-crawler-reset`.

## Active ABSA model

The active implementation is `src/absa_system/`. It uses PhoBERT as the text
encoder, nine learned aspect queries, aspect-specific mention attention, and
three polarity-specific evidence heads. Each aspect is predicted as an
independent set over `negative`, `positive`, and `neutral`, so a genuine mixed
case can retain both positive and negative labels. Neutral is made exclusive
at decision time. No Monte Carlo component is used.

The current model-ready engineering release is:

```text
data/model_ready/absa_pseudo_v1_2_20260729/
release_id: absa-model-ready-5e6d8c9306664b405624
records: 28,266 (22,508 train / 2,861 dev / 2,897 test)
status: DEVELOPMENT_PSEUDO_MODEL_READY_NOT_GOLD
```

The split is isolated by connected leakage groups and excludes the two
human-reference reservation ledgers. It is suitable for model development,
not for the final Q1 benchmark claim because the labels are still
LLM-generated pseudo labels. Use a separately locked human-adjudicated test
set for the final reported result.

Run the implemented workflow from PowerShell:

```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path

python -X utf8 -m absa_system validate-data `
  .\data\model_ready\absa_pseudo_v1_2_20260729

python -X utf8 -m absa_system train `
  --data .\data\model_ready\absa_pseudo_v1_2_20260729 `
  --output .\artifacts\models\absa_phobert_v1_<RUN_DATE> `
  --config .\configs\training_v1.json `
  --device cuda

python -X utf8 -m absa_system validate-run `
  .\artifacts\models\absa_phobert_v1_<RUN_DATE>

python -X utf8 -m absa_system predict `
  --checkpoint .\artifacts\models\absa_phobert_v1_<RUN_DATE>\model.pt `
  --device cuda `
  --local-files-only `
  --text "Máy hút mạnh nhưng đóng gói móp và giao hàng chậm."
```

Training tunes per-aspect/per-polarity thresholds only on dev, selects the
best epoch by end-to-end macro-F1, evaluates test once, and seals every
completed run with a manifest and SHA-256 checksums. The
`absa_arch_smoke_v2_20260729` artifact only proves that the full GPU path
works; its two-sample metrics are deliberately not model-performance results.

## Deploy to an Ubuntu GPU server

The maintained deployment path for this branch is documented in
[`docs/SERVER_SETUP_FINAL_ABSA.md`](docs/SERVER_SETUP_FINAL_ABSA.md). From
Windows PowerShell, setup and validate a remote server with:

```powershell
.\scripts\deploy_final_absa_server.ps1 `
  -Server "user@server-ip" `
  -Action setup `
  -Device cuda
```

Then run the mandatory capacity pilot in a detached `tmux` session:

```powershell
.\scripts\deploy_final_absa_server.ps1 `
  -Server "user@server-ip" `
  -Action pilot `
  -Device cuda `
  -Detach `
  -TmuxSession "absa-pilot"
```

The scripts validate the immutable model-ready release and CUDA environment
before training. They do not install NVIDIA drivers, upload raw/private data,
or commit generated checkpoints.

## Processed 2026-07-28 incremental delta

The 18 closed crawl runs under `data/raw/2026-07-28/` contain 2,615 review
candidates: 990 accepted unique reviews, 1,056 quality rejections and 569
cross-run duplicates. The accepted records were frozen and processed as:

```text
data/releases/lazada_vi_reviews_delta_v1_20260728/
data/releases/lazada_vi_absa_delta_curation_v1_20260728/
data/annotations/absa_ai_delta_v1_20260728/final/
```

Curation rule 2.1.2 produced 613 clean-core records, 375 quarantine records
and two confirmed duplicate exclusions, covering all 990 parent records
exactly once. The 613 clean-core records have zero exact text-hash overlap
with the 200 human-reference records or their 6,646 reserved leakage-group
records. They were processed with the same frozen Codex generation setup as
the previous 8,976-record tranche: `gpt-5.6-terra`, reasoning `medium`, batch
20, eight workers and identical prompt/schema hashes.

The final pseudo-label release contains 554 `LABELED`, 46 `ESCALATE` and 13
`REJECT_NON_REVIEW` records; 275 records are in the human-review queue. Its
status is `AI_PSEUDO_LABEL_PENDING_HUMAN_VERIFICATION`, not human gold. The
independent validator returns `VALID`:

```powershell
.\.venv\Scripts\python -X utf8 `
  .\scripts\validate_ai_annotation_tranche.py `
  --package .\data\annotations\absa_ai_delta_v1_20260728 `
  --release .\data\annotations\absa_ai_delta_v1_20260728\final `
  --expected-records 613
```

The machine-readable reconciliation report is
`docs/audits/DELTA_V1_PROCESSING_REPORT_20260728.json`.

## Current step: verify the 13,976-review safe-frame pseudo-label corpus

The complete leakage-controlled safe frame is now labeled in two immutable,
non-overlapping releases:

```text
data/annotations/absa_ai_tranche_5000_v1_20260727/final/
data/annotations/absa_ai_remainder_8976_v1_20260728/final/
```

The releases contain 5,000 and 8,976 records respectively, with zero
`sample_id` or review-text-hash overlap; their union is exactly the 13,976-row
safe frame left after reserving all human-reference leakage groups. Aggregate
status is 12,827 `LABELED`, 936 `ESCALATE`, and 213
`REJECT_NON_REVIEW`. Both were generated with the same frozen execution
configuration: `gpt-5.6-terra`, Codex backend, reasoning `medium`, prompt
`absa-ai-compact-v1.0.0`, and the same prompt/schema hashes.

Each release uses `ai_pseudo_labels.jsonl` as canonical and provides a CSV
compatibility projection, decision ledger, mandatory/stratified human-review
queue, sealed run provenance and checksums. The two queues contain 1,978 and
3,592 disjoint records. Every label remains
`AI_PSEUDO_LABEL_PENDING_HUMAN_VERIFICATION`; neither release is human gold or
an independently annotated benchmark.

Replay the independent release validator at any time:

```powershell
.\.venv\Scripts\python -X utf8 `
  .\scripts\validate_ai_annotation_tranche.py `
  --package .\data\annotations\absa_ai_tranche_5000_v1_20260727 `
  --release .\data\annotations\absa_ai_tranche_5000_v1_20260727\final

.\.venv\Scripts\python -X utf8 `
  .\scripts\validate_ai_annotation_tranche.py `
  --package .\data\annotations\absa_ai_remainder_8976_v1_20260728 `
  --release .\data\annotations\absa_ai_remainder_8976_v1_20260728\final
```

The final manifest SHA-256 values are
`bc01e6727b73e5962aa14affea4f3628917554572087a84da951ad69e03f5da1`
and
`c3c5f6e77553cd148258b5808ce756881ed6e03c1351277be1d92329666fbe47`.
The next action is human verification and expert adjudication of the queue,
followed by new versioned human-verified releases; do not edit either
pseudo-label release in place.

## Legacy old-dataset re-annotation

The ten historical XLSX files contain 10,105 non-empty source rows, not an
exact 11,000. After NFKC/casefold/whitespace deduplication there are 9,772
unique review texts and 333 duplicate aliases. A new label-blind source and
pseudo-label package are available at:

```text
data/releases/legacy_old_reviews_label_blind_v1_20260728/
data/annotations/absa_legacy_old_relabel_9772_v1_20260728/
```

All nine historical aspect-label values were excluded from clean-core,
selection, LLM targets, prompt calibration, and validation. The original
workbooks remain immutable provenance; they were not overwritten. The new
canonical release contains 9,772 unique pseudo-label records: 8,553
`LABELED`, 950 `ESCALATE`, and 269 `REJECT_NON_REVIEW`. Its review queue has
3,570 records. It was generated with exactly the same frozen execution config
as the 8,976-record tranche: `gpt-5.6-terra`, Codex backend, reasoning
`medium`, batch 20, eight workers, prompt hash
`4184cc3dc33980f5d985f40628b319f9e69eb4f183dd363d1832f1101dbdb08c`,
and schema hash
`0205bac4e8a379b19972da98d1df2869c9b20d20c2c22087d5f84b4ccfd88e9e`.

For compatibility with the old code, all 10,105 source rows and ten new
10-column workbooks are projected under:

```text
data/annotations/absa_legacy_old_relabel_9772_v1_20260728/legacy_projection/
```

Duplicate aliases inherit their canonical review annotation. The projection
does not reuse old labels and must not be reported as 10,105 independently
annotated reviews. JSONL is canonical because XLSX omits evidence,
uncertainty, status, and generation provenance. Every label remains pending
human verification; this release is silver/pseudo data, not human gold or an
independent benchmark.

Replay both the standard release validator and the legacy row/XLSX validator:

```powershell
.\.venv\Scripts\python -X utf8 `
  .\scripts\validate_ai_annotation_tranche.py `
  --package .\data\annotations\absa_legacy_old_relabel_9772_v1_20260728 `
  --release .\data\annotations\absa_legacy_old_relabel_9772_v1_20260728\final

# System Python is used here because the legacy XLSX audit environment owns
# openpyxl 3.1.5.
python -X utf8 .\scripts\validate_legacy_old_reannotation.py
```

The canonical and projection manifest SHA-256 values are
`350be6f16cebdb588c779cc4c1befdc8c42db0c29fd69f55394cd162894a94df`
and
`4483cd5e7e41da756f6f824ebf597d215a2d85b5e4be24f804828f31f4a37dc5`.

## Human-check source: 200 AI pre-annotations

The separate AI-assisted review package is ready at:

```text
data/annotations/human_reference_ai_preannotation_v1_20260726/
```

It contains 200 guideline-V2 suggestions with exact evidence, a calibration
comparison against the first 10 completed human records, cross-audit ledgers
and checksums. Start the dedicated human-check UI with:

```powershell
.\human_annotation_ui\start_ai_review.ps1
```

This opens a distinct assignment/session. Read and confirm or correct every
record before exporting FINAL. Until then the package is
`AI_PREANNOTATION_PENDING_HUMAN_VERIFICATION`, not human gold.

## Original double-blind reference input

The locked double-blind input is:

```text
data/annotations/human_reference_v1_20260726/
```

It contains a 150-review representative panel and a 50-review challenge panel.
This is **not human gold yet**. Annotator A can start the local UI with:

```powershell
.\human_annotation_ui\start.ps1 -Role A
```

Annotator B must use the B assignment independently, preferably on another
machine or browser profile:

```powershell
.\human_annotation_ui\start.ps1 -Role B -Port 8766
```

The workbench is entirely under `human_annotation_ui/`, outside `src/`.
See [human_annotation_ui/README.md](human_annotation_ui/README.md) for the
label/evidence workflow, backup procedure and strict FINAL validation command.
For a genuine double-blind experiment, do not expose A's file or AI
suggestions to B and do not use these 200 reviews as practice examples. The
AI-assisted workflow above is a different protocol and must be reported as
such.

## Frozen 2026-07-25 corpus

The frozen release is:

```text
data/releases/lazada_vi_reviews_v1_20260725/
```

It contains 31,928 canonical `substantive_vi_v2` reviews. Of these, 31,797
are in 32 primary annotation batches and 131 are in a separate adjudication
queue for near-duplicate or repetition review. Nothing was automatically
deleted from the canonical JSONL.

Re-run the full offline validation at any time:

```powershell
.\.venv\Scripts\python -X utf8 .\scripts\validate_corpus_release.py
```

Annotate files under `annotation/primary_batches/`. They use the exact legacy
10-column schema and all nine label cells intentionally start blank. Use only
`-1`, `0`, `1`, `2`, or `1, -1`. Record keep/drop decisions for the 131
quarantined rows in `annotation/review_decisions.csv`; do not put those
decisions into an ABSA label column.

When opening a CSV in Excel, import it through **Data > From Text/CSV** and set
`reviewContent` to the Text data type. This preserves reviews beginning with
characters such as `-` or `+` instead of letting Excel interpret them as
formulas.

The raw crawl tree remains unchanged and is frozen by 1,573 SHA-256 entries
under `provenance/SOURCE_SHA256SUMS.txt`. Do not resume the crawler merely to
increase this release; a future collection should be published as a new
snapshot and release ID.

## Safety and collection policy

- The collector stops when Lazada returns a CAPTCHA or HTML challenge.
- It does not contain CAPTCHA-bypass logic.
- The recommended collector uses a local Lazada cookie export with the review
  JSON endpoint. Rendered DOM collection remains available as a fallback.
- Buyer names, avatars and cookies are never written to the dataset.
- Rating filter `0` does not impose fixed per-star quotas. The resulting
  rating distribution is natural within the accepted substantive reviews of
  the sampled products; it is not claimed to represent all Lazada reviews.
- Review collection must follow the platform terms, privacy requirements and
  the research protocol approved for the project.

## Repository layout

```text
src/lazada_collector/     collector package
configs/collector.toml    rate, transport and output settings
configs/collection_plan.toml
                           automatic segments, quotas and quality thresholds
data/raw/                 ignored incremental crawl runs
data/manifests/           reserved for published aggregate manifests
data/releases/            frozen canonical and annotation-ready releases
docs/                     annotation and collection documentation
human_annotation_ui/      local double-blind human annotation workbench
legacy/data/              historical datasets, not used by the new collector
legacy/system/            read-only snapshot of the former system source code
tests/                    offline contract tests
```

Each crawl run writes:

```text
data/raw/YYYY-MM-DD/<crawl-id>/
  products.jsonl
  reviews.jsonl
  rejections.jsonl
  manifest.json
```

The JSONL records contain source/product/review IDs, collection time, sampling
frame, transport, page number and response hash. They intentionally exclude
buyer identity. `rejections.jsonl` contains only review ID, text hash, quality
measurements and rejection reasons; it never contains rejected review text.

## Setup on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[browser,dev]"
```

Chrome is required for the Selenium fallback. Selenium Manager resolves the
matching driver automatically.

## Recommended: hybrid cookie API + Selenium DOM

Export a Netscape-format cookie file to `src/cookies.txt`. Although the loader
uses only active `lazada.vn` cookies, the safest export contains Lazada cookies
only. The file and `browser-profile/` are ignored by Git. Cookie values are
never written to dataset records, logs or manifests; Chrome may retain them
inside the ignored local profile.

Validate both stages without contacting Lazada or opening Chrome:

```powershell
.\run_crawl_automatic.cmd -Mode hybrid -TargetReviews 30000 -ValidateOnly
```

Start or resume the unattended collection:

```powershell
.\run_crawl_automatic.cmd -TargetReviews 30000
```

`hybrid` is the default. The supervisor starts with the cookie-authenticated
API. On `paused_rate_limit` it immediately starts a rendered Selenium DOM
burst; after that bounded burst it retries the API. If DOM itself is
rate-limited, it waits 2-3 minutes before retrying API. A browser challenge
stops safely instead of attempting a bypass.

Both stages automatically discover products and use the same
`substantive_vi_v2` filter, cross-run review-ID/text deduplication, provenance,
privacy controls and target. API sampling uses `filter=0`, scans at most five
50-review pages per product and retains at most 50 accepted reviews per
product. DOM sampling scrolls to rendered review cards and follows the normal
pagination controls. The filtered Lazada cookies are also imported into
Chrome in memory; a dedicated ignored profile preserves the normal browser
session between DOM bursts.

The target is the total number of accepted current-policy records under
`data/raw/`. The historical old/augmented dataset under `legacy/data/` remains
separate and is not counted. No URL or query argument is required.

Use `-AdaptiveCooldown` if repeated short retries make no progress. Use
`Ctrl+C` to stop safely. A bounded two-stage pilot is:

```powershell
.\run_crawl_automatic.cmd `
  -Mode hybrid `
  -TargetReviews 400 `
  -MaxCycles 2 `
  -MaxProductsPerCycle 1
```

Rate limiting can still occur with valid cookies. The supervisor records
each transition and starts another deduplicated run; it does not bypass
CAPTCHA or platform controls. Locking the Windows screen does not stop the
default headless hybrid mode, but sleep, hibernation, shutdown or loss of
network does.

Run the compatibility entry point once without the supervisor:

```powershell
.\.venv\Scripts\python .\src\crawl.py --target-reviews 30000
```

It stops safely on the first rate-limit response. Use the supervisor when
automatic retrying is desired.

## Run only one transport

Force only rendered Chrome:

```powershell
.\run_crawl_automatic.cmd -Mode dom -TargetReviews 30000
```

Force only the cookie API:

```powershell
.\run_crawl_automatic.cmd -Mode cookie -TargetReviews 30000
```

Show the Selenium browser during hybrid collection:

```powershell
.\run_crawl_automatic.cmd `
  -Mode hybrid `
  -TargetReviews 30000 `
  -HeadedDom
```

## Offline verification

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```

## Search smoke test

```powershell
.\.venv\Scripts\lazada-collect search "dau goi" --limit 5
```

## Fully automatic collection

Validate the category sampling and quality plan without network access:

```powershell
.\.venv\Scripts\lazada-collect crawl-auto --dry-run
```

Start the complete automatic run. No product URL or search query is required:

```powershell
.\.venv\Scripts\lazada-collect crawl-auto
```

The default plan rotates through eight product segments and 24 reproducible
queries. It selects up to 24 non-sponsored products with at least 50 available
reviews and retains every accepted review found on the first 50-review page,
up to 50 per product. Only one review page is requested per product, which
reduces challenge cascades and increases product diversity.

If live catalogue search is temporarily challenged, discovery stops after two
consecutive failures and falls back to products saved by prior runs. Products
already attempted are skipped by default, and accepted review IDs/text hashes
are deduplicated across all runs under `data/raw/`.

The default substantive-review policy requires at least 80 characters, 15
words, 40% unique-word ratio, eight meaningful words and a combined quality
score of 0.55. Edit `configs/collection_plan.toml` to change the sampling frame
or thresholds.

Run a very small automatic pilot:

```powershell
.\.venv\Scripts\lazada-collect crawl-auto `
  --max-products 1 `
  --reviews-per-product 3
```

## Cookie API scale-collection details

`crawl-scale` is the underlying command used by `src/crawl.py` and the
supervisor:

```powershell
.\.venv\Scripts\lazada-collect crawl-scale --target-reviews 30000
```

For a 50,000-review corpus:

```powershell
.\.venv\Scripts\lazada-collect crawl-scale --target-reviews 50000
```

The target is the total number of unique, accepted reviews already stored
under `data/raw/` with the current `substantive_vi_v2` policy plus those
written by the current run. Older/unfiltered records and historical datasets
under `legacy/data/` are not counted. Each product contributes at most 50
accepted reviews from up to five natural-rating API pages (normally up to 250
candidates). The number of products required therefore depends on how many
reviews pass the substantive-review policy.

Before starting, inspect the local capacity estimate without contacting
Lazada:

```powershell
.\.venv\Scripts\lazada-collect crawl-scale `
  --target-reviews 30000 `
  --dry-run
```

The command searches all configured categories page by page, excludes
sponsored and low-review-count products, retains at most 50 substantive
Vietnamese reviews per product, rejects suspicious text encoding and
foreign-script-dominant content, and checkpoints after every product. HTTP
JSON is decoded directly from UTF-8 bytes even when the server declares the
wrong charset. Repeated transport failures trigger bounded exponential
cooldowns instead of rapid retries.

When the public review endpoint returns a challenge, the default scale job
waits 5, 10 and then 15 minutes for consecutive failures. The cooldown counter
resets after a successful product, allowing one long-running command to
continue through temporary limits. It exits with `paused_rate_limit` after
three consecutive blocked retries. It does not send the same blocked API URL
through Selenium because that merely returns challenge HTML and creates
misleading `SCHEMA_CHANGED` errors.

Keep the terminal and computer running for a scale job. Pressing `Ctrl+C`
saves an `interrupted` manifest. If the final status is `paused_rate_limit`,
`interrupted`, or `exhausted_catalogue`, run the exact same command again:
cross-run deduplication and the total target make it continue from the
remaining amount. The collector does not bypass CAPTCHA or other platform
controls, so no program can guarantee that one uninterrupted invocation will
reach 30k-50k.

Use a one-product end-to-end pilot before the full run:

```powershell
.\.venv\Scripts\lazada-collect crawl-scale `
  --target-reviews 30000 `
  --max-products 1 `
  --max-cooldowns 0
```

## Crawl one product

`auto` first tries the lightweight requests transport. Review API challenges
are surfaced as `BLOCKED`; Selenium fallback remains opt-in because challenge
HTML is not review JSON:

```powershell
.\.venv\Scripts\lazada-collect crawl-product `
  "https://www.lazada.vn/products/example-i123456.html" `
  --max-reviews 20 `
  --transport auto
```

Use `--headed` during the first browser run to inspect login, CAPTCHA or page
failures:

```powershell
.\.venv\Scripts\lazada-collect crawl-product `
  "https://www.lazada.vn/products/example-i123456.html" `
  --max-reviews 20 `
  --transport selenium `
  --headed
```

## Pilot collection by query

```powershell
.\.venv\Scripts\lazada-collect crawl-query "dien thoai" `
  --products 5 `
  --reviews-per-product 50 `
  --transport auto
```

Start with a small pilot. Audit its manifest and review schema before expanding
to many categories or products. Augmentation and annotation do not belong in
the raw collection stage.

## Continuing after a shortfall

`completed_with_shortfall` means every accepted review was saved correctly, but
one or more products did not reach the requested quota. This is expected when
the platform temporarily challenges catalogue/review requests or when a
product has too few substantive reviews.

Wait before the next run and execute the same command again:

```powershell
.\.venv\Scripts\lazada-collect crawl-auto
```

The next run reuses the cached product catalogue, skips products already
attempted when possible, and deduplicates accepted review IDs/text hashes
against every earlier run. Do not merge JSONL files manually before the
dataset audit stage.
