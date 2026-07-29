# Model-ready ABSA releases

These directories are immutable generated releases. Do not edit their files
in place because each release is closed by `manifest.json` and `SHA256SUMS`.

## Current

`absa_pseudo_v1_2_20260729/`

- Release ID: `absa-model-ready-5e6d8c9306664b405624`
- Status: `DEVELOPMENT_PSEUDO_MODEL_READY_NOT_GOLD`
- Unique records: 28,266
- Train/dev/test: 22,508 / 2,861 / 2,897
- Leakage-group overlap: zero
- Reserved human sample/text overlap: zero
- Legacy and newly crawled source families were split independently, then
  merged, so both source families are represented in all three partitions.

Validate it with:

```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path
python -X utf8 -m absa_system validate-data `
  .\data\model_ready\absa_pseudo_v1_2_20260729
```

This is a pseudo-label engineering corpus. It is not the final
human-adjudicated test benchmark.

## Superseded experiments

- `absa_pseudo_v1_20260729/`: valid and immutable, but its global group split
  left the legacy source family severely underrepresented in dev/test.
- `absa_pseudo_v1_1_20260729/`: improved balancing, but legacy coverage in
  dev/test remained lower than intended.

Both superseded releases remain only for auditability and must not be used as
the primary experiment release.
