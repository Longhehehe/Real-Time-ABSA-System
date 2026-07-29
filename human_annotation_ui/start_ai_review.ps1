param(
    [int]$Port = 8770,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
$packageRoot = Join-Path $repoRoot (
    "data\annotations\human_reference_ai_preannotation_v1_20260726"
)
$assignmentPath = Join-Path $packageRoot (
    "human_check\ai_review.assignment.json"
)
$suggestionsPath = Join-Path $packageRoot (
    "human_check\ai_suggestions.json"
)
$guidelinePath = Join-Path $packageRoot (
    "provenance\ABSA_ANNOTATION_GUIDELINE_V2.md"
)

foreach ($requiredPath in @(
    $pythonPath,
    $assignmentPath,
    $suggestionsPath,
    $guidelinePath
)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Không tìm thấy file bắt buộc: $requiredPath"
    }
}

$arguments = @(
    "-X", "utf8",
    "-m", "human_annotation_ui.serve",
    "--mode", "ai-review",
    "--assignment", $assignmentPath,
    "--guideline", $guidelinePath,
    "--suggestions", $suggestionsPath,
    "--port", "$Port"
)
if (-not $NoOpen) {
    $arguments += "--open"
}

Push-Location $repoRoot
try {
    & $pythonPath @arguments
} finally {
    Pop-Location
}
