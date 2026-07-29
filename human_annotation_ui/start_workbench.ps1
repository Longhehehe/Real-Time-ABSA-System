param(
    [ValidateSet("Blind", "AIReview", "Adjudication")]
    [string]$Mode = "Blind",
    [ValidateSet("A", "B")]
    [string]$Role = "A",
    [string]$PackageRoot = (
        "data\annotations\q1_human_gold_1200_v1_20260728"
    ),
    [string]$Assignment,
    [string]$Suggestions,
    [string]$Adjudication,
    [string]$Guideline,
    [int]$Port = 8765,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
$packagePath = if ([System.IO.Path]::IsPathRooted($PackageRoot)) {
    $PackageRoot
} else {
    Join-Path $repoRoot $PackageRoot
}

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Không tìm thấy Python venv: $pythonPath"
}

$modeArgument = switch ($Mode) {
    "Blind" { "blind" }
    "AIReview" { "ai-review" }
    "Adjudication" { "adjudication" }
}

if (-not $Assignment) {
    if ($Mode -ne "Blind") {
        throw "-Assignment là bắt buộc cho mode $Mode"
    }
    $assignmentName = if ($Role -eq "A") {
        "annotator_a.assignment.json"
    } else {
        "annotator_b.assignment.json"
    }
    $Assignment = Join-Path $packagePath "assignments\$assignmentName"
}
if (-not [System.IO.Path]::IsPathRooted($Assignment)) {
    $Assignment = Join-Path $repoRoot $Assignment
}
if (-not $Guideline) {
    $Guideline = Join-Path $packagePath "ABSA_ANNOTATION_GUIDELINE_V2.md"
} elseif (-not [System.IO.Path]::IsPathRooted($Guideline)) {
    $Guideline = Join-Path $repoRoot $Guideline
}

foreach ($requiredPath in @($Assignment, $Guideline)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Không tìm thấy file bắt buộc: $requiredPath"
    }
}

$arguments = @(
    "-X", "utf8",
    "-m", "human_annotation_ui.serve",
    "--mode", $modeArgument,
    "--assignment", $Assignment,
    "--guideline", $Guideline,
    "--port", "$Port"
)
if ($Mode -eq "AIReview") {
    if (-not $Suggestions) {
        throw "-Suggestions là bắt buộc cho AIReview"
    }
    if (-not [System.IO.Path]::IsPathRooted($Suggestions)) {
        $Suggestions = Join-Path $repoRoot $Suggestions
    }
    $arguments += @("--suggestions", $Suggestions)
}
if ($Mode -eq "Adjudication") {
    if (-not $Adjudication) {
        throw "-Adjudication là bắt buộc cho Adjudication"
    }
    if (-not [System.IO.Path]::IsPathRooted($Adjudication)) {
        $Adjudication = Join-Path $repoRoot $Adjudication
    }
    $arguments += @("--adjudication", $Adjudication)
}
if (-not $NoOpen) {
    $arguments += "--open"
}

Push-Location $repoRoot
try {
    & $pythonPath @arguments
} finally {
    Pop-Location
}
