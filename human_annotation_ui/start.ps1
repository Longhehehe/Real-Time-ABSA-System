param(
    [ValidateSet("A", "B")]
    [string]$Role = "A",
    [int]$Port = 8765,
    [string]$PackageRoot = (
        "data\annotations\q1_human_gold_1200_v1_20260728"
    ),
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $repoRoot ".venv\Scripts\python.exe"
$packageRoot = if ([System.IO.Path]::IsPathRooted($PackageRoot)) {
    $PackageRoot
} else {
    Join-Path $repoRoot $PackageRoot
}
$assignmentName = if ($Role -eq "A") {
    "annotator_a.assignment.json"
} else {
    "annotator_b.assignment.json"
}
$assignmentPath = Join-Path $packageRoot "assignments\$assignmentName"

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Không tìm thấy Python venv: $pythonPath"
}
if (-not (Test-Path -LiteralPath $assignmentPath -PathType Leaf)) {
    throw "Không tìm thấy assignment: $assignmentPath"
}

$arguments = @(
    "-X", "utf8",
    "-m", "human_annotation_ui.serve",
    "--mode", "blind",
    "--assignment", $assignmentPath,
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
