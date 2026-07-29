[CmdletBinding()]
param(
    [ValidateRange(1, 1000000)]
    [int]$TargetReviews = 30000,

    [ValidateRange(1, 1440)]
    [int]$MinCooldownMinutes = 2,

    [ValidateRange(1, 1440)]
    [int]$MaxCooldownMinutes = 3,

    [ValidateRange(0, 1000000)]
    [int]$MaxCycles = 0,

    [ValidateSet("hybrid", "cookie", "dom")]
    [string]$Mode = "hybrid",

    [ValidateRange(0, 1000000)]
    [int]$MaxProductsPerCycle = 0,

    [ValidateRange(1, 100)]
    [int]$HybridDomProductsPerCycle = 1,

    [string]$BrowserProfile = "browser-profile/lazada-hybrid",

    [string]$CookieFile = "src/cookies.txt",

    [switch]$HeadedDom,

    [switch]$AdaptiveCooldown,

    [switch]$ValidateOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$collectorPath = Join-Path $projectRoot ".venv\Scripts\lazada-collect.exe"
$rawRoot = Join-Path $projectRoot "data\raw"
$logRoot = Join-Path $projectRoot "logs"
$logPath = Join-Path $logRoot "crawl-supervisor.log"

if ($MaxCooldownMinutes -lt $MinCooldownMinutes) {
    throw "MaxCooldownMinutes must be greater than or equal to MinCooldownMinutes."
}
if (-not (Test-Path -LiteralPath $collectorPath)) {
    throw "Collector executable not found: $collectorPath"
}

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null

function Write-SupervisorLog {
    param([Parameter(Mandatory = $true)][string]$Message)

    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -LiteralPath $logPath -Value $line -Encoding utf8
}

function New-CrawlSpec {
    param([Parameter(Mandatory = $true)][ValidateSet("cookie", "dom")][string]$Stage)

    if ($Stage -eq "cookie") {
        $arguments = @(
            "crawl-scale",
            "--target-reviews", [string]$TargetReviews,
            "--transport", "requests",
            "--cookie-file", $CookieFile,
            "--review-pages-per-product", "5",
            "--max-cooldowns", "0"
        )
        if ($MaxProductsPerCycle -gt 0) {
            $arguments += @("--max-products", [string]$MaxProductsPerCycle)
        }
        return [pscustomobject]@{
            Stage = "cookie"
            Command = "crawl-scale"
            Arguments = $arguments
        }
    }

    $arguments = @(
        "crawl-dom-scale",
        "--target-reviews", [string]$TargetReviews,
        "--cookie-file", $CookieFile
    )
    if (-not [string]::IsNullOrWhiteSpace($BrowserProfile)) {
        $arguments += @("--profile-dir", $BrowserProfile)
    }
    if ($HeadedDom) {
        $arguments += "--headed"
    }
    if ($MaxProductsPerCycle -gt 0) {
        $arguments += @("--max-products", [string]$MaxProductsPerCycle)
    }
    elseif ($Mode -eq "hybrid") {
        # A bounded DOM burst returns control to the API periodically and
        # restarts Chrome before a degraded session can linger indefinitely.
        $arguments += @(
            "--max-products",
            [string]$HybridDomProductsPerCycle
        )
    }
    return [pscustomobject]@{
        Stage = "dom"
        Command = "crawl-dom-scale"
        Arguments = $arguments
    }
}

function Get-LatestRunManifest {
    param(
        [Parameter(Mandatory = $true)][datetime]$NotBefore,
        [Parameter(Mandatory = $true)][string]$ExpectedCommand
    )

    if (-not (Test-Path -LiteralPath $rawRoot)) {
        return $null
    }
    $candidates = Get-ChildItem `
        -LiteralPath $rawRoot `
        -Recurse `
        -Filter "manifest.json" `
        -File |
        Where-Object { $_.LastWriteTime -ge $NotBefore } |
        Sort-Object LastWriteTime -Descending

    foreach ($file in $candidates) {
        try {
            $manifest = Get-Content `
                -LiteralPath $file.FullName `
                -Raw `
                -Encoding utf8 |
                ConvertFrom-Json
        }
        catch {
            continue
        }
        if ([string]$manifest.command -eq $ExpectedCommand) {
            return [pscustomobject]@{
                File = $file
                Manifest = $manifest
            }
        }
    }
    return $null
}

function Wait-Cooldown {
    param([Parameter(Mandatory = $true)][int]$Minutes)

    if ($MaxCycles -gt 0 -and $cycle -ge $MaxCycles) {
        Write-SupervisorLog (
            "Skipping cooldown because MaxCycles=$MaxCycles was reached."
        )
        return
    }
    Write-SupervisorLog "Cooldown ${Minutes} minute(s). Press Ctrl+C to stop safely."
    $remainingSeconds = $Minutes * 60
    while ($remainingSeconds -gt 0) {
        $remainingMinutes = [math]::Ceiling($remainingSeconds / 60)
        Write-Host -NoNewline (
            "`r[cooldown] next attempt in {0} minute(s)   " -f $remainingMinutes
        )
        $step = [math]::Min(60, $remainingSeconds)
        Start-Sleep -Seconds $step
        $remainingSeconds -= $step
    }
    Write-Host ""
}

function Get-CooldownMinutes {
    param([Parameter(Mandatory = $true)][int]$EmptyCycles)

    $factor = 1.0
    if ($AdaptiveCooldown) {
        $exponent = [math]::Min(
            [math]::Max($EmptyCycles - 1, 0),
            3
        )
        $factor = [math]::Pow(1.5, $exponent)
    }
    $lower = [math]::Min(
        1440,
        [math]::Ceiling($MinCooldownMinutes * $factor)
    )
    $upper = [math]::Min(
        1440,
        [math]::Ceiling($MaxCooldownMinutes * $factor)
    )
    if ($upper -lt $lower) {
        $upper = $lower
    }
    return Get-Random -Minimum $lower -Maximum ($upper + 1)
}

function Get-OtherStage {
    param([Parameter(Mandatory = $true)][string]$Stage)

    if ($Stage -eq "cookie") {
        return "dom"
    }
    return "cookie"
}

Push-Location $projectRoot
try {
    if ($ValidateOnly) {
        $validationStages = if ($Mode -eq "hybrid") {
            @("cookie", "dom")
        }
        else {
            @($Mode)
        }
        Write-SupervisorLog (
            "Validation only: mode=$Mode, target=$TargetReviews, cooldown=" +
            "$MinCooldownMinutes-$MaxCooldownMinutes minutes, " +
            "adaptive=$AdaptiveCooldown."
        )
        foreach ($stage in $validationStages) {
            $spec = New-CrawlSpec -Stage $stage
            Write-SupervisorLog "Validating $stage stage."
            $validationArguments = @($spec.Arguments) + "--dry-run"
            & $collectorPath @validationArguments
            if ($LASTEXITCODE -ne 0) {
                throw "$stage validation failed with exit code $LASTEXITCODE."
            }
        }
        return
    }

    $cycle = 0
    $emptyCycles = 0
    $consecutiveFailures = 0
    $currentStage = if ($Mode -eq "hybrid") { "cookie" } else { $Mode }
    $stageExhausted = @{
        cookie = $false
        dom = $false
    }

    while ($true) {
        if ($MaxCycles -gt 0 -and $cycle -ge $MaxCycles) {
            Write-SupervisorLog "Stopped after MaxCycles=$MaxCycles."
            return
        }

        $cycle += 1
        $spec = New-CrawlSpec -Stage $currentStage
        $startedAt = Get-Date
        Write-SupervisorLog (
            "Starting $currentStage crawl cycle $cycle with " +
            "target=$TargetReviews."
        )

        $crawlArguments = @($spec.Arguments)
        & $collectorPath @crawlArguments
        $collectorExitCode = $LASTEXITCODE

        if ($collectorExitCode -eq 130) {
            Write-SupervisorLog "Crawler was interrupted by the user."
            return
        }

        $runResult = Get-LatestRunManifest `
            -NotBefore $startedAt `
            -ExpectedCommand $spec.Command
        if ($null -eq $runResult) {
            if ($collectorExitCode -eq 0) {
                Write-SupervisorLog (
                    "Target was already satisfied; no new run was created."
                )
                return
            }
            if ($Mode -ne "hybrid") {
                throw "$currentStage crawler exited without a readable manifest."
            }

            $consecutiveFailures += 1
            if ($consecutiveFailures -ge 3) {
                Write-SupervisorLog (
                    "Stopped after three stage launches without a manifest."
                )
                return
            }
            $previousStage = $currentStage
            $currentStage = Get-OtherStage -Stage $currentStage
            Write-SupervisorLog (
                "$previousStage failed before creating a manifest; " +
                "switching to $currentStage."
            )
            if ($previousStage -eq "dom") {
                $emptyCycles += 1
                Wait-Cooldown -Minutes (
                    Get-CooldownMinutes -EmptyCycles $emptyCycles
                )
            }
            continue
        }

        $manifest = $runResult.Manifest
        $manifestFile = $runResult.File
        $status = [string]$manifest.status
        $newReviews = [int]$manifest.counts.reviews_written
        if ($null -ne $manifest.PSObject.Properties["scale"]) {
            $totalReviews = [int]$manifest.scale.reviews_total
        }
        elseif ($null -ne $manifest.PSObject.Properties["dom"]) {
            $totalReviews = [int]$manifest.dom.reviews_total
        }
        else {
            $totalReviews = 0
        }

        if ($newReviews -gt 0) {
            $emptyCycles = 0
            $consecutiveFailures = 0
            $stageExhausted.cookie = $false
            $stageExhausted.dom = $false
        }
        else {
            $emptyCycles += 1
        }

        Write-SupervisorLog (
            "Cycle $cycle ($currentStage) ended: status=$status, " +
            "new=$newReviews, total=$totalReviews/$TargetReviews."
        )

        switch ($status) {
            "completed_target" {
                Write-SupervisorLog "Collection target reached."
                return
            }
            "paused_rate_limit" {
                if ($Mode -eq "hybrid" -and $currentStage -eq "cookie") {
                    $currentStage = "dom"
                    Write-SupervisorLog (
                        "Cookie API was rate-limited; switching immediately " +
                        "to rendered Selenium DOM."
                    )
                    continue
                }
                if ($Mode -eq "hybrid") {
                    $currentStage = "cookie"
                    Write-SupervisorLog (
                        "DOM was rate-limited; cooling down before retrying API."
                    )
                }
                Wait-Cooldown -Minutes (
                    Get-CooldownMinutes -EmptyCycles $emptyCycles
                )
                continue
            }
            "stopped_max_products" {
                if ($Mode -ne "hybrid") {
                    Write-SupervisorLog (
                        "Clean pilot/product-cap stop for $currentStage."
                    )
                    return
                }
                $previousStage = $currentStage
                $currentStage = Get-OtherStage -Stage $currentStage
                Write-SupervisorLog (
                    "$previousStage burst reached its product cap; " +
                    "switching to $currentStage."
                )
                if ($previousStage -eq "dom" -and $newReviews -eq 0) {
                    Wait-Cooldown -Minutes (
                        Get-CooldownMinutes -EmptyCycles $emptyCycles
                    )
                }
                continue
            }
            "paused_challenge" {
                Write-SupervisorLog (
                    "Selenium encountered a browser challenge. The collector " +
                    "stopped without bypassing it. Rerun with -HeadedDom after " +
                    "normal manual verification if permitted."
                )
                return
            }
            "interrupted" {
                Write-SupervisorLog "Crawler checkpointed an interruption."
                return
            }
            "exhausted_catalogue" {
                $stageExhausted[$currentStage] = $true
                if ($Mode -ne "hybrid") {
                    Write-SupervisorLog (
                        "All configured products/search pages were exhausted."
                    )
                    return
                }
                $nextStage = Get-OtherStage -Stage $currentStage
                if ($stageExhausted[$nextStage]) {
                    Write-SupervisorLog (
                        "Both API and DOM catalogues were exhausted."
                    )
                    return
                }
                $previousStage = $currentStage
                $currentStage = $nextStage
                Write-SupervisorLog (
                    "$previousStage catalogue exhausted; switching to " +
                    "$currentStage."
                )
                if ($previousStage -eq "dom") {
                    Wait-Cooldown -Minutes (
                        Get-CooldownMinutes -EmptyCycles $emptyCycles
                    )
                }
                continue
            }
            "failed" {
                $consecutiveFailures += 1
                if ($consecutiveFailures -ge 3) {
                    Write-SupervisorLog (
                        "Stopped after three failed stages. Inspect: " +
                        $manifestFile.FullName
                    )
                    return
                }
                if ($Mode -eq "hybrid") {
                    $previousStage = $currentStage
                    $currentStage = Get-OtherStage -Stage $currentStage
                    Write-SupervisorLog (
                        "$previousStage failed; switching to $currentStage."
                    )
                    if ($previousStage -eq "dom") {
                        Wait-Cooldown -Minutes (
                            Get-CooldownMinutes -EmptyCycles $emptyCycles
                        )
                    }
                    continue
                }
                $failureWait = [math]::Min(30, 5 * $consecutiveFailures)
                Write-SupervisorLog (
                    "Transient failed cycle; retrying after " +
                    "$failureWait minute(s)."
                )
                Wait-Cooldown -Minutes $failureWait
                continue
            }
            default {
                if ($collectorExitCode -eq 0) {
                    Write-SupervisorLog (
                        "Crawler completed with status=$status."
                    )
                    return
                }
                Write-SupervisorLog (
                    "Unhandled status=$status (exit=$collectorExitCode). " +
                    "Inspect: " + $manifestFile.FullName
                )
                return
            }
        }
    }
}
finally {
    Pop-Location
}
