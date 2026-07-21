[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Server,

    [ValidateRange(1, 65535)]
    [int]$Port = 22,

    [string]$IdentityFile,

    [string]$DataPath,

    [switch]$FullRun,

    [switch]$InstallDriver,

    [switch]$Cpu,

    [switch]$SkipSmoke,

    [switch]$SkipSystemPackages,

    [switch]$SkipUpload,

    [switch]$KeepArchive,

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$ServerSetupScript = Join-Path $RepositoryRoot "scripts\setup_experiment_server.sh"
$RemoteArchive = "absa-data.zip"
$RemoteSetupScript = "setup_experiment_server.sh"
$ArchivePath = $null

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Format-NativeCommand {
    param(
        [string]$Command,
        [string[]]$Arguments
    )

    $rendered = foreach ($argument in $Arguments) {
        if ($argument -match '[\s"]') {
            '"{0}"' -f ($argument -replace '"', '\"')
        }
        else {
            $argument
        }
    }
    return "$Command $($rendered -join ' ')"
}

function Invoke-NativeChecked {
    param(
        [string]$Command,
        [string[]]$Arguments
    )

    if ($DryRun) {
        Write-Host (Format-NativeCommand -Command $Command -Arguments $Arguments)
        return
    }

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $(Format-NativeCommand -Command $Command -Arguments $Arguments)"
    }
}

function New-SshArguments {
    param([switch]$AllocateTty)

    $arguments = @()
    if ($IdentityFile) {
        $arguments += @("-i", $IdentityFile)
    }
    $arguments += @("-p", $Port.ToString())
    if ($AllocateTty) {
        $arguments += "-tt"
    }
    return $arguments
}

function New-ScpArguments {
    $arguments = @()
    if ($IdentityFile) {
        $arguments += @("-i", $IdentityFile)
    }
    $arguments += @("-P", $Port.ToString())
    return $arguments
}

if ($InstallDriver -and $Cpu) {
    throw "Do not combine -InstallDriver and -Cpu."
}

if (-not (Test-Path -LiteralPath $ServerSetupScript -PathType Leaf)) {
    throw "Missing server setup script: $ServerSetupScript"
}

if ($IdentityFile) {
    $IdentityFile = (Resolve-Path -LiteralPath $IdentityFile).Path
}

foreach ($command in @("ssh", "scp", "tar.exe")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required local command is missing: $command"
    }
}

if (-not $DataPath) {
    $DataPath = Join-Path $RepositoryRoot "absa data"
}

if (-not $SkipUpload) {
    $DataPath = (Resolve-Path -LiteralPath $DataPath).Path
    if (-not (Test-Path -LiteralPath $DataPath -PathType Container)) {
        throw "Data directory does not exist: $DataPath"
    }
    if ((Split-Path -Leaf $DataPath) -ne "absa data") {
        throw "DataPath must point to a directory named exactly 'absa data': $DataPath"
    }
}

$sshArguments = New-SshArguments
$scpArguments = New-ScpArguments

try {
    Write-Step "Checking SSH connectivity to $Server"
    Invoke-NativeChecked -Command "ssh" -Arguments ($sshArguments + @(
        $Server,
        "echo 'SSH connection OK'"
    ))

    Write-Step "Uploading the Ubuntu bootstrap script"
    Invoke-NativeChecked -Command "scp" -Arguments ($scpArguments + @(
        $ServerSetupScript,
        "${Server}:${RemoteSetupScript}"
    ))

    if (-not $SkipUpload) {
        $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $ArchivePath = Join-Path ([System.IO.Path]::GetTempPath()) "absa-data-$timestamp.zip"
        $dataParent = Split-Path -Parent $DataPath
        $dataLeaf = Split-Path -Leaf $DataPath

        Write-Step "Creating ZIP archive from $DataPath"
        Invoke-NativeChecked -Command "tar.exe" -Arguments @(
            "-a", "-c", "-f", $ArchivePath,
            "-C", $dataParent,
            $dataLeaf
        )

        if (-not $DryRun) {
            if (-not (Test-Path -LiteralPath $ArchivePath -PathType Leaf)) {
                throw "Archive was not created: $ArchivePath"
            }
            $localHash = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
            $archiveSize = (Get-Item -LiteralPath $ArchivePath).Length
            Write-Host "Archive: $ArchivePath"
            Write-Host "Bytes:   $archiveSize"
            Write-Host "SHA256:  $localHash"
        }

        Write-Step "Uploading data archive"
        Invoke-NativeChecked -Command "scp" -Arguments ($scpArguments + @(
            $ArchivePath,
            "${Server}:${RemoteArchive}"
        ))

        if (-not $DryRun) {
            Write-Step "Verifying remote SHA-256"
            $hashArguments = $sshArguments + @($Server, "sha256sum ~/$RemoteArchive")
            $remoteHashOutput = & ssh @hashArguments
            if ($LASTEXITCODE -ne 0) {
                throw "Unable to calculate the remote SHA-256."
            }
            $remoteHashText = ($remoteHashOutput | Out-String).Trim()
            if ($remoteHashText -notmatch '^([0-9a-fA-F]{64})\s') {
                throw "Unexpected sha256sum output: $remoteHashText"
            }
            $remoteHash = $Matches[1].ToLowerInvariant()
            Write-Host "Remote SHA256: $remoteHash"
            if ($remoteHash -ne $localHash) {
                throw "Checksum mismatch. The remote setup was not started."
            }
            Write-Host "Checksum OK" -ForegroundColor Green
        }
    }
    else {
        Write-Step "Skipping data upload; the server must already contain ~/$RemoteArchive"
    }

    $remoteOptions = @(
        "--repo-dir", "~/Real-Time-ABSA-System",
        "--data-archive", "~/$RemoteArchive"
    )
    if ($FullRun) {
        $remoteOptions += "--full-run"
    }
    if ($InstallDriver) {
        $remoteOptions += "--install-driver"
    }
    if ($Cpu) {
        $remoteOptions += "--cpu"
    }
    if ($SkipSmoke) {
        $remoteOptions += "--skip-smoke"
    }
    if ($SkipSystemPackages) {
        $remoteOptions += "--skip-system-packages"
    }

    $remoteCommand = "bash ~/$RemoteSetupScript " + ($remoteOptions -join " ")
    Write-Step "Running the Ubuntu setup workflow"
    $remoteSshArguments = (New-SshArguments -AllocateTty) + @($Server, $remoteCommand)
    if ($DryRun) {
        Invoke-NativeChecked -Command "ssh" -Arguments $remoteSshArguments
    }
    else {
        & ssh @remoteSshArguments
        $remoteExitCode = $LASTEXITCODE
        if ($remoteExitCode -eq 20 -and $InstallDriver) {
            Write-Step "NVIDIA driver was installed; reboot is required"
            Write-Host "Reboot the server, wait for SSH to return, then rerun with:"
            Write-Host "  .\scripts\upload_and_setup_server.ps1 -Server `"$Server`" -Port $Port -SkipUpload -FullRun"
            return
        }
        if ($remoteExitCode -ne 0) {
            throw "Ubuntu setup failed with exit code $remoteExitCode."
        }
    }

    Write-Step "Remote workflow started successfully"
    if ($FullRun) {
        Write-Host "The 72-run matrix is running in tmux on the server."
        Write-Host "Reconnect and monitor with:"
        Write-Host "  ssh -p $Port $Server"
        Write-Host "  tmux attach -t absa-experiments"
        Write-Host "  tail -f ~/Real-Time-ABSA-System/logs/experiments_full.log"
    }
    else {
        Write-Host "Setup, data audit and smoke tests are complete."
        Write-Host "Rerun this launcher with -FullRun to start all 72 runs."
    }
}
finally {
    if ($ArchivePath -and -not $KeepArchive -and (Test-Path -LiteralPath $ArchivePath -PathType Leaf)) {
        Write-Step "Removing temporary local archive"
        Remove-Item -LiteralPath $ArchivePath -Force
    }
    elseif ($ArchivePath -and $KeepArchive) {
        Write-Host "Temporary archive kept at: $ArchivePath"
    }
}
