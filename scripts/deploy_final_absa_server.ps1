<#
.SYNOPSIS
Deploys and runs the reviewed final_absa server bootstrap over SSH.

.DESCRIPTION
Uploads scripts/setup_final_absa_server.sh to a unique temporary path on an
Ubuntu server, executes setup, validation, pilot training, or full training,
and removes only that temporary script. Dataset v1.2 is obtained from the
final_absa Git branch; this command does not upload local raw data, secrets,
or checkpoints.

.EXAMPLE
.\scripts\deploy_final_absa_server.ps1 `
  -Server "user@server-ip" `
  -Action setup `
  -Device cuda

.EXAMPLE
.\scripts\deploy_final_absa_server.ps1 `
  -Server "user@server-ip" `
  -Action pilot `
  -Device cuda `
  -Detach `
  -TmuxSession "absa-pilot"
#>


[CmdletBinding()]
param(
    [string]$Server,

    [ValidateRange(1, 65535)]
    [int]$Port = 22,

    [string]$IdentityFile,

    [ValidateSet("setup", "validate", "pilot", "full")]
    [string]$Action = "setup",

    [string]$RepoUrl = "https://github.com/Longhehehe/Real-Time-ABSA-System.git",

    [string]$Branch = "final_absa",

    [string]$RepoDir = "~/Real-Time-ABSA-System",

    [string]$PythonBin = "python3",

    [string]$VenvName = ".venv-model",

    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "auto",

    [string]$TorchIndexUrl,

    [string]$RunName,

    [ValidateRange(1, 1024)]
    [int]$BatchSize,

    [ValidateRange(8, 8192)]
    [int]$MaxLength,

    [ValidateRange(1, 4096)]
    [int]$GradientAccumulationSteps,

    [string]$HfHome,

    [switch]$WithBrowser,

    [switch]$SkipSystemPackages,

    [switch]$SkipInstall,

    [switch]$SkipModelDownload,

    [switch]$NoUpdate,

    [switch]$Offline,

    [switch]$Detach,

    [switch]$AllowCpuFull,

    [string]$TmuxSession = "final-absa",

    [switch]$DryRun,

    [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$SetupScript = Join-Path $RepositoryRoot "scripts\setup_final_absa_server.sh"
$RemoteToken = [guid]::NewGuid().ToString("N")
$RemoteScript = "/tmp/setup_final_absa_server_$RemoteToken.sh"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Quote-Posix {
    param([string]$Value)
    if ($Value.Contains("'")) {
        throw "Remote arguments containing a single quote are not supported."
    }
    return "'$Value'"
}

function Format-NativeCommand {
    param(
        [string]$Command,
        [string[]]$Arguments
    )
    $rendered = foreach ($argument in $Arguments) {
        if ($argument -match '[\s"]') {
            '"' + ($argument -replace '"', '\"') + '"'
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

if ($Help) {
    @"
Usage:
  .\scripts\deploy_final_absa_server.ps1 -Server user@host [options]

Required:
  -Server DESTINATION

Actions:
  -Action setup|validate|pilot|full

Common options:
  -Port N
  -IdentityFile PATH
  -Device auto|cuda|cpu
  -TorchIndexUrl URL
  -SkipSystemPackages
  -SkipInstall
  -NoUpdate
  -Offline
  -Detach
  -TmuxSession NAME
  -DryRun

Examples:
  .\scripts\deploy_final_absa_server.ps1 -Server user@host -Action setup -Device cuda
  .\scripts\deploy_final_absa_server.ps1 -Server user@host -Action pilot -Device cuda -Detach
"@
    exit 0
}
if ([string]::IsNullOrWhiteSpace($Server)) {
    throw "-Server is required. Use -Help to see examples."
}
if (-not (Test-Path -LiteralPath $SetupScript -PathType Leaf)) {
    throw "Missing server setup script: $SetupScript"
}
if ($Server.StartsWith("-") -or $Server -match "\s") {
    throw "Server must be an SSH destination such as user@host, without whitespace."
}
if ($IdentityFile) {
    $IdentityFile = (Resolve-Path -LiteralPath $IdentityFile).Path
}
foreach ($command in @("ssh", "scp")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        throw "Required local command is missing: $command"
    }
}

$sshArguments = @("-p", $Port.ToString())
$scpArguments = @("-P", $Port.ToString())
if ($IdentityFile) {
    $sshArguments = @("-i", $IdentityFile) + $sshArguments
    $scpArguments = @("-i", $IdentityFile) + $scpArguments
}

$remoteArguments = @(
    "--action", $Action,
    "--repo-url", $RepoUrl,
    "--branch", $Branch,
    "--repo-dir", $RepoDir,
    "--python-bin", $PythonBin,
    "--venv-name", $VenvName,
    "--device", $Device,
    "--tmux-session", $TmuxSession
)
if ($TorchIndexUrl) {
    $remoteArguments += @("--torch-index-url", $TorchIndexUrl)
}
if ($RunName) {
    $remoteArguments += @("--run-name", $RunName)
}
if ($PSBoundParameters.ContainsKey("BatchSize")) {
    $remoteArguments += @("--batch-size", $BatchSize.ToString())
}
if ($PSBoundParameters.ContainsKey("MaxLength")) {
    $remoteArguments += @("--max-length", $MaxLength.ToString())
}
if ($PSBoundParameters.ContainsKey("GradientAccumulationSteps")) {
    $remoteArguments += @(
        "--gradient-accumulation-steps",
        $GradientAccumulationSteps.ToString()
    )
}
if ($HfHome) {
    $remoteArguments += @("--hf-home", $HfHome)
}
if ($WithBrowser) {
    $remoteArguments += "--with-browser"
}
if ($SkipSystemPackages) {
    $remoteArguments += "--skip-system-packages"
}
if ($SkipInstall) {
    $remoteArguments += "--skip-install"
}
if ($SkipModelDownload) {
    $remoteArguments += "--skip-model-download"
}
if ($NoUpdate) {
    $remoteArguments += "--no-update"
}
if ($Offline) {
    $remoteArguments += "--offline"
}
if ($Detach) {
    $remoteArguments += "--detach"
}
if ($AllowCpuFull) {
    $remoteArguments += "--allow-cpu-full"
}

$quotedRemoteArguments = $remoteArguments | ForEach-Object { Quote-Posix $_ }
$remoteCommand = @(
    "set -o pipefail"
    "bash $(Quote-Posix $RemoteScript) $($quotedRemoteArguments -join ' ')"
    "status=`$?"
    "rm -f -- $(Quote-Posix $RemoteScript)"
    "exit `$status"
) -join "; "

Write-Step "Checking SSH connectivity to $Server"
Invoke-NativeChecked -Command "ssh" -Arguments (
    $sshArguments + @($Server, "printf 'SSH connection OK\n'")
)

Write-Step "Uploading the reviewed final_absa bootstrap script"
Invoke-NativeChecked -Command "scp" -Arguments (
    $scpArguments + @($SetupScript, "${Server}:${RemoteScript}")
)

Write-Step "Running action '$Action' on the server"
Invoke-NativeChecked -Command "ssh" -Arguments (
    $sshArguments + @("-tt", $Server, $remoteCommand)
)

if ($DryRun) {
    Write-Host ""
    Write-Host "Dry run completed; no SSH connection or remote change was made."
    exit 0
}
if ($Detach -and $Action -in @("pilot", "full")) {
    Write-Host ""
    Write-Host "Remote training was launched in tmux session '$TmuxSession'."
    Write-Host "Attach with:"
    Write-Host "  ssh -p $Port $Server"
    Write-Host "  tmux attach -t $TmuxSession"
}
else {
    Write-Host ""
    Write-Host "Remote action completed successfully."
}
