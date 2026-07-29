#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

REPO_URL="https://github.com/Longhehehe/Real-Time-ABSA-System.git"
BRANCH="final_absa"
REPO_DIR="$HOME/Real-Time-ABSA-System"
PYTHON_BIN="python3"
VENV_NAME=".venv-model"
ACTION="setup"
DEVICE="auto"
TORCH_INDEX_URL=""
DATA_RELEASE="data/model_ready/absa_pseudo_v1_2_20260729"
TRAINING_CONFIG="configs/training_v1.json"
RUN_NAME=""
HF_HOME_DIR="$HOME/.cache/huggingface"
SKIP_SYSTEM_PACKAGES=false
SKIP_INSTALL=false
SKIP_MODEL_DOWNLOAD=false
WITH_BROWSER=false
NO_UPDATE=false
OFFLINE=false
DETACH=false
ALLOW_CPU_FULL=false
TMUX_SESSION="final-absa"
BATCH_SIZE=""
MAX_LENGTH=""
GRADIENT_ACCUMULATION=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_final_absa_server.sh [options]

Actions:
  --action setup       Clone/update, install, validate data and cache PhoBERT.
  --action validate    Validate environment, GPU, data and cached/downloaded model.
  --action pilot       Train 1,000/200/200 records for one epoch.
  --action full        Train the full frozen v1 configuration.

Repository and environment:
  --repo-url URL       Git repository URL.
  --branch NAME        Branch to clone/update (default: final_absa).
  --repo-dir PATH      Server checkout path.
  --python-bin PATH    Python >=3.11 executable (default: python3).
  --venv-name NAME     Virtual environment directory name.
  --torch-index-url URL
                       Optional official PyTorch wheel index selected for the
                       server GPU/driver. If omitted, pip uses its normal index.
  --with-browser       Install collector Selenium dependencies too.
  --skip-system-packages
                       Do not install apt packages.
  --skip-install       Reuse an existing virtual environment without pip install.
  --no-update          Do not fetch/pull an existing checkout.

Model/data:
  --device auto|cuda|cpu
  --data-release PATH  Model-ready release relative to the repository.
  --training-config PATH
  --hf-home PATH       Hugging Face cache directory.
  --skip-model-download
                       Do not prefetch/verify PhoBERT during setup.
  --offline            Require all Hugging Face files to exist locally.

Training:
  --run-name NAME      Artifact directory name under artifacts/models.
  --batch-size N       Runtime override; use 1 first if VRAM is limited.
  --max-length N       Runtime override; changing it changes the experiment.
  --gradient-accumulation-steps N
  --detach             Run pilot/full in a detached tmux session.
  --tmux-session NAME  tmux session name (default: final-absa).
  --allow-cpu-full     Explicitly permit a very slow full CPU run.

Examples:
  bash scripts/setup_final_absa_server.sh --action setup
  bash scripts/setup_final_absa_server.sh --action pilot --device cuda --detach
  bash scripts/setup_final_absa_server.sh --action full --device cuda --detach
EOF
}

log() {
  printf '\n[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

warn() {
  printf '\nWARNING: %s\n' "$*" >&2
}

die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

require_value() {
  [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

while (($#)); do
  case "$1" in
    --action)
      require_value "$@"
      ACTION="$2"
      shift 2
      ;;
    --repo-url)
      require_value "$@"
      REPO_URL="$2"
      shift 2
      ;;
    --branch)
      require_value "$@"
      BRANCH="$2"
      shift 2
      ;;
    --repo-dir)
      require_value "$@"
      REPO_DIR="$2"
      shift 2
      ;;
    --python-bin)
      require_value "$@"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv-name)
      require_value "$@"
      VENV_NAME="$2"
      shift 2
      ;;
    --torch-index-url)
      require_value "$@"
      TORCH_INDEX_URL="$2"
      shift 2
      ;;
    --with-browser)
      WITH_BROWSER=true
      shift
      ;;
    --skip-system-packages)
      SKIP_SYSTEM_PACKAGES=true
      shift
      ;;
    --skip-install)
      SKIP_INSTALL=true
      shift
      ;;
    --no-update)
      NO_UPDATE=true
      shift
      ;;
    --device)
      require_value "$@"
      DEVICE="$2"
      shift 2
      ;;
    --data-release)
      require_value "$@"
      DATA_RELEASE="$2"
      shift 2
      ;;
    --training-config)
      require_value "$@"
      TRAINING_CONFIG="$2"
      shift 2
      ;;
    --hf-home)
      require_value "$@"
      HF_HOME_DIR="$2"
      shift 2
      ;;
    --skip-model-download)
      SKIP_MODEL_DOWNLOAD=true
      shift
      ;;
    --offline)
      OFFLINE=true
      shift
      ;;
    --run-name)
      require_value "$@"
      RUN_NAME="$2"
      shift 2
      ;;
    --batch-size)
      require_value "$@"
      BATCH_SIZE="$2"
      shift 2
      ;;
    --max-length)
      require_value "$@"
      MAX_LENGTH="$2"
      shift 2
      ;;
    --gradient-accumulation-steps)
      require_value "$@"
      GRADIENT_ACCUMULATION="$2"
      shift 2
      ;;
    --detach)
      DETACH=true
      shift
      ;;
    --tmux-session)
      require_value "$@"
      TMUX_SESSION="$2"
      shift 2
      ;;
    --allow-cpu-full)
      ALLOW_CPU_FULL=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ "$(uname -s)" == "Linux" ]] || die "this script must run on a Linux server"
[[ "$ACTION" =~ ^(setup|validate|pilot|full)$ ]] || die "invalid action: $ACTION"
[[ "$DEVICE" =~ ^(auto|cuda|cpu)$ ]] || die "invalid device: $DEVICE"
[[ "$BRANCH" =~ ^[A-Za-z0-9._/-]+$ ]] || die "invalid branch name"
[[ "$VENV_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid venv name"
[[ "$TMUX_SESSION" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid tmux session name"
if [[ -n "$RUN_NAME" ]]; then
  [[ "$RUN_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid run name"
fi
for numeric in "$BATCH_SIZE" "$MAX_LENGTH" "$GRADIENT_ACCUMULATION"; do
  if [[ -n "$numeric" ]]; then
    [[ "$numeric" =~ ^[1-9][0-9]*$ ]] || die "training overrides must be positive integers"
  fi
done

REPO_DIR="${REPO_DIR/#\~/$HOME}"
HF_HOME_DIR="${HF_HOME_DIR/#\~/$HOME}"

run_apt() {
  local apt_prefix=()
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    command -v sudo >/dev/null 2>&1 || die "sudo is required to install system packages"
    apt_prefix=(sudo)
  fi
  log "Installing required Ubuntu packages"
  "${apt_prefix[@]}" apt-get update
  "${apt_prefix[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git ca-certificates build-essential python3-dev python3-pip python3-venv \
    tmux
}

if ! $SKIP_SYSTEM_PACKAGES; then
  command -v apt-get >/dev/null 2>&1 \
    || die "automatic system-package setup currently supports apt-based Linux"
  run_apt
fi

command -v git >/dev/null 2>&1 || die "git is missing"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BIN"

"$PYTHON_BIN" -c '
import sys
if sys.version_info < (3, 11):
    raise SystemExit(
        f"Python >=3.11 is required, found {sys.version.split()[0]}. "
        "On Ubuntu 22.04, install a separate Python 3.11+ interpreter and "
        "pass --python-bin."
    )
print("Python:", sys.version.split()[0])
'

if [[ -e "$REPO_DIR" && ! -d "$REPO_DIR/.git" ]]; then
  [[ -z "$(find "$REPO_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]] \
    || die "repo-dir exists and is not an empty Git checkout: $REPO_DIR"
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
  log "Cloning $BRANCH into $REPO_DIR"
  mkdir -p "$(dirname "$REPO_DIR")"
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$REPO_DIR"
elif ! $NO_UPDATE; then
  log "Updating existing checkout without rewriting local work"
  tracked_changes="$(git -C "$REPO_DIR" status --porcelain --untracked-files=no)"
  [[ -z "$tracked_changes" ]] \
    || die "tracked changes exist in $REPO_DIR; commit/stash them before setup"
  git -C "$REPO_DIR" fetch origin "$BRANCH"
  git -C "$REPO_DIR" switch "$BRANCH"
  git -C "$REPO_DIR" pull --ff-only origin "$BRANCH"
fi

[[ -f "$REPO_DIR/pyproject.toml" ]] || die "pyproject.toml is missing"
[[ -f "$REPO_DIR/$TRAINING_CONFIG" ]] || die "training config is missing"
[[ -f "$REPO_DIR/$DATA_RELEASE/manifest.json" ]] || die "data release is missing"

VENV_DIR="$REPO_DIR/$VENV_NAME"
VENV_PYTHON="$VENV_DIR/bin/python"

if ! $SKIP_INSTALL; then
  if [[ ! -x "$VENV_PYTHON" ]]; then
    log "Creating virtual environment: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  log "Installing final_absa dependencies"
  "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
  if [[ -n "$TORCH_INDEX_URL" ]]; then
    "$VENV_PYTHON" -m pip install \
      --index-url "$TORCH_INDEX_URL" \
      "torch>=2.4,<3"
  fi
  extras="ml"
  if $WITH_BROWSER; then
    extras="ml,browser"
  fi
  (
    cd "$REPO_DIR"
    "$VENV_PYTHON" -m pip install -e ".[$extras]"
  )
fi

[[ -x "$VENV_PYTHON" ]] || die "virtual environment is unavailable: $VENV_DIR"
mkdir -p "$HF_HOME_DIR"
export HF_HOME="$HF_HOME_DIR"
if $OFFLINE; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

log "Runtime inventory"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total \
    --format=csv,noheader || true
else
  warn "nvidia-smi is unavailable"
fi
df -h "$REPO_DIR" | tail -n 1
free -h || true

runtime_json="$("$VENV_PYTHON" -c '
import json
import numpy
import torch
import transformers
print(json.dumps({
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}))
')"
printf '%s\n' "$runtime_json"

CUDA_AVAILABLE="$("$VENV_PYTHON" -c 'import torch; print("1" if torch.cuda.is_available() else "0")')"
RESOLVED_DEVICE="$DEVICE"
if [[ "$RESOLVED_DEVICE" == "auto" ]]; then
  if [[ "$CUDA_AVAILABLE" == "1" ]]; then
    RESOLVED_DEVICE="cuda"
  else
    RESOLVED_DEVICE="cpu"
  fi
fi
if [[ "$RESOLVED_DEVICE" == "cuda" && "$CUDA_AVAILABLE" != "1" ]]; then
  die "CUDA was requested but torch.cuda.is_available() is false"
fi
if [[ "$ACTION" == "full" && "$RESOLVED_DEVICE" == "cpu" ]] && ! $ALLOW_CPU_FULL; then
  die "full CPU training is blocked; use a CUDA build/GPU or pass --allow-cpu-full"
fi

log "Validating immutable model-ready data"
(
  cd "$REPO_DIR"
  "$VENV_PYTHON" -X utf8 -m absa_system validate-data "$DATA_RELEASE"
)

if ! $SKIP_MODEL_DOWNLOAD; then
  log "Downloading or verifying vinai/phobert-base"
  local_only="False"
  if $OFFLINE; then
    local_only="True"
  fi
  "$VENV_PYTHON" -X utf8 -c "
from transformers import AutoModel
from absa_system.tokenization import load_offset_tokenizer
name = 'vinai/phobert-base'
local = $local_only
tokenizer = load_offset_tokenizer(name, local_files_only=local)
model = AutoModel.from_pretrained(name, local_files_only=local)
print({
    'model': name,
    'tokenizer_fast': tokenizer.is_fast,
    'hidden_size': model.config.hidden_size,
})
"
fi

if [[ "$ACTION" == "setup" || "$ACTION" == "validate" ]]; then
  log "Server action '$ACTION' completed successfully"
  exit 0
fi

timestamp="$(date -u '+%Y%m%dT%H%M%SZ')"
if [[ -z "$RUN_NAME" ]]; then
  if [[ "$ACTION" == "pilot" ]]; then
    RUN_NAME="absa_capacity_pilot_${timestamp}"
  else
    RUN_NAME="absa_phobert_full_${timestamp}"
  fi
fi
OUTPUT_DIR="$REPO_DIR/artifacts/models/$RUN_NAME"
LOG_DIR="$REPO_DIR/artifacts/server_logs"
LOG_FILE="$LOG_DIR/$RUN_NAME.log"
[[ ! -e "$OUTPUT_DIR" ]] || die "refusing to overwrite existing run: $OUTPUT_DIR"
mkdir -p "$LOG_DIR"

train_command=(
  "$VENV_PYTHON" -X utf8 -m absa_system train
  --data "$REPO_DIR/$DATA_RELEASE"
  --output "$OUTPUT_DIR"
  --config "$REPO_DIR/$TRAINING_CONFIG"
  --device "$RESOLVED_DEVICE"
)
if [[ "$ACTION" == "pilot" ]]; then
  train_command+=(
    --max-train-samples 1000
    --max-dev-samples 200
    --max-test-samples 200
    --max-epochs 1
  )
fi
if [[ -n "$BATCH_SIZE" ]]; then
  train_command+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "$MAX_LENGTH" ]]; then
  train_command+=(--max-length "$MAX_LENGTH")
fi
if [[ -n "$GRADIENT_ACCUMULATION" ]]; then
  train_command+=(--gradient-accumulation-steps "$GRADIENT_ACCUMULATION")
fi
validate_command=(
  "$VENV_PYTHON" -X utf8 -m absa_system validate-run "$OUTPUT_DIR"
)

printf -v quoted_train '%q ' "${train_command[@]}"
printf -v quoted_validate '%q ' "${validate_command[@]}"
printf -v quoted_repo '%q' "$REPO_DIR"
printf -v quoted_log '%q' "$LOG_FILE"
run_line="cd $quoted_repo && set -o pipefail && $quoted_train 2>&1 | tee $quoted_log && $quoted_validate"

if $DETACH; then
  command -v tmux >/dev/null 2>&1 || die "tmux is required for --detach"
  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    die "tmux session already exists: $TMUX_SESSION"
  fi
  log "Launching $ACTION in tmux session '$TMUX_SESSION'"
  tmux new-session -d -s "$TMUX_SESSION" "bash -lc $(printf '%q' "$run_line")"
  printf 'Attach: tmux attach -t %q\n' "$TMUX_SESSION"
  printf 'Log:    %s\n' "$LOG_FILE"
  printf 'Output: %s\n' "$OUTPUT_DIR"
  exit 0
fi

log "Starting foreground $ACTION run"
bash -lc "$run_line"
log "Training and sealed-run validation completed: $OUTPUT_DIR"
