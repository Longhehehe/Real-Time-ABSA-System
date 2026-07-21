#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"

if [[ -f "$SCRIPT_DIR/../configs/datasets.json" ]]; then
  DEFAULT_REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
else
  DEFAULT_REPO_DIR="$HOME/Real-Time-ABSA-System"
fi

REPO_DIR="$DEFAULT_REPO_DIR"
REPO_URL="https://github.com/Longhehehe/Real-Time-ABSA-System.git"
DATA_ARCHIVE=""
CONDA_ENV="absa"
PYTHON_VERSION="3.11"
TORCH_VERSION="2.12.1"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
TORCH_INDEX_EXPLICIT=false
CPU_ONLY=false
INSTALL_DRIVER=false
SKIP_SYSTEM_PACKAGES=false
SKIP_SMOKE=false
FULL_RUN=false
RUN_ONLY=false
TMUX_SESSION="absa-experiments"
STAGING_DIR=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_experiment_server.sh [options]

Setup options:
  --repo-dir PATH          Repository path (default: ~/Real-Time-ABSA-System)
  --repo-url URL           Git repository used when --repo-dir is absent
  --data-archive PATH      ZIP/RAR/7Z containing the "absa data" directory
  --conda-env NAME         Conda environment name (default: absa)
  --python-version VERSION Python version for a new env (default: 3.11)
  --torch-version VERSION  PyTorch version (default: 2.12.1)
  --torch-index-url URL    PyTorch wheel index (default: CUDA 12.6 index)
  --cpu                    Install/use CPU-only PyTorch
  --install-driver         Install recommended NVIDIA driver, then request reboot
  --skip-system-packages   Do not run apt-get
  --skip-smoke             Skip Logistic Regression and PhoBERT smoke tests

Execution options:
  --full-run               Launch the official 72-run matrix in tmux, then validate/publish
  --tmux-session NAME      tmux session name (default: absa-experiments)
  -h, --help               Show this help

Examples:
  bash scripts/setup_experiment_server.sh --data-archive ~/absa-data.zip
  bash scripts/setup_experiment_server.sh --data-archive ~/absa-data.zip --full-run

The script is idempotent: existing Conda envs, prepared data and compatible
experiment runs are reused. Full runs always use --resume.
EOF
}

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

warn() {
  printf '\nWARNING: %s\n' "$*" >&2
}

die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

cleanup() {
  if [[ -n "$STAGING_DIR" && -d "$STAGING_DIR" && "$STAGING_DIR" == "$HOME"/absa-data-staging.* ]]; then
    rm -rf -- "$STAGING_DIR"
  fi
}
trap cleanup EXIT

while (($#)); do
  case "$1" in
    --repo-dir)
      [[ $# -ge 2 ]] || die "--repo-dir requires a value"
      REPO_DIR="$2"
      shift 2
      ;;
    --repo-url)
      [[ $# -ge 2 ]] || die "--repo-url requires a value"
      REPO_URL="$2"
      shift 2
      ;;
    --data-archive)
      [[ $# -ge 2 ]] || die "--data-archive requires a value"
      DATA_ARCHIVE="$2"
      shift 2
      ;;
    --conda-env)
      [[ $# -ge 2 ]] || die "--conda-env requires a value"
      CONDA_ENV="$2"
      shift 2
      ;;
    --python-version)
      [[ $# -ge 2 ]] || die "--python-version requires a value"
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --torch-version)
      [[ $# -ge 2 ]] || die "--torch-version requires a value"
      TORCH_VERSION="$2"
      shift 2
      ;;
    --torch-index-url)
      [[ $# -ge 2 ]] || die "--torch-index-url requires a value"
      TORCH_INDEX_URL="$2"
      TORCH_INDEX_EXPLICIT=true
      shift 2
      ;;
    --cpu)
      CPU_ONLY=true
      shift
      ;;
    --install-driver)
      INSTALL_DRIVER=true
      shift
      ;;
    --skip-system-packages)
      SKIP_SYSTEM_PACKAGES=true
      shift
      ;;
    --skip-smoke)
      SKIP_SMOKE=true
      shift
      ;;
    --full-run)
      FULL_RUN=true
      shift
      ;;
    --tmux-session)
      [[ $# -ge 2 ]] || die "--tmux-session requires a value"
      TMUX_SESSION="$2"
      shift 2
      ;;
    --run-only)
      RUN_ONLY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

[[ "$CONDA_ENV" =~ ^[A-Za-z0-9._-]+$ ]] || die "Invalid Conda env name: $CONDA_ENV"
[[ "$PYTHON_VERSION" =~ ^[0-9]+\.[0-9]+([.][0-9]+)?$ ]] || die "Invalid Python version"
[[ "$TORCH_VERSION" =~ ^[0-9]+\.[0-9]+([.][0-9]+)?$ ]] || die "Invalid PyTorch version"
[[ "$TMUX_SESSION" =~ ^[A-Za-z0-9._-]+$ ]] || die "Invalid tmux session name"

if $CPU_ONLY && ! $TORCH_INDEX_EXPLICIT; then
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  die "This script must run on the Ubuntu server, not on the local Windows machine"
fi

REPO_DIR="${REPO_DIR/#\~/$HOME}"
if [[ -n "$DATA_ARCHIVE" ]]; then
  DATA_ARCHIVE="${DATA_ARCHIVE/#\~/$HOME}"
fi

find_conda_base() {
  if command -v conda >/dev/null 2>&1; then
    conda info --base
    return
  fi
  if [[ -x "$HOME/anaconda3/bin/conda" ]]; then
    printf '%s\n' "$HOME/anaconda3"
    return
  fi
  if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    printf '%s\n' "$HOME/miniconda3"
    return
  fi
  return 1
}

activate_conda_env() {
  local conda_base
  conda_base="$(find_conda_base)" || die "Conda was not found"
  # shellcheck disable=SC1091
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  unset PYTHONPATH || true
  hash -r
  PYTHON_BIN="$(command -v python)"
  [[ "$CONDA_DEFAULT_ENV" == "$CONDA_ENV" ]] || die "Failed to activate Conda env $CONDA_ENV"
}

verify_repo_branch() {
  [[ -d "$REPO_DIR/.git" ]] || die "Not a Git repository: $REPO_DIR"
  local branch
  branch="$(git -C "$REPO_DIR" branch --show-current)"
  [[ "$branch" == "experiments" ]] || die "Expected branch experiments, found $branch"
}

run_full_pipeline() {
  activate_conda_env
  verify_repo_branch
  cd "$REPO_DIR"

  mkdir -p logs .cache/huggingface artifacts/experiments
  export HF_HOME="$REPO_DIR/.cache/huggingface"
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false

  local device_args=()
  if $CPU_ONLY; then
    device_args=(--device cpu)
  fi

  log "Running the official 72-run matrix with resume"
  "$PYTHON_BIN" scripts/run_experiments.py \
    --profiles uit_visd4sa vlsp2018_hotel vlsp2018_restaurant vlsp2016 \
    --models logistic_regression naive_bayes bilstm cnn_bilstm phobert xlm_roberta \
    --seeds 42 52 62 \
    --resume \
    "${device_args[@]}" \
    2>&1 | tee logs/experiments_full.log

  log "Validating the complete artifact tree"
  "$PYTHON_BIN" scripts/validate_experiment_artifacts.py \
    2>&1 | tee logs/experiments_validate.log

  log "Publishing lightweight reports"
  "$PYTHON_BIN" scripts/publish_experiment_results.py \
    2>&1 | tee logs/experiments_publish.log

  log "All experiments completed, validated and published"
  printf 'Reports: %s\n' "$REPO_DIR/reports/experiments"
}

if $RUN_ONLY; then
  run_full_pipeline
  exit 0
fi

if [[ ! -r /etc/os-release ]]; then
  die "Unable to identify the Linux distribution"
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
  die "This bootstrap is tested for Ubuntu Server; detected ${PRETTY_NAME:-unknown}"
fi

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=()
else
  command -v sudo >/dev/null 2>&1 || die "sudo is required for system package installation"
  SUDO=(sudo)
fi

if ! $SKIP_SYSTEM_PACKAGES; then
  log "Installing Ubuntu system packages"
  "${SUDO[@]}" apt-get update
  "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git curl wget ca-certificates gnupg build-essential pkg-config \
    python3 python3-dev python3-pip python3-venv \
    libgomp1 jq zip unzip p7zip-full rsync tmux htop nvme-cli pciutils

  if ! "${SUDO[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y unrar; then
    warn "Package unrar is unavailable; RAR extraction will use 7z"
  fi
fi

command -v git >/dev/null 2>&1 || die "git is missing"
command -v curl >/dev/null 2>&1 || die "curl is missing"
command -v rsync >/dev/null 2>&1 || die "rsync is missing"
command -v tmux >/dev/null 2>&1 || die "tmux is missing"

if ! $CPU_ONLY; then
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    if $INSTALL_DRIVER; then
      log "Installing the recommended NVIDIA driver"
      "${SUDO[@]}" apt-get install -y ubuntu-drivers-common
      "${SUDO[@]}" ubuntu-drivers install --gpgpu
      printf '\nNVIDIA driver installation finished. Reboot the server, SSH again,\n'
      printf 'then rerun the same setup command without --install-driver.\n'
      exit 20
    fi
    die "nvidia-smi is unavailable. Rerun with --install-driver, or use --cpu"
  fi
  log "NVIDIA GPU detected"
  nvidia-smi
else
  warn "CPU-only mode selected; transformer training will be very slow"
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
  if [[ -e "$REPO_DIR" && -n "$(find "$REPO_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    die "Repository target exists and is not empty: $REPO_DIR"
  fi
  log "Cloning the experiments branch"
  mkdir -p "$(dirname -- "$REPO_DIR")"
  git clone --branch experiments --single-branch "$REPO_URL" "$REPO_DIR"
else
  log "Updating the experiments branch"
  if [[ -n "$(git -C "$REPO_DIR" status --porcelain --untracked-files=no)" ]]; then
    die "Repository has tracked local changes. Resolve them before running the bootstrap"
  fi
  git -C "$REPO_DIR" fetch origin experiments
  if [[ "$(git -C "$REPO_DIR" branch --show-current)" != "experiments" ]]; then
    git -C "$REPO_DIR" switch experiments
  fi
  git -C "$REPO_DIR" merge --ff-only origin/experiments
fi
verify_repo_branch

if ! CONDA_BASE="$(find_conda_base)"; then
  log "Installing Miniconda under $HOME/miniconda3"
  case "$(uname -m)" in
    x86_64) miniconda_arch="x86_64" ;;
    aarch64|arm64) miniconda_arch="aarch64" ;;
    *) die "Unsupported CPU architecture for Miniconda: $(uname -m)" ;;
  esac
  installer="$(mktemp --suffix=.sh)"
  curl -fsSLo "$installer" \
    "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${miniconda_arch}.sh"
  bash "$installer" -b -p "$HOME/miniconda3"
  rm -f -- "$installer"
  CONDA_BASE="$HOME/miniconda3"
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; then
  log "Creating Conda env $CONDA_ENV with Python $PYTHON_VERSION"
  conda create -n "$CONDA_ENV" "python=$PYTHON_VERSION" pip -y
fi
activate_conda_env

log "Python environment"
printf 'Python: %s\n' "$PYTHON_BIN"
"$PYTHON_BIN" --version
"$PYTHON_BIN" -m pip --version
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

torch_cuda_required="true"
if $CPU_ONLY; then
  torch_cuda_required="false"
fi

if ! "$PYTHON_BIN" - "$TORCH_VERSION" "$torch_cuda_required" <<'PY'
import sys

expected, cuda_required = sys.argv[1], sys.argv[2] == "true"
try:
    import torch
except Exception:
    raise SystemExit(1)

actual = torch.__version__.split("+")[0]
if actual != expected:
    raise SystemExit(1)
if cuda_required and not torch.cuda.is_available():
    raise SystemExit(1)
PY
then
  log "Installing PyTorch $TORCH_VERSION from $TORCH_INDEX_URL"
  "$PYTHON_BIN" -m pip install \
    "torch==$TORCH_VERSION" \
    --index-url "$TORCH_INDEX_URL"
else
  log "Compatible PyTorch installation already exists"
fi

log "Installing project training requirements"
"$PYTHON_BIN" -m pip install -r "$REPO_DIR/requirements_training.txt"
"$PYTHON_BIN" -m pip check

"$PYTHON_BIN" - <<'PY'
import sys
import torch

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("Torch file:", torch.__file__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

mkdir -p "$REPO_DIR/.experiment_cache/environment"
"$PYTHON_BIN" -m pip freeze \
  > "$REPO_DIR/.experiment_cache/environment/requirements-server.lock.txt"

if [[ -n "$DATA_ARCHIVE" ]]; then
  [[ -f "$DATA_ARCHIVE" ]] || die "Data archive not found: $DATA_ARCHIVE"
  log "Data archive checksum"
  sha256sum "$DATA_ARCHIVE"

  STAGING_DIR="$(mktemp -d "$HOME/absa-data-staging.XXXXXX")"
  case "${DATA_ARCHIVE,,}" in
    *.zip)
      unzip -n "$DATA_ARCHIVE" -d "$STAGING_DIR"
      ;;
    *.rar)
      if command -v unrar >/dev/null 2>&1; then
        unrar x -o- "$DATA_ARCHIVE" "$STAGING_DIR/"
      else
        7z x "$DATA_ARCHIVE" -o"$STAGING_DIR" -aos
      fi
      ;;
    *.7z)
      7z x "$DATA_ARCHIVE" -o"$STAGING_DIR" -aos
      ;;
    *)
      die "Unsupported data archive. Use ZIP, RAR or 7Z"
      ;;
  esac

  if [[ -d "$STAGING_DIR/absa data" ]]; then
    DATA_SOURCE="$STAGING_DIR/absa data"
  elif [[ -d "$STAGING_DIR/UIT-ViSD4SA-main" \
       && -d "$STAGING_DIR/absa-vlsp-2018-main" \
       && -d "$STAGING_DIR/vlsp2016" ]]; then
    DATA_SOURCE="$STAGING_DIR"
  else
    find "$STAGING_DIR" -maxdepth 4 -type d -print >&2
    die "Archive does not contain the expected absa data layout"
  fi

  log "Copying raw datasets into the repository"
  mkdir -p "$REPO_DIR/absa data"
  rsync -a --info=progress2 "$DATA_SOURCE/" "$REPO_DIR/absa data/"
fi

git -C "$REPO_DIR" check-ignore -q "absa data" \
  || die "Raw data directory is not ignored by Git"

log "Validating all raw paths from configs/datasets.json"
REPO_DIR_ENV="$REPO_DIR" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["REPO_DIR_ENV"])
registry = json.loads((root / "configs/datasets.json").read_text(encoding="utf-8"))
missing = []
for profile_id, profile in registry["profiles"].items():
    for split, relative in profile["splits"].items():
        path = root / relative
        if not path.is_file():
            missing.append(f"{profile_id}/{split}: {path}")

if missing:
    print("Missing raw files:")
    print("\n".join(f"- {item}" for item in missing))
    raise SystemExit(1)
print("OK: every configured raw split exists")
PY

cd "$REPO_DIR"
mkdir -p .cache/huggingface logs
export HF_HOME="$REPO_DIR/.cache/huggingface"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

log "Preparing and auditing all four profiles"
"$PYTHON_BIN" scripts/prepare_experiment_data.py --profiles all

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

profiles = ["uit_visd4sa", "vlsp2018_hotel", "vlsp2018_restaurant", "vlsp2016"]
for profile in profiles:
    path = Path(".experiment_cache/processed") / profile / "dataset_audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    overlaps = audit["clean_overlaps"]
    assert overlaps["train_dev_unique_texts"] == 0, profile
    assert overlaps["train_test_unique_texts"] == 0, profile
    print(profile, "clean_train=", audit["clean_train_rows"], "overlap=0")
print("OK: exact-text train leakage is zero")
PY

log "Running automated experiment checks"
"$PYTHON_BIN" -m unittest discover -s tests -p "test_experiments.py" -v
"$PYTHON_BIN" -m compileall experiments scripts methods app/absa_predictor.py

log "Checking the 72-run matrix"
"$PYTHON_BIN" scripts/run_experiments.py --dry-run \
  > .experiment_cache/dry_run.json
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

data = json.loads(Path(".experiment_cache/dry_run.json").read_text(encoding="utf-8"))
assert data["n_runs"] == 72, data["n_runs"]
print("OK: dry-run contains 72 runs")
PY

device_args=()
if $CPU_ONLY; then
  device_args=(--device cpu)
fi

if ! $SKIP_SMOKE; then
  log "Smoke testing Logistic Regression on all profiles"
  "$PYTHON_BIN" scripts/run_experiments.py \
    --profiles all \
    --models logistic_regression \
    --seeds 42 \
    --smoke \
    --fail-fast \
    --artifact-root .experiment_cache/smoke_artifacts \
    "${device_args[@]}"

  log "Smoke testing PhoBERT on all profiles"
  "$PYTHON_BIN" scripts/run_experiments.py \
    --profiles all \
    --models phobert \
    --seeds 42 \
    --smoke \
    --fail-fast \
    --artifact-root .experiment_cache/smoke_artifacts \
    "${device_args[@]}"
fi

available_kb="$(df -Pk "$REPO_DIR" | awk 'NR == 2 {print $4}')"
required_kb=$((40 * 1024 * 1024))
if ((available_kb < required_kb)); then
  if $FULL_RUN; then
    die "Less than 40 GB is free; full experiments were not started"
  fi
  warn "Less than 40 GB is free; reserve more space before --full-run"
fi

if $FULL_RUN; then
  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    die "tmux session $TMUX_SESSION already exists; attach with: tmux attach -t $TMUX_SESSION"
  fi

  runner_script="$REPO_DIR/scripts/setup_experiment_server.sh"
  [[ -f "$runner_script" ]] || runner_script="$SCRIPT_PATH"
  runner_args=(
    --run-only
    --repo-dir "$REPO_DIR"
    --conda-env "$CONDA_ENV"
    --tmux-session "$TMUX_SESSION"
  )
  if $CPU_ONLY; then
    runner_args+=(--cpu)
  fi
  printf -v tmux_command '%q ' bash "$runner_script" "${runner_args[@]}"

  log "Launching full experiments in detached tmux session $TMUX_SESSION"
  tmux new-session -d -s "$TMUX_SESSION" "$tmux_command"
  printf 'Attach:  tmux attach -t %s\n' "$TMUX_SESSION"
  printf 'Log:     tail -f %q\n' "$REPO_DIR/logs/experiments_full.log"
else
  log "Setup, data audit and smoke tests completed"
  printf 'Start the full run later with:\n'
  printf '  bash %q --full-run --skip-system-packages --data-archive %q\n' \
    "$REPO_DIR/scripts/setup_experiment_server.sh" "${DATA_ARCHIVE:-$HOME/absa-data.zip}"
fi
